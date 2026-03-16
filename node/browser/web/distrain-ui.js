/**
 * Distrain Node — Shared UI Logic
 * Used by: browser (WASM), desktop (Tauri), iOS, Android
 *
 * Each platform provides a `DistrainAdapter` that implements:
 *   - startTraining()        → Promise<string>       Start training, return status message
 *   - stopTraining()         → Promise<string>       Stop training, return status message
 *   - getStatus()            → Promise<NodeStatus>   { is_training, connected, node_id, gpu_info, h_mini }
 *   - getStats()             → Promise<TrainingStats> { current_step, total_steps, current_loss, ... }
 *   - getLogs()              → Promise<string[]>     New log messages since last poll
 *   - platformName           → string                "browser", "desktop", "ios", "android"
 */

// Globals
let adapter = null;
let lossHistory = [];
let pollInterval = null;
let trainingStartedAt = 0; // client-side timestamp for smooth elapsed display

// --- Helpers ---
function $(id) { return document.getElementById(id); }

function log(msg) {
  const el = $('log');
  if (!el) return;
  const time = new Date().toLocaleTimeString();
  el.textContent += `\n[${time}] ${msg}`;
  el.scrollTop = el.scrollHeight;
  // Trim old lines
  const lines = el.textContent.split('\n');
  if (lines.length > 500) {
    el.textContent = lines.slice(-300).join('\n');
  }
}

function formatTokens(n) {
  if (n < 1000) return `${n}`;
  if (n < 1e6) return `${(n/1000).toFixed(1)}K`;
  if (n < 1e9) return `${(n/1e6).toFixed(1)}M`;
  return `${(n/1e9).toFixed(2)}B`;
}

function updateChart() {
  const chart = $('loss-chart');
  if (!chart) return;
  chart.innerHTML = '';
  if (lossHistory.length === 0) return;
  const maxLoss = Math.max(...lossHistory);
  if (maxLoss === 0) return;
  const w = chart.clientWidth;
  const barWidth = Math.max(2, w / Math.max(lossHistory.length, 1));
  lossHistory.forEach((loss, i) => {
    const bar = document.createElement('div');
    bar.className = 'loss-bar';
    bar.style.left = `${i * barWidth}px`;
    bar.style.width = `${barWidth - 1}px`;
    bar.style.height = `${(loss / maxLoss) * 100}%`;
    chart.appendChild(bar);
  });
}

function setStatus(connected) {
  const dot = $('status-dot');
  const text = $('status-text');
  if (dot) dot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
  if (text) text.textContent = connected ? 'Connected' : 'Not connected';
}

// --- Actions ---
let isTraining = false;

function updateToggleButton() {
  const btn = $('btn-toggle');
  if (!btn) return;
  if (isTraining) {
    btn.textContent = 'Stop Training';
    btn.classList.add('running');
  } else {
    btn.textContent = 'Start Training';
    btn.classList.remove('running');
  }
  btn.disabled = false;
}

window.distrainToggle = async function() {
  if (!adapter) return;
  const btn = $('btn-toggle');
  if (isTraining) {
    if (!confirm('Stop training? Current round progress will be lost.')) return;
    btn.textContent = 'Stopping...';
    btn.disabled = true;
    try {
      const msg = await adapter.stopTraining();
      log(msg);
      isTraining = false;
    } catch (e) {
      log(`Failed to stop: ${e}`);
    }
    updateToggleButton();
  } else {
    btn.textContent = 'Starting...';
    btn.disabled = true;
    try {
      const msg = await adapter.startTraining();
      log(msg);
      isTraining = true;
      trainingStartedAt = Date.now() / 1000;
    } catch (e) {
      log(`Failed to start: ${e}`);
    }
    updateToggleButton();
  }
};

// --- Polling ---
async function refreshStatus() {
  if (!adapter) return;
  try {
    const status = await adapter.getStatus();
    setStatus(status.connected);
    if (status.is_training !== isTraining) {
      isTraining = status.is_training;
      updateToggleButton();
    }
  } catch (e) {}
}

function formatTime(secs) {
  if (!secs || secs <= 0) return '--';
  if (secs < 60) return `${Math.floor(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

async function refreshStats() {
  if (!adapter) return;
  try {
    const s = await adapter.getStats();
    $('step').textContent = s.current_step || s.total_steps || 0;
    $('loss').textContent = s.current_loss > 0 ? s.current_loss.toFixed(4) : '--';
    $('tokens').textContent = formatTokens(s.tokens_processed || 0);
    $('tps').textContent = s.tokens_per_sec > 0 ? formatTokens(Math.floor(s.tokens_per_sec)) : '--';
    $('round').textContent = s.rounds_completed || 0;
    $('ckpt-version').textContent = s.checkpoint_version || 0;
    $('push-count').textContent = s.rounds_completed || 0;
    // Use client-side timer for smooth ticking; fall back to backend value
    const clientElapsed = isTraining && trainingStartedAt > 0
      ? (Date.now() / 1000) - trainingStartedAt
      : s.elapsed_secs;
    $('elapsed').textContent = formatTime(clientElapsed || s.elapsed_secs);

    if (s.current_loss > 0) {
      lossHistory.push(s.current_loss);
      if (lossHistory.length > 500) lossHistory = lossHistory.slice(-300);
      updateChart();
    }
  } catch (e) {}
}

async function refreshLogs() {
  if (!adapter || !adapter.getLogs) return;
  try {
    const logs = await adapter.getLogs();
    for (const msg of logs) {
      log(msg);
    }
  } catch (e) {}
}

function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    refreshStatus();
    refreshStats();
    refreshLogs();
  }, 1000);
}

// --- Settings ---
window.distrainToggleSettings = function() {
  const panel = $('settings-panel');
  const btn = $('settings-toggle');
  if (!panel || !btn) return;
  if (panel.classList.contains('hidden')) {
    panel.classList.remove('hidden');
    btn.textContent = '\u2212';
    loadSettings();
  } else {
    panel.classList.add('hidden');
    btn.textContent = '+';
  }
};

async function loadSettings() {
  if (!adapter || !adapter.getTrainingParams) return;
  try {
    const p = await adapter.getTrainingParams();
    if ($('cfg-lr-max')) $('cfg-lr-max').value = p.lr_max;
    if ($('cfg-lr-min')) $('cfg-lr-min').value = p.lr_min;
    if ($('cfg-warmup')) $('cfg-warmup').value = Math.round((p.warmup_fraction || 0.2) * 100);
    if ($('cfg-grad-clip')) $('cfg-grad-clip').value = p.grad_clip_norm;
    if ($('cfg-weight-decay')) $('cfg-weight-decay').value = p.weight_decay;
    if ($('cfg-batch-size')) $('cfg-batch-size').value = p.batch_size;
    if ($('cfg-seq-len')) $('cfg-seq-len').value = p.seq_len;
    if ($('cfg-shards')) $('cfg-shards').value = Math.round((p.shards_fraction || 0.2) * 100);
  } catch (e) {
    log(`Failed to load settings: ${e}`);
  }
}

window.distrainSaveSettings = async function() {
  if (!adapter || !adapter.saveTrainingParams) return;
  const params = {
    lr_max: parseFloat($('cfg-lr-max').value),
    lr_min: parseFloat($('cfg-lr-min').value),
    warmup_fraction: (parseInt($('cfg-warmup').value) || 20) / 100,
    grad_clip_norm: parseFloat($('cfg-grad-clip').value),
    weight_decay: parseFloat($('cfg-weight-decay').value),
    batch_size: parseInt($('cfg-batch-size').value) || 4,
    seq_len: parseInt($('cfg-seq-len').value) || 512,
    shards_fraction: (parseInt($('cfg-shards').value) || 20) / 100,
  };
  try {
    await adapter.saveTrainingParams(params);
    const hint = $('settings-hint');
    if (hint) { hint.textContent = 'Saved'; setTimeout(() => hint.textContent = 'Defaults from coordinator', 2000); }
    log('Training params saved');
  } catch (e) {
    log(`Failed to save settings: ${e}`);
  }
};

// --- Init ---
window.distrainInit = function(platformAdapter) {
  adapter = platformAdapter;
  const sub = $('subtitle');
  if (sub && adapter.platformName) {
    const names = {
      browser: 'Browser-based distributed training via WebAssembly + Burn',
      desktop: 'Desktop distributed training via wgpu (Metal/Vulkan) + Burn',
      ios: 'iOS distributed training via Metal + Burn',
      android: 'Android distributed training via Vulkan + Burn',
    };
    sub.textContent = names[adapter.platformName] || `Distributed training (${adapter.platformName})`;
  }
  log(`Platform: ${adapter.platformName || 'unknown'}`);
  refreshStatus();
  startPolling();
};
