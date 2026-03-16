// Distrain Desktop — Frontend logic
// Communicates with Tauri Rust backend via IPC commands.

const { invoke } = window.__TAURI__.core;

// --- State ---
let pollInterval = null;

// --- UI Helpers ---
function $(id) { return document.getElementById(id); }

function addLog(message, level = "info") {
  const container = $("log-container");
  const entry = document.createElement("div");
  entry.className = `log-entry ${level}`;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${message}`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
  // Keep last 200 entries
  while (container.children.length > 200) {
    container.removeChild(container.firstChild);
  }
}

function setBadge(state) {
  const badge = $("status-badge");
  badge.className = `badge ${state}`;
  badge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function formatBytes(bytes) {
  if (bytes === 0) return "--";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function formatTokens(n) {
  if (n === 0) return "0";
  if (n < 1000) return `${n}`;
  if (n < 1000000) return `${(n / 1000).toFixed(1)}K`;
  if (n < 1000000000) return `${(n / 1000000).toFixed(1)}M`;
  return `${(n / 1000000000).toFixed(2)}B`;
}

function formatTime(secs) {
  if (secs < 60) return `${Math.floor(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h ${m}m`;
}

// --- Commands ---
async function startTraining() {
  try {
    const result = await invoke("start_training");
    addLog(result, "success");
    setBadge("training");
    $("btn-start").disabled = true;
    $("btn-stop").disabled = false;
    $("btn-calibrate").disabled = true;
    startPolling();
  } catch (e) {
    addLog(`Failed to start: ${e}`, "error");
  }
}

async function stopTraining() {
  try {
    const result = await invoke("stop_training");
    addLog(result, "warning");
    setBadge("idle");
    $("btn-start").disabled = false;
    $("btn-stop").disabled = true;
    $("btn-calibrate").disabled = false;
    stopPolling();
  } catch (e) {
    addLog(`Failed to stop: ${e}`, "error");
  }
}

async function calibrateDevice() {
  setBadge("calibrating");
  $("btn-calibrate").disabled = true;
  addLog("Calibrating device...");
  try {
    const hMini = await invoke("calibrate_device");
    $("h-mini").textContent = hMini;
    addLog(`Calibration complete: H_mini = ${hMini}`, "success");
    setBadge("idle");
  } catch (e) {
    addLog(`Calibration failed: ${e}`, "error");
    setBadge("error");
  }
  $("btn-calibrate").disabled = false;
}

async function refreshStatus() {
  try {
    const status = await invoke("get_node_status");
    $("node-status").textContent = status.is_training ? "Training" : "Idle";
    $("gpu-info").textContent = status.gpu_info;
    if (status.h_mini) $("h-mini").textContent = status.h_mini;
  } catch (e) {
    // Silently retry
  }
}

async function refreshStats() {
  try {
    const stats = await invoke("get_training_stats");
    $("current-loss").textContent = stats.current_loss > 0 ? stats.current_loss.toFixed(4) : "--";
    $("current-step").textContent = `${stats.current_step} / ${stats.total_steps}`;
    $("tokens-per-sec").textContent = stats.tokens_per_sec > 0 ? stats.tokens_per_sec.toFixed(0) : "--";
    $("tokens-processed").textContent = formatTokens(stats.tokens_processed);
    $("checkpoint-version").textContent = `v${stats.checkpoint_version}`;
    $("rounds-completed").textContent = stats.rounds_completed;
    $("delta-size").textContent = formatBytes(stats.delta_upload_size_bytes);
    $("compression-ratio").textContent = stats.compression_ratio > 0 ? `${stats.compression_ratio.toFixed(1)}x` : "--";
    $("elapsed").textContent = formatTime(stats.elapsed_secs);

    if (stats.total_steps > 0) {
      const pct = (stats.current_step / stats.total_steps) * 100;
      $("progress-fill").style.width = `${pct}%`;
    }
  } catch (e) {
    // Silently retry
  }
}

async function refreshLogs() {
  try {
    const logs = await invoke("get_logs");
    for (const msg of logs) {
      addLog(msg);
    }
  } catch (e) {
    // Silently retry
  }
}

function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    refreshStatus();
    refreshStats();
    refreshLogs();
  }, 1000);
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

function toggleConfig() {
  const panel = $("config-panel");
  const toggle = $("config-toggle");
  if (panel.classList.contains("hidden")) {
    panel.classList.remove("hidden");
    toggle.textContent = "-";
  } else {
    panel.classList.add("hidden");
    toggle.textContent = "+";
  }
}

async function saveConfig() {
  const config = {
    coordinator_url: $("coordinator-url").value,
    api_key: $("api-key").value,
    cache_dir: $("cache-dir").value,
    min_inner_steps: parseInt($("min-h").value) || 100,
    max_inner_steps: parseInt($("max-h").value) || 1000,
    target_push_interval_secs: 300.0,
  };
  try {
    const result = await invoke("save_config", { config });
    addLog(result, "success");
  } catch (e) {
    addLog(`Failed to save config: ${e}`, "error");
  }
}

async function loadConfig() {
  try {
    const config = await invoke("get_config");
    $("coordinator-url").value = config.coordinator_url;
    $("api-key").value = config.api_key;
    $("cache-dir").value = config.cache_dir;
    $("min-h").value = config.min_inner_steps;
    $("max-h").value = config.max_inner_steps;
  } catch (e) {
    // Use defaults
  }
}

// --- Init ---
window.addEventListener("DOMContentLoaded", () => {
  refreshStatus();
  loadConfig();
  startPolling();
});
