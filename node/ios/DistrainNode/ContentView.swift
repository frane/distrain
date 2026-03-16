import SwiftUI

struct ContentView: View {
    @StateObject private var manager = TrainingManager()

    @State private var lossHistory: [Double] = []
    @State private var showSettings = false

    private let accent = Color(red: 99.0/255, green: 102.0/255, blue: 241.0/255)

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                header
                coordinatorCard
                trainingCard
                lossChartCard
                settingsCard
                logsCard
                if let error = manager.errorMessage {
                    errorCard(error)
                }
            }
            .padding()
        }
        .background(Color(red: 15.0/255, green: 17.0/255, blue: 23.0/255))
        .onChange(of: manager.currentLoss) { newLoss in
            if newLoss > 0 {
                lossHistory.append(newLoss)
                if lossHistory.count > 500 {
                    lossHistory = Array(lossHistory.suffix(300))
                }
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Distrain Node")
                .font(.title2.bold())
                .foregroundColor(.white)
            Text(backendSubtitle)
                .font(.caption)
                .foregroundColor(Color(white: 0.44))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 8)
    }

    // MARK: - Coordinator + Start/Stop

    private var coordinatorCard: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                Text("Coordinator")
                    .font(.caption)
                    .foregroundColor(Color(white: 0.44))
                TextField("URL", text: $manager.coordinatorUrl)
                    .font(.system(.caption, design: .monospaced))
                    .textFieldStyle(.plain)
                    .padding(6)
                    .background(Color(red: 15.0/255, green: 17.0/255, blue: 23.0/255))
                    .cornerRadius(6)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(white: 0.16), lineWidth: 1))
                    .foregroundColor(.white)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .disabled(manager.isRunning)
            }

            HStack {
                HStack(spacing: 6) {
                    Circle()
                        .fill(statusDotColor)
                        .frame(width: 8, height: 8)
                    Text(statusText)
                        .font(.caption2)
                        .foregroundColor(Color(white: 0.44))
                }
                Spacer()
                Button(action: {
                    if manager.isRunning {
                        manager.stop()
                    } else {
                        lossHistory.removeAll()
                        manager.start()
                    }
                }) {
                    Text(manager.isRunning ? "Stop Training" : "Start Training")
                        .font(.caption.bold())
                        .padding(.horizontal, 12)
                        .padding(.vertical, 5)
                        .background(manager.isRunning
                            ? Color.red.opacity(0.15)
                            : accent)
                        .foregroundColor(manager.isRunning ? .red : .white)
                        .cornerRadius(6)
                        .overlay(manager.isRunning
                            ? RoundedRectangle(cornerRadius: 6).stroke(Color.red.opacity(0.3), lineWidth: 1)
                            : nil)
                }
            }
        }
        .card()
    }

    private var statusDotColor: Color {
        switch manager.phase {
        case .training, .computingDelta, .uploadingDelta, .pushingMeta:
            return Color(red: 34.0/255, green: 197.0/255, blue: 94.0/255) // green
        case .idle:
            return Color(red: 239.0/255, green: 68.0/255, blue: 68.0/255) // red
        default:
            return Color(red: 245.0/255, green: 158.0/255, blue: 11.0/255) // amber
        }
    }

    private var statusText: String {
        if !manager.isRunning && manager.round == 0 { return "Not connected" }
        if manager.isRunning { return "Connected \u{2014} \(manager.phase.rawValue)" }
        return "Disconnected"
    }

    // MARK: - Training Stats (2x4 grid)

    private var trainingCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("TRAINING")
                .font(.caption2)
                .foregroundColor(Color(white: 0.44))
                .tracking(0.8)

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 12) {
                stat(label: "Step", value: "\(manager.currentStep)")
                stat(label: "Loss", value: manager.currentLoss > 0
                    ? String(format: "%.4f", manager.currentLoss) : "--")
                stat(label: "Tokens", value: formatTokens(manager.totalTokens))
                stat(label: "Tok/s", value: manager.tokensPerSec > 0
                    ? formatTokens(UInt64(manager.tokensPerSec)) : "--")
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 12) {
                stat(label: "Round", value: "\(manager.round)")
                stat(label: "Checkpoint", value: "\(manager.checkpointVersion)")
                stat(label: "Pushes", value: "\(manager.round)")
                stat(label: "Time", value: formatTime(manager.elapsedSecs))
            }
        }
        .card()
    }

    private func stat(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(Color(white: 0.44))
                .textCase(.uppercase)
            Text(value)
                .font(.system(size: 18, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)
        }
    }

    // MARK: - Loss Chart

    private var lossChartCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("LOSS HISTORY")
                .font(.caption2)
                .foregroundColor(Color(white: 0.44))
                .tracking(0.8)

            LossChartView(data: lossHistory, accentColor: accent)
                .frame(height: 120)
                .background(Color(red: 15.0/255, green: 17.0/255, blue: 23.0/255))
                .cornerRadius(6)
        }
        .card()
    }

    // MARK: - Settings (collapsible)

    private var settingsCard: some View {
        VStack(spacing: 0) {
            HStack {
                Text("SETTINGS")
                    .font(.caption2)
                    .foregroundColor(Color(white: 0.44))
                    .tracking(0.8)
                Spacer()
                Button(action: { withAnimation { showSettings.toggle() } }) {
                    Text(showSettings ? "\u{2212}" : "+")
                        .font(.system(size: 14, design: .monospaced))
                        .frame(width: 24, height: 24)
                        .foregroundColor(Color(white: 0.44))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color(white: 0.16), lineWidth: 1))
                }
            }

            if showSettings {
                VStack(spacing: 6) {
                    settingsRow(label: "LR max", text: binding({ manager.lrMax }, { manager.lrMax = $0 }, format: "%.2e"))
                    settingsRow(label: "LR min", text: binding({ manager.lrMin }, { manager.lrMin = $0 }, format: "%.2e"))
                    settingsRow(label: "Warmup %", text: binding({ manager.warmupPct }, { manager.warmupPct = $0 }, format: "%.0f"))
                    settingsRow(label: "Grad clip", text: binding({ manager.gradClipNorm }, { manager.gradClipNorm = $0 }, format: "%.1f"))
                    settingsRow(label: "Weight decay", text: binding({ manager.weightDecay }, { manager.weightDecay = $0 }, format: "%.2f"))
                    settingsRow(label: "Batch size", text: binding({ manager.batchSize }, { manager.batchSize = $0 }, format: "%.0f"))
                    settingsRow(label: "Seq length", text: binding({ manager.seqLen }, { manager.seqLen = $0 }, format: "%.0f"))
                    settingsRow(label: "Shards %", text: binding({ manager.shardsPct }, { manager.shardsPct = $0 }, format: "%.0f"))
                }
                .padding(.top, 10)

                HStack {
                    Text("Defaults from coordinator")
                        .font(.caption2)
                        .foregroundColor(Color(white: 0.44))
                    Spacer()
                }
                .padding(.top, 8)
            }
        }
        .card()
    }

    // Settings binding helper
    private func binding(_ get: @escaping () -> Double, _ set: @escaping (Double) -> Void, format: String) -> Binding<String> {
        Binding(
            get: { String(format: format, get()) },
            set: { if let v = Double($0) { set(v) } }
        )
    }

    private func settingsRow(label: String, text: Binding<String>) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2)
                .foregroundColor(Color(white: 0.44))
                .frame(width: 80, alignment: .leading)
            TextField("", text: text)
                .font(.system(.caption, design: .monospaced))
                .textFieldStyle(.plain)
                .padding(4)
                .background(Color(red: 15.0/255, green: 17.0/255, blue: 23.0/255))
                .cornerRadius(4)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color(white: 0.16), lineWidth: 1))
                .foregroundColor(.white)
                .frame(width: 80)
                .disabled(manager.isRunning)
        }
    }

    // MARK: - Logs

    private var logsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("LOGS")
                .font(.caption2)
                .foregroundColor(Color(white: 0.44))
                .tracking(0.8)

            ScrollViewReader { proxy in
                ScrollView {
                    Text(manager.logText)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(Color(white: 0.44))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("logBottom")
                }
                .frame(maxHeight: 200)
                .background(Color(red: 15.0/255, green: 17.0/255, blue: 23.0/255))
                .cornerRadius(6)
                .onChange(of: manager.logText) { _ in
                    proxy.scrollTo("logBottom", anchor: .bottom)
                }
            }
        }
        .card()
    }

    // MARK: - Error

    private func errorCard(_ error: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(error)
                .font(.caption)
                .foregroundColor(.red)
            Spacer()
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(10)
    }

    private var backendSubtitle: String {
        switch manager.gpuActive {
        case .none: return "Distributed training via Burn"
        case .some(true): return "Distributed training via Metal GPU + Burn"
        case .some(false): return "Distributed training via CPU + Burn (GPU incompatible)"
        }
    }

    // MARK: - Formatters

    private func formatTokens(_ n: UInt64) -> String {
        if n >= 1_000_000_000 { return String(format: "%.2fB", Double(n) / 1e9) }
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1e6) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1e3) }
        return "\(n)"
    }

    private func formatTime(_ secs: Double) -> String {
        if secs <= 0 { return "--" }
        if secs < 60 { return "\(Int(secs))s" }
        if secs < 3600 { return "\(Int(secs / 60))m \(Int(secs.truncatingRemainder(dividingBy: 60)))s" }
        return "\(Int(secs / 3600))h \(Int((secs.truncatingRemainder(dividingBy: 3600)) / 60))m"
    }
}

// MARK: - Card modifier

extension View {
    func card() -> some View {
        self
            .padding(16)
            .background(Color(red: 26.0/255, green: 29.0/255, blue: 39.0/255))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(white: 0.16), lineWidth: 1))
            .cornerRadius(10)
    }
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}
