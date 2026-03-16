import Foundation
import Combine

/// Protocol phase indicator for the UI.
public enum ProtocolPhase: String {
    case idle = "Idle"
    case registering = "Registering"
    case downloadingCheckpoint = "Downloading Checkpoint"
    case loadingCheckpoint = "Loading Checkpoint"
    case downloadingShards = "Downloading Shards"
    case training = "Training"
    case computingDelta = "Computing Delta"
    case uploadingDelta = "Uploading Delta"
    case pushingMeta = "Pushing Metadata"
    case waitingForCheckpoint = "Waiting for Checkpoint"
}

/// Manages the full distributed training protocol on iOS.
///
/// Protocol loop (mirrors CLI `run_training_loop`):
///   register → calibrate → download manifest → loop {
///     get latest checkpoint → download checkpoint → load → shard assignment →
///     download shards → snapshot params → train H_mini steps →
///     compute delta → upload delta → push metadata → poll for new checkpoint
///   }
///
/// Swift handles all networking (coordinator API + R2 via coordinator proxy).
/// Rust FFI handles all compute (model init, training, delta compression).
@MainActor
public class TrainingManager: ObservableObject {

    // MARK: - Published State

    @Published public var phase: ProtocolPhase = .idle
    @Published public var isRunning = false
    @Published public var currentStep: UInt64 = 0
    @Published public var currentLoss: Double = 0
    @Published public var totalTokens: UInt64 = 0
    @Published public var tokensPerSec: Double = 0
    @Published public var hMini: UInt64 = 0
    @Published public var round: UInt64 = 0
    @Published public var checkpointVersion: UInt64 = 0
    @Published public var gpuActive: Bool?  // nil = not probed yet
    @Published public var errorMessage: String?
    @Published public var statusDetail: String = ""
    @Published public var elapsedSecs: Double = 0
    @Published public var logText: String = "Ready. Click \"Start Training\" to begin."

    // MARK: - Settings (exposed for UI binding)

    @Published public var coordinatorUrl: String = "http://localhost:8000"
    @Published public var lrMax: Double = 3e-4
    @Published public var lrMin: Double = 1e-6
    @Published public var warmupPct: Double = 20
    @Published public var gradClipNorm: Double = 1.0
    @Published public var weightDecay: Double = 0.1
    // iPad defaults — conservative for devices with limited shared GPU/CPU memory.
    // seq_len gets clamped to model's max_seq_len after checkpoint load.
    @Published public var batchSize: Double = 1
    @Published public var seqLen: Double = 32
    @Published public var shardsPct: Double = 20

    // MARK: - Private

    private var protocolTask: Task<Void, Never>?
    private var protocolStartTime: Date?
    private var nodeId: String = ""
    private var apiKey: String = ""
    private var seqNum: UInt64 = 0
    /// Steps per training round. Kept low for iPads to push deltas frequently
    /// and avoid long GPU occupancy on memory-constrained devices.
    private static let stepsPerRound: UInt64 = 10

    /// Manifest data (cached after first download).
    private var manifest: [[String: Any]]?
    private var totalShards: UInt32 = 0

    private var shardsDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("shards")
    }

    private var deltasDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("deltas")
    }

    // MARK: - Public API

    public init() {}

    /// Start the full distributed training protocol.
    public func start() {
        guard !isRunning else { return }
        isRunning = true
        errorMessage = nil
        round = 0
        elapsedSecs = 0
        protocolStartTime = Date()

        protocolTask = Task.detached { [weak self] in
            do {
                try await self?.runProtocol()
            } catch is CancellationError {
                // Normal cancellation
            } catch {
                await MainActor.run {
                    self?.errorMessage = error.localizedDescription
                    self?.isRunning = false
                    self?.phase = .idle
                }
            }
        }
    }

    /// Stop the protocol loop.
    public func stop() {
        protocolTask?.cancel()
        isRunning = false
        phase = .idle
        statusDetail = "Stopped"
    }

    /// Shut down engine and reset state.
    public func shutdown() {
        stop()
        DistrainBridge.shutdown()
        nodeId = ""
        apiKey = ""
        seqNum = 0
        round = 0
        checkpointVersion = 0
        currentStep = 0
        currentLoss = 0
        totalTokens = 0
        manifest = nil
        statusDetail = ""
    }

    deinit {
        DistrainBridge.shutdown()
    }

    // MARK: - Protocol Loop

    private var baseUrl: String {
        coordinatorUrl.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func runProtocol() async throws {
        // 0. Detect GPU capability before any Rust GPU calls
        let gpuOk = DistrainBridge.gpuSupported()
        distrain_set_backend(gpuOk ? 1 : 0)
        await MainActor.run { [weak self] in self?.gpuActive = gpuOk }
        await appendLog("Backend: \(gpuOk ? "GPU (Metal)" : "CPU (GPU not supported on this device)")")

        // 1. Register
        await setPhase(.registering, detail: "Registering with coordinator...")
        let reg = try await register()
        nodeId = reg.nodeId
        apiKey = reg.apiKey
        if let params = reg.trainingParams {
            // Apply coordinator params (LR, shards, warmup) — keep local batch/seq (mobile defaults)
            await MainActor.run { [weak self] in
                self?.lrMax = params.lrMax
                self?.lrMin = params.lrMin
                self?.warmupPct = params.warmupFraction * 100
                self?.gradClipNorm = params.gradClipNorm
                self?.weightDecay = params.weightDecay
                self?.shardsPct = params.shardsFraction * 100
                // batch_size and seq_len kept as mobile defaults (same as browser)
            }
        }
        await setDetail("Registered as \(nodeId)")

        // 2. Download manifest
        await setDetail("Downloading data manifest...")
        try await downloadManifest()

        // Ensure directories exist
        try FileManager.default.createDirectory(at: shardsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: deltasDir, withIntermediateDirectories: true)

        // Fixed steps per round — kept low for iPad to avoid long GPU occupancy
        let hMiniVal = Self.stepsPerRound
        await MainActor.run { [weak self] in
            self?.hMini = hMiniVal
        }
        await appendLog("Steps per round: \(hMiniVal)")

        // 5. Training loop
        while !Task.isCancelled {
            try Task.checkCancellation()
            await MainActor.run { [weak self] in
                self?.round += 1
            }
            let currentRound = await round

            // 4a. Get latest checkpoint
            await setPhase(.downloadingCheckpoint, detail: "Fetching checkpoint info...")
            let ckptInfo = try await getLatestCheckpoint()
            let version = ckptInfo.version
            let ckptKey = ckptInfo.checkpointKey
            await MainActor.run { [weak self] in
                self?.checkpointVersion = version
            }

            // 4b. Download checkpoint to file (stream to disk, not RAM)
            let ckptFile = deltasDir.appendingPathComponent("v\(version)_model.safetensors")
            if !FileManager.default.fileExists(atPath: ckptFile.path) {
                await setDetail("Downloading checkpoint v\(version)...")
                try await downloadToFile(key: ckptKey, destination: ckptFile)
            }
            let ckptSizeKB = (try? FileManager.default.attributesOfItem(atPath: ckptFile.path)[.size] as? Int)
                .map { $0 / 1024 } ?? 0
            let ckptSizeStr = ckptSizeKB >= 1024 ? "\(ckptSizeKB / 1024)MB" : "\(ckptSizeKB)KB"

            // 4c. Load checkpoint from file into Rust engine
            await setPhase(.loadingCheckpoint, detail: "Loading checkpoint v\(version) (\(ckptSizeStr))...")
            let ckptPath = ckptFile.path
            let ckptResult = try await Task.detached {
                try DistrainBridge.loadCheckpointFile(path: ckptPath)
            }.value

            // Update GPU status and clamp seq_len to model's max
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.gpuActive = ckptResult.gpu
                if Int(self.seqLen) > ckptResult.maxSeqLen {
                    self.seqLen = Double(ckptResult.maxSeqLen)
                }
            }
            await appendLog("Backend: \(ckptResult.gpu ? "GPU (Metal)" : "CPU (fallback)")")

            // 4d. Compute shard assignment
            let shardsPerNode = max(2, UInt32(Double(totalShards) * (await shardsPct / 100)))
            let shardIds = try await Task.detached { [nodeId = self.nodeId, totalShards = self.totalShards] in
                try DistrainBridge.computeShardAssignment(
                    nodeId: nodeId, version: version,
                    totalShards: totalShards, shardsPerNode: shardsPerNode
                )
            }.value

            // 4e. Download assigned shards
            await setPhase(.downloadingShards, detail: "Downloading \(shardIds.count) shards...")
            let shardPaths = try await downloadShards(shardIds: shardIds)

            // Load shards into engine
            await setDetail("Loading shards into engine...")
            let batchSize = Int(await self.batchSize)
            let seqLen = Int(await self.seqLen)
            try await Task.detached {
                try DistrainBridge.loadShards(paths: shardPaths, batchSize: batchSize, seqLen: seqLen)
            }.value

            // 4f. Snapshot params (baseline for delta)
            try await Task.detached {
                _ = try DistrainBridge.snapshotParams()
            }.value

            // 4g. Train H_mini steps
            await setPhase(.training, detail: "Training round \(currentRound)...")
            let lrMax = await self.lrMax
            let lrMin = await self.lrMin
            let warmupFraction = await self.warmupPct / 100
            let totalSteps = hMiniVal
            let warmupSteps = max(UInt64(Double(totalSteps) * warmupFraction), 1)

            let startTime = Date()
            var lastLoss: Double = 0
            var stepsRun: UInt64 = 0

            for step in 0..<hMiniVal {
                try Task.checkCancellation()
                let lr = DistrainBridge.cosineLr(
                    step: step, warmupSteps: warmupSteps,
                    totalSteps: totalSteps, maxLr: lrMax, minLr: lrMin
                )
                let result = try await Task.detached {
                    try DistrainBridge.trainStep(learningRate: lr)
                }.value

                stepsRun = step + 1
                lastLoss = result.loss
                let elapsed = -startTime.timeIntervalSinceNow
                let tps = elapsed > 0 ? Double(result.tokensProcessed) / elapsed : 0

                await MainActor.run { [weak self] in
                    self?.currentStep = result.step
                    self?.currentLoss = result.loss
                    self?.totalTokens = result.tokensProcessed
                    self?.tokensPerSec = tps
                    self?.elapsedSecs = -(self?.protocolStartTime?.timeIntervalSinceNow ?? 0)
                    self?.statusDetail = "Step \(step + 1)/\(hMiniVal) | loss \(String(format: "%.3f", result.loss))"
                }
            }

            let trainingElapsed = -startTime.timeIntervalSinceNow

            // 4h. Compute + compress delta
            await setPhase(.computingDelta, detail: "Compressing delta...")
            seqNum += 1
            let deltaFilename = "delta_\(nodeId)_\(seqNum).delta.zst"
            let deltaPath = deltasDir.appendingPathComponent(deltaFilename).path
            let deltaSize = try await Task.detached { [deltaPath] in
                try DistrainBridge.computeDelta(outputPath: deltaPath)
            }.value
            await setDetail("Delta compressed: \(deltaSize / 1024)KB")

            // 4i. Upload delta to R2 via coordinator proxy
            await setPhase(.uploadingDelta, detail: "Uploading delta (\(deltaSize / 1024)KB)...")
            let deltaKey = "deltas/v\(version)/\(nodeId)_\(seqNum).delta.zst"
            let deltaData = try Data(contentsOf: URL(fileURLWithPath: deltaPath))
            try await uploadToStorage(key: deltaKey, data: deltaData)

            // Clean up local delta file
            try? FileManager.default.removeItem(atPath: deltaPath)

            // 4j. Push metadata to coordinator
            await setPhase(.pushingMeta, detail: "Pushing delta metadata...")
            let pushResp = try await pushDelta(
                version: version,
                innerSteps: stepsRun,
                deltaKey: deltaKey,
                trainingLoss: lastLoss,
                tokensProcessed: await totalTokens,
                trainingTimeSecs: trainingElapsed
            )

            if pushResp.accepted {
                await appendLog("Round \(currentRound): pushed (loss=\(String(format: "%.4f", lastLoss)), \(String(format: "%.1f", trainingElapsed))s)")
                await setDetail("Push accepted (ckpt v\(pushResp.checkpointVersion))")
            } else {
                await appendLog("Round \(currentRound): rejected — \(pushResp.reason ?? "unknown")")
                await setDetail("Push rejected: \(pushResp.reason ?? "unknown")")
            }

            // 4k. Brief pause before next round (let coordinator merge)
            await setPhase(.waitingForCheckpoint, detail: "Waiting before next round...")
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2s
        }
    }

    // MARK: - Networking (Coordinator API)

    private struct RegistrationInfo {
        let nodeId: String
        let apiKey: String
        let trainingParams: DistrainBridge.TrainingParamsResponse?
    }

    private func register() async throws -> RegistrationInfo {
        guard let url = URL(string: "\(baseUrl)/nodes/register") else {
            throw DistrainBridge.DistrainError.ffiError("invalid coordinator URL: \(coordinatorUrl)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "gpu_model": "ios-metal",
            "gpu_memory_gb": 0,
            "bandwidth_mbps": 100,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try checkHTTPResponse(response, context: "register")

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DistrainBridge.DistrainError.decodingError("invalid register response")
        }

        let nodeId: String
        if let idDict = json["node_id"] as? [String: String], let id = idDict.values.first {
            nodeId = id
        } else if let id = json["node_id"] as? String {
            nodeId = id
        } else {
            throw DistrainBridge.DistrainError.decodingError("missing node_id in register response")
        }

        let apiKey = json["api_key"] as? String ?? ""

        var params: DistrainBridge.TrainingParamsResponse?
        if let paramsDict = json["training_params"] as? [String: Any] {
            params = DistrainBridge.TrainingParamsResponse(
                batchSize: paramsDict["batch_size"] as? Int ?? 4,
                seqLen: paramsDict["seq_len"] as? Int ?? 512,
                lrMax: paramsDict["lr_max"] as? Double ?? 3e-4,
                lrMin: paramsDict["lr_min"] as? Double ?? 1e-6,
                weightDecay: paramsDict["weight_decay"] as? Double ?? 0.1,
                gradClipNorm: paramsDict["grad_clip_norm"] as? Double ?? 1.0,
                warmupFraction: paramsDict["warmup_fraction"] as? Double ?? 0.2,
                shardsFraction: paramsDict["shards_fraction"] as? Double ?? 0.2
            )
        }

        return RegistrationInfo(nodeId: nodeId, apiKey: apiKey, trainingParams: params)
    }

    private struct CheckpointInfoResponse {
        let version: UInt64
        let checkpointKey: String
    }

    private func getLatestCheckpoint() async throws -> CheckpointInfoResponse {
        guard let url = URL(string: "\(baseUrl)/checkpoint/latest") else {
            throw DistrainBridge.DistrainError.ffiError("invalid coordinator URL")
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        try checkHTTPResponse(response, context: "checkpoint/latest")

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DistrainBridge.DistrainError.decodingError("invalid checkpoint response")
        }

        let version = (json["version"] as? UInt64) ?? (json["version"] as? Int).map { UInt64($0) } ?? 0
        let checkpointKey = json["checkpoint_key"] as? String ?? "checkpoints/v\(version)/model.safetensors"

        return CheckpointInfoResponse(version: version, checkpointKey: checkpointKey)
    }

    private struct DeltaPushResponse {
        let accepted: Bool
        let checkpointVersion: UInt64
        let reason: String?
    }

    private func pushDelta(
        version: UInt64,
        innerSteps: UInt64,
        deltaKey: String,
        trainingLoss: Double,
        tokensProcessed: UInt64,
        trainingTimeSecs: Double
    ) async throws -> DeltaPushResponse {
        let baseUrl = coordinatorUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: "\(baseUrl)/delta") else {
            throw DistrainBridge.DistrainError.ffiError("invalid coordinator URL")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "node_id": nodeId,
            "seq_num": seqNum,
            "checkpoint_version": version,
            "inner_steps": innerSteps,
            "delta_key": deltaKey,
            "training_loss": trainingLoss,
            "tokens_processed": tokensProcessed,
            "training_time_secs": trainingTimeSecs,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try checkHTTPResponse(response, context: "push delta")

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DistrainBridge.DistrainError.decodingError("invalid delta push response")
        }

        return DeltaPushResponse(
            accepted: json["accepted"] as? Bool ?? false,
            checkpointVersion: (json["checkpoint_version"] as? UInt64)
                ?? (json["checkpoint_version"] as? Int).map { UInt64($0) } ?? version,
            reason: json["reason"] as? String
        )
    }

    // MARK: - Storage (via coordinator proxy)

    /// Download to file (streaming — doesn't hold full response in RAM).
    private func downloadToFile(key: String, destination: URL) async throws {
        let urlString = "\(baseUrl)/download/\(key)"
        guard let url = URL(string: urlString) else {
            throw DistrainBridge.DistrainError.ffiError("invalid download URL: \(urlString)")
        }
        let (tempURL, response) = try await URLSession.shared.download(from: url)
        try checkHTTPResponse(response, context: "download \(key)")
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    private func downloadFromStorage(key: String) async throws -> Data {
        let urlString = "\(coordinatorUrl.trimmingCharacters(in: .whitespacesAndNewlines))/download/\(key)"
        guard let url = URL(string: urlString) else {
            throw DistrainBridge.DistrainError.ffiError("invalid download URL: \(urlString)")
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        try checkHTTPResponse(response, context: "download \(key)")
        return data
    }

    private func uploadToStorage(key: String, data: Data) async throws {
        let urlString = "\(coordinatorUrl.trimmingCharacters(in: .whitespacesAndNewlines))/upload/\(key)"
        guard let url = URL(string: urlString) else {
            throw DistrainBridge.DistrainError.ffiError("invalid upload URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.httpBody = data
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")

        let (_, response) = try await URLSession.shared.data(for: request)
        try checkHTTPResponse(response, context: "upload \(key)")
    }

    // MARK: - Data Management

    private func downloadManifest() async throws {
        let manifestData = try await downloadFromStorage(key: "data/manifest.json")
        guard let json = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              let shards = json["shards"] as? [[String: Any]] else {
            throw DistrainBridge.DistrainError.decodingError("invalid manifest format")
        }
        manifest = shards
        totalShards = UInt32(shards.count)
        await setDetail("Manifest: \(shards.count) shards")
    }

    private func downloadShards(shardIds: [Int]) async throws -> [String] {
        guard let manifest = manifest else {
            throw DistrainBridge.DistrainError.ffiError("manifest not loaded")
        }

        var paths: [String] = []
        for (i, shardIdx) in shardIds.enumerated() {
            try Task.checkCancellation()
            guard shardIdx < manifest.count,
                  let filename = manifest[shardIdx]["filename"] as? String else { continue }

            let localPath = shardsDir.appendingPathComponent(filename)

            // Skip if already cached
            if !FileManager.default.fileExists(atPath: localPath.path) {
                let key = "data/\(filename)"
                let shardData = try await downloadFromStorage(key: key)
                try shardData.write(to: localPath)
            }

            paths.append(localPath.path)
            await setDetail("Downloading shards \(i + 1)/\(shardIds.count)...")
        }

        return paths
    }

    // MARK: - Helpers

    private func appendLog(_ msg: String) async {
        let time = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        await MainActor.run { [weak self] in
            guard let self else { return }
            self.logText += "\n[\(time)] \(msg)"
            // Trim old lines
            let lines = self.logText.split(separator: "\n", omittingEmptySubsequences: false)
            if lines.count > 500 {
                self.logText = lines.suffix(300).joined(separator: "\n")
            }
        }
    }

    private func setPhase(_ phase: ProtocolPhase, detail: String) async {
        await appendLog(detail)
        await MainActor.run { [weak self] in
            self?.phase = phase
            self?.statusDetail = detail
        }
    }

    private func setDetail(_ detail: String) async {
        await appendLog(detail)
        await MainActor.run { [weak self] in
            self?.statusDetail = detail
        }
    }

    private func checkHTTPResponse(_ response: URLResponse, context: String) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200...299).contains(http.statusCode) else {
            throw DistrainBridge.DistrainError.ffiError(
                "\(context) failed: HTTP \(http.statusCode)"
            )
        }
    }
}
