import Foundation
import Metal

/// Swift bridge to the Distrain Rust training engine (distrain-ffi).
///
/// Wraps the C FFI functions from crates/ffi/distrain.h.
/// All communication uses JSON strings for cross-language simplicity.
///
/// Build steps:
///   1. Cross-compile FFI crate for iOS:
///      cargo lipo --release -p distrain-ffi
///   2. Copy libdistrain_ffi.a to the Xcode project
///   3. Add crates/ffi/distrain.h as a bridging header
public class DistrainBridge {

    // MARK: - Response Types

    public struct StatusResponse: Codable {
        public let initialized: Bool
        public let step: UInt64
        public let lastLoss: Double
        public let totalTokens: UInt64
        public let modelPreset: String

        enum CodingKeys: String, CodingKey {
            case initialized, step
            case lastLoss = "last_loss"
            case totalTokens = "total_tokens"
            case modelPreset = "model_preset"
        }
    }

    public struct StepResult: Codable {
        public let loss: Double
        public let step: UInt64
        public let tokensProcessed: UInt64

        enum CodingKeys: String, CodingKey {
            case loss, step
            case tokensProcessed = "tokens_processed"
        }
    }

    public struct CalibrationResult: Codable {
        public let secsPerStep: Double
        public let recommendedHMini: UInt64

        enum CodingKeys: String, CodingKey {
            case secsPerStep = "secs_per_step"
            case recommendedHMini = "recommended_h_mini"
        }
    }

    public enum DistrainError: Error, LocalizedError {
        case ffiError(String)
        case decodingError(String)

        public var errorDescription: String? {
            switch self {
            case .ffiError(let msg): return "FFI error: \(msg)"
            case .decodingError(let msg): return "Decoding error: \(msg)"
            }
        }
    }

    // MARK: - Private Helpers

    private static func parseJSON<T: Codable>(_ ptr: UnsafeMutablePointer<CChar>?) throws -> T {
        guard let ptr = ptr else {
            throw DistrainError.ffiError("null pointer from FFI")
        }
        let jsonString = String(cString: ptr)
        distrain_free_string(ptr)

        // Check for error response
        if let data = jsonString.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let error = dict["error"] as? String {
            throw DistrainError.ffiError(error)
        }

        guard let data = jsonString.data(using: .utf8) else {
            throw DistrainError.decodingError("invalid UTF-8")
        }

        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }

    // MARK: - GPU Detection

    /// Check if device GPU supports the Metal features cubecl/wgpu needs.
    /// Metal GPU Family 4+ (A11 chip, 2017+) is required.
    /// A10 and earlier generate InnocentVictim command buffer errors.
    public static func gpuSupported() -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        // GPU Family 4 = Apple A11+ (iPhone 8, iPad 8th gen+)
        return device.supportsFamily(.apple4)
    }

    // MARK: - Public API

    /// Initialize the training engine with a model preset.
    /// - Parameter preset: One of "Tiny", "Small", "Medium", "Large"
    public static func initialize(preset: String = "Tiny") throws -> StatusResponse {
        let result = preset.withCString { distrain_init($0) }
        return try parseJSON(result)
    }

    /// Load real training data from binary shard files.
    /// - Parameters:
    ///   - paths: Local file paths to .bin shard files (flat little-endian uint16 tokens)
    ///   - batchSize: Sequences per batch
    ///   - seqLen: Tokens per sequence
    public static func loadShards(paths: [String], batchSize: Int, seqLen: Int) throws {
        let json = try JSONSerialization.data(withJSONObject: paths)
        let jsonStr = String(data: json, encoding: .utf8)!
        let result = jsonStr.withCString { distrain_load_shards($0, batchSize, seqLen) }
        let _: OkResponse = try parseJSON(result)
    }

    /// Run one training step using loaded shard data.
    /// Requires `loadShards` to have been called first.
    /// - Parameter learningRate: Learning rate for this step
    public static func trainStep(learningRate: Double = 3e-4) throws -> StepResult {
        let result = distrain_train_step(learningRate)
        return try parseJSON(result)
    }

    /// Get current training status.
    public static func status() throws -> StatusResponse {
        let result = distrain_status()
        return try parseJSON(result)
    }

    /// Run device calibration to determine optimal H_mini.
    /// - Parameter targetIntervalSecs: Target push interval in seconds
    public static func calibrate(targetIntervalSecs: Double = 300.0) throws -> CalibrationResult {
        let result = distrain_calibrate(targetIntervalSecs)
        return try parseJSON(result)
    }

    /// Shut down the training engine and free all resources.
    public static func shutdown() {
        distrain_shutdown()
    }

    /// Compute cosine LR with linear warmup (calls Rust FFI — no reimplementation needed).
    public static func cosineLr(step: UInt64, warmupSteps: UInt64, totalSteps: UInt64, maxLr: Double, minLr: Double) -> Double {
        return distrain_cosine_lr(step, warmupSteps, totalSteps, maxLr, minLr)
    }

    public struct TrainingParamsResponse: Codable {
        public let batchSize: Int
        public let seqLen: Int
        public let lrMax: Double
        public let lrMin: Double
        public let weightDecay: Double
        public let gradClipNorm: Double
        public let warmupFraction: Double
        public let shardsFraction: Double

        enum CodingKeys: String, CodingKey {
            case batchSize = "batch_size"
            case seqLen = "seq_len"
            case lrMax = "lr_max"
            case lrMin = "lr_min"
            case weightDecay = "weight_decay"
            case gradClipNorm = "grad_clip_norm"
            case warmupFraction = "warmup_fraction"
            case shardsFraction = "shards_fraction"
        }
    }

    /// Get default training parameters from the Rust engine.
    public static func defaultTrainingParams() throws -> TrainingParamsResponse {
        let result = distrain_default_training_params()
        return try parseJSON(result)
    }

    // MARK: - Protocol API (checkpoint, delta, shard assignment)

    public struct OkResponse: Codable {
        public let ok: Bool
        public let numParams: Int?
        public let sizeBytes: Int?
        public let maxSeqLen: Int?
        public let gpu: Bool?

        enum CodingKeys: String, CodingKey {
            case ok
            case numParams = "num_params"
            case sizeBytes = "size_bytes"
            case maxSeqLen = "max_seq_len"
            case gpu
        }
    }

    public struct CheckpointLoadResult {
        public let maxSeqLen: Int
        public let gpu: Bool
    }

    /// Load a checkpoint from a file path (memory-efficient — doesn't hold checkpoint in Swift memory).
    public static func loadCheckpointFile(path: String) throws -> CheckpointLoadResult {
        let result = path.withCString { distrain_load_checkpoint_file($0) }
        let resp: OkResponse = try parseJSON(result)
        return CheckpointLoadResult(maxSeqLen: resp.maxSeqLen ?? 4096, gpu: resp.gpu ?? false)
    }

    /// Load a checkpoint from in-memory safetensors bytes.
    public static func loadCheckpoint(data: Data) throws -> CheckpointLoadResult {
        let result = data.withUnsafeBytes { buf -> UnsafeMutablePointer<CChar>? in
            let ptr = buf.baseAddress?.assumingMemoryBound(to: UInt8.self)
            return distrain_load_checkpoint(ptr, buf.count)
        }
        let resp: OkResponse = try parseJSON(result)
        return CheckpointLoadResult(maxSeqLen: resp.maxSeqLen ?? 4096, gpu: resp.gpu ?? false)
    }

    /// Snapshot current model parameters as baseline for delta computation.
    /// Returns the number of parameters snapshotted.
    public static func snapshotParams() throws -> Int {
        let result = distrain_snapshot_params()
        let resp: OkResponse = try parseJSON(result)
        return resp.numParams ?? 0
    }

    /// Compute delta between snapshot and current params, compress, and write to file.
    /// Returns the size of the compressed delta in bytes.
    public static func computeDelta(outputPath: String) throws -> Int {
        let result = outputPath.withCString { distrain_compute_delta($0) }
        let resp: OkResponse = try parseJSON(result)
        return resp.sizeBytes ?? 0
    }

    /// Compute deterministic shard assignment for a node and checkpoint version.
    /// Returns an array of shard indices.
    public static func computeShardAssignment(
        nodeId: String,
        version: UInt64,
        totalShards: UInt32,
        shardsPerNode: UInt32
    ) throws -> [Int] {
        let result = nodeId.withCString {
            distrain_compute_shard_assignment($0, version, totalShards, shardsPerNode)
        }
        guard let ptr = result else {
            throw DistrainError.ffiError("null pointer from FFI")
        }
        let jsonString = String(cString: ptr)
        distrain_free_string(ptr)

        // Check for error response
        if let data = jsonString.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let error = dict["error"] as? String {
            throw DistrainError.ffiError(error)
        }

        guard let data = jsonString.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [Int] else {
            throw DistrainError.decodingError("expected JSON array of ints, got: \(jsonString)")
        }
        return array
    }
}
