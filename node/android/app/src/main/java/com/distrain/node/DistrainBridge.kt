package com.distrain.node

import org.json.JSONObject

/**
 * JNI bridge to the Distrain Rust training engine (distrain-ffi).
 *
 * All methods return JSON strings parsed from the Rust FFI.
 * The native library is compiled from crates/ffi as libdistrain_ffi.so.
 *
 * Build steps:
 *   1. Cross-compile FFI crate for Android targets:
 *      cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o app/src/main/jniLibs build --release -p distrain-ffi
 *   2. The .so files land in app/src/main/jniLibs/<abi>/libdistrain_ffi.so
 */
object DistrainBridge {

    init {
        System.loadLibrary("distrain_ffi")
    }

    // --- Native declarations (match crates/ffi/src/lib.rs) ---

    private external fun nativeInit(presetJson: String): String
    private external fun nativeTrainStep(learningRate: Double): String
    private external fun nativeStatus(): String
    private external fun nativeCalibrate(targetIntervalSecs: Double): String
    private external fun nativeShutdown()
    private external fun nativeCosineLr(step: Long, warmupSteps: Long, totalSteps: Long, maxLr: Double, minLr: Double): Double
    private external fun nativeDefaultTrainingParams(): String

    // --- Kotlin-friendly wrappers ---

    data class StatusResponse(
        val initialized: Boolean,
        val step: Long,
        val lastLoss: Double,
        val totalTokens: Long,
        val modelPreset: String,
    )

    data class StepResult(
        val loss: Double,
        val step: Long,
        val tokensProcessed: Long,
    )

    data class CalibrationResult(
        val secsPerStep: Double,
        val recommendedHMini: Long,
    )

    fun init(preset: String = "Tiny"): StatusResponse {
        val json = JSONObject(nativeInit(preset))
        return if (json.has("error")) {
            throw RuntimeException(json.getString("error"))
        } else {
            StatusResponse(
                initialized = json.getBoolean("initialized"),
                step = json.getLong("step"),
                lastLoss = json.getDouble("last_loss"),
                totalTokens = json.getLong("total_tokens"),
                modelPreset = json.getString("model_preset"),
            )
        }
    }

    fun trainStep(learningRate: Double = 3e-4): StepResult {
        val json = JSONObject(nativeTrainStep(learningRate))
        return if (json.has("error")) {
            throw RuntimeException(json.getString("error"))
        } else {
            StepResult(
                loss = json.getDouble("loss"),
                step = json.getLong("step"),
                tokensProcessed = json.getLong("tokens_processed"),
            )
        }
    }

    fun status(): StatusResponse {
        val json = JSONObject(nativeStatus())
        return StatusResponse(
            initialized = json.getBoolean("initialized"),
            step = json.getLong("step"),
            lastLoss = json.getDouble("last_loss"),
            totalTokens = json.getLong("total_tokens"),
            modelPreset = json.getString("model_preset"),
        )
    }

    fun calibrate(targetIntervalSecs: Double = 300.0): CalibrationResult {
        val json = JSONObject(nativeCalibrate(targetIntervalSecs))
        return if (json.has("error")) {
            throw RuntimeException(json.getString("error"))
        } else {
            CalibrationResult(
                secsPerStep = json.getDouble("secs_per_step"),
                recommendedHMini = json.getLong("recommended_h_mini"),
            )
        }
    }

    fun shutdown() = nativeShutdown()

    /** Compute cosine LR with linear warmup (calls Rust FFI). */
    fun cosineLr(step: Long, warmupSteps: Long, totalSteps: Long, maxLr: Double, minLr: Double): Double {
        return nativeCosineLr(step, warmupSteps, totalSteps, maxLr, minLr)
    }

    data class TrainingParams(
        val batchSize: Int,
        val seqLen: Int,
        val lrMax: Double,
        val lrMin: Double,
        val weightDecay: Double,
        val gradClipNorm: Double,
        val warmupFraction: Double,
        val shardsFraction: Double,
    )

    fun defaultTrainingParams(): TrainingParams {
        val json = JSONObject(nativeDefaultTrainingParams())
        return TrainingParams(
            batchSize = json.getInt("batch_size"),
            seqLen = json.getInt("seq_len"),
            lrMax = json.getDouble("lr_max"),
            lrMin = json.getDouble("lr_min"),
            weightDecay = json.getDouble("weight_decay"),
            gradClipNorm = json.getDouble("grad_clip_norm"),
            warmupFraction = json.getDouble("warmup_fraction"),
            shardsFraction = json.getDouble("shards_fraction"),
        )
    }
}
