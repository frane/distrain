/* Distrain FFI — C header for mobile platform bindings.
 *
 * All functions return JSON strings that the caller must free
 * with distrain_free_string(). */

#ifndef DISTRAIN_H
#define DISTRAIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the training engine with a model preset.
 * @param preset_json  C string, e.g. "Tiny", "Small", "Base"
 * @return JSON StatusResponse. Caller must free.
 */
char* distrain_init(const char* preset_json);

/**
 * Load real training data from binary shard files.
 * @param paths_json  JSON array of file paths, e.g. ["/path/shard_0001.bin"]
 * @param batch_size  Sequences per batch
 * @param seq_len     Tokens per sequence
 * @return JSON status. Caller must free.
 */
char* distrain_load_shards(const char* paths_json, size_t batch_size,
                           size_t seq_len);

/**
 * Run one training step using loaded shard data.
 * Requires distrain_load_shards to have been called first.
 * @param learning_rate  Learning rate for this step
 * @return JSON StepResult { loss, step, tokens_processed }. Caller must free.
 */
char* distrain_train_step(double learning_rate);

/**
 * Get current training status.
 * @return JSON StatusResponse. Caller must free.
 */
char* distrain_status(void);

/**
 * Run device calibration.
 * @param target_interval_secs  Target push interval in seconds
 * @return JSON CalibrationResult { secs_per_step, recommended_h_mini }. Caller must free.
 */
char* distrain_calibrate(double target_interval_secs);

/**
 * Shut down the training engine and free all resources.
 */
void distrain_shutdown(void);

/**
 * Free a string returned by any distrain_* function.
 * @param ptr  Pointer previously returned by a distrain_* function.
 */
void distrain_free_string(char* ptr);

/**
 * Compute cosine learning rate with linear warmup.
 * @return The learning rate for the given step.
 */
double distrain_cosine_lr(uint64_t step, uint64_t warmup_steps,
                          uint64_t total_steps, double max_lr, double min_lr);

/**
 * Get default training parameters as JSON.
 * @return JSON TrainingParams. Caller must free.
 */
char* distrain_default_training_params(void);

/**
 * Set backend: 1 = GPU (Metal/Vulkan), 0 = CPU (ndarray).
 * Call before any other distrain_* function.
 * On iOS, check Metal GPU family from Swift to decide.
 */
void distrain_set_backend(uint8_t use_gpu);

/**
 * Load a checkpoint from a file path (memory-efficient for large checkpoints).
 * @param path  Path to .safetensors file
 * @return JSON {"ok": true, "num_params": N, "max_seq_len": M, "gpu": bool}. Caller must free.
 */
char* distrain_load_checkpoint_file(const char* path);

/**
 * Load a checkpoint from in-memory safetensors bytes.
 * Infers model config from tensor shapes, reinitializes engine.
 * @param data     Pointer to safetensors bytes
 * @param len      Length of data in bytes
 * @return JSON {"ok": true, "num_params": N}. Caller must free.
 */
char* distrain_load_checkpoint(const uint8_t* data, size_t len);

/**
 * Snapshot current model parameters as baseline for delta computation.
 * @return JSON {"ok": true, "num_params": N}. Caller must free.
 */
char* distrain_snapshot_params(void);

/**
 * Compute delta between snapshot and current params, compress, write to file.
 * @param output_path  File path to write compressed delta (.delta.zst)
 * @return JSON {"ok": true, "size_bytes": N}. Caller must free.
 */
char* distrain_compute_delta(const char* output_path);

/**
 * Compute deterministic shard assignment for a node and checkpoint version.
 * @param node_id         Node identifier string
 * @param version         Checkpoint version
 * @param total_shards    Total number of shards in the dataset
 * @param shards_per_node Number of shards to assign to this node
 * @return JSON array of shard indices, e.g. [3, 7, 12]. Caller must free.
 */
char* distrain_compute_shard_assignment(const char* node_id, uint64_t version,
                                         uint32_t total_shards,
                                         uint32_t shards_per_node);

#ifdef __cplusplus
}
#endif

#endif /* DISTRAIN_H */
