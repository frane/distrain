package com.distrain.node

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

/**
 * Android foreground service that runs Distrain training in the background.
 *
 * Uses the Rust FFI via DistrainBridge to run training steps.
 * Runs as a foreground service so Android doesn't kill it.
 */
class TrainingService : Service() {

    companion object {
        private const val TAG = "DistrainTraining"
        private const val CHANNEL_ID = "distrain_training"
        private const val NOTIFICATION_ID = 1
    }

    private val binder = LocalBinder()
    private val isTraining = AtomicBoolean(false)
    private var trainingThread: Thread? = null

    var onStepCallback: ((DistrainBridge.StepResult) -> Unit)? = null

    // Training params (from coordinator or defaults)
    var lrMax: Double = 3e-4
    var lrMin: Double = 1e-6
    var warmupFraction: Double = 0.2
    var coordinatorUrl: String = "http://localhost:8000"

    inner class LocalBinder : Binder() {
        fun getService(): TrainingService = this@TrainingService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    fun startTraining(preset: String = "Tiny", maxSteps: Long = 1000) {
        if (isTraining.getAndSet(true)) {
            Log.w(TAG, "Training already running")
            return
        }

        // Start foreground service
        val notification = buildNotification("Initializing...")
        startForeground(NOTIFICATION_ID, notification)

        val warmupSteps = maxOf((maxSteps * warmupFraction).toLong(), 1L)

        trainingThread = Thread {
            try {
                DistrainBridge.init(preset)
                Log.i(TAG, "Model initialized: $preset")

                var step = 0L
                while (isTraining.get() && step < maxSteps) {
                    val lr = DistrainBridge.cosineLr(step, warmupSteps, maxSteps, lrMax, lrMin)
                    val result = DistrainBridge.trainStep(lr)
                    step = result.step
                    onStepCallback?.invoke(result)

                    if (step % 10 == 0L) {
                        Log.i(TAG, "Step $step/$maxSteps: loss=${"%.4f".format(result.loss)}, lr=${"%.2e".format(lr)}")
                        updateNotification("Step $step | Loss: ${"%.4f".format(result.loss)}")
                    }
                }

                Log.i(TAG, "Training complete: $step steps")
            } catch (e: Exception) {
                Log.e(TAG, "Training failed", e)
            } finally {
                isTraining.set(false)
                DistrainBridge.shutdown()
                stopForeground(STOP_FOREGROUND_REMOVE)
            }
        }.also { it.start() }
    }

    /** Fetch training params from the coordinator. Call before startTraining. */
    fun fetchTrainingParams() {
        Thread {
            try {
                val conn = URL("$coordinatorUrl/nodes/register").openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                conn.doOutput = true
                conn.outputStream.write("""{"gpu_model":"android","gpu_memory_gb":0,"bandwidth_mbps":100}""".toByteArray())
                val body = conn.inputStream.bufferedReader().readText()
                val json = JSONObject(body)
                if (json.has("training_params")) {
                    val p = json.getJSONObject("training_params")
                    lrMax = p.optDouble("lr_max", lrMax)
                    lrMin = p.optDouble("lr_min", lrMin)
                    warmupFraction = p.optDouble("warmup_fraction", warmupFraction)
                    Log.i(TAG, "Training params: lr=${"%.2e".format(lrMax)}→${"%.2e".format(lrMin)}, warmup=${(warmupFraction*100).toInt()}%")
                }
                conn.disconnect()
            } catch (e: Exception) {
                Log.w(TAG, "Could not fetch training params: ${e.message}")
            }
        }.start()
    }

    fun stopTraining() {
        isTraining.set(false)
    }

    fun isRunning(): Boolean = isTraining.get()

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Distrain Training",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Shows training progress"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("Distrain Node")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("Distrain Node")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .build()
        }
    }

    private fun updateNotification(text: String) {
        val notification = buildNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        stopTraining()
        super.onDestroy()
    }
}
