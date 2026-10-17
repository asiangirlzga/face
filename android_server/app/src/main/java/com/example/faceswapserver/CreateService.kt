package com.example.faceswapserver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File

/**
 * Foreground service for the long jobs of the Create tab: model download, text-to-image and the
 * motion video. It keeps the process alive while the app is in the background.
 */
class CreateService : Service() {
    private var worker: Thread? = null
    private var wake: PowerManager.WakeLock? = null
    private var lastNotify = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            cancelWork()
            return START_NOT_STICKY
        }
        startForegroundCompat(notification("Starting…"))
        if (worker?.isAlive == true || intent == null) return START_NOT_STICKY

        val mode = intent.getStringExtra(EXTRA_MODE) ?: "image"
        wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "faceswap:create").apply { acquire(3 * 60 * 60 * 1000L) }

        CreateState.mode = mode
        CreateState.busy = true
        CreateState.error = null
        CreateState.progress = 0f
        CreateState.stage = "Starting"
        CreateState.message = ""

        worker = Thread {
            try {
                when (mode) {
                    "download" -> download(intent)
                    "image" -> generateImage(intent)
                    "video" -> makeVideo(intent)
                }
            } catch (t: Throwable) {
                CreateState.error = t.message ?: t.toString()
                CreateState.message = "Failed"
            } finally {
                CreateState.busy = false
                runCatching { wake?.release() }
                if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_REMOVE) else stopForeground(true)
                stopSelf()
            }
        }.also { it.name = "create-worker"; it.start() }
        return START_NOT_STICKY
    }

    private fun cancelWork() {
        ModelStore.abort = true
        if (Python.isStarted()) {
            runCatching { Python.getInstance().getModule("sdgen.pipeline").callAttr("cancel") }
        }
        CreateState.message = "Cancelling…"
    }

    private fun ensurePython(): Python {
        synchronized(CreateService::class.java) {
            if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
        }
        return Python.getInstance()
    }

    // ------------------------------------------------------------------ jobs
    private fun download(intent: Intent) {
        val base = intent.getStringExtra(EXTRA_BASE_URL) ?: SdModelStore.DEFAULT_BASE
        CreateState.stage = "Downloading image model"
        SdModelStore.downloadAll(this, base) { label, done, total ->
            CreateState.progress = if (total > 0) done.toFloat() / total else 0f
            CreateState.stage = "Downloading $label"
            CreateState.message = "${done / 1_000_000} MB" + if (total > 0) " of ${total / 1_000_000} MB" else ""
            maybeNotify("Downloading $label ${(CreateState.progress * 100).toInt()}%")
        }
        CreateState.stage = "Done"
        CreateState.message = "Image model downloaded"
    }

    private fun generateImage(intent: Intent) {
        val py = ensurePython()
        val pipeline = py.getModule("sdgen.pipeline")
        val out = File(cacheDir, "t2i_${System.currentTimeMillis()}.jpg")
        val seed: Long? = if (intent.hasExtra(EXTRA_SEED)) intent.getLongExtra(EXTRA_SEED, -1L) else null
        startPolling(pipeline)
        val result = pipeline.callAttr(
            "generate_image",
            intent.getStringExtra(EXTRA_PROMPT) ?: "",
            intent.getIntExtra(EXTRA_STEPS, 1),
            intent.getIntExtra(EXTRA_SIZE, 512),
            seed,
            SdModelStore.dir(this).absolutePath,
            out.absolutePath,
            intent.getBooleanExtra(EXTRA_TILED, true),
        ).toString()
        if (result.isEmpty()) {
            CreateState.message = "Cancelled"
            return
        }
        val where = MediaSaver.saveImage(this, File(result), "txt2img_${System.currentTimeMillis()}.jpg")
        CreateState.imagePath = result
        CreateState.imageVersion++
        CreateState.progress = 1f
        CreateState.stage = "Done"
        CreateState.message = "Saved to $where"
    }

    private fun makeVideo(intent: Intent) {
        val py = ensurePython()
        val pipeline = py.getModule("sdgen.pipeline")
        val motion = py.getModule("sdgen.motion")
        val out = File(cacheDir, "motion_${System.currentTimeMillis()}.mp4")
        startPolling(pipeline)
        val result = motion.callAttr(
            "make_video",
            intent.getStringExtra(EXTRA_IMAGE) ?: "",
            out.absolutePath,
            intent.getIntExtra(EXTRA_SECONDS, 4),
            intent.getIntExtra(EXTRA_FPS, 24),
            intent.getStringExtra(EXTRA_STYLE) ?: "zoom_in",
            768,
        ).toString()
        if (result.isEmpty()) {
            CreateState.message = "Cancelled"
            return
        }
        val where = MediaSaver.saveVideo(this, File(result), "motion_${System.currentTimeMillis()}.mp4")
        File(result).delete()
        CreateState.progress = 1f
        CreateState.stage = "Done"
        CreateState.message = "Video saved to $where"
    }

    /** Mirrors the Python progress (stage text and fraction) into CreateState while the job runs. */
    private fun startPolling(pipeline: PyObject) {
        Thread {
            while (CreateState.busy) {
                try {
                    val j = JSONObject(pipeline.callAttr("get_progress").toString())
                    CreateState.stage = j.optString("stage", CreateState.stage)
                    CreateState.progress = j.optDouble("frac", 0.0).toFloat()
                    if (j.has("seed") && !j.isNull("seed")) CreateState.lastSeed = j.optLong("seed")
                    maybeNotify(CreateState.stage)
                } catch (_: Throwable) {
                }
                Thread.sleep(700)
            }
        }.also { it.name = "create-poll"; it.isDaemon = true; it.start() }
    }

    // ------------------------------------------------------------------ notification
    private fun maybeNotify(text: String) {
        val now = System.currentTimeMillis()
        if (now - lastNotify < 1000) return
        lastNotify = now
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, notification(text))
    }

    private fun startForegroundCompat(n: Notification) {
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Create", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        val cancel = PendingIntent.getService(
            this, 2, Intent(this, CreateService::class.java).setAction(ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Text to image")
            .setContentText(text)
            .setContentIntent(open)
            .addAction(0, "Cancel", cancel)
            .setOngoing(true)
            .build()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {   // Android 15 limits dataSync services
        cancelWork()
    }

    companion object {
        const val ACTION_CANCEL = "com.example.faceswapserver.CREATE_CANCEL"
        const val EXTRA_MODE = "mode"
        const val EXTRA_BASE_URL = "base_url"
        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_STEPS = "steps"
        const val EXTRA_SIZE = "size"
        const val EXTRA_SEED = "seed"
        const val EXTRA_TILED = "tiled"
        const val EXTRA_IMAGE = "image"
        const val EXTRA_SECONDS = "seconds"
        const val EXTRA_FPS = "fps"
        const val EXTRA_STYLE = "style"
        private const val CHANNEL = "create"
        private const val NOTIF_ID = 2
    }
}
