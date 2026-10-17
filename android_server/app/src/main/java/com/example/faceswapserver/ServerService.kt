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
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File

/** Foreground service that runs the Python face-swap HTTP server (server.py) inside this app. */
class ServerService : Service() {
    private var serverThread: Thread? = null
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopServer()
            return START_NOT_STICKY
        }
        startForegroundCompat(notification("Starting server…"))
        if (serverThread != null) return START_NOT_STICKY

        val host = intent?.getStringExtra(EXTRA_HOST) ?: "127.0.0.1"
        val port = intent?.getIntExtra(EXTRA_PORT, 8000) ?: 8000
        val token = intent?.getStringExtra(EXTRA_TOKEN) ?: ""

        wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "faceswap:server").apply { acquire() }

        ServerState.error = null
        serverThread = Thread {
            try {
                if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
                val server = Python.getInstance().getModule("server")
                ServerState.running = true
                startPolling(server)
                val work = File(cacheDir, "jobs").absolutePath
                server.callAttr(
                    "serve", host, port, token, ModelStore.dir(this).absolutePath, work,
                    SdModelStore.dir(this).absolutePath,   // image model folder for POST /jobs/text2img
                )  // blocks
            } catch (t: Throwable) {
                ServerState.error = t.message ?: t.toString()
            } finally {
                ServerState.running = false
                stopSelf()
            }
        }.also { it.name = "python-server"; it.start() }
        return START_NOT_STICKY
    }

    private fun startPolling(server: com.chaquo.python.PyObject) {
        Thread {
            while (ServerState.running) {
                try {
                    val j = JSONObject(server.callAttr("get_state").toString())
                    ServerState.engine = j.optString("engine", "-") +
                        (j.optString("engine_error").takeIf { it.isNotEmpty() && it != "null" }?.let { ": $it" } ?: "")
                    val last = j.optJSONObject("last")
                    ServerState.job = if (last == null) "No jobs yet"
                    else "${last.optString("kind")} ${last.optString("status")} " +
                        "${last.optInt("done")}/${last.optInt("total")}" +
                        (last.optString("error").takeIf { it.isNotEmpty() && it != "null" }?.let { " – $it" } ?: "")
                    notify(notification("Engine: ${j.optString("engine")} · ${ServerState.job}"))
                } catch (_: Throwable) {
                }
                Thread.sleep(1500)
            }
        }.also { it.name = "state-poll"; it.isDaemon = true; it.start() }
    }

    private fun stopServer() {
        Thread {
            runCatching { Python.getInstance().getModule("server").callAttr("stop") }
            stopSelf()
        }.start()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {   // Android 15 limits dataSync services
        stopServer()
    }

    override fun onDestroy() {
        runCatching { wake?.release() }
        super.onDestroy()
    }

    private fun startForegroundCompat(n: Notification) {
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    private fun notify(n: Notification) {
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, n)
    }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Face swap server", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, ServerService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Face Swap Server")
            .setContentText(text)
            .setContentIntent(open)
            .addAction(0, "Stop", stop)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val ACTION_STOP = "com.example.faceswapserver.STOP"
        const val EXTRA_HOST = "host"
        const val EXTRA_PORT = "port"
        const val EXTRA_TOKEN = "token"
        private const val CHANNEL = "server"
        private const val NOTIF_ID = 1
    }
}
