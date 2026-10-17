package com.example.faceswap

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

/** State and actions for the "Text to image" tab (talks to the Face Swap Server app on this phone). */
class CreateViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("faceswap", Context.MODE_PRIVATE)

    var prompt by mutableStateOf(prefs.getString("t2i_prompt", "") ?: "")
    var steps by mutableStateOf(prefs.getInt("t2i_steps", 1))
    var size by mutableStateOf(prefs.getInt("t2i_size", 512))
    var seedText by mutableStateOf("")
    var tiled by mutableStateOf(prefs.getBoolean("t2i_tiled", true))
    var styleIndex by mutableStateOf(0)
    var seconds by mutableStateOf(4)

    var running by mutableStateOf(false)
    var progress by mutableStateOf(0f)
    var stage by mutableStateOf("")
    var message by mutableStateOf("")
    var lastSeed by mutableStateOf<Long?>(null)
    var imageFile by mutableStateOf<File?>(null)
    var imageVersion by mutableStateOf(0)

    val styles = listOf(
        "zoom_in" to "Zoom in", "zoom_out" to "Zoom out",
        "pan_left" to "Pan left", "pan_right" to "Pan right",
        "pan_up" to "Pan up", "pan_down" to "Pan down", "drift" to "Drift",
    )

    private var api: CreateApi? = null
    private var jobId: String? = null

    fun generate(url: String, token: String) {
        prefs.edit().putString("t2i_prompt", prompt).putInt("t2i_steps", steps).putInt("t2i_size", size)
            .putBoolean("t2i_tiled", tiled).apply()
        val seed = seedText.trim().toLongOrNull()?.takeIf { it >= 0 }
        runJob(url, token, "text2img") { a ->
            if (a.imageModelState() != "ready") {
                throw IOException(
                    "The image model is not on the phone yet. Open the Face Swap Server app, " +
                        "go to its Text to image tab and tap Download image model."
                )
            }
            a.submitTextToImage(prompt.trim(), steps, size, seed, tiled)
        }
    }

    /** Animates the image generated above (or any image file). */
    fun animate(url: String, token: String, file: File) {
        runJob(url, token, "img2video") { a ->
            a.submitImageToVideo(file, styles[styleIndex].first, seconds, 24)
        }
    }

    /** Copies a picked photo into the cache, then animates it. */
    fun animatePicked(url: String, token: String, uri: Uri) {
        if (running) return
        val ctx = getApplication<Application>()
        viewModelScope.launch {
            val f = withContext(Dispatchers.IO) {
                File(ctx.cacheDir, "anim_src_${System.currentTimeMillis()}.jpg").also { dest ->
                    ctx.contentResolver.openInputStream(uri)?.use { input -> dest.outputStream().use { input.copyTo(it) } }
                }
            }
            animate(url, token, f)
        }
    }

    fun cancel() {
        val a = api
        val id = jobId
        if (a == null || id == null) return
        message = "Cancelling…"
        viewModelScope.launch(Dispatchers.IO) { runCatching { a.cancel(id) } }
    }

    private fun runJob(url: String, token: String, kind: String, submit: (CreateApi) -> String) {
        if (running) return
        prefs.edit().putString("url", url).putString("token", token).apply()
        viewModelScope.launch {
            running = true
            progress = 0f
            stage = "Sending request"
            message = ""
            try {
                withContext(Dispatchers.IO) {
                    val a = CreateApi(url, token)
                    api = a
                    val id = submit(a)
                    jobId = id
                    while (true) {
                        val st = a.status(id)
                        progress = st.fraction
                        stage = st.stage.ifEmpty { st.status }
                        if (st.seed != null) lastSeed = st.seed
                        if (st.status == "error") throw IOException(st.error ?: "Server error")
                        if (st.status == "cancelled") {
                            message = "Cancelled"
                            break
                        }
                        if (st.status == "done") {
                            finish(a, id, kind)
                            break
                        }
                        delay(1500)
                    }
                }
            } catch (e: Exception) {
                message = "Failed: ${e.message}"
            } finally {
                running = false
                api = null
                jobId = null
            }
        }
    }

    private fun finish(a: CreateApi, id: String, kind: String) {
        val ctx = getApplication<Application>()
        val stamp = System.currentTimeMillis()
        stage = "Downloading result"
        if (kind == "text2img") {
            val f = File(ctx.cacheDir, "t2i_$stamp.jpg")
            a.download(id, f)
            val where = MediaSaver.saveImage(ctx.contentResolver, f, "txt2img_$stamp.jpg")
            imageFile = f
            imageVersion++
            message = "Saved to $where"
        } else {
            val f = File(ctx.cacheDir, "motion_$stamp.mp4")
            a.download(id, f)
            val where = MediaSaver.saveVideo(ctx.contentResolver, f, "motion_$stamp.mp4")
            f.delete()
            message = "Video saved to $where"
        }
        progress = 1f
        stage = "Done"
    }
}
