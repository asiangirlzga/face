package com.example.faceswapserver

import android.app.Application
import android.content.ContentResolver
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

const val MAX_IMAGES = 50

/** State and actions for the face swap screen. The engine runs inside this app; it is started on demand. */
class SwapViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("faceswap", Context.MODE_PRIVATE)
    private val serverPrefs = app.getSharedPreferences("server", Context.MODE_PRIVATE)

    var consent by mutableStateOf(false)
    var sourceFace by mutableStateOf<Uri?>(null)
    var targetImages by mutableStateOf<List<Uri>>(emptyList())
    var targetVideo by mutableStateOf<Uri?>(null)

    var running by mutableStateOf(false)
    var progress by mutableStateOf(0f)
    var message by mutableStateOf("Ready")

    val canRun get() = consent && !running && sourceFace != null

    fun setImages(uris: List<Uri>) {
        targetImages = uris.take(MAX_IMAGES)
        if (uris.size > MAX_IMAGES) message = "Only the first $MAX_IMAGES images were kept"
    }

    fun swapImages() = launchJob("images") { api, cr -> api.submitImages(cr, sourceFace!!, targetImages) }

    fun swapVideo() = launchJob("video") { api, cr -> api.submitVideo(cr, sourceFace!!, targetVideo!!) }

    fun resumeLastJob() {
        val id = prefs.getString("job_id", null) ?: run { message = "No previous job"; return }
        val kind = prefs.getString("job_kind", "images") ?: "images"
        launchJob(kind, resumeId = id) { _, _ -> id }
    }

    /** Starts the on-phone engine if it is not running yet and waits until its models are loaded. */
    private fun ensureEngine(ctx: Context, api: SwapApi, token: String) {
        var started = false
        val deadline = System.currentTimeMillis() + 180_000
        while (System.currentTimeMillis() < deadline) {
            val h = api.health()
            if (h != null) {
                when (h.first) {
                    "ready" -> return
                    "missing", "error" -> throw IOException(h.second ?: "The face engine could not start")
                }
                message = "Loading the face models…"
            } else {
                if (!started) {
                    startEngine(ctx, token)
                    started = true
                }
                message = "Starting the face engine…"
            }
            Thread.sleep(1000)
        }
        throw IOException("The face engine did not start in time. Try again.")
    }

    private fun launchJob(
        kind: String,
        resumeId: String? = null,
        submit: (SwapApi, ContentResolver) -> String,
    ) {
        val ctx = getApplication<Application>()
        if (!ModelStore.allPresent(ctx)) {
            message = "Download the models first (step 1)."
            return
        }
        if (running) return
        viewModelScope.launch {
            running = true
            progress = 0f
            try {
                withContext(Dispatchers.IO) {
                    val token = serverPrefs.getString("token", "") ?: ""
                    val api = SwapApi("http://127.0.0.1:$PORT", token)
                    val cr = ctx.contentResolver
                    ensureEngine(ctx, api, token)
                    message = if (resumeId == null) "Preparing…" else "Resuming…"
                    val jobId = submit(api, cr)
                    prefs.edit().putString("job_id", jobId).putString("job_kind", kind).apply()

                    while (true) {
                        val st = api.status(jobId)
                        progress = if (st.total > 0) st.done.toFloat() / st.total else 0f
                        message = "Swapping: ${st.done}/${st.total}"
                        if (st.status == "error") error(st.error ?: "Engine error")
                        if (st.status == "cancelled") error("The job was cancelled")
                        if (st.status == "done") {
                            message = "Saving the result…"
                            val tmp = File(ctx.cacheDir, if (kind == "images") "result.zip" else "result.mp4")
                            api.downloadResult(jobId, tmp)
                            if (kind == "images") {
                                val n = Gallery.saveImagesFromZip(cr, tmp)
                                val sk = if (st.skipped.isEmpty()) "" else
                                    " Skipped ${st.skipped.size} (no face / unreadable)."
                                message = "Saved $n images to Pictures/FaceSwap.$sk"
                            } else {
                                Gallery.saveVideo(cr, tmp, "faceswap_${System.currentTimeMillis()}.mp4")
                                message = "Saved video to Movies/FaceSwap"
                            }
                            tmp.delete()
                            progress = 1f
                            break
                        }
                        delay(1500)
                    }
                }
            } catch (e: Exception) {
                message = "Failed: ${e.message}"
            } finally {
                running = false
            }
        }
    }
}
