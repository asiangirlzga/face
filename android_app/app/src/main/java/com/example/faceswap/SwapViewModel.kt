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

const val MAX_IMAGES = 50

class SwapViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("faceswap", Context.MODE_PRIVATE)

    var serverUrl by mutableStateOf(prefs.getString("url", "") ?: "")
    var token by mutableStateOf(prefs.getString("token", "") ?: "")
    var consent by mutableStateOf(false)

    var sourceFace by mutableStateOf<Uri?>(null)
    var targetImages by mutableStateOf<List<Uri>>(emptyList())
    var targetVideo by mutableStateOf<Uri?>(null)

    var running by mutableStateOf(false)
    var progress by mutableStateOf(0f)
    var message by mutableStateOf("Ready")

    val canRun get() = consent && !running && serverUrl.isNotBlank() && token.isNotBlank() && sourceFace != null

    fun setImages(uris: List<Uri>) {
        targetImages = uris.take(MAX_IMAGES)
        if (uris.size > MAX_IMAGES) message = "Only the first $MAX_IMAGES images were kept"
    }

    fun swapImages() = launchJob("images") { api, cr ->
        api.submitImages(cr, sourceFace!!, targetImages)
    }

    fun swapVideo() = launchJob("video") { api, cr ->
        api.submitVideo(cr, sourceFace!!, targetVideo!!)
    }

    fun resumeLastJob() {
        val id = prefs.getString("job_id", null) ?: run { message = "No previous job"; return }
        val kind = prefs.getString("job_kind", "images") ?: "images"
        launchJob(kind, resumeId = id) { _, _ -> id }
    }

    private fun launchJob(
        kind: String,
        resumeId: String? = null,
        submit: (SwapApi, android.content.ContentResolver) -> String,
    ) {
        prefs.edit().putString("url", serverUrl).putString("token", token).apply()
        val ctx = getApplication<Application>()
        viewModelScope.launch {
            running = true
            progress = 0f
            try {
                withContext(Dispatchers.IO) {
                    val api = SwapApi(serverUrl, token)
                    val cr = ctx.contentResolver
                    message = if (resumeId == null) "Uploading…" else "Resuming…"
                    val jobId = submit(api, cr)
                    prefs.edit().putString("job_id", jobId).putString("job_kind", kind).apply()

                    while (true) {
                        val st = api.status(jobId)
                        progress = if (st.total > 0) st.done.toFloat() / st.total else 0f
                        message = "Processing on server: ${st.done}/${st.total}"
                        if (st.status == "error") error(st.error ?: "Server error")
                        if (st.status == "done") {
                            message = "Downloading result…"
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
                        delay(2000)
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
