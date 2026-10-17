package com.example.faceswap

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

data class CreateStatus(
    val status: String,      // queued | running | done | error | cancelled
    val fraction: Float,     // 0..1
    val stage: String,       // e.g. "Generating: step 1 of 1"
    val seed: Long?,
    val error: String?,
)

/** Text-to-image and image-to-video requests to the Face Swap Server app (offline on the phone). */
class CreateApi(baseUrl: String, private val token: String) {
    private val base = baseUrl.trim().trimEnd('/')

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.MINUTES)
        .readTimeout(5, TimeUnit.MINUTES)
        .build()

    private fun execute(req: Request): Response {
        val resp = http.newCall(req).execute()
        if (!resp.isSuccessful) {
            val raw = resp.body?.string().orEmpty()
            val code = resp.code
            resp.close()
            // The server answers errors as {"detail": "..."}; show just the sentence.
            val detail = runCatching { JSONObject(raw).getString("detail") }.getOrNull() ?: raw.take(200)
            throw IOException(if (code == 401) "Wrong token (server said 401)" else detail.ifEmpty { "Server returned $code" })
        }
        return resp
    }

    private fun jobIdFrom(req: Request): String =
        execute(req).use { JSONObject(it.body!!.string()).getString("job_id") }

    /** "ready" when the image model is on the phone, "missing" when it still has to be downloaded. */
    fun imageModelState(): String {
        val req = Request.Builder().url("$base/health").build()
        return execute(req).use { JSONObject(it.body!!.string()).optString("t2i", "missing") }
    }

    fun submitTextToImage(prompt: String, steps: Int, size: Int, seed: Long?, tiled: Boolean): String {
        val json = JSONObject().put("prompt", prompt).put("steps", steps).put("size", size).put("tiled", tiled)
        if (seed != null) json.put("seed", seed)
        val body = json.toString().toRequestBody("application/json".toMediaType())
        return jobIdFrom(Request.Builder().url("$base/jobs/text2img").header("X-Token", token).post(body).build())
    }

    fun submitImageToVideo(image: File, style: String, seconds: Int, fps: Int): String {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("image", image.name, image.asRequestBody("image/jpeg".toMediaType()))
            .addFormDataPart("style", style)
            .addFormDataPart("seconds", seconds.toString())
            .addFormDataPart("fps", fps.toString())
            .build()
        return jobIdFrom(Request.Builder().url("$base/jobs/img2video").header("X-Token", token).post(body).build())
    }

    fun status(jobId: String): CreateStatus {
        val req = Request.Builder().url("$base/jobs/$jobId").header("X-Token", token).build()
        execute(req).use {
            val j = JSONObject(it.body!!.string())
            val total = j.optInt("total", 0)
            val frac = if (total > 0) (j.optInt("done", 0).toFloat() / total).coerceIn(0f, 1f) else 0f
            return CreateStatus(
                status = j.getString("status"),
                fraction = frac,
                stage = j.optString("stage", ""),
                seed = if (j.isNull("seed")) null else j.optLong("seed"),
                error = if (j.isNull("error")) null else j.getString("error"),
            )
        }
    }

    fun cancel(jobId: String) {
        val req = Request.Builder().url("$base/jobs/$jobId/cancel").header("X-Token", token)
            .post("".toRequestBody(null)).build()
        execute(req).close()
    }

    fun download(jobId: String, dest: File) {
        val req = Request.Builder().url("$base/jobs/$jobId/result").header("X-Token", token).build()
        execute(req).use { resp -> dest.outputStream().use { out -> resp.body!!.byteStream().copyTo(out) } }
    }
}
