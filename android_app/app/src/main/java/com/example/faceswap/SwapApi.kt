package com.example.faceswap

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import okio.BufferedSink
import okio.source
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

data class JobStatus(
    val status: String,
    val done: Int,
    val total: Int,
    val error: String?,
    val skipped: List<String>,
)

/** Talks to server.py running in Google Colab. */
class SwapApi(baseUrl: String, private val token: String) {
    private val base = baseUrl.trim().trimEnd('/')

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.MINUTES)
        .readTimeout(15, TimeUnit.MINUTES)
        .build()

    private fun uriBody(cr: ContentResolver, uri: Uri, mime: String): RequestBody =
        object : RequestBody() {
            override fun contentType() = mime.toMediaType()
            override fun contentLength(): Long =
                cr.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use {
                    if (it.moveToFirst()) it.getLong(0) else -1L
                } ?: -1L

            override fun writeTo(sink: BufferedSink) {
                cr.openInputStream(uri)!!.use { sink.writeAll(it.source()) }
            }
        }

    private fun displayName(cr: ContentResolver, uri: Uri, fallback: String): String =
        cr.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        } ?: fallback

    private fun execute(req: Request): Response {
        val resp = http.newCall(req).execute()
        if (!resp.isSuccessful) {
            val msg = resp.body?.string().orEmpty().take(200)
            resp.close()
            throw IOException("Server returned ${resp.code}: $msg")
        }
        return resp
    }

    private fun post(path: String, body: MultipartBody): String {
        val req = Request.Builder().url("$base$path").header("X-Token", token).post(body).build()
        execute(req).use { return JSONObject(it.body!!.string()).getString("job_id") }
    }

    fun submitImages(cr: ContentResolver, source: Uri, targets: List<Uri>): String {
        val b = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("source", "source.jpg", uriBody(cr, source, "image/jpeg"))
        targets.forEachIndexed { i, uri ->
            b.addFormDataPart("targets", displayName(cr, uri, "img_$i.jpg"), uriBody(cr, uri, "image/jpeg"))
        }
        return post("/jobs/images", b.build())
    }

    fun submitVideo(cr: ContentResolver, source: Uri, video: Uri): String {
        val b = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("source", "source.jpg", uriBody(cr, source, "image/jpeg"))
            .addFormDataPart("video", displayName(cr, video, "video.mp4"), uriBody(cr, video, "video/mp4"))
        return post("/jobs/video", b.build())
    }

    fun status(jobId: String): JobStatus {
        val req = Request.Builder().url("$base/jobs/$jobId").header("X-Token", token).build()
        execute(req).use {
            val j = JSONObject(it.body!!.string())
            val sk = j.optJSONArray("skipped")
            return JobStatus(
                j.getString("status"), j.getInt("done"), j.getInt("total"),
                if (j.isNull("error")) null else j.getString("error"),
                (0 until (sk?.length() ?: 0)).map { i -> sk!!.getString(i) },
            )
        }
    }

    fun downloadResult(jobId: String, dest: File) {
        val req = Request.Builder().url("$base/jobs/$jobId/result").header("X-Token", token).build()
        execute(req).use { resp ->
            dest.outputStream().use { out -> resp.body!!.byteStream().copyTo(out) }
        }
    }
}

/** Saves results into the gallery (no storage permission needed on Android 10+). */
object Gallery {
    fun saveImagesFromZip(cr: ContentResolver, zip: File): Int {
        var count = 0
        ZipInputStream(zip.inputStream().buffered()).use { zin ->
            var entry = zin.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) {
                    val values = ContentValues().apply {
                        put(MediaStore.Images.Media.DISPLAY_NAME, entry!!.name)
                        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/FaceSwap")
                    }
                    val uri = cr.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    if (uri != null) {
                        cr.openOutputStream(uri)!!.use { zin.copyTo(it) }
                        count++
                    }
                }
                entry = zin.nextEntry
            }
        }
        return count
    }

    fun saveVideo(cr: ContentResolver, file: File, name: String) {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/FaceSwap")
        }
        val uri = cr.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Could not create video entry")
        cr.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
    }
}
