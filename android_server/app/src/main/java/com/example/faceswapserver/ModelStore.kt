package com.example.faceswapserver

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/** Downloads the three ONNX models into app storage (resumable). */
object ModelStore {
    const val DET = "det_10g.onnx"
    const val REC = "w600k_r50.onnx"
    const val SWAP = "inswapper_128.onnx"

    private const val BUFFALO_URL =
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    private val SWAP_URLS = listOf(
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    )

    private val MIN_SIZE = mapOf(DET to 10_000_000L, REC to 100_000_000L, SWAP to 400_000_000L)

    fun dir(ctx: Context): File = File(ctx.filesDir, "models").also { it.mkdirs() }

    fun has(ctx: Context, name: String): Boolean {
        val f = File(dir(ctx), name)
        return f.isFile && f.length() >= (MIN_SIZE[name] ?: 1L)
    }

    /** progress(label, doneBytes, totalBytes) */
    fun downloadAll(ctx: Context, customSwapUrl: String, progress: (String, Long, Long) -> Unit) {
        val d = dir(ctx)
        if (!has(ctx, DET) || !has(ctx, REC)) {
            val zip = File(ctx.cacheDir, "buffalo_l.zip")
            downloadWithRetry(BUFFALO_URL, zip) { a, b -> progress("Detector + recognizer", a, b) }
            ZipInputStream(zip.inputStream().buffered()).use { zin ->
                var e = zin.nextEntry
                while (e != null) {
                    val name = e.name.substringAfterLast('/')
                    if (name == DET || name == REC) {
                        FileOutputStream(File(d, name)).use { zin.copyTo(it) }
                    }
                    e = zin.nextEntry
                }
            }
            zip.delete()
            if (!has(ctx, DET) || !has(ctx, REC)) throw IOException("buffalo_l.zip did not contain the expected models")
        }
        if (!has(ctx, SWAP)) {
            val urls = (listOf(customSwapUrl.trim()).filter { it.isNotEmpty() }) + SWAP_URLS
            var last: Exception? = null
            for (u in urls) {
                try {
                    downloadWithRetry(u, File(d, SWAP)) { a, b -> progress("Swap model", a, b) }
                    if (has(ctx, SWAP)) return
                    File(d, SWAP).delete()
                    throw IOException("Downloaded file is too small to be inswapper_128.onnx")
                } catch (e: Exception) {
                    last = e
                    File(d, SWAP + ".part").delete()
                }
            }
            throw IOException("Could not download the swap model (${last?.message}). Paste a direct URL for inswapper_128.onnx in the custom URL box.")
        }
    }

    /** Set to true to make a running download stop (it can be resumed later). */
    @Volatile
    var abort = false

    fun downloadWithRetry(url: String, dest: File, onProgress: (Long, Long) -> Unit) {
        var last: Exception? = null
        repeat(5) {
            try {
                download(url, dest, onProgress)
                return
            } catch (e: IOException) {
                if (abort) throw e
                last = e
                Thread.sleep(2000)
            }
        }
        throw last ?: IOException("download failed")
    }

    private fun download(url: String, dest: File, onProgress: (Long, Long) -> Unit) {
        val part = File(dest.path + ".part")
        val have = if (part.exists()) part.length() else 0L
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "FaceSwapServer/1.0")
            if (have > 0) setRequestProperty("Range", "bytes=$have-")
        }
        try {
            val code = conn.responseCode
            if (code == 416 && have > 0) {          // already complete
                part.renameTo(dest); return
            }
            if (code != 200 && code != 206) throw IOException("HTTP $code for $url")
            val append = code == 206 && have > 0
            var done = if (append) have else 0L
            val total = if (conn.contentLengthLong > 0) done + conn.contentLengthLong else -1L
            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    val buf = ByteArray(256 * 1024)
                    var lastReport = 0L
                    while (true) {
                        if (abort) throw IOException("Download cancelled")
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        val now = System.currentTimeMillis()
                        if (now - lastReport > 400) { onProgress(done, total); lastReport = now }
                    }
                }
            }
            onProgress(done, total)
            dest.delete()
            if (!part.renameTo(dest)) throw IOException("could not move download into place")
        } finally {
            conn.disconnect()
        }
    }
}
