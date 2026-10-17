package com.example.faceswapserver

import android.content.Context
import android.os.StatFs
import java.io.File
import java.io.IOException

/**
 * SD-Turbo as a CPU-compatible ONNX export (optimum, fp32). About 5.3 GB in total.
 * The NVIDIA-only "onnxruntime/sd-turbo" repo does not run on a phone CPU, so it is not used.
 * Models are downloaded into app storage (resumable) and loaded one at a time by sdgen/pipeline.py.
 */
object SdModelStore {
    const val DEFAULT_BASE = "https://huggingface.co/Heliosoph/sd-turbo-onnx/resolve/main/"

    class Item(val path: String, val label: String, val approxBytes: Long)

    val ITEMS = listOf(
        Item("tokenizer/vocab.json", "tokenizer", 1_109_372L),
        Item("tokenizer/merges.txt", "tokenizer", 573_514L),
        Item("tokenizer/special_tokens_map.json", "tokenizer", 604L),
        Item("tokenizer/tokenizer_config.json", "tokenizer", 924L),
        Item("vae_decoder/model.onnx", "image decoder", 198_078_154L),
        Item("text_encoder/model.onnx", "text encoder", 1_361_986_798L),
        Item("unet/model.onnx", "image model", 1_086_817L),
        Item("unet/model.onnx_data", "image model", 3_463_659_136L),
    )

    fun dir(ctx: Context): File = File(ctx.filesDir, "sd-turbo").also { it.mkdirs() }

    fun file(ctx: Context, rel: String): File = File(dir(ctx), rel)

    /** A file only gets its final name after it has been downloaded completely. */
    fun ready(ctx: Context): Boolean = ITEMS.all { file(ctx, it.path).isFile }

    fun freeBytes(ctx: Context): Long = StatFs(ctx.filesDir.path).availableBytes

    /** progress(label, doneBytes, totalBytes) */
    fun downloadAll(ctx: Context, baseUrl: String, progress: (String, Long, Long) -> Unit) {
        val base = baseUrl.trim().ifEmpty { DEFAULT_BASE }.let { if (it.endsWith("/")) it else "$it/" }
        val missing = ITEMS.filter { !file(ctx, it.path).isFile }
        val partial = missing.sumOf { File(file(ctx, it.path).path + ".part").let { p -> if (p.isFile) p.length() else 0L } }
        val need = missing.sumOf { it.approxBytes } - partial + 300_000_000L
        if (missing.isNotEmpty() && freeBytes(ctx) < need) {
            throw IOException("Not enough free storage: about ${need / 1_000_000} MB needed, ${freeBytes(ctx) / 1_000_000} MB free.")
        }
        ModelStore.abort = false
        for (item in missing) {
            val dest = file(ctx, item.path)
            dest.parentFile?.mkdirs()
            ModelStore.downloadWithRetry(base + item.path, dest) { done, total -> progress(item.label, done, total) }
        }
    }
}
