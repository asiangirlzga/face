package com.example.faceswapserver

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.IOException

/** Saves results to the gallery (Pictures/TextToImage, Movies/TextToImage). */
object MediaSaver {
    /** Returns a human-readable location. */
    fun saveImage(ctx: Context, file: File, name: String): String =
        save(ctx, file, name, "image/jpeg", MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.MIME_TYPE,
            MediaStore.Images.Media.RELATIVE_PATH, "Pictures/TextToImage", Environment.DIRECTORY_PICTURES)

    fun saveVideo(ctx: Context, file: File, name: String): String =
        save(ctx, file, name, "video/mp4", MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Video.Media.DISPLAY_NAME, MediaStore.Video.Media.MIME_TYPE,
            MediaStore.Video.Media.RELATIVE_PATH, "Movies/TextToImage", Environment.DIRECTORY_MOVIES)

    private fun save(
        ctx: Context, file: File, name: String, mime: String, collection: android.net.Uri,
        nameCol: String, mimeCol: String, pathCol: String, relativePath: String, legacyDir: String,
    ): String {
        if (Build.VERSION.SDK_INT < 29) {
            // No scoped storage: keep it in the app's own media folder.
            val dir = File(ctx.getExternalFilesDir(legacyDir), "TextToImage").also { it.mkdirs() }
            file.copyTo(File(dir, name), overwrite = true)
            return dir.absolutePath
        }
        val values = ContentValues().apply {
            put(nameCol, name)
            put(mimeCol, mime)
            put(pathCol, relativePath)
        }
        val uri = ctx.contentResolver.insert(collection, values) ?: throw IOException("Could not create the gallery entry")
        ctx.contentResolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
        return relativePath
    }
}
