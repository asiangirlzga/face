package com.example.faceswap

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import java.io.File
import java.io.IOException

/** Saves generated images and videos to the gallery (no storage permission needed on Android 10+). */
object MediaSaver {
    fun saveImage(cr: ContentResolver, file: File, name: String): String =
        save(cr, file, name, "image/jpeg", MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.MIME_TYPE,
            MediaStore.Images.Media.RELATIVE_PATH, "Pictures/TextToImage")

    fun saveVideo(cr: ContentResolver, file: File, name: String): String =
        save(cr, file, name, "video/mp4", MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Video.Media.DISPLAY_NAME, MediaStore.Video.Media.MIME_TYPE,
            MediaStore.Video.Media.RELATIVE_PATH, "Movies/TextToImage")

    private fun save(
        cr: ContentResolver, file: File, name: String, mime: String, collection: Uri,
        nameCol: String, mimeCol: String, pathCol: String, relativePath: String,
    ): String {
        val values = ContentValues().apply {
            put(nameCol, name)
            put(mimeCol, mime)
            put(pathCol, relativePath)
        }
        val uri = cr.insert(collection, values) ?: throw IOException("Could not create the gallery entry")
        cr.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
        return relativePath
    }
}
