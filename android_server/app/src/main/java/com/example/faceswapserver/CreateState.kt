package com.example.faceswapserver

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** Observable status of the text-to-image / video / download work (same process as the UI). */
object CreateState {
    var busy by mutableStateOf(false)
    var mode by mutableStateOf("")            // "download", "image" or "video"
    var stage by mutableStateOf("")
    var progress by mutableStateOf(0f)
    var message by mutableStateOf("")
    var error by mutableStateOf<String?>(null)
    var imagePath by mutableStateOf<String?>(null)
    var imageVersion by mutableStateOf(0)
    var lastSeed by mutableStateOf<Long?>(null)
}
