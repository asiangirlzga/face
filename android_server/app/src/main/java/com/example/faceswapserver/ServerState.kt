package com.example.faceswapserver

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** Observable server status shared by the service and the UI (same process). */
object ServerState {
    var running by mutableStateOf(false)
    var engine by mutableStateOf("-")
    var job by mutableStateOf("")
    var error by mutableStateOf<String?>(null)
}
