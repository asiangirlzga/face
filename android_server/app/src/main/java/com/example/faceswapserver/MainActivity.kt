package com.example.faceswapserver

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.security.SecureRandom

/** The engine listens on this phone only (127.0.0.1); the token keeps other apps out. */
const val PORT = 8000

fun startEngine(ctx: Context, token: String) {
    val i = Intent(ctx, ServerService::class.java)
        .putExtra(ServerService.EXTRA_HOST, "127.0.0.1")
        .putExtra(ServerService.EXTRA_PORT, PORT)
        .putExtra(ServerService.EXTRA_TOKEN, token)
    ContextCompat.startForegroundService(ctx, i)
}

fun stopEngine(ctx: Context) {
    ctx.startService(Intent(ctx, ServerService::class.java).setAction(ServerService.ACTION_STOP))
}

/** Model download state. Models live in the app's private storage and survive app updates. */
class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("server", Context.MODE_PRIVATE)

    val token: String = prefs.getString("token", null) ?: newToken().also { prefs.edit().putString("token", it).apply() }
    var customUrl by mutableStateOf(prefs.getString("custom_url", "") ?: "")
    var haveDet by mutableStateOf(false)
    var haveRec by mutableStateOf(false)
    var haveSwap by mutableStateOf(false)
    var busy by mutableStateOf(false)
    var progress by mutableStateOf(0f)
    var message by mutableStateOf("")

    init {
        // Earlier builds could download a 5 GB text-to-image model. That feature is gone, so free the space.
        Thread { runCatching { File(app.filesDir, "sd-turbo").deleteRecursively() } }.start()
        refresh()
    }

    private fun newToken(): String {
        val b = ByteArray(12); SecureRandom().nextBytes(b)
        return android.util.Base64.encodeToString(b, android.util.Base64.URL_SAFE or android.util.Base64.NO_WRAP or android.util.Base64.NO_PADDING)
    }

    fun refresh() {
        val c = getApplication<Application>()
        haveDet = ModelStore.has(c, ModelStore.DET)
        haveRec = ModelStore.has(c, ModelStore.REC)
        haveSwap = ModelStore.has(c, ModelStore.SWAP)
    }

    val modelsReady get() = haveDet && haveRec && haveSwap

    fun download() {
        if (busy) return
        prefs.edit().putString("custom_url", customUrl).apply()
        val c = getApplication<Application>()
        viewModelScope.launch {
            busy = true; message = "Starting download…"; progress = 0f
            try {
                withContext(Dispatchers.IO) {
                    ModelStore.downloadAll(c, customUrl) { label, done, total ->
                        progress = if (total > 0) done.toFloat() / total else 0f
                        message = "$label: ${done / 1_000_000} MB" + if (total > 0) " / ${total / 1_000_000} MB" else ""
                    }
                }
                message = "Models downloaded"
            } catch (e: Exception) {
                message = "Download failed: ${e.message}"
            } finally {
                busy = false; refresh()
            }
        }
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Surface(Modifier.fillMaxSize()) { Screen() } } }
    }
}

@Composable
fun Screen(mvm: MainViewModel = viewModel(), vm: SwapViewModel = viewModel()) {
    val ctx = LocalContext.current
    val askNotif = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    LaunchedEffect(Unit) {   // lets Android show the "engine running" notification
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
    val pickFace = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) vm.sourceFace = it
    }
    val pickImages = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(MAX_IMAGES)
    ) { vm.setImages(it) }
    val pickVideo = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) vm.targetVideo = it
    }
    val version = remember(ctx) {
        runCatching { ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName }.getOrNull() ?: ""
    }

    Column(
        Modifier.fillMaxSize().statusBarsPadding().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Face Swap", style = MaterialTheme.typography.headlineSmall)
        Text("Everything runs on this phone. Use it only with photos and videos of people who agreed to be swapped.")

        Text("1. Models (about 850 MB, downloaded once)", style = MaterialTheme.typography.titleMedium)
        Text("Detector ${if (mvm.haveDet) "✓" else "–"}   Recognizer ${if (mvm.haveRec) "✓" else "–"}   Swap model ${if (mvm.haveSwap) "✓" else "–"}")
        if (mvm.modelsReady) {
            Text("Models are saved in the app and stay there when you update it.")
        } else {
            OutlinedTextField(
                mvm.customUrl, { mvm.customUrl = it },
                label = { Text("Custom URL for inswapper_128.onnx (optional)") },
                singleLine = true, modifier = Modifier.fillMaxWidth(),
            )
            Button(onClick = { mvm.download(); vm.message = "Ready" }, enabled = !mvm.busy) {
                Text("Download models (use Wi-Fi)")
            }
        }
        if (mvm.busy) LinearProgressIndicator(progress = { mvm.progress }, modifier = Modifier.fillMaxWidth())
        if (mvm.message.isNotEmpty()) Text(mvm.message)

        Text("2. Face to use", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickFace.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) { Text(if (vm.sourceFace == null) "Pick source face photo" else "Source face selected ✓") }

        Text("3a. Batch images (max $MAX_IMAGES)", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickImages.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) { Text("Pick images (${vm.targetImages.size} selected)") }

        Text("3b. Video", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickVideo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
        }) { Text(if (vm.targetVideo == null) "Pick video" else "Video selected ✓") }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(vm.consent, { vm.consent = it })
            Text("Everyone shown has agreed to this, and I won't use the results to deceive or harm anyone.")
        }

        val ready = mvm.modelsReady && vm.canRun
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { vm.swapImages() }, enabled = ready && vm.targetImages.isNotEmpty()) { Text("Swap images") }
            Button(onClick = { vm.swapVideo() }, enabled = ready && vm.targetVideo != null) { Text("Swap video") }
        }
        OutlinedButton(onClick = { vm.resumeLastJob() }, enabled = !vm.running && mvm.modelsReady) {
            Text("Resume last job")
        }

        if (vm.running) LinearProgressIndicator(progress = { vm.progress }, modifier = Modifier.fillMaxWidth())
        Text(vm.message)

        Text(
            "The face engine starts by itself when you tap Swap and keeps running (see the notification) " +
                "so the next swap starts faster. Phones are slow at this: about 1–3 s per photo and 1–3 s per video frame.",
            style = MaterialTheme.typography.bodySmall,
        )
        ServerState.error?.let { Text("Engine error: $it", color = MaterialTheme.colorScheme.error) }
        if (ServerState.running) {
            Text("Engine: ${ServerState.engine}", style = MaterialTheme.typography.bodySmall)
            OutlinedButton(onClick = { stopEngine(ctx) }) { Text("Stop engine (frees memory)") }
        }
        Text("Version $version", style = MaterialTheme.typography.bodySmall)
    }
}
