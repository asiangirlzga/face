package com.example.faceswapserver

import android.Manifest
import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.Inet4Address
import java.net.NetworkInterface
import java.security.SecureRandom

const val PORT = 8000

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("server", Context.MODE_PRIVATE)

    var token by mutableStateOf(prefs.getString("token", null) ?: newToken().also { prefs.edit().putString("token", it).apply() })
    var allowLan by mutableStateOf(prefs.getBoolean("lan", false))
    var customUrl by mutableStateOf(prefs.getString("custom_url", "") ?: "")
    var haveDet by mutableStateOf(false)
    var haveRec by mutableStateOf(false)
    var haveSwap by mutableStateOf(false)
    var busy by mutableStateOf(false)
    var progress by mutableStateOf(0f)
    var message by mutableStateOf("")

    init { refresh() }

    private fun newToken(): String {
        val b = ByteArray(12); SecureRandom().nextBytes(b)
        return android.util.Base64.encodeToString(b, android.util.Base64.URL_SAFE or android.util.Base64.NO_WRAP or android.util.Base64.NO_PADDING)
    }

    fun regenerateToken() { token = newToken(); prefs.edit().putString("token", token).apply() }

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

    fun start(ctx: Context) {
        prefs.edit().putBoolean("lan", allowLan).apply()
        val i = Intent(ctx, ServerService::class.java)
            .putExtra(ServerService.EXTRA_HOST, if (allowLan) "0.0.0.0" else "127.0.0.1")
            .putExtra(ServerService.EXTRA_PORT, PORT)
            .putExtra(ServerService.EXTRA_TOKEN, token)
        ContextCompat.startForegroundService(ctx, i)
    }

    fun stop(ctx: Context) {
        ctx.startService(Intent(ctx, ServerService::class.java).setAction(ServerService.ACTION_STOP))
    }

    fun urls(): List<String> {
        val out = mutableListOf("http://127.0.0.1:$PORT   (client app on this phone)")
        if (allowLan) {
            runCatching {
                NetworkInterface.getNetworkInterfaces().toList().filter { it.isUp && !it.isLoopback }.forEach { ni ->
                    ni.inetAddresses.toList().filterIsInstance<Inet4Address>().forEach {
                        out.add("http://${it.hostAddress}:$PORT   (other devices on this Wi-Fi)")
                    }
                }
            }
        }
        return out
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Surface(Modifier.fillMaxSize()) { Screen() } } }
    }
}

@Composable
fun Screen(vm: MainViewModel = viewModel()) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val askNotif = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { vm.start(ctx) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Face Swap Server", style = MaterialTheme.typography.headlineSmall)
        Text("Runs the face-swap engine on this phone. Use it only with photos and videos of people who agreed to be swapped.")

        Text("1. Models (about 850 MB, downloaded once)", style = MaterialTheme.typography.titleMedium)
        Text("Detector ${if (vm.haveDet) "✓" else "–"}   Recognizer ${if (vm.haveRec) "✓" else "–"}   Swap model ${if (vm.haveSwap) "✓" else "–"}")
        OutlinedTextField(
            vm.customUrl, { vm.customUrl = it },
            label = { Text("Custom URL for inswapper_128.onnx (optional)") },
            singleLine = true, modifier = Modifier.fillMaxWidth(),
        )
        Button(onClick = { vm.download() }, enabled = !vm.busy && !vm.modelsReady) {
            Text(if (vm.modelsReady) "Models ready" else "Download models (use Wi-Fi)")
        }
        if (vm.busy) LinearProgressIndicator(progress = { vm.progress }, modifier = Modifier.fillMaxWidth())
        if (vm.message.isNotEmpty()) Text(vm.message)

        Text("2. Server", style = MaterialTheme.typography.titleMedium)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Switch(vm.allowLan, { vm.allowLan = it }, enabled = !ServerState.running)
            Text("Allow other devices on my Wi-Fi")
        }
        if (!ServerState.running) {
            Button(onClick = {
                if (Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                ) askNotif.launch(Manifest.permission.POST_NOTIFICATIONS) else vm.start(ctx)
            }, enabled = vm.modelsReady) { Text("Start server") }
        } else {
            OutlinedButton(onClick = { vm.stop(ctx) }) { Text("Stop server") }
        }
        ServerState.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }

        if (ServerState.running) {
            Text("Engine: ${ServerState.engine}")
            Text("Last job: ${ServerState.job}")
            Text("Server URL for the client app:", style = MaterialTheme.typography.titleSmall)
            vm.urls().forEach { Text(it) }
        }
        Text("Token (paste into the client app):", style = MaterialTheme.typography.titleSmall)
        Text(vm.token)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = {
                (ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager)
                    .setPrimaryClip(ClipData.newPlainText("token", vm.token))
            }) { Text("Copy token") }
            OutlinedButton(onClick = { vm.regenerateToken() }, enabled = !ServerState.running) { Text("New token") }
        }
        Text("Keep this app running while you swap. Phones are slow at this: roughly 1–3 s per photo, and a video takes about 1–3 s per frame.")
    }
}
