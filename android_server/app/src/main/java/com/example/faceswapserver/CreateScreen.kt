package com.example.faceswapserver

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class CreateViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("create", Context.MODE_PRIVATE)

    var prompt by mutableStateOf(prefs.getString("prompt", "") ?: "")
    var steps by mutableStateOf(prefs.getInt("steps", 1))
    var size by mutableStateOf(prefs.getInt("size", 512))
    var seedText by mutableStateOf("")
    var tiled by mutableStateOf(prefs.getBoolean("tiled", true))
    var baseUrl by mutableStateOf(prefs.getString("base", SdModelStore.DEFAULT_BASE) ?: SdModelStore.DEFAULT_BASE)
    var styleIndex by mutableStateOf(0)
    var seconds by mutableStateOf(4)
    var modelsReady by mutableStateOf(SdModelStore.ready(app))

    val styles = listOf(
        "zoom_in" to "Zoom in", "zoom_out" to "Zoom out",
        "pan_left" to "Pan left", "pan_right" to "Pan right",
        "pan_up" to "Pan up", "pan_down" to "Pan down", "drift" to "Drift",
    )

    fun refreshModels() {
        modelsReady = SdModelStore.ready(getApplication())
    }

    private fun savePrefs() {
        prefs.edit().putString("prompt", prompt).putInt("steps", steps).putInt("size", size)
            .putBoolean("tiled", tiled).putString("base", baseUrl).apply()
    }

    private fun intent(ctx: Context, mode: String) =
        Intent(ctx, CreateService::class.java).putExtra(CreateService.EXTRA_MODE, mode)

    fun download(ctx: Context) {
        savePrefs()
        ContextCompat.startForegroundService(ctx, intent(ctx, "download").putExtra(CreateService.EXTRA_BASE_URL, baseUrl))
    }

    fun generate(ctx: Context) {
        savePrefs()
        val i = intent(ctx, "image")
            .putExtra(CreateService.EXTRA_PROMPT, prompt)
            .putExtra(CreateService.EXTRA_STEPS, steps)
            .putExtra(CreateService.EXTRA_SIZE, size)
            .putExtra(CreateService.EXTRA_TILED, tiled)
        seedText.trim().toLongOrNull()?.let { if (it >= 0) i.putExtra(CreateService.EXTRA_SEED, it) }
        ContextCompat.startForegroundService(ctx, i)
    }

    fun animate(ctx: Context, imagePath: String) {
        ContextCompat.startForegroundService(
            ctx,
            intent(ctx, "video")
                .putExtra(CreateService.EXTRA_IMAGE, imagePath)
                .putExtra(CreateService.EXTRA_SECONDS, seconds)
                .putExtra(CreateService.EXTRA_FPS, 24)
                .putExtra(CreateService.EXTRA_STYLE, styles[styleIndex].first),
        )
    }

    /** Copies a picked photo into the cache and starts the motion video for it. */
    fun animatePicked(ctx: Context, uri: Uri) {
        viewModelScope.launch {
            val f = withContext(Dispatchers.IO) {
                File(ctx.cacheDir, "anim_src_${System.currentTimeMillis()}.jpg").also { dest ->
                    ctx.contentResolver.openInputStream(uri)?.use { input -> dest.outputStream().use { input.copyTo(it) } }
                }
            }
            animate(ctx, f.absolutePath)
        }
    }

    fun cancel(ctx: Context) {
        ctx.startService(Intent(ctx, CreateService::class.java).setAction(CreateService.ACTION_CANCEL))
    }
}

@Composable
private fun Choice(label: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    if (selected) Button(onClick = onClick, enabled = enabled) { Text(label) }
    else OutlinedButton(onClick = onClick, enabled = enabled) { Text(label) }
}

@Composable
fun CreateScreen(vm: CreateViewModel = viewModel()) {
    val ctx = LocalContext.current
    val busy = CreateState.busy

    var pending by remember { mutableStateOf<(() -> Unit)?>(null) }
    val askNotif = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        pending?.invoke()
        pending = null
    }
    fun withNotifPermission(action: () -> Unit) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            pending = action
            askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            action()
        }
    }
    val pickPhoto = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) withNotifPermission { vm.animatePicked(ctx, uri) }
    }

    LaunchedEffect(busy) { if (!busy) vm.refreshModels() }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Text to image (offline)", style = MaterialTheme.typography.headlineSmall)
        Text("Runs entirely on this phone. After the one-time model download it needs no internet. On a 6 GB phone expect several minutes per image.")

        Text("1. Image model (about 5.3 GB, one time, use Wi-Fi)", style = MaterialTheme.typography.titleMedium)
        Text(if (vm.modelsReady) "Model files: ready ✓" else "Model files: not downloaded yet")
        OutlinedTextField(
            vm.baseUrl, { vm.baseUrl = it },
            label = { Text("Model repo URL (advanced)") },
            singleLine = true, enabled = !busy, modifier = Modifier.fillMaxWidth(),
        )
        Button(
            onClick = { withNotifPermission { vm.download(ctx) } },
            enabled = !busy && !vm.modelsReady,
        ) { Text(if (vm.modelsReady) "Model ready" else "Download image model") }

        Text("2. Prompt", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            vm.prompt, { vm.prompt = it },
            label = { Text("Describe the image (English works best)") },
            minLines = 2, enabled = !busy, modifier = Modifier.fillMaxWidth(),
        )
        Text("Steps (1 is fastest, 4 is a bit sharper)")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(1, 2, 4).forEach { n -> Choice("$n", vm.steps == n, !busy) { vm.steps = n } }
        }
        Text("Size")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(512, 384).forEach { s -> Choice("${s}px", vm.size == s, !busy) { vm.size = s } }
        }
        OutlinedTextField(
            vm.seedText, { vm.seedText = it },
            label = { Text("Seed (optional, same seed + prompt = same image)") },
            singleLine = true, enabled = !busy,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth(),
        )
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Switch(vm.tiled, { vm.tiled = it }, enabled = !busy)
            Text("Low-memory decoding (recommended)")
        }

        if (!busy) {
            Button(
                onClick = { withNotifPermission { vm.generate(ctx) } },
                enabled = vm.modelsReady && vm.prompt.isNotBlank(),
            ) { Text("Generate image") }
        } else {
            OutlinedButton(onClick = { vm.cancel(ctx) }) { Text("Cancel") }
        }

        if (busy) LinearProgressIndicator(progress = { CreateState.progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        if (busy && CreateState.stage.isNotEmpty()) Text(CreateState.stage)
        if (CreateState.message.isNotEmpty()) Text(CreateState.message)
        CreateState.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }
        CreateState.lastSeed?.let { if (!busy && CreateState.imagePath != null) Text("Seed used: $it") }

        val path = CreateState.imagePath
        if (path != null) {
            val bmp = remember(path, CreateState.imageVersion) { BitmapFactory.decodeFile(path)?.asImageBitmap() }
            if (bmp != null) {
                Image(
                    bitmap = bmp, contentDescription = "Generated image",
                    modifier = Modifier.fillMaxWidth(), contentScale = ContentScale.Fit,
                )
            }
        }

        Text("3. Motion video from an image", style = MaterialTheme.typography.titleMedium)
        Text("Adds a camera zoom or pan over a still image. The subject itself does not move (this is not AI animation).")
        OutlinedButton(
            onClick = { vm.styleIndex = (vm.styleIndex + 1) % vm.styles.size },
            enabled = !busy,
        ) { Text("Style: ${vm.styles[vm.styleIndex].second} (tap to change)") }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(3, 4, 6).forEach { s -> Choice("$s s", vm.seconds == s, !busy) { vm.seconds = s } }
        }
        Button(
            onClick = { path?.let { p -> withNotifPermission { vm.animate(ctx, p) } } },
            enabled = !busy && path != null,
        ) { Text("Animate the image above") }
        OutlinedButton(
            onClick = { pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            enabled = !busy,
        ) { Text("Pick a photo and animate it") }

        Text(
            "Results are saved to Pictures/TextToImage and Movies/TextToImage. " +
                "SD-Turbo is licensed under the Stability AI Community License, so check its terms before commercial use. " +
                "Do not create images of real people without their consent or content that harms others.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
