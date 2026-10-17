package com.example.faceswap

import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
private fun Choice(label: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    if (selected) Button(onClick = onClick, enabled = enabled) { Text(label) }
    else OutlinedButton(onClick = onClick, enabled = enabled) { Text(label) }
}

/**
 * Text to image and motion video, generated offline by the Face Swap Server app on this phone.
 * The server URL and token come from the shared fields above the tabs.
 */
@Composable
fun CreateScreen(serverUrl: String, token: String, vm: CreateViewModel = viewModel()) {
    val busy = vm.running
    val connected = serverUrl.isNotBlank() && token.isNotBlank()
    val pickPhoto = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) vm.animatePicked(serverUrl, token, uri)
    }

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Text to image (offline)", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Uses the Face Swap Server app on this phone. Open it, download the image model once " +
                "(Text to image tab, about 5.3 GB), tap Start server, then paste its URL and token above. " +
                "Everything runs on the phone, so a 6 GB phone needs several minutes per image."
        )

        Text("Prompt", style = MaterialTheme.typography.titleMedium)
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
                onClick = { vm.generate(serverUrl, token) },
                enabled = connected && vm.prompt.isNotBlank(),
            ) { Text("Generate image") }
        } else {
            OutlinedButton(onClick = { vm.cancel() }) { Text("Cancel") }
        }
        if (!connected) Text("Enter the server URL and token above first.")

        if (busy) LinearProgressIndicator(progress = { vm.progress }, modifier = Modifier.fillMaxWidth())
        if (busy && vm.stage.isNotEmpty()) Text(vm.stage)
        if (vm.message.isNotEmpty()) Text(vm.message)

        val file = vm.imageFile
        if (file != null && file.isFile) {
            vm.lastSeed?.let { if (!busy) Text("Seed used: $it") }
            val bmp = remember(file.path, vm.imageVersion) { BitmapFactory.decodeFile(file.path)?.asImageBitmap() }
            if (bmp != null) {
                Image(
                    bitmap = bmp, contentDescription = "Generated image",
                    modifier = Modifier.fillMaxWidth(), contentScale = ContentScale.Fit,
                )
            }
        }

        Text("Motion video from an image", style = MaterialTheme.typography.titleMedium)
        Text("Adds a camera zoom or pan over a still image. The subject itself does not move; this is not AI animation.")
        OutlinedButton(
            onClick = { vm.styleIndex = (vm.styleIndex + 1) % vm.styles.size },
            enabled = !busy,
        ) { Text("Style: ${vm.styles[vm.styleIndex].second} (tap to change)") }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(3, 4, 6).forEach { s -> Choice("$s s", vm.seconds == s, !busy) { vm.seconds = s } }
        }
        Button(
            onClick = { file?.let { vm.animate(serverUrl, token, it) } },
            enabled = connected && !busy && file != null && file.isFile,
        ) { Text("Animate the image above") }
        OutlinedButton(
            onClick = { pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            enabled = connected && !busy,
        ) { Text("Pick a photo and animate it") }

        Text(
            "Results are saved to Pictures/TextToImage and Movies/TextToImage. " +
                "The SD-Turbo model is under the Stability AI Community License (free below US$1M yearly revenue). " +
                "Do not create images of real people without their consent, or content that harms others.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
