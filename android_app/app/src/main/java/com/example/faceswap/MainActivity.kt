package com.example.faceswap

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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) { Screen() }
            }
        }
    }
}

@Composable
fun Screen(vm: SwapViewModel = viewModel()) {
    val pickFace = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) vm.sourceFace = it
    }
    val pickImages = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(MAX_IMAGES)
    ) { vm.setImages(it) }
    val pickVideo = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) vm.targetVideo = it
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Face Swap (Colab server)", style = MaterialTheme.typography.headlineSmall)

        OutlinedTextField(
            vm.serverUrl, { vm.serverUrl = it },
            label = { Text("Server URL (Colab https://… or phone http://127.0.0.1:8000)") },
            singleLine = true, modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            vm.token, { vm.token = it },
            label = { Text("Token") },
            singleLine = true, modifier = Modifier.fillMaxWidth(),
        )

        Text("1. Face to use", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickFace.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) { Text(if (vm.sourceFace == null) "Pick source face photo" else "Source face selected ✓") }

        Text("2a. Batch images (max $MAX_IMAGES)", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickImages.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) { Text("Pick images (${vm.targetImages.size} selected)") }

        Text("2b. Video", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = {
            pickVideo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
        }) { Text(if (vm.targetVideo == null) "Pick video" else "Video selected ✓") }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(vm.consent, { vm.consent = it })
            Text("Everyone shown has agreed to this, and I won't use the results to deceive or harm anyone.")
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { vm.swapImages() },
                enabled = vm.canRun && vm.targetImages.isNotEmpty(),
            ) { Text("Swap images") }
            Button(
                onClick = { vm.swapVideo() },
                enabled = vm.canRun && vm.targetVideo != null,
            ) { Text("Swap video") }
        }
        OutlinedButton(onClick = { vm.resumeLastJob() }, enabled = !vm.running) {
            Text("Resume last job")
        }

        if (vm.running) LinearProgressIndicator(progress = { vm.progress }, modifier = Modifier.fillMaxWidth())
        Text(vm.message)
    }
}
