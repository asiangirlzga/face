plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.chaquo.python")
}

android {
    namespace = "com.example.faceswapserver"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.faceswapserver"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Chaquopy's numpy/opencv wheels and ONNX Runtime are packaged for 64-bit ARM phones only.
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

chaquopy {
    defaultConfig {
        version = "3.10"   // opencv-python-headless is only published for Python 3.10 on Chaquopy
        pip {
            install("numpy==1.26.2")
            install("opencv-python-headless==4.5.1.48")
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.09.03")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
