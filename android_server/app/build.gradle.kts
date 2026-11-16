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
        minSdk = 29
        targetSdk = 35
        // Every GitHub build gets a higher number, so a new APK installs OVER the old one (keeping the
        // downloaded models) instead of being refused. Local builds use 1.
        val run = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 1
        versionCode = run
        versionName = "2.$run"
        // Chaquopy's numpy/opencv wheels and ONNX Runtime are packaged for 64-bit ARM phones only.
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    // One fixed signing key for every build. Android only installs an update when the signature matches the
    // installed app; CI used to generate a new random debug key each run, which forced an uninstall (and
    // deleted the models). Keep faceswap.keystore in the repo so every build is signed the same way.
    signingConfigs {
        create("shared") {
            storeFile = rootProject.file("faceswap.keystore")
            storePassword = "faceswap"
            keyAlias = "faceswap"
            keyPassword = "faceswap"
        }
    }
    buildTypes {
        getByName("debug") { signingConfig = signingConfigs.getByName("shared") }
        getByName("release") { signingConfig = signingConfigs.getByName("shared") }
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
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
