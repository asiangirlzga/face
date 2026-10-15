pluginManagement {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://maven-central.storage-download.googleapis.com/maven2/") }  // fallback mirror of Maven Central
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://maven-central.storage-download.googleapis.com/maven2/") }  // fallback mirror of Maven Central
    }
}
rootProject.name = "FaceSwapServer"
include(":app")
