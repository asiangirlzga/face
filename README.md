# Face Swap

- `android_server/`: **the app** (one APK). Face swap for up to 50 photos or a video, fully on the phone. The engine runs inside the app and starts by itself when you tap Swap.
- `colab_server/`: optional GPU server for Google Colab. It is not used by the app; it is kept for running the same API on a GPU.
- `.github/workflows/android-build.yml`: builds the APK on GitHub Actions.

See `android_server/README.md`. Use only with photos and videos of people who agreed to be swapped.

The earlier separate client app (`android_app/`) and the offline text-to-image feature were removed when the app became a single APK. They are still in the git history (commit `e904cfe` and earlier).
