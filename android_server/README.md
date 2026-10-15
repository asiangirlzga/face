# Face Swap Server (runs on the phone)

An Android app that hosts the Python face-swap server on the phone itself, with the same API as the Colab server.
The client app (`android_app/`) works with it unchanged.

```
client app ──HTTP──> server.py (Python 3.10 via Chaquopy, in this app)
                         └─ engine.py: OpenCV + numpy pre/post-processing
                              └─ OrtBridge.kt: ONNX Runtime runs det_10g, w600k_r50, inswapper_128
                         └─ VideoBridge.kt: MediaCodec/MediaMuxer for video frames + audio
```

## Build the APK on GitHub
1. Put both folders (`android_app/` and `android_server/`) and `.github/workflows/android-build.yml` in the repo, then push.
2. Actions → "Build Android APKs" → open the run → download the `faceswap-apks` artifact. It holds both APKs.
3. Install `android_server-app-debug.apk` (and the client APK) on a 64-bit ARM phone.

## Use it
1. Open **Face Swap Server**, tap **Download models** on Wi-Fi (~850 MB, once). Detector and recognizer come from the insightface `buffalo_l.zip` GitHub release; the swap model from Hugging Face. If that link is down, paste a direct URL for `inswapper_128.onnx` in the custom URL box.
2. Tap **Start server**. Copy the token.
3. In the client app: Server URL `http://127.0.0.1:8000`, paste the token, pick the source face, then 1–50 images or a video.
   Other devices on your Wi-Fi: switch on "Allow other devices" and use the phone's IP shown in the app.

## Requirements and limits
- Android 7+ (API 24), **64-bit ARM only** (no emulator), 6 GB+ RAM recommended (the swap model alone is 554 MB).
- Speed: expect roughly 1–3 s per photo and 1–3 s per video frame, so a 10 s clip at 30 fps takes 5–15+ minutes. Keep the phone charging; the server runs as a foreground service with a wake lock.
- Video frames are downscaled to 1280 px max; images to 4096 px max.
- `inswapper_128` is licensed for non-commercial research use only.
- Use only with photos and videos of people who agreed.

## Tests (desktop)
`tests/test_engine.py` and `tests/test_server.py` run the Python side against the real detector/recognizer models:
`python -I tests/test_server.py /path/to/models` (needs numpy, opencv-python-headless, onnxruntime, requests, scikit-image).
