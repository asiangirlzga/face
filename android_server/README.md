# Face Swap (single app, runs on the phone)

One Android app: pick a face and up to 50 photos or a video, tap Swap, results land in the gallery.
The Python engine runs inside the app (Chaquopy) and starts by itself when you tap Swap.

```
app UI (Compose) ──HTTP 127.0.0.1──> server.py (Python 3.10 via Chaquopy, same process, foreground service)
                                         └─ engine.py: OpenCV + numpy pre/post-processing
                                              └─ OrtBridge.kt: ONNX Runtime runs det_10g, w600k_r50, inswapper_128
                                         └─ VideoBridge.kt: MediaCodec/MediaMuxer for video frames + audio
```

## Build the APK on GitHub
Push to `main`; Actions → "Build Android APKs" → open the run → download the `faceswap-apks` artifact → install `android_server-app-debug.apk` on a 64-bit ARM phone (Android 10+).

## Updating without re-downloading the models
- Models (~850 MB) are saved in the app's private storage and are only downloaded if a file is missing.
- Every build is signed with the same key (`faceswap.keystore`) and has a higher version number (the GitHub run number), so a new APK installs **over** the old one and keeps the models. Just open the new APK and tap Update. **Do not uninstall first.**
- If the app is ever uninstalled, Android asks whether to keep its data (`hasFragileUserData`). Choose keep and the models are still there after reinstalling.
- The very first time you move from an older build (different signature) you must uninstall it once; that one time the models are downloaded again.
- `faceswap.keystore` is committed with a known password, so anyone with the repo can sign an APK that Android treats as an update to yours. That is fine for personal use. If you ever publish the app, make a private key and keep it out of the repo.

## Use it
1. Open **Face Swap**. Tap **Download models** on Wi-Fi (once). Detector and recognizer come from the insightface `buffalo_l.zip` GitHub release; the swap model from Hugging Face. If that link is down, paste a direct URL for `inswapper_128.onnx` in the custom URL box.
2. Pick the source face, then 1–50 images or a video, tick the consent box, tap **Swap images** or **Swap video**.
3. Results are saved to `Pictures/FaceSwap` and `Movies/FaceSwap`. **Stop engine** (shown while it runs) frees the memory.

## Result quality
The swap model works on a 128 px face, which is then enlarged onto the photo. The paste-back step (`Swapper.swap` in `engine.py`) therefore:
- blends through a face-shaped mask with a wide feather (no visible seam),
- matches colour and lighting to the photo's own skin (low-frequency difference over the skin area),
- smooths the enlarged pixel grid, boosts feature-sized detail, and adds the photo's own grain.
It cannot invent detail the model never produced, so a very large face still looks softer than the rest of the photo. A face-restoration model (GFPGAN/CodeFormer) would fix that but adds ~300 MB and is slow on a phone.

## Requirements and limits
- Android 10+ (API 29), **64-bit ARM only** (no emulator), 6 GB+ RAM recommended (the swap model alone is 554 MB).
- Speed: roughly 1–3 s per photo and 1–3 s per video frame, so a 10 s clip at 30 fps takes 5–15+ minutes. Keep the phone charging; the engine runs as a foreground service with a wake lock.
- Video frames are downscaled to 1280 px max; images to 4096 px max.
- `inswapper_128` is licensed for non-commercial research use only.
- Use only with photos and videos of people who agreed.

## Tests (desktop)
`tests/test_engine.py` and `tests/test_server.py` run the Python side against the real detector/recognizer models:
`python -I tests/test_server.py /path/to/models` (needs numpy, opencv-python-headless, onnxruntime, requests, scikit-image).
The Kotlin code is only compiled by GitHub Actions and exercised on a phone.
