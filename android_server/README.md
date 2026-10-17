# Face Swap Server + offline Text to image (runs on the phone)

An Android app that hosts the Python face-swap server on the phone itself, with the same API as the Colab server.
It also generates images from a text prompt and turns an image into a camera-motion video, fully offline.
The client app (`android_app/`) works with all of it.

```
client app ──HTTP──> server.py (Python 3.10 via Chaquopy, in this app)
                         └─ engine.py: OpenCV + numpy pre/post-processing
                              └─ OrtBridge.kt: ONNX Runtime runs det_10g, w600k_r50, inswapper_128
                         └─ VideoBridge.kt: MediaCodec/MediaMuxer for video frames + audio
                         └─ sdgen/: text to image (SD-Turbo via ONNX Runtime) and image to video (OpenCV)
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

## Text to image and image to video (offline)
1. In the server app open the **Text to image** tab and tap **Download image model** (about 5.3 GB, once, on Wi-Fi; needs about 6 GB free storage). After that nothing needs the internet.
   The files come from the Hugging Face repo `Heliosoph/sd-turbo-onnx` (SD-Turbo exported to ONNX, fp32 for phone CPUs): text encoder 1.36 GB, UNet 3.46 GB, VAE decoder 198 MB, tokenizer files. You can point the "Model repo URL" box at another repo with the same folder layout.
2. Type a prompt, pick steps (1 is fastest, 4 a bit sharper) and size (512 is the model's native size), tap **Generate image**. The result is saved to `Pictures/TextToImage`.
3. **Animate** adds a zoom, pan or drift over any still image (`Movies/TextToImage`). This is a camera-motion effect: the subject does not move. Real AI image-to-video needs a GPU and cannot run on a phone.
4. From the client app: Text to image tab, same URL and token (needs **Start server** in the server app). HTTP API:
   `POST /jobs/text2img` JSON `{prompt, steps 1-8, size 256-768 (multiple of 64), seed?, tiled?}`,
   `POST /jobs/img2video` form `image, style, seconds 1-10, fps 12-30`,
   `POST /jobs/<id>/cancel`, then the usual `GET /jobs/<id>` and `GET /jobs/<id>/result` (JPEG or MP4).
- Memory: models are loaded one at a time (text encoder, then UNet, then decoder) and the face models are unloaded first, so a 6 GB phone can run it. Close other apps first; if Android still kills the app, choose 384 px and keep "Low-memory decoding" on.
- Speed on a phone CPU: roughly one to several minutes per image (not measured on a device).
- If the image model rejects its `timestep` input (error mentioning `/time_embedding/linear_1/Gemm`), the app retries once with a patched copy of the small graph file (`unet/model_ts0.onnx` or `model_ts2.onnx`), remembers what worked in `unet/timestep_rank.txt`, and carries on. If nothing works, the error text lists how the graph declares and uses `timestep`; send that text to whoever maintains the app.
- Licence: SD-Turbo is under the Stability AI Community License (free for research and for commercial use below US$1M annual revenue; its Acceptable Use Policy applies to derivatives). Selling anything built on it needs a lawyer's check.

## Requirements and limits
- Android 7+ (API 24), **64-bit ARM only** (no emulator), 6 GB+ RAM recommended (the swap model alone is 554 MB).
- Speed: expect roughly 1–3 s per photo and 1–3 s per video frame, so a 10 s clip at 30 fps takes 5–15+ minutes. Keep the phone charging; the server runs as a foreground service with a wake lock.
- Video frames are downscaled to 1280 px max; images to 4096 px max.
- `inswapper_128` is licensed for non-commercial research use only.
- Use only with photos and videos of people who agreed.

## Tests (desktop)
`tests/test_engine.py` and `tests/test_server.py` run the face-swap Python side against the real detector/recognizer models:
`python -I tests/test_server.py /path/to/models` (needs numpy, opencv-python-headless, onnxruntime, requests, scikit-image).
`tests/test_sdgen.py` and `tests/test_server_create.py` cover text to image and image to video with a stand-in ONNX backend (no model download needed):
`python -I tests/test_sdgen.py [path/to/openai-CLIP/clip]` (the optional folder from `git clone https://github.com/openai/CLIP` also checks the tokenizer against OpenAI's reference).
The real SD-Turbo model and the Kotlin code are only exercised on a phone.
