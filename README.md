# Face Swap: two ways to run the server

- `android_app/`: client app. Face swap tab (pick a face, up to 50 images or a video) and Text to image tab (prompt to image, image to camera-motion video). Results go to the gallery.
- `colab_server/`: GPU server for Google Colab (fast; needs a Colab runtime).
- `android_server/`: the server running on the phone itself (slower, no Colab needed). Also generates images from text and makes motion videos, fully offline.
- `.github/workflows/android-build.yml`: builds every APK on GitHub Actions.

See `android_server/README.md` for the on-phone server. Colab: upload `colab_server/FaceSwap_Server.ipynb`, use a GPU runtime, run all cells, paste the printed URL and token into the client app.

Use only with photos and videos of people who agreed to be swapped.
