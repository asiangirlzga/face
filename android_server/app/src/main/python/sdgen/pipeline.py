"""On-device text-to-image with SD-Turbo (Stable Diffusion 2.1 distilled, 1-4 steps, no guidance).

Neural-net inference runs in ONNX Runtime (Kotlin side, see OrtBridge) through a *backend*:

    backend.load(name, path, opt)         opt: "none" | "basic" | "all" (graph optimisation level)
    backend.unload(name)
    backend.input_info(name)  -> [(input_name, dtype_str)]   dtype_str: float32|float16|int64|int32
    backend.output_names(name) -> [str]
    backend.run(name, {input_name: ndarray}) -> [flat float32 ndarray per output]

Only one model is held in memory at a time (text encoder -> UNet -> VAE decoder), because the
UNet alone is ~3.5 GB in fp32. Everything else (tokenizer, Euler scheduler, tiled VAE decode,
progress, cancel) is plain numpy.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

from . import scheduler
from .tokenizer import ClipTokenizer

VAE_SCALE = 0.18215
MAX_TOKENS = 77
REQUIRED_FILES = (
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "text_encoder/model.onnx",
    "unet/model.onnx",
    "vae_decoder/model.onnx",
)
_DTYPES = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "int32": np.int32}


class Cancelled(Exception):
    pass


def missing_files(models_dir):
    """Names of required model files that are not in models_dir (empty list = ready to run)."""
    def p(rel):
        return os.path.join(models_dir, *rel.split("/"))

    missing = [f for f in REQUIRED_FILES if not os.path.isfile(p(f))]
    # A small unet/model.onnx is only the graph; the weights live in the external-data file beside it.
    if "unet/model.onnx" not in missing and os.path.getsize(p("unet/model.onnx")) < 100_000_000 \
            and not os.path.isfile(p("unet/model.onnx_data")):
        missing.append("unet/model.onnx_data")
    return missing


# --------------------------------------------------------------------------- progress / cancel
_lock = threading.Lock()
_cancel = threading.Event()
_state = {}
# One image/video job at a time: they share the progress state and each needs most of the phone's RAM.
_busy = threading.Lock()
BUSY_MESSAGE = "Another image or video job is already running. Wait for it to finish or cancel it."


def _reset():
    _cancel.clear()
    with _lock:
        _state.clear()
        _state.update(stage="Starting", frac=0.0, done=False, error=None, result=None, seed=None)


def set_progress(**kw):
    with _lock:
        _state.update(kw)


def get_progress():
    """JSON string polled by the Android UI."""
    with _lock:
        return json.dumps(_state)


def cancel():
    _cancel.set()


def check_cancel():
    if _cancel.is_set():
        raise Cancelled()


_reset()


# --------------------------------------------------------------------------- Android backend
def _to_bytes(arr):
    try:
        return bytes(memoryview(arr))
    except Exception:
        return bytes((b & 0xFF) for b in arr)


class AndroidOrtGeneric:
    """Calls com.example.faceswapserver.OrtBridge (Kotlin) through Chaquopy."""

    def __init__(self):
        from java import jclass
        self._b = jclass("com.example.faceswapserver.OrtBridge")

    def load(self, name, path, opt):
        self._b.loadModel(name, path, opt)

    def unload(self, name):
        self._b.unload(name)

    def input_info(self, name):
        return list(zip([str(n) for n in self._b.inputNames(name)], [str(t) for t in self._b.inputTypes(name)]))

    def output_names(self, name):
        return [str(n) for n in self._b.outputNames(name)]

    def run(self, name, feeds):
        from java import jarray
        self._b.feedClear()
        for key, arr in feeds.items():
            arr = np.ascontiguousarray(arr)
            self._b.feed(key, arr.dtype.name, arr.tobytes(), jarray("J")([int(s) for s in arr.shape]))
        res = self._b.runFeed(name)
        return [np.frombuffer(_to_bytes(res[i]), dtype=np.float32) for i in range(len(res))]


def default_backend():
    return AndroidOrtGeneric()


# --------------------------------------------------------------------------- pipeline
class SDTurbo:
    def __init__(self, backend, models_dir):
        self.backend = backend
        self.dir = models_dir

    def _path(self, rel):
        return os.path.join(self.dir, *rel.split("/"))

    def check_files(self):
        missing = missing_files(self.dir)
        if missing:
            raise RuntimeError("Model files missing: " + ", ".join(missing) + ". Download the image model first.")

    def _run(self, model, canonical_names, arrays):
        """Run with arrays cast to each input's dtype; map by name, else by position."""
        info = self.backend.input_info(model)
        names = [n for n, _ in info]
        dtypes = dict(info)
        if all(c in dtypes for c in canonical_names):
            order = list(canonical_names)
        elif len(names) == len(arrays):
            order = names
        else:
            raise RuntimeError("Unexpected inputs for %s model: %s (expected %s)" % (model, names, list(canonical_names)))
        feeds = {}
        for key, arr in zip(order, arrays):
            dt = _DTYPES.get(dtypes[key])
            if dt is None:
                raise RuntimeError("Unsupported input type %s for %s.%s" % (dtypes[key], model, key))
            feeds[key] = np.ascontiguousarray(arr, dtype=dt)
        return self.backend.run(model, feeds)

    # -- stage 1: text
    def encode_prompt(self, prompt):
        set_progress(stage="Reading prompt", frac=0.02)
        ids = ClipTokenizer(self._path("tokenizer")).encode(prompt, MAX_TOKENS)
        check_cancel()
        set_progress(stage="Loading text encoder", frac=0.04)
        self.backend.load("te", self._path("text_encoder/model.onnx"), "basic")
        try:
            check_cancel()
            set_progress(stage="Encoding prompt", frac=0.10)
            outs = self._run("te", ["input_ids"], [ids])
            names = self.backend.output_names("te")
            idx = names.index("last_hidden_state") if "last_hidden_state" in names else 0
            hidden = outs[idx]
            if hidden.size % MAX_TOKENS:
                raise RuntimeError("Unexpected text encoder output size %d" % hidden.size)
            return hidden.reshape(1, MAX_TOKENS, hidden.size // MAX_TOKENS).astype(np.float32)
        finally:
            self.backend.unload("te")

    # -- stage 2: denoise
    def denoise(self, hidden, steps, size, seed):
        lat = size // 8
        ts, sigmas, init_sigma = scheduler.euler_trailing(steps)
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((1, 4, lat, lat)) * init_sigma).astype(np.float32)
        set_progress(stage="Loading image model (first step is slow)", frac=0.14)
        self.backend.load("unet", self._path("unet/model.onnx"), "none")
        try:
            for i, t in enumerate(ts):
                check_cancel()
                set_progress(stage="Generating: step %d of %d" % (i + 1, steps), frac=0.18 + 0.67 * i / steps)
                sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
                scaled = x / np.sqrt(sigma * sigma + 1.0)
                eps = self._run("unet", ["sample", "timestep", "encoder_hidden_states"],
                                [scaled, np.array([t]), hidden])[0].reshape(x.shape)
                x = (x + eps * (sigma_next - sigma)).astype(np.float32)
            return x
        finally:
            self.backend.unload("unet")

    # -- stage 3: decode
    def _vae(self, z):
        out = self._run("vae", ["latent_sample"], [z])[0]
        h, w = z.shape[2] * 8, z.shape[3] * 8
        return out.reshape(1, 3, h, w)

    @staticmethod
    def _starts(length, tile, stride):
        if length <= tile:
            return [0]
        s = list(range(0, length - tile, stride))
        s.append(length - tile)
        return s

    def _vae_tiled(self, z, tile=40, overlap=16):
        _, _, lh, lw = z.shape
        if lh <= tile and lw <= tile:
            return self._vae(z)
        stride = tile - overlap
        ys, xs = self._starts(lh, tile, stride), self._starts(lw, tile, stride)
        acc = np.zeros((3, lh * 8, lw * 8), dtype=np.float32)
        wsum = np.zeros((lh * 8, lw * 8), dtype=np.float32)
        ramp = float(overlap * 8)
        total, n = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                check_cancel()
                set_progress(stage="Decoding image: tile %d of %d" % (n + 1, total), frac=0.86 + 0.12 * n / total)
                part = self._vae(np.ascontiguousarray(z[:, :, y:y + tile, x:x + tile]))[0]
                th, tw = part.shape[1], part.shape[2]
                wy = np.minimum(np.minimum(np.arange(1, th + 1), np.arange(th, 0, -1)) / ramp, 1.0)
                wx = np.minimum(np.minimum(np.arange(1, tw + 1), np.arange(tw, 0, -1)) / ramp, 1.0)
                w = (wy[:, None] * wx[None, :]).astype(np.float32)
                acc[:, y * 8:y * 8 + th, x * 8:x * 8 + tw] += part * w
                wsum[y * 8:y * 8 + th, x * 8:x * 8 + tw] += w
                n += 1
        return (acc / wsum)[None]

    def decode(self, x, tiled=True):
        z = (x / VAE_SCALE).astype(np.float32)
        set_progress(stage="Loading image decoder", frac=0.85)
        self.backend.load("vae", self._path("vae_decoder/model.onnx"), "all")
        try:
            if tiled:
                img = self._vae_tiled(z)
            else:
                set_progress(stage="Decoding image", frac=0.90)
                img = self._vae(z)
        finally:
            self.backend.unload("vae")
        rgb = np.clip(img[0] / 2.0 + 0.5, 0.0, 1.0).transpose(1, 2, 0)
        return np.ascontiguousarray((rgb[:, :, ::-1] * 255.0).round().astype(np.uint8))  # BGR

    def generate(self, prompt, steps=1, size=512, seed=None, tiled=True):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Please type a prompt")
        if not 1 <= int(steps) <= 8:
            raise ValueError("Steps must be between 1 and 8")
        if size % 64 or not 256 <= size <= 768:
            raise ValueError("Image size must be a multiple of 64 between 256 and 768")
        self.check_files()
        if seed is None or int(seed) < 0:
            seed = int(time.time() * 1000) % (2 ** 32)
        set_progress(seed=int(seed))
        hidden = self.encode_prompt(prompt)
        x = self.denoise(hidden, int(steps), int(size), int(seed))
        # Tiles only pay off at 512 px; a 384 px image (48x48 latents) is small enough to decode at once.
        return self.decode(x, bool(tiled) and int(size) > 384)


def generate_image(prompt, steps, size, seed, models_dir, out_path, tiled=True, backend=None):
    """Entry point called from Kotlin and from server.py. Returns out_path, or "" if cancelled; raises on errors."""
    if not _busy.acquire(False):
        raise RuntimeError(BUSY_MESSAGE)   # leave the running job's progress untouched
    try:
        _reset()
        try:
            pipe = SDTurbo(backend or default_backend(), models_dir)
            bgr = pipe.generate(prompt, int(steps), int(size), None if seed is None else int(seed), bool(tiled))
            set_progress(stage="Saving image", frac=0.99)
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("Could not encode the image")
            with open(out_path, "wb") as fh:
                fh.write(enc.tobytes())
            set_progress(stage="Done", frac=1.0, done=True, result=out_path)
            return out_path
        except Cancelled:
            set_progress(stage="Cancelled", done=True, error="Cancelled")
            return ""
        except Exception as e:
            set_progress(stage="Failed", done=True, error=str(e))
            raise
    finally:
        _busy.release()
