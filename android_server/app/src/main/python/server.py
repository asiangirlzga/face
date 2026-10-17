"""HTTP face-swap / text-to-image server that runs on the phone (Chaquopy) or on any desktop for testing.

Same face-swap API as the Colab server (header  X-Token: <token>):
  GET  /health                                   -> {"ok":true,"engine":"ready|idle|loading|missing|error","t2i":"ready|missing"}  (no token)
  POST /jobs/images   form: source, targets x1..50  -> {"job_id"}
  POST /jobs/video    form: source, video           -> {"job_id"}
  GET  /jobs/<id>                                   -> status JSON (status: queued|running|done|error|cancelled, stage, seed)
  GET  /jobs/<id>/result                            -> zip of JPEGs (images), jpeg (text2img) or mp4 (video, img2video)
  POST /jobs/<id>/cancel                            -> {"ok":true}

Offline image generation (needs the SD-Turbo model files downloaded in the app):
  POST /jobs/text2img  JSON {prompt, steps 1-8, size 256-768 (x64), seed?, tiled?}  -> {"job_id"}
  POST /jobs/img2video form: image, style, seconds 1-10, fps 12-30                  -> {"job_id"}  (camera zoom/pan clip)
"""
import io
import json
import os
import queue
import re
import secrets
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import engine as E
from sdgen import motion as sdm
from sdgen import pipeline as sdp

MAX_IMAGES = 50
MAX_IMAGE_SIDE = 4096     # downscale anything bigger (memory)
MAX_VIDEO_SIDE = 1280     # video frames are downscaled to this (speed on a phone)
JOB_TTL = 6 * 3600
FINISHED = ("done", "error", "cancelled")

_cfg = {"token": "", "models": "", "work": "", "sd": "", "host": "127.0.0.1", "port": 8000}
_jobs = {}
_jobs_lock = threading.Lock()
_queue = queue.Queue()
_engine = {"obj": None, "status": "loading", "error": None}
_engine_lock = threading.Lock()
_httpd = None
# Test hooks: override to run off-device.
backend_factory = lambda: E.AndroidOrt()
video_factory = None  # () -> object with open/read/open_writer/write/close_writer
sd_backend_factory = None  # () -> ONNX backend for sdgen.pipeline (default: Android OrtBridge)
sd_writer_factory = None   # () -> frame writer for sdgen.motion (default: Android VideoBridge)


class _JobCancelled(Exception):
    pass


# --------------------------------------------------------------------------- multipart
def parse_multipart(rfile, content_type, length, tmp_dir):
    """Streaming multipart/form-data parser. Files go to tmp_dir. Returns (files, fields):
    files = [(field, filename, path)], fields = {name: str}."""
    m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type or "")
    if not m:
        raise ValueError("expected multipart/form-data")
    delim = b"\r\n--" + (m.group(1) or m.group(2)).encode()
    state = {"buf": b"\r\n", "left": length}

    def fill():
        if state["left"] <= 0:
            return False
        chunk = rfile.read(min(262144, state["left"]))
        if not chunk:
            state["left"] = 0
            return False
        state["left"] -= len(chunk)
        state["buf"] += chunk
        return True

    while True:  # skip preamble
        i = state["buf"].find(delim)
        if i >= 0:
            state["buf"] = state["buf"][i + len(delim):]
            break
        state["buf"] = state["buf"][-len(delim):]
        if not fill():
            raise ValueError("malformed multipart body")

    files, fields, n = [], {}, 0
    while True:
        while len(state["buf"]) < 2:
            if not fill():
                raise ValueError("truncated multipart body")
        if state["buf"][:2] == b"--":
            break
        if state["buf"][:2] != b"\r\n":
            raise ValueError("malformed multipart boundary")
        state["buf"] = state["buf"][2:]
        while b"\r\n\r\n" not in state["buf"]:
            if not fill():
                raise ValueError("truncated multipart headers")
        h = state["buf"].index(b"\r\n\r\n")
        head = state["buf"][:h].decode("utf-8", "replace")
        state["buf"] = state["buf"][h + 4:]
        name = filename = None
        for line in head.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                a = re.search(r'\bname="([^"]*)"', line)
                b = re.search(r'filename="([^"]*)"', line)
                name = a.group(1) if a else None
                filename = b.group(1) if b else None
        if filename is not None:
            n += 1
            path = os.path.join(tmp_dir, "part_%03d" % n)
            out = open(path, "wb")
        else:
            path, out = None, io.BytesIO()
        try:
            while True:
                i = state["buf"].find(delim)
                if i >= 0:
                    out.write(state["buf"][:i])
                    state["buf"] = state["buf"][i + len(delim):]
                    break
                keep = len(delim) - 1
                if len(state["buf"]) > keep:
                    out.write(state["buf"][:-keep])
                    state["buf"] = state["buf"][-keep:]
                if not fill():
                    raise ValueError("truncated multipart body")
            if path:
                files.append((name, os.path.basename(filename.replace("\\", "/")) or "file", path))
            else:
                fields[name] = out.getvalue().decode("utf-8", "replace")
        finally:
            out.close()
    return files, fields


# --------------------------------------------------------------------------- engine / jobs
def _load_engine():
    try:
        for f in (E.DET_FILE, E.REC_FILE, E.SWAP_FILE):
            if not os.path.isfile(os.path.join(_cfg["models"], f)):
                _engine["status"] = "missing"
                _engine["error"] = "Model file missing: " + f
                return
        _engine["obj"] = E.Engine(backend_factory(), _cfg["models"])
        _engine["status"] = "ready"
    except Exception as e:
        traceback.print_exc()
        _engine["status"], _engine["error"] = "error", str(e)


def _wait_engine_settled():
    for _ in range(600):   # a load that is still in progress finishes first
        if _engine["status"] != "loading":
            break
        time.sleep(0.5)


def _free_face_engine():
    """Unload the face models (about 0.8 GB) before the image model needs the RAM; reloaded on demand."""
    _wait_engine_settled()
    with _engine_lock:
        eng = _engine["obj"]
        if eng is not None:
            eng.unload()
            _engine.update(obj=None, status="idle", error=None)


def _engine_or_fail():
    _wait_engine_settled()
    if _engine["status"] == "idle":
        with _engine_lock:
            if _engine["status"] == "idle":
                _engine["status"] = "loading"
                _load_engine()
    if _engine["status"] != "ready":
        raise RuntimeError(_engine["error"] or "Face swap models are not ready")
    return _engine["obj"]


def _cleanup_old():
    now = time.time()
    with _jobs_lock:
        for jid, j in list(_jobs.items()):
            if now - j["created"] > JOB_TTL and j["status"] in FINISHED:
                shutil.rmtree(j["dir"], ignore_errors=True)
                del _jobs[jid]


def _new_job(kind, total):
    _cleanup_old()
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(_cfg["work"], jid)
    os.makedirs(d)
    job = {"status": "queued", "done": 0, "total": total, "error": None, "kind": kind,
           "skipped": [], "dir": d, "result": None, "created": time.time(),
           "stage": "", "seed": None, "cancel": False}
    with _jobs_lock:
        _jobs[jid] = job
    return jid, job, d


def read_image(path, max_side=MAX_IMAGE_SIDE):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    s = max(img.shape[:2])
    if s > max_side:
        k = max_side / float(s)
        img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return img


def _run_images(job, src_path, targets):
    try:
        job["status"] = "running"
        eng = _engine_or_fail()
        src = read_image(src_path)
        if src is None:
            raise RuntimeError("Could not read the source image")
        latent = eng.source_latent(src)
        zpath = os.path.join(job["dir"], "result.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for i, (orig, path) in enumerate(targets):
                if job["cancel"]:
                    raise _JobCancelled()
                try:
                    img = read_image(path)
                    if img is None:
                        job["skipped"].append(orig + " (unreadable)")
                    else:
                        out = eng.swap_image(img, latent)
                        if out is None:
                            job["skipped"].append(orig + " (no face)")
                        else:
                            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            stem = os.path.splitext(orig)[0][:60] or "img"
                            z.writestr("%03d_%s_swap.jpg" % (i + 1, stem), enc.tobytes())
                except Exception as e:  # one bad image must not kill the batch
                    traceback.print_exc()
                    job["skipped"].append("%s (%s)" % (orig, e))
                finally:
                    os.remove(path)
                    job["done"] += 1
        job["result"], job["status"] = zpath, "done"
    except _JobCancelled:
        job["status"] = "cancelled"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)


def _video_bridge():
    if video_factory:
        return video_factory()
    from java import jclass
    return _AndroidVideo(jclass("com.example.faceswapserver.VideoBridge"))


class _AndroidVideo:
    def __init__(self, cls):
        self.c = cls
        self._src = None

    def open(self, path):
        self._src = path
        info = self.c.openReader(path)  # [frames, fps*1000]
        return int(info[0]), int(info[1]) / 1000.0

    def read(self, i):
        data = self.c.readFrame(int(i))
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(engine_bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)

    def open_writer(self, path, w, h, fps):
        self.c.openWriter(path, int(w), int(h), float(fps), self._src)

    def write(self, frame):
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.c.writeFrame(enc.tobytes())

    def close_writer(self, src_video):
        self.c.closeWriter()

    def close_reader(self):
        self.c.closeReader()


def engine_bytes(arr):
    return E._to_bytes(arr)


def _run_video(job, src_path, video_path):
    vb = None
    try:
        job["status"] = "running"
        eng = _engine_or_fail()
        src = read_image(src_path)
        if src is None:
            raise RuntimeError("Could not read the source image")
        latent = eng.source_latent(src)
        vb = _video_bridge()
        frames, fps = vb.open(video_path)
        if frames <= 0:
            raise RuntimeError("Could not read the video")
        job["total"] = frames
        out_path = os.path.join(job["dir"], "result.mp4")
        writer_open, last, size = False, None, None
        for i in range(frames):
            if job["cancel"]:
                raise _JobCancelled()
            frame = vb.read(i)
            if frame is None:
                frame = last  # tail frames the decoder cannot seek to
                if frame is None:
                    continue
            s = max(frame.shape[:2])
            if s > MAX_VIDEO_SIDE:
                k = MAX_VIDEO_SIDE / float(s)
                frame = cv2.resize(frame, (int(frame.shape[1] * k), int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
            if size is None:
                size = (frame.shape[1] & ~1, frame.shape[0] & ~1)  # H.264 needs even dimensions
            frame = frame[:size[1], :size[0]]
            res = eng.swap_image(frame, latent)
            res = frame if res is None else res
            if not writer_open:
                vb.open_writer(out_path, size[0], size[1], fps)
                writer_open = True
            vb.write(res)
            last = frame
            job["done"] = i + 1
        if not writer_open:
            raise RuntimeError("No frames could be decoded")
        vb.close_writer(video_path)
        job["result"], job["status"] = out_path, "done"
    except _JobCancelled:
        job["status"] = "cancelled"
        try:   # finish the half-written file so the encoder is released
            if writer_open:
                vb.close_writer(video_path)
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)
    finally:
        try:
            if vb is not None and hasattr(vb, "close_reader"):
                vb.close_reader()
        except Exception:
            pass
        for p in (video_path,):
            try:
                os.remove(p)
            except OSError:
                pass


# --------------------------------------------------------------------------- text-to-image / image-to-video
NO_IMAGE_MODEL = ("The image model is not on the phone yet. Open the Face Swap Server app, go to the "
                  "Text to image tab and tap Download image model.")


def _sd_ready():
    return bool(_cfg["sd"]) and not sdp.missing_files(_cfg["sd"])


def _t2i_params(b):
    """Validate the JSON body of POST /jobs/text2img (ValueError -> HTTP 400)."""
    if not isinstance(b, dict):
        raise ValueError("expected a JSON object")
    prompt = str(b.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Please type a prompt")
    if len(prompt) > 1000:
        raise ValueError("The prompt is too long (max 1000 characters)")
    try:
        steps = int(b.get("steps", 1))
        size = int(b.get("size", 512))
        seed = b.get("seed")
        seed = None if seed in (None, "") else int(seed)
    except (TypeError, ValueError):
        raise ValueError("steps, size and seed must be numbers")
    if not 1 <= steps <= 8:
        raise ValueError("Steps must be between 1 and 8")
    if size % 64 or not 256 <= size <= 768:
        raise ValueError("Image size must be a multiple of 64 between 256 and 768")
    if seed is not None and seed < 0:
        seed = None
    return {"prompt": prompt, "steps": steps, "size": size, "seed": seed, "tiled": bool(b.get("tiled", True))}


def _i2v_params(fields):
    """Validate the form fields of POST /jobs/img2video (ValueError -> HTTP 400)."""
    style = (fields.get("style") or "zoom_in").strip()
    if style not in sdm.STYLES:
        raise ValueError("Unknown style: %s (use one of %s)" % (style, ", ".join(sdm.STYLES)))
    try:
        seconds = int(fields.get("seconds") or 4)
        fps = int(fields.get("fps") or 24)
    except ValueError:
        raise ValueError("seconds and fps must be numbers")
    if not 1 <= seconds <= 10 or not 12 <= fps <= 30:
        raise ValueError("Use 1-10 seconds and 12-30 fps")
    return {"style": style, "seconds": seconds, "fps": fps}


def _sync_progress(job):
    """Copy the live sdgen progress (stage text, fraction, seed) into the job record."""
    try:
        st = json.loads(sdp.get_progress())
    except ValueError:
        return
    job["stage"] = st.get("stage") or ""
    job["done"] = int(float(st.get("frac") or 0.0) * 1000)
    job["total"] = 1000
    if st.get("seed") is not None:
        job["seed"] = st["seed"]


def _watch_progress(job, stop):
    while not stop.wait(0.5):
        _sync_progress(job)


def _run_text2img(job, params):
    stop = threading.Event()
    try:
        if job["cancel"]:
            raise _JobCancelled()
        job["status"] = "running"
        job["stage"] = "Freeing memory"
        _free_face_engine()
        out = os.path.join(job["dir"], "result.jpg")
        backend = sd_backend_factory() if sd_backend_factory else None
        threading.Thread(target=_watch_progress, args=(job, stop), daemon=True).start()
        res = sdp.generate_image(params["prompt"], params["steps"], params["size"], params["seed"],
                                 _cfg["sd"], out, params["tiled"], backend)
        stop.set()
        _sync_progress(job)
        if not res:
            raise _JobCancelled()
        job["result"], job["status"] = out, "done"
    except _JobCancelled:
        job["status"] = "cancelled"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)
    finally:
        stop.set()


def _run_img2video(job, image_path, params):
    stop = threading.Event()
    try:
        if job["cancel"]:
            raise _JobCancelled()
        job["status"] = "running"
        out = os.path.join(job["dir"], "result.mp4")
        writer = sd_writer_factory() if sd_writer_factory else None
        threading.Thread(target=_watch_progress, args=(job, stop), daemon=True).start()
        res = sdm.make_video(image_path, out, params["seconds"], params["fps"], params["style"], 768, writer)
        stop.set()
        _sync_progress(job)
        if not res:
            raise _JobCancelled()
        job["result"], job["status"] = out, "done"
    except _JobCancelled:
        job["status"] = "cancelled"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)
    finally:
        stop.set()
        try:
            os.remove(image_path)
        except OSError:
            pass


def _worker():
    while True:
        fn, args = _queue.get()
        try:
            fn(*args)
        except Exception:   # a runner bug must not kill the only worker thread
            traceback.print_exc()


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "FaceSwapPhone/1.0"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        tok = self.headers.get("X-Token", "")
        if not secrets.compare_digest(tok.encode(), _cfg["token"].encode()):
            self._json(401, {"detail": "bad token"})
            return False
        return True

    def do_GET(self):
        try:
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                return self._json(200, {"ok": True, "engine": _engine["status"], "error": _engine["error"],
                                        "t2i": "ready" if _sd_ready() else "missing"})
            if not self._auth():
                return
            m = re.fullmatch(r"/jobs/([0-9a-f]+)(/result)?", path)
            job = _jobs.get(m.group(1)) if m else None
            if not job:
                return self._json(404, {"detail": "unknown job"})
            if not m.group(2):
                return self._json(200, {k: job[k] for k in (
                    "status", "done", "total", "error", "kind", "skipped", "stage", "seed")})
            if job["status"] != "done":
                return self._json(404, {"detail": "result not ready"})
            size = os.path.getsize(job["result"])
            self.send_response(200)
            self.send_header("Content-Type", {"images": "application/zip", "text2img": "image/jpeg"}.get(job["kind"], "video/mp4"))
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(job["result"], "rb") as fh:
                shutil.copyfileobj(fh, self.wfile, 262144)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"detail": str(e)})

    def do_POST(self):
        try:
            if not self._auth():
                return
            path = self.path.split("?")[0].rstrip("/")
            cancel = re.fullmatch(r"/jobs/([0-9a-f]+)/cancel", path)
            if cancel:
                job = _jobs.get(cancel.group(1))
                if not job:
                    return self._json(404, {"detail": "unknown job"})
                job["cancel"] = True
                if job["status"] == "running" and job["kind"] in ("text2img", "img2video"):
                    sdp.cancel()   # stops after the current step / frame
                return self._json(200, {"ok": True})
            if path == "/jobs/text2img":
                length = int(self.headers.get("Content-Length") or 0)
                if length > 100_000:
                    return self._json(413, {"detail": "request too large"})
                try:
                    params = _t2i_params(json.loads(self.rfile.read(length).decode("utf-8") or "{}"))
                except ValueError as e:   # includes bad JSON
                    return self._json(400, {"detail": str(e)})
                if not _sd_ready():
                    return self._json(409, {"detail": NO_IMAGE_MODEL})
                jid, job, _ = _new_job("text2img", 1000)
                _queue.put((_run_text2img, (job, params)))
                return self._json(200, {"job_id": jid})
            if path not in ("/jobs/images", "/jobs/video", "/jobs/img2video"):
                return self._json(404, {"detail": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            kind = {"/jobs/images": "images", "/jobs/video": "video", "/jobs/img2video": "img2video"}[path]
            jid, job, d = _new_job(kind, 1)
            try:
                files, fields = parse_multipart(self.rfile, self.headers.get("Content-Type"), length, d)
            except Exception as e:
                shutil.rmtree(d, ignore_errors=True)
                with _jobs_lock:
                    _jobs.pop(jid, None)
                return self._json(400, {"detail": str(e)})
            if kind == "img2video":
                img = [f for f in files if f[0] == "image"]
                if not img:
                    return self._fail(jid, d, "missing 'image' file")
                try:
                    params = _i2v_params(fields)
                except ValueError as e:
                    return self._fail(jid, d, str(e))
                job["total"] = 1000
                _queue.put((_run_img2video, (job, img[0][2], params)))
                return self._json(200, {"job_id": jid})
            src = [f for f in files if f[0] == "source"]
            if not src:
                return self._fail(jid, d, "missing 'source' file")
            if kind == "images":
                targets = [(n, p) for f, n, p in files if f == "targets"]
                if not 1 <= len(targets) <= MAX_IMAGES:
                    return self._fail(jid, d, "send 1..%d target images" % MAX_IMAGES)
                job["total"] = len(targets)
                _queue.put((_run_images, (job, src[0][2], targets)))
            else:
                vid = [f for f in files if f[0] == "video"]
                if not vid:
                    return self._fail(jid, d, "missing 'video' file")
                _queue.put((_run_video, (job, src[0][2], vid[0][2])))
            self._json(200, {"job_id": jid})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"detail": str(e)})

    def _fail(self, jid, d, msg):
        shutil.rmtree(d, ignore_errors=True)
        with _jobs_lock:
            _jobs.pop(jid, None)
        self._json(400, {"detail": msg})


# --------------------------------------------------------------------------- entry points (called from Kotlin)
def serve(host, port, token, models_dir, work_dir, sd_dir=""):
    """Blocks until stop() is called. sd_dir holds the SD-Turbo files for text-to-image (optional)."""
    global _httpd
    _cfg.update(host=host, port=int(port), token=token, models=models_dir, work=work_dir, sd=sd_dir or "")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    _jobs.clear()
    _engine.update(obj=None, status="loading", error=None)
    threading.Thread(target=_load_engine, daemon=True).start()
    if not getattr(serve, "_worker_started", False):
        threading.Thread(target=_worker, daemon=True).start()
        serve._worker_started = True
    _httpd = ThreadingHTTPServer((host, int(port)), Handler)
    _httpd.daemon_threads = True
    try:
        _httpd.serve_forever()
    finally:
        _httpd.server_close()
        _httpd = None


def stop():
    if _httpd is not None:
        _httpd.shutdown()


def get_state():
    """JSON string for the UI."""
    with _jobs_lock:
        active = [j for j in _jobs.values() if j["status"] in ("queued", "running")]
        last = max(_jobs.values(), key=lambda j: j["created"]) if _jobs else None
    return json.dumps({
        "engine": _engine["status"], "engine_error": _engine["error"],
        "t2i": "ready" if _sd_ready() else "missing",
        "active": len(active),
        "last": None if not last else {k: last[k] for k in ("kind", "status", "done", "total", "error", "stage")},
    })
