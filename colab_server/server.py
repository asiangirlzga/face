"""Face swap server for Google Colab (GPU runtime).

Endpoints (all need header  X-Token: <token>):
  POST /jobs/images   form: source (1 file), targets (1..50 files)  -> {"job_id"}
  POST /jobs/video    form: source (1 file), video (1 file)         -> {"job_id"}
  GET  /jobs/{id}                                                   -> status JSON
  GET  /jobs/{id}/result                                            -> zip (images) or mp4 (video)
"""
import os
import shutil
import subprocess
import threading
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import List

import cv2
import insightface
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis

TOKEN = os.environ.get("FS_TOKEN", "change-me")
MODEL_PATH = os.environ.get("SWAP_MODEL", "/content/models/inswapper_128.onnx")
MAX_IMAGES = 50
WORK = Path("/content/jobs")
WORK.mkdir(parents=True, exist_ok=True)

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
analyser = FaceAnalysis(name="buffalo_l", providers=providers)
analyser.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=providers)

jobs = {}
gpu_lock = threading.Lock()  # one job on the GPU at a time
app = FastAPI()


def auth(token):
    if token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def largest_face(img):
    faces = analyser.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def swap_all_faces(img, src_face):
    out = img
    for f in analyser.get(img):
        out = swapper.get(out, f, src_face, paste_back=True)
    return out


def save_upload(up: UploadFile, dest: Path):
    with open(dest, "wb") as fh:
        shutil.copyfileobj(up.file, fh)


def new_job(kind, total):
    jid = uuid.uuid4().hex[:12]
    d = WORK / jid
    d.mkdir(parents=True)
    jobs[jid] = {"status": "queued", "done": 0, "total": total, "error": None,
                 "kind": kind, "skipped": [], "dir": d, "result": None}
    return jid, d


def run_images(jid, src_path, target_paths):
    job = jobs[jid]
    try:
        with gpu_lock:
            job["status"] = "running"
            src = cv2.imread(str(src_path))
            src_face = largest_face(src) if src is not None else None
            if src_face is None:
                raise RuntimeError("No face found in the source image")
            out_dir = job["dir"] / "out"
            out_dir.mkdir()
            for p in target_paths:
                img = cv2.imread(str(p))
                if img is None:
                    job["skipped"].append(p.name + " (unreadable)")
                elif not analyser.get(img):
                    job["skipped"].append(p.name + " (no face)")
                else:
                    res = swap_all_faces(img, src_face)
                    cv2.imwrite(str(out_dir / (p.stem + "_swap.jpg")), res,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                job["done"] += 1
            zpath = job["dir"] / "result.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(out_dir.iterdir()):
                    z.write(f, f.name)
            job["result"] = zpath
            job["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)


def run_video(jid, src_path, video_path):
    job = jobs[jid]
    try:
        with gpu_lock:
            job["status"] = "running"
            src = cv2.imread(str(src_path))
            src_face = largest_face(src) if src is not None else None
            if src_face is None:
                raise RuntimeError("No face found in the source image")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            job["total"] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            silent = job["dir"] / "silent.mp4"
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(swap_all_faces(frame, src_face))
                job["done"] += 1
            cap.release()
            writer.release()
            final = job["dir"] / "result.mp4"
            # re-encode to H.264 (plays on Android) and copy original audio if present
            cmd = ["ffmpeg", "-y", "-i", str(silent), "-i", str(video_path),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)]
            subprocess.run(cmd, check=True, capture_output=True)
            job["result"] = final
            job["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)


@app.post("/jobs/images")
def submit_images(source: UploadFile = File(...), targets: List[UploadFile] = File(...),
                  x_token: str = Header(None)):
    auth(x_token)
    if not 1 <= len(targets) <= MAX_IMAGES:
        raise HTTPException(400, f"send 1..{MAX_IMAGES} target images")
    jid, d = new_job("images", len(targets))
    src_path = d / "source.jpg"
    save_upload(source, src_path)
    paths = []
    for i, t in enumerate(targets):
        p = d / f"in_{i:03d}{Path(t.filename or '').suffix or '.jpg'}"
        save_upload(t, p)
        paths.append(p)
    threading.Thread(target=run_images, args=(jid, src_path, paths), daemon=True).start()
    return {"job_id": jid}


@app.post("/jobs/video")
def submit_video(source: UploadFile = File(...), video: UploadFile = File(...),
                 x_token: str = Header(None)):
    auth(x_token)
    jid, d = new_job("video", 1)
    src_path, vid_path = d / "source.jpg", d / "input.mp4"
    save_upload(source, src_path)
    save_upload(video, vid_path)
    threading.Thread(target=run_video, args=(jid, src_path, vid_path), daemon=True).start()
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def status(jid: str, x_token: str = Header(None)):
    auth(x_token)
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return {k: j[k] for k in ("status", "done", "total", "error", "kind", "skipped")}


@app.get("/jobs/{jid}/result")
def result(jid: str, x_token: str = Header(None)):
    auth(x_token)
    j = jobs.get(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "result not ready")
    media = "application/zip" if j["kind"] == "images" else "video/mp4"
    return FileResponse(j["result"], media_type=media)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
