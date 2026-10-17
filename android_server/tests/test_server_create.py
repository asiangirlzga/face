"""HTTP tests for the text-to-image / image-to-video endpoints of server.py (no face models needed).

Usage: python -I tests/test_server_create.py
The ONNX models are replaced by the fake backend from test_sdgen.py, the video encoder by a cv2 writer.
Needs numpy, opencv-python-headless and requests.
"""
import io
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app", "src", "main", "python"))
sys.path.insert(0, HERE)
import server as S  # noqa: E402
import test_sdgen as T  # noqa: E402
from sdgen import pipeline as P  # noqa: E402

PORT = 18766
BASE = "http://127.0.0.1:%d" % PORT
H = {"X-Token": "tok"}


def ok(msg):
    print("ok -", msg)


class FakeFace:
    """Stands in for engine.Engine: only the memory handling is under test."""

    def __init__(self):
        self.unloaded = 0

    def unload(self):
        self.unloaded += 1


def wait(jid, want=("done", "error", "cancelled"), timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        st = requests.get("%s/jobs/%s" % (BASE, jid), headers=H).json()
        if st["status"] in want:
            return st
        time.sleep(0.05)
    raise AssertionError("timeout waiting for %s, last %s" % (want, st))


def post_t2i(body, headers=H):
    return requests.post(BASE + "/jobs/text2img", headers=headers, data=json.dumps(body))


def main():
    md = T.make_models_dir()
    T.tiny_tokenizer_files(md)
    face_models = os.path.join(T.TMP, "no_face_models")
    created, slow = [], {"on": False}

    def backend():
        be = T.FakeBackend()
        be.on_unet = (lambda: time.sleep(0.3)) if slow["on"] else None
        created.append(be)
        return be

    S.sd_backend_factory = backend
    S.sd_writer_factory = T.FakeWriter
    threading.Thread(target=S.serve, args=("127.0.0.1", PORT, "tok", face_models, os.path.join(T.TMP, "work"), md),
                     daemon=True).start()
    for _ in range(100):
        try:
            if requests.get(BASE + "/health").json()["engine"] != "loading":
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    h = requests.get(BASE + "/health").json()
    assert h["t2i"] == "ready" and h["engine"] == "missing", h
    ok("health reports t2i ready (face models absent here)")

    # ---- text2img happy path
    r = post_t2i({"prompt": "a red fox in snow", "steps": 1, "size": 512, "seed": 42})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    st = wait(jid)
    assert st["status"] == "done" and st["done"] == st["total"] == 1000 and st["seed"] == 42, st
    assert st["kind"] == "text2img" and st["stage"] == "Done", st
    res = requests.get("%s/jobs/%s/result" % (BASE, jid), headers=H)
    assert res.headers["Content-Type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(res.content, np.uint8), 1)
    assert img.shape == (512, 512, 3)
    ok("text2img: job runs, status carries stage/seed, result is a 512x512 jpeg")

    again = post_t2i({"prompt": "a red fox in snow", "steps": 1, "size": 512, "seed": 42}).json()["job_id"]
    wait(again)
    assert requests.get("%s/jobs/%s/result" % (BASE, again), headers=H).content == res.content
    r = post_t2i({"prompt": "fox", "steps": 2, "size": 384})   # no seed -> random seed reported back
    st = wait(r.json()["job_id"])
    assert st["status"] == "done" and isinstance(st["seed"], int)
    ok("text2img: same seed -> identical jpeg, missing seed -> random seed returned")

    # ---- validation and auth
    for body, why in (({"prompt": "  "}, "empty prompt"), ({"prompt": "x", "size": 500}, "size"),
                      ({"prompt": "x", "steps": 99}, "steps"), ({"prompt": "x" * 1001}, "long prompt"),
                      ({"prompt": "x", "steps": "abc"}, "non-numeric"), ([1, 2], "not an object")):
        r = post_t2i(body)
        assert r.status_code == 400 and r.json()["detail"], (why, r.status_code, r.text)
    assert requests.post(BASE + "/jobs/text2img", headers=H, data="{not json").status_code == 400
    assert post_t2i({"prompt": "x"}, headers={"X-Token": "wrong"}).status_code == 401
    assert requests.post(BASE + "/jobs/text2img", headers=H, data=b"x" * 200_000).status_code == 413
    ok("text2img: bad input -> 400, bad token -> 401, oversized body -> 413")

    S._cfg["sd"] = os.path.join(T.TMP, "empty_sd")
    r = post_t2i({"prompt": "fox"})
    assert r.status_code == 409 and "Download image model" in r.json()["detail"], r.text
    assert requests.get(BASE + "/health").json()["t2i"] == "missing"
    S._cfg["sd"] = md
    ok("text2img: model not downloaded -> 409 with instructions, health says missing")

    # ---- the face models are released before the big model runs, and come back on demand
    fake = FakeFace()
    S._engine.update(obj=fake, status="ready", error=None)
    st = wait(post_t2i({"prompt": "fox", "steps": 1, "size": 384}).json()["job_id"])
    assert st["status"] == "done" and fake.unloaded == 1 and S._engine["status"] == "idle" and S._engine["obj"] is None
    reloaded = FakeFace()

    def fake_load():
        S._engine.update(obj=reloaded, status="ready", error=None)

    orig_load, S._load_engine = S._load_engine, fake_load
    try:
        assert S._engine_or_fail() is reloaded and S._engine["status"] == "ready"
    finally:
        S._load_engine = orig_load
    ok("memory: face models unloaded before text2img, reloaded lazily for the next face job")

    # ---- cancel while running
    slow["on"] = True
    jid = post_t2i({"prompt": "fox", "steps": 8, "size": 384, "seed": 1}).json()["job_id"]
    st = wait(jid, want=("running",))
    for _ in range(100):   # until the denoising loop is going
        st = requests.get("%s/jobs/%s" % (BASE, jid), headers=H).json()
        if st["stage"].startswith("Generating"):
            break
        time.sleep(0.05)
    assert st["stage"].startswith("Generating"), st
    try:   # an in-app (Kotlin) job at the same moment is refused, not mixed in
        P.generate_image("fox", 1, 512, 1, md, os.path.join(T.TMP, "x.jpg"), True, T.FakeBackend())
    except RuntimeError as e:
        assert "already running" in str(e)
    else:
        raise AssertionError("second image job was not refused")
    assert requests.post("%s/jobs/%s/cancel" % (BASE, jid), headers=H).json() == {"ok": True}
    st = wait(jid)
    assert st["status"] == "cancelled", st
    assert requests.get("%s/jobs/%s/result" % (BASE, jid), headers=H).status_code == 404
    assert not created[-1].loaded and created[-1].runs.count("unet") < 8
    ok("cancel: running job stops after the current step, models unloaded, no result; parallel in-app job refused")

    # ---- cancel while queued: never starts
    n_before = len(created)
    a = post_t2i({"prompt": "fox", "steps": 4, "size": 384}).json()["job_id"]
    b = post_t2i({"prompt": "fox", "steps": 4, "size": 384}).json()["job_id"]
    assert requests.post("%s/jobs/%s/cancel" % (BASE, b), headers=H).status_code == 200
    assert wait(a)["status"] == "done"
    assert wait(b)["status"] == "cancelled" and len(created) == n_before + 1
    assert requests.post("%s/jobs/deadbeef/cancel" % BASE, headers=H).status_code == 404
    slow["on"] = False
    ok("cancel: a queued job is skipped without loading anything; unknown job -> 404")

    # ---- img2video
    src = np.random.default_rng(0).integers(0, 255, (400, 300, 3), dtype=np.uint8)
    src = cv2.GaussianBlur(src, (0, 0), 5)
    jpg = cv2.imencode(".jpg", src)[1].tobytes()

    def post_video(files, data):
        return requests.post(BASE + "/jobs/img2video", headers=H, files=files, data=data)

    r = post_video({"image": ("a.jpg", jpg, "image/jpeg")}, {"style": "pan_left", "seconds": "2", "fps": "12"})
    assert r.status_code == 200, r.text
    st = wait(r.json()["job_id"])
    assert st["status"] == "done" and st["kind"] == "img2video" and st["done"] == st["total"] == 1000, st
    vid = requests.get("%s/jobs/%s/result" % (BASE, r.json()["job_id"]), headers=H)
    assert vid.headers["Content-Type"] == "video/mp4" and len(vid.content) > 1000
    vpath = os.path.join(T.TMP, "got.mp4")
    open(vpath, "wb").write(vid.content)
    cap = cv2.VideoCapture(vpath)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    ok("img2video: 2 s at 12 fps -> mp4 with 24 frames")

    for files, data, why in (({}, {"style": "zoom_in"}, "no image"),
                             ({"image": ("a.jpg", jpg)}, {"style": "spin"}, "bad style"),
                             ({"image": ("a.jpg", jpg)}, {"seconds": "99"}, "bad seconds"),
                             ({"image": ("a.jpg", jpg)}, {"fps": "5"}, "bad fps")):
        r = post_video(files or {"x": ("x", b"1")}, data)
        assert r.status_code == 400, (why, r.status_code, r.text)
    bad = post_video({"image": ("a.jpg", b"not an image")}, {})
    assert wait(bad.json()["job_id"])["status"] == "error"
    ok("img2video: invalid input -> 400, unreadable image -> job error")

    slow_w = T.FakeWriter

    class SlowWriter(slow_w):
        def write(self, frame):
            time.sleep(0.05)
            super().write(frame)

    S.sd_writer_factory = SlowWriter
    r = post_video({"image": ("a.jpg", jpg)}, {"seconds": "10", "fps": "30"})
    jid = r.json()["job_id"]
    wait(jid, want=("running",))
    time.sleep(0.4)
    requests.post("%s/jobs/%s/cancel" % (BASE, jid), headers=H)
    assert wait(jid)["status"] == "cancelled"
    ok("img2video: cancel mid-render")

    # job records show the new fields for the old job kinds too
    st = json.loads(S.get_state())
    assert st["t2i"] == "ready" and st["last"]["stage"] is not None
    S.stop()
    print("SERVER CREATE ALL OK")


if __name__ == "__main__":
    main()
