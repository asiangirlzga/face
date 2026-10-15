"""End-to-end HTTP test of server.py on a desktop with real det/rec models and a stand-in swapper.
Usage: python -I test_server.py <models_dir>"""
import json, os, sys, tempfile, threading, time, zipfile, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "main", "python"))
import cv2, numpy as np, requests
from skimage import data
import engine as E, server as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_engine import DesktopOrt

models = sys.argv[1]
class FakeSwapBE(DesktopOrt):
    def load(self, name, path):
        if name != "swap": super().load(name, path)
    def run(self, name, arrays):
        if name == "swap":
            return [np.clip(arrays[0] * 0.5 + 0.4, 0, 1).astype(np.float32).reshape(-1)]
        return super().run(name, arrays)
# stand-in swap model file with a valid emap tail so Engine() can be built
from test_engine import tensor, ld, field, varint
tmp = tempfile.mkdtemp()
for f in (E.DET_FILE, E.REC_FILE):
    os.symlink(os.path.abspath(os.path.join(models, f)), os.path.join(tmp, f))
emap = np.eye(512, dtype=np.float32)
open(os.path.join(tmp, E.SWAP_FILE), "wb").write(field(1, 0, varint(8)) + ld(7, ld(5, tensor("emap", emap))))
S.backend_factory = FakeSwapBE

class FakeVideo:  # stands in for the Kotlin VideoBridge
    def open(self, path):
        self.frames = [cv2.imread(os.path.join(tmp, "frame.jpg"))] * 6
        return 6, 10.0
    def read(self, i): return self.frames[i] if i < 5 else None  # last frame undecodable -> reuses previous
    def open_writer(self, path, w, h, fps):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)); self.path = path
    def write(self, f): self.w.write(f)
    def close_writer(self, src): self.w.release()
S.video_factory = FakeVideo

astro = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(tmp, "frame.jpg"), astro)
good = cv2.imencode(".jpg", astro)[1].tobytes()
blank = cv2.imencode(".jpg", np.full((300, 400, 3), 120, np.uint8))[1].tobytes()

port = 18765
t = threading.Thread(target=S.serve, args=("127.0.0.1", port, "tok", tmp, os.path.join(tmp, "work")), daemon=True); t.start()
base = "http://127.0.0.1:%d" % port; H = {"X-Token": "tok"}
for _ in range(100):
    try:
        if requests.get(base + "/health").json()["engine"] == "ready": break
    except Exception: pass
    time.sleep(0.2)
print("health", requests.get(base + "/health").json())
assert requests.get(base + "/jobs/abc", headers={"X-Token": "x"}).status_code == 401

def wait(jid):
    for _ in range(300):
        st = requests.get("%s/jobs/%s" % (base, jid), headers=H).json()
        if st["status"] in ("done", "error"): return st
        time.sleep(0.3)
    raise SystemExit("timeout")

# 1) batch of 50: mixed good / blank / garbage / same filenames
files = [("source", ("s.jpg", good, "image/jpeg"))]
for i in range(47): files.append(("targets", ("p.jpg", good, "image/jpeg")))
files += [("targets", ("blank.jpg", blank, "image/jpeg")), ("targets", ("bad.jpg", b"notanimage", "image/jpeg")), ("targets", ("p.jpg", good, "image/jpeg"))]
r = requests.post(base + "/jobs/images", headers=H, files=files); print("submit", r.status_code, r.text)
jid = r.json()["job_id"]; st = wait(jid); print("status", st["status"], st["done"], st["total"], st["skipped"])
assert st["status"] == "done" and st["done"] == 50 and len(st["skipped"]) == 2
z = zipfile.ZipFile(io.BytesIO(requests.get("%s/jobs/%s/result" % (base, jid), headers=H).content))
assert len(z.namelist()) == 48 and len(set(z.namelist())) == 48, z.namelist()[:3]
out = cv2.imdecode(np.frombuffer(z.read(z.namelist()[0]), np.uint8), 1)
assert out.shape == astro.shape and np.abs(out.astype(int) - astro.astype(int)).mean() > 1
print("images ok:", z.namelist()[0], out.shape)

# 2) limits and errors
assert requests.post(base + "/jobs/images", headers=H, files=[("source", ("s.jpg", good))] + [("targets", ("a.jpg", good))] * 51).status_code == 400
assert requests.post(base + "/jobs/images", headers=H, files=[("targets", ("a.jpg", good))]).status_code == 400
r = requests.post(base + "/jobs/images", headers=H, files=[("source", ("s.jpg", blank)), ("targets", ("a.jpg", good))])
st = wait(r.json()["job_id"]); assert st["status"] == "error" and "source" in st["error"], st
print("error paths ok:", st["error"])

# 3) video via stand-in bridge
r = requests.post(base + "/jobs/video", headers=H, files=[("source", ("s.jpg", good)), ("video", ("v.mp4", b"x" * 100000))])
st = wait(r.json()["job_id"]); print("video", st["status"], st["done"], st["total"])
assert st["status"] == "done" and st["done"] == 6
res = requests.get("%s/jobs/%s/result" % (base, r.json()["job_id"]), headers=H)
assert res.headers["Content-Type"] == "video/mp4" and len(res.content) > 1000

# 4) large upload streaming (60 MB) stays correct
big = os.urandom(60 * 1024 * 1024)
files, _ = S.parse_multipart(io.BytesIO(requests.Request("POST", "http://x", files=[("source", ("s.bin", big)), ("note", (None, "hi"))]).prepare().body), 
    requests.Request("POST", "http://x", files=[("a", ("a", b"1"))]).prepare().headers["Content-Type"], 0, tmp) if False else (None, None)
req = requests.Request("POST", "http://x", files=[("source", ("s.bin", big)), ("note", (None, "hi"))]).prepare()
files, fields = S.parse_multipart(io.BytesIO(req.body), req.headers["Content-Type"], len(req.body), tmp)
assert open(files[0][2], "rb").read() == big and fields == {"note": "hi"}
print("multipart 60MB ok"); print(json.loads(S.get_state())); S.stop(); print("SERVER ALL OK")
