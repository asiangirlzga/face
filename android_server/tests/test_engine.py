"""Desktop tests for engine.py. Usage: python -I test_engine.py <models_dir>
(models_dir needs det_10g.onnx and w600k_r50.onnx from buffalo_l.zip)."""
import os, struct, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "main", "python"))
import cv2, numpy as np, onnxruntime as ort
import engine as E

class DesktopOrt:
    def __init__(self): self.s = {}
    def load(self, name, path): self.s[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    def run(self, name, arrays):
        sess = self.s[name]
        feed = {i.name: a for i, a in zip(sess.get_inputs(), arrays)}
        return [np.asarray(o, dtype=np.float32).reshape(-1) for o in sess.run(None, feed)]

def varint(n):
    out = b""
    while True:
        b = n & 0x7F; n >>= 7
        if n: out += bytes([b | 0x80])
        else: return out + bytes([b])
def field(f, w, payload): return varint(f << 3 | w) + payload
def ld(f, data): return field(f, 2, varint(len(data)) + data)

def tensor(name, arr):
    t = b"".join(field(1, 0, varint(d)) for d in arr.shape) + field(2, 0, varint(1)) + ld(8, name.encode()) + ld(9, arr.astype("<f4").tobytes())
    return t

def test_emap():
    emap = np.random.rand(512, 512).astype(np.float32)
    graph = ld(1, b"fake node") + ld(5, tensor("a", np.ones((3, 3)))) + ld(5, tensor("b", np.ones((2,)))) + ld(5, tensor("emap", emap)) + ld(2, b"g")
    model = field(1, 0, varint(8)) + ld(2, b"x") + ld(7, graph)
    p = tempfile.mktemp(suffix=".onnx"); open(p, "wb").write(model)
    got = E.load_emap(p); os.remove(p)
    assert got.shape == (512, 512) and np.array_equal(got, emap)
    print("emap parser ok")

def test_umeyama():
    from skimage import transform as T
    kps = E.ARC_DST + np.random.randn(5, 2) * 3 + 50
    for size in (112, 128):
        ref = T.SimilarityTransform(); dst = E.ARC_DST * (size / 112.0 if size % 112 == 0 else size / 128.0)
        if size % 112: dst[:, 0] += 8.0 * size / 128.0
        ref.estimate(kps, dst)
        assert np.allclose(ref.params[:2], E.estimate_norm(kps, size), atol=1e-6)
    print("alignment matches skimage")

def main(models):
    from skimage import data
    test_emap(); test_umeyama()
    be = DesktopOrt()
    det = E.Detector(be, os.path.join(models, E.DET_FILE))
    rec = E.Recognizer(be, os.path.join(models, E.REC_FILE))
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    faces = det.detect(img)
    print("faces:", [(f.bbox.round(0).tolist(), round(f.score, 3)) for f in faces])
    assert len(faces) >= 1
    f = max(faces, key=lambda x: x.area)
    # landmarks must lie inside the box
    assert (f.kps[:, 0] > f.bbox[0]).all() and (f.kps[:, 0] < f.bbox[2]).all()
    # portrait + landscape + tiny letterbox shapes
    for im in (img, cv2.resize(img, (200, 640)), cv2.resize(img, (900, 300))):
        det.detect(im)
    e1 = rec.embed(img, f.kps)
    flip = cv2.flip(img, 1); ff = max(det.detect(flip), key=lambda x: x.area)
    e2 = rec.embed(flip, ff.kps)
    cos = float(e1 @ e2); print("flip cosine", round(cos, 3), "norm", round(float(np.linalg.norm(e1)), 3))
    assert cos > 0.6
    # swapper plumbing with a stand-in model (identity-ish generator)
    class FakeBE:
        def load(self, n, p): pass
        def run(self, name, arrays):
            blob = arrays[0]; out = np.clip(blob * 0.5 + 0.4, 0, 1)  # visibly different face region
            return [out.astype(np.float32).reshape(-1)]
    sw = E.Swapper.__new__(E.Swapper); sw.backend = FakeBE(); sw.emap = np.eye(512, dtype=np.float32)
    lat = sw.latent(e1); assert lat.shape == (1, 512)
    out = sw.swap(img, f.kps, lat)
    diff = np.abs(out.astype(int) - img.astype(int)).sum(2)
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    inside = diff[y1:y2, x1:x2].mean(); outside = diff[:max(y1 - 80, 1), :].mean()
    print("diff inside face", round(inside, 1), "far outside", round(outside, 2))
    assert inside > 5 and outside < 0.5 and out.shape == img.shape and out.dtype == np.uint8
    # blank image: no faces
    assert det.detect(np.full((480, 640, 3), 127, np.uint8)) == []
    print("ALL OK")

if __name__ == "__main__":
    main(sys.argv[1])
