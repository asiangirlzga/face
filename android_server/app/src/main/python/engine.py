"""On-device face swap pipeline: SCRFD det_10g + ArcFace w600k_r50 + inswapper_128.

Neural-net inference is delegated to a *backend* (on Android: ONNX Runtime running in
Kotlin, reached through OrtBridge). Everything else (pre/post-processing, alignment,
paste-back, emap extraction) is plain numpy + OpenCV.

Backend contract:
    backend.load(name, path)
    backend.run(name, [float32 arrays in model input order]) -> [flat float32 arrays in model output order]
"""
import mmap
import os

import cv2
import numpy as np

ARC_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)

DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"
SWAP_FILE = "inswapper_128.onnx"


# --------------------------------------------------------------------------- backend (Android)
def _to_bytes(arr):
    try:
        return bytes(memoryview(arr))
    except Exception:
        return bytes((b & 0xFF) for b in arr)


class AndroidOrt:
    """Calls com.example.faceswapserver.OrtBridge (Kotlin) through Chaquopy."""

    def __init__(self):
        from java import jclass
        self._b = jclass("com.example.faceswapserver.OrtBridge")

    def load(self, name, path):
        self._b.load(name, path)

    def unload(self, name):
        self._b.unload(name)

    def run(self, name, arrays):
        from java import jarray
        arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
        shapes = [jarray("J")([int(s) for s in a.shape]) for a in arrays]
        datas = [a.tobytes() for a in arrays]
        if len(arrays) == 1:
            res = self._b.run1(name, datas[0], shapes[0])
        elif len(arrays) == 2:
            res = self._b.run2(name, datas[0], shapes[0], datas[1], shapes[1])
        else:
            raise ValueError("1 or 2 inputs supported")
        return [np.frombuffer(_to_bytes(res[i]), dtype=np.float32) for i in range(len(res))]


# --------------------------------------------------------------------------- geometry
def umeyama(src, dst):
    """Similarity transform (3x3) mapping src -> dst (same maths as skimage SimilarityTransform)."""
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    A = dd.T @ sd / n
    d = np.ones(2)
    if np.linalg.det(A) < 0:
        d[1] = -1
    U, S, Vt = np.linalg.svd(A)
    R = U @ np.diag(d) @ Vt
    scale = 1.0 / sd.var(axis=0).sum() * (S @ d)
    T = np.eye(3)
    T[:2, :2] = R * scale
    T[:2, 2] = dm - scale * (R @ sm)
    return T


def estimate_norm(kps, size):
    if size % 112 == 0:
        ratio, dx = size / 112.0, 0.0
    else:
        ratio = size / 128.0
        dx = 8.0 * ratio
    dst = ARC_DST * ratio
    dst[:, 0] += dx
    return umeyama(np.asarray(kps, dtype=np.float64), dst)[:2]


def norm_crop(img, kps, size):
    M = estimate_norm(kps, size)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0), M


def _nms(dets, thresh):
    x1, y1, x2, y2, sc = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


# --------------------------------------------------------------------------- models
class Face:
    __slots__ = ("bbox", "score", "kps")

    def __init__(self, bbox, score, kps):
        self.bbox, self.score, self.kps = bbox, score, kps

    @property
    def area(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class Detector:
    SIZE = 640
    STRIDES = (8, 16, 32)

    def __init__(self, backend, path, thresh=0.5, nms=0.4):
        self.backend, self.thresh, self.nms = backend, thresh, nms
        self._centers = {}
        backend.load("det", path)

    def _anchor_centers(self, stride):
        c = self._centers.get(stride)
        if c is None:
            n = self.SIZE // stride
            grid = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32)
            c = (grid * stride).reshape(-1, 2)
            c = np.stack([c, c], axis=1).reshape(-1, 2)  # 2 anchors per cell
            self._centers[stride] = c
        return c

    def detect(self, img):
        h, w = img.shape[:2]
        if h / w > 1.0:
            nh, nw = self.SIZE, int(self.SIZE / (h / w))
        else:
            nw, nh = self.SIZE, int(self.SIZE * (h / w))
        scale = nh / float(h)
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (self.SIZE, self.SIZE),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.backend.run("det", [blob])
        if len(outs) != 9:
            raise RuntimeError("unexpected detector outputs: %d" % len(outs))
        sc_all, bb_all, kp_all = [], [], []
        for i, stride in enumerate(self.STRIDES):
            scores = outs[i].reshape(-1)
            bbox = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride
            ctr = self._anchor_centers(stride)
            pos = np.where(scores >= self.thresh)[0]
            if pos.size == 0:
                continue
            c, b, k = ctr[pos], bbox[pos], kps[pos]
            boxes = np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1],
                              c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=1)
            pts = np.empty((len(pos), 10), dtype=np.float32)
            pts[:, 0::2] = c[:, 0:1] + k[:, 0::2]
            pts[:, 1::2] = c[:, 1:2] + k[:, 1::2]
            sc_all.append(scores[pos])
            bb_all.append(boxes / scale)
            kp_all.append(pts / scale)
        if not sc_all:
            return []
        scores = np.concatenate(sc_all)
        boxes = np.concatenate(bb_all)
        kps = np.concatenate(kp_all).reshape(-1, 5, 2)
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms)
        return [Face(boxes[i], float(scores[i]), kps[i]) for i in keep]


class Recognizer:
    SIZE = 112

    def __init__(self, backend, path):
        self.backend = backend
        backend.load("rec", path)

    def embed(self, img, kps):
        aimg, _ = norm_crop(img, kps, self.SIZE)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 127.5, (self.SIZE, self.SIZE),
                                     (127.5, 127.5, 127.5), swapRB=True)
        emb = self.backend.run("rec", [blob])[0].reshape(-1)
        return emb / np.linalg.norm(emb)


# --------------------------------------------------------------------------- emap extraction
def _varint(m, i):
    shift = v = 0
    while True:
        b = m[i]
        i += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v, i
        shift += 7


def _fields(m, i, end):
    """Yield (field_number, wire_type, value) over a protobuf message in [i, end)."""
    while i < end:
        tag, i = _varint(m, i)
        f, w = tag >> 3, tag & 7
        if w == 0:
            v, i = _varint(m, i)
            yield f, w, v
        elif w == 2:
            ln, i = _varint(m, i)
            yield f, w, (i, i + ln)
            i += ln
        elif w == 1:
            yield f, w, (i, i + 8)
            i += 8
        elif w == 5:
            yield f, w, (i, i + 4)
            i += 4
        else:
            raise ValueError("unsupported wire type %d" % w)


def load_emap(onnx_path):
    """Return the last initializer of an ONNX graph (inswapper's `emap`) without the onnx package."""
    with open(onnx_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        graph = None
        for f, w, v in _fields(m, 0, len(m)):
            if f == 7 and w == 2:
                graph = v
        if graph is None:
            raise ValueError("no graph in model")
        last = None
        for f, w, v in _fields(m, graph[0], graph[1]):
            if f == 5 and w == 2:
                last = v
        if last is None:
            raise ValueError("model has no initializers")
        dims, dtype, raw, floats = [], 1, None, None
        for f, w, v in _fields(m, last[0], last[1]):
            if f == 1:
                if w == 0:
                    dims.append(v)
                else:  # packed
                    j = v[0]
                    while j < v[1]:
                        d, j = _varint(m, j)
                        dims.append(d)
            elif f == 2:
                dtype = v
            elif f == 9:
                raw = bytes(m[v[0]:v[1]])
            elif f == 4 and w == 2:
                floats = bytes(m[v[0]:v[1]])
        np_dtype = {1: np.float32, 10: np.float16, 11: np.float64}.get(dtype)
        if np_dtype is None:
            raise ValueError("unsupported emap dtype %d" % dtype)
        buf = raw if raw is not None else floats
        if buf is None:
            raise ValueError("emap has no data")
        return np.frombuffer(buf, dtype=np_dtype).reshape(dims).astype(np.float32)


class Swapper:
    SIZE = 128
    # Face-shaped blend region in the 128px aligned crop: (cx, cy, rx, ry). It stays inside the skin so the
    # generated hair/ear/background corners of the crop are never pasted back.
    FACE = (64.0, 66.0, 38.0, 52.0)
    FEATHER = 3.5          # blend edge softness, in 128px-crop pixels (scaled up to the photo)

    def __init__(self, backend, path, color_match=True, sharpen=0.8, grain=True):
        self.color_match, self.sharpen, self.grain = color_match, sharpen, grain
        self.backend = backend
        backend.load("swap", path)
        self.emap = load_emap(path)

    def latent(self, normed_embedding):
        lat = normed_embedding.reshape(1, -1) @ self.emap
        lat /= np.linalg.norm(lat)
        return lat.astype(np.float32)

    def _ellipse_mask(self, T, size, shrink=1.0):
        """Hard 0..1 face ellipse (model space) warped into the photo ROI by the 2x3 matrix T."""
        cx, cy, rx, ry = self.FACE
        up = 4
        big = np.zeros((self.SIZE * up, self.SIZE * up), np.uint8)
        cv2.ellipse(big, (int(cx * up), int(cy * up)), (int(rx * shrink * up), int(ry * shrink * up)),
                    0, 0, 360, 255, -1, cv2.LINE_AA)
        T4 = T.copy()
        T4[:, :2] = T[:, :2] / up
        return cv2.warpAffine(big, T4, size, flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0

    def swap(self, img, kps, latent):
        S = self.SIZE
        aimg, M = norm_crop(img, kps, S)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (S, S), (0, 0, 0), swapRB=True)
        pred = self.backend.run("swap", [blob, latent])[0].reshape(1, 3, S, S)
        fake128 = np.clip(255 * pred.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[:, :, ::-1]

        H, W = img.shape[:2]
        IM = cv2.invertAffineTransform(M)                       # model px -> photo px
        corners = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float64)
        pts = corners @ IM[:, :2].T + IM[:, 2]
        x0 = max(int(np.floor(pts[:, 0].min())) - 4, 0)
        y0 = max(int(np.floor(pts[:, 1].min())) - 4, 0)
        x1 = min(int(np.ceil(pts[:, 0].max())) + 4, W)
        y1 = min(int(np.ceil(pts[:, 1].max())) + 4, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return img
        rw, rh = x1 - x0, y1 - y0
        T = IM.copy()
        T[0, 2] -= x0
        T[1, 2] -= y0
        scale = float(np.sqrt(abs(np.linalg.det(IM[:, :2]))))   # photo px per model px

        fake = cv2.warpAffine(fake128, T, (rw, rh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        orig = img[y0:y1, x0:x1].astype(np.float32)
        inner = self._ellipse_mask(T, (rw, rh))
        if inner.sum() < 50:
            return img
        sk = max(self.FEATHER * scale, 1.0)
        alpha = cv2.GaussianBlur(inner, (0, 0), sk)[:, :, None]
        stats = self._ellipse_mask(T, (rw, rh), 0.8)            # skin-only area used to measure tone

        if scale > 1.6:                                         # remove the enlarged 128px pixel grid first
            fake = cv2.GaussianBlur(fake, (0, 0), 0.4 * scale)
        if self.color_match:
            fake = _match_lighting(fake, orig, stats, scale)
        if self.sharpen > 0 and scale > 1.0:                    # boost feature-sized detail, not the pixel grid
            blur = cv2.GaussianBlur(fake, (0, 0), 0.9 * scale)
            fake = fake + self.sharpen * (fake - blur)
        if self.grain:
            fake = _match_grain(fake, orig, stats > 0.5)
        merged = alpha * np.clip(fake, 0, 255) + (1 - alpha) * orig
        out = img.copy()
        out[y0:y1, x0:x1] = np.clip(merged + 0.5, 0, 255).astype(np.uint8)
        return out


def _match_lighting(fake, orig, stats, scale):
    """Make the swapped face's colour AND lighting gradient follow the photo's own skin.

    Measures the smooth (low-frequency) colour of both faces over the skin area and adds the difference to the
    swapped face. Unlike one global mean/std shift, a face lit from one side keeps that lighting, and the
    difference fades smoothly, so no seam appears where the swapped face meets the photo."""
    h, w = fake.shape[:2]
    ds = max(1, int(min(h, w) / 160))                           # do the heavy blur on a small copy
    sw_, sh_ = max(w // ds, 2), max(h // ds, 2)
    def lab(a):
        small = cv2.resize(a, (sw_, sh_), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(np.clip(small, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    f, o = lab(fake), lab(orig)
    m = cv2.resize(stats, (sw_, sh_), interpolation=cv2.INTER_AREA)
    sigma = max(10.0 * scale / ds, 2.0)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-4
    corr = np.empty_like(f)
    for c in range(3):
        corr[..., c] = (cv2.GaussianBlur(o[..., c] * m, (0, 0), sigma) -
                        cv2.GaussianBlur(f[..., c] * m, (0, 0), sigma)) / den
    corr = cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)
    fl = cv2.cvtColor(np.clip(fake, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    fl = np.clip(fl + corr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(fl, cv2.COLOR_LAB2BGR).astype(np.float32)


def _match_grain(fake, orig, core):
    """The model output is noise-free; add the photo's own fine noise so the face does not look airbrushed."""
    if core.sum() < 200:
        return fake
    def hf(a):
        g = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 1.2)
    want = float(hf(orig)[core].std())
    have = float(hf(fake)[core].std())
    extra = (max(want * want - have * have, 0.0)) ** 0.5
    if extra < 0.3:
        return fake
    rng = np.random.default_rng(1234)                           # fixed seed: same input, same output
    noise = rng.normal(0.0, min(extra, 6.0), fake.shape[:2]).astype(np.float32)
    return fake + noise[..., None]


# --------------------------------------------------------------------------- facade
class Engine:
    def __init__(self, backend, models_dir):
        self.backend = backend
        self.detector = Detector(backend, os.path.join(models_dir, DET_FILE))
        self.recognizer = Recognizer(backend, os.path.join(models_dir, REC_FILE))
        self.swapper = Swapper(backend, os.path.join(models_dir, SWAP_FILE))

    def unload(self):
        """Release the ONNX sessions (RAM) so a bigger model can use the memory."""
        for name in ("det", "rec", "swap"):
            try:
                self.backend.unload(name)
            except Exception:
                pass

    def source_latent(self, img):
        faces = self.detector.detect(img)
        if not faces:
            raise RuntimeError("No face found in the source image")
        face = max(faces, key=lambda f: f.area)
        return self.swapper.latent(self.recognizer.embed(img, face.kps))

    def swap_image(self, img, latent):
        """Swap every detected face; returns None when the image has no face."""
        faces = self.detector.detect(img)
        if not faces:
            return None
        out = img
        for f in faces:
            out = self.swapper.swap(out, f.kps, latent)
        return out
