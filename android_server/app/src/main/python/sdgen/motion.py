"""Image -> video "camera motion" clip (zoom / pan / drift) rendered from a single still image.

This is NOT generative animation: the subject does not move, only the virtual camera does.
It is cheap enough to run on any phone. Frames go to a writer (Android: VideoBridge/MediaCodec).
"""
import cv2
import numpy as np

from . import pipeline

STYLES = ("zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "drift")


class AndroidVideoWriter:
    """Writes BGR frames through com.example.faceswapserver.VideoBridge (H.264 MP4, no audio)."""

    def __init__(self):
        from java import jclass
        self._c = jclass("com.example.faceswapserver.VideoBridge")

    def open(self, path, w, h, fps):
        self._c.openWriter(path, int(w), int(h), float(fps), None)

    def write(self, frame):
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Could not encode a video frame")
        self._c.writeFrame(enc.tobytes())

    def close(self):
        self._c.closeWriter()


def _ease(t):
    return t * t * (3.0 - 2.0 * t)


def crop_rect(style, t, w, h, zoom=1.25):
    """Source-pixel crop window (x0, y0, cw, ch) at eased time t in [0, 1]; keeps the image aspect."""
    if style == "zoom_in":
        k = 1.0 - (1.0 - 1.0 / zoom) * t
        fx, fy = 0.5, 0.5
    elif style == "zoom_out":
        k = 1.0 / zoom + (1.0 - 1.0 / zoom) * t
        fx, fy = 0.5, 0.5
    elif style == "pan_left":
        k, fx, fy = 1.0 / 1.2, 1.0 - t, 0.5
    elif style == "pan_right":
        k, fx, fy = 1.0 / 1.2, t, 0.5
    elif style == "pan_up":
        k, fx, fy = 1.0 / 1.2, 0.5, 1.0 - t
    elif style == "pan_down":
        k, fx, fy = 1.0 / 1.2, 0.5, t
    elif style == "drift":
        k = 1.0 - (1.0 - 1.0 / 1.15) * t
        fx, fy = 0.3 + 0.4 * t, 0.6 - 0.3 * t
    else:
        raise ValueError("Unknown style: %s" % style)
    cw, ch = w * k, h * k
    return (w - cw) * fx, (h - ch) * fy, cw, ch


def make_video(image_path, out_path, seconds=4, fps=24, style="zoom_in", max_side=768, writer=None):
    """Entry point called from Kotlin and from server.py. Returns out_path, or "" if cancelled."""
    if not pipeline._busy.acquire(False):
        raise RuntimeError(pipeline.BUSY_MESSAGE)   # leave the running job's progress untouched
    try:
        pipeline._reset()
        try:
            if style not in STYLES:
                raise ValueError("Unknown style: %s" % style)
            seconds, fps = int(seconds), int(fps)
            if not 1 <= seconds <= 10 or not 12 <= fps <= 30:
                raise ValueError("Use 1-10 seconds and 12-30 fps")
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Could not read the image")
            h, w = img.shape[:2]
            scale = max_side / float(max(h, w))
            out_w, out_h = max(2, int(round(w * scale)) // 2 * 2), max(2, int(round(h * scale)) // 2 * 2)
            keep = 1.3 * max(out_w, out_h) / float(max(h, w))   # pre-shrink huge photos, keep zoom headroom
            if keep < 1.0:
                img = cv2.resize(img, (int(w * keep), int(h * keep)), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            n = seconds * fps
            writer = writer or AndroidVideoWriter()
            writer.open(out_path, out_w, out_h, fps)
            try:
                for i in range(n):
                    pipeline.check_cancel()
                    pipeline.set_progress(stage="Rendering video: frame %d of %d" % (i + 1, n), frac=i / float(n))
                    x0, y0, cw, ch = crop_rect(style, _ease(i / float(max(n - 1, 1))), w, h)
                    sx, sy = out_w / cw, out_h / ch
                    m = np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]], dtype=np.float32)
                    frame = cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
                    writer.write(frame)
            finally:
                pipeline.set_progress(stage="Finishing video", frac=0.99)
                writer.close()
            pipeline.set_progress(stage="Done", frac=1.0, done=True, result=out_path)
            return out_path
        except pipeline.Cancelled:
            pipeline.set_progress(stage="Cancelled", done=True, error="Cancelled")
            return ""
        except Exception as e:
            pipeline.set_progress(stage="Failed", done=True, error=str(e))
            raise
    finally:
        pipeline._busy.release()
