"""Desktop tests for the sdgen package (tokenizer, scheduler, pipeline with a fake ONNX backend, motion video).

Usage: python -I tests/test_sdgen.py [path/to/openai-CLIP/clip]
The optional directory (from `git clone https://github.com/openai/CLIP`) enables the tokenizer
comparison against OpenAI's reference implementation. Needs numpy and opencv-python-headless.
"""
import gzip
import importlib.util
import json
import os
import random
import sys
import tempfile
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app", "src", "main", "python"))
from sdgen import motion, pipeline as P, scheduler  # noqa: E402
from sdgen.tokenizer import ClipTokenizer, bytes_to_unicode  # noqa: E402

TMP = tempfile.mkdtemp()


def ok(msg):
    print("ok -", msg)


# --------------------------------------------------------------------------- tokenizer
def write_hf_tokenizer(gz_path, out_dir):
    """Build HF-format vocab.json / merges.txt from OpenAI's bpe_simple_vocab_16e6.txt.gz."""
    os.makedirs(out_dir, exist_ok=True)
    lines = gzip.open(gz_path).read().decode("utf-8").split("\n")
    merges = lines[1:49152 - 256 - 2 + 1]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v + "</w>" for v in vocab] + ["".join(m.split()) for m in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump({v: i for i, v in enumerate(vocab)}, fh, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as fh:
        fh.write("#version: 0.2\n" + "\n".join(merges) + "\n")
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as fh:
        json.dump({"bos_token": {"content": "<|startoftext|>"}, "eos_token": {"content": "<|endoftext|>"},
                   "pad_token": "!"}, fh)


def load_reference(clip_dir):
    """OpenAI's SimpleTokenizer, with `regex` replaced by an `re` approximation and ftfy by identity."""
    import re

    class Compiled:
        def __init__(self, pat):
            pat = (pat.replace(r"[\p{L}]+", r"[^\W\d_]+").replace(r"[\p{N}]", r"\d")
                   .replace(r"[^\s\p{L}\p{N}]+", r"(?:[^\s\w]|_)+"))
            self.rx = re.compile(pat, re.IGNORECASE)

        def findall(self, text):
            return self.rx.findall(text)

    shim = types.ModuleType("regex")
    shim.compile = lambda p, flags=0: Compiled(p)
    shim.findall = lambda pat, text: pat.findall(text)
    shim.sub = re.sub
    shim.IGNORECASE = re.IGNORECASE
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda t: t
    sys.modules["regex"], sys.modules["ftfy"] = shim, ftfy
    spec = importlib.util.spec_from_file_location("clip_simple_tokenizer", os.path.join(clip_dir, "simple_tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SimpleTokenizer(os.path.join(clip_dir, "bpe_simple_vocab_16e6.txt.gz"))


PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "starry night over Golden Gate Bridge by van gogh",
    "A cat's tiny paws, they're 3 inches long!!",
    "  extra   spaces\tand\nnewlines  ",
    "portrait, 8k, ultra-detailed, (masterpiece:1.2), [bokeh]",
    "I'll be there; you've been here; we'd go, I'm ok. It's 100% fine",
    "Ünïcödé café naïve résumé façade",
    "东京的夜景 neon lights 🌃🚀 cyberpunk",
    "snake_case_words and CamelCaseWords and numbers 12345 6.78",
    "&amp; &lt;b&gt; html entities &quot;quoted&quot;",
    "<|startoftext|> special <|endoftext|> tokens",
    "a" * 300,
    "word " * 120,
    "!!!???...,,,;;;",
    "x",
    "ok",
]


def test_tokenizer(clip_dir):
    d = os.path.join(TMP, "tokenizer")
    write_hf_tokenizer(os.path.join(clip_dir, "bpe_simple_vocab_16e6.txt.gz"), d)
    mine = ClipTokenizer(d)
    ref = load_reference(clip_dir)
    assert mine.bos == 49406 and mine.eos == 49407 and mine.pad == 0, (mine.bos, mine.eos, mine.pad)
    for p in PROMPTS:
        assert mine.tokens(p) == ref.encode(p), p
    ok("tokenizer matches OpenAI reference on %d prompts" % len(PROMPTS))
    rnd = random.Random(1)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ABCDEF0123456789 '.,!?-_()[]:;/%&") + list("éüñçßøåÆ东京猫あ한🌃🚀")
    for _ in range(400):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 60)))
        assert mine.tokens(s) == ref.encode(s), repr(s)
    ok("tokenizer matches reference on 400 random strings")
    arr = mine.encode("hello world")
    assert arr.shape == (1, 77) and arr.dtype == np.int64
    assert arr[0, 0] == 49406 and arr[0, 3] == 49407 and (arr[0, 4:] == 0).all()
    long = mine.encode("word " * 200)
    assert long.shape == (1, 77) and long[0, 76] == 49407 and long[0, 0] == 49406
    ok("encode(): BOS/EOS/pad(!=0)/truncation to 77 correct")


# --------------------------------------------------------------------------- scheduler
def test_scheduler():
    ts, sig, init = scheduler.euler_trailing(1)
    assert list(ts) == [999.0] and abs(sig[0] - 14.6146) < 1e-3 and sig[1] == 0.0, (ts, sig)
    assert abs(init - 14.6146) < 1e-3  # trailing spacing: init_noise_sigma == max(sigmas)
    ts4, sig4, _ = scheduler.euler_trailing(4)
    assert list(ts4) == [999.0, 749.0, 499.0, 249.0] and (np.diff(sig4[:-1]) < 0).all() and sig4[-1] == 0
    for n in (2, 3, 5, 7, 8):
        t, s, _ = scheduler.euler_trailing(n)
        assert len(t) == n and len(s) == n + 1 and (np.diff(t) < 0).all(), n
    ok("scheduler: trailing timesteps and sigmas (sigma_max=14.6146)")


# --------------------------------------------------------------------------- fake ONNX backend
INFO = {
    "te": [("input_ids", "int32")],
    "unet": [("sample", "float16"), ("timestep", "int64"), ("encoder_hidden_states", "float16")],
    "vae": [("latent_sample", "float32")],
}
OUTS = {"te": ["last_hidden_state", "pooler_output"], "unet": ["out_sample"], "vae": ["sample"]}


class FakeBackend:
    def __init__(self, unet_fn=None):
        self.loaded, self.log, self.runs = set(), [], []
        self.unet_fn = unet_fn
        self.on_unet = None

    def load(self, name, path, opt):
        assert os.path.isfile(path), path
        assert not self.loaded, "two models resident at once: %s + %s" % (self.loaded, name)
        self.loaded.add(name)
        self.log.append(("load", name, opt))

    def unload(self, name):
        self.loaded.remove(name)
        self.log.append(("unload", name))

    def input_info(self, name):
        return INFO[name]

    def output_names(self, name):
        return OUTS[name]

    def run(self, name, feeds):
        want = dict(INFO[name])
        assert set(feeds) == set(want), (set(feeds), set(want))
        for k, v in feeds.items():
            assert v.dtype == np.dtype(want[k]), (name, k, v.dtype, want[k])
        self.runs.append(name)
        if name == "te":
            ids = feeds["input_ids"]
            hid = np.sin(ids[..., None] * 0.01 + np.arange(1024) * 0.001).astype(np.float32)
            return [hid.ravel(), hid[:, 0].ravel()]
        if name == "unet":
            if self.on_unet:
                self.on_unet()
            s = feeds["sample"].astype(np.float32)
            eps = self.unet_fn(s) if self.unet_fn else 0.1 * s + 0.01 * float(feeds["encoder_hidden_states"].astype(np.float32).mean())
            return [eps.astype(np.float32).ravel()]
        z = feeds["latent_sample"]  # vae: local (nearest-neighbour) decoder, so tiling must be exact
        img = np.repeat(np.repeat(z[:, :3] * 0.5 + z[:, 3:4] * 0.1, 8, axis=2), 8, axis=3)
        return [img.astype(np.float32).ravel()]


def make_models_dir():
    d = os.path.join(TMP, "models_%d" % random.randrange(10 ** 9))
    for rel in P.REQUIRED_FILES:
        path = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"x")
    open(os.path.join(d, "unet", "model.onnx_data"), "wb").write(b"x")   # external weights of the unet graph
    return d


def make_tokenizer_files(models_dir, src):
    for f in ("vocab.json", "merges.txt", "special_tokens_map.json"):
        open(os.path.join(models_dir, "tokenizer", f), "wb").write(open(os.path.join(src, f), "rb").read())


def tiny_tokenizer_files(models_dir):
    """Minimal HF-style tokenizer (enough merges for tests that do not have the real vocab)."""
    d = os.path.join(models_dir, "tokenizer")
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    n = len(vocab)
    for i, c in enumerate(list(bytes_to_unicode().values())):
        vocab[c + "</w>"] = n + i
    vocab["<|startoftext|>"], vocab["<|endoftext|>"] = 49406, 49407
    json.dump(vocab, open(os.path.join(d, "vocab.json"), "w"))
    open(os.path.join(d, "merges.txt"), "w").write("#version: 0.2\n")
    json.dump({"pad_token": "!"}, open(os.path.join(d, "special_tokens_map.json"), "w"))


def test_pipeline(real_tok_dir):
    md = make_models_dir()
    if real_tok_dir:
        make_tokenizer_files(md, real_tok_dir)
    else:
        tiny_tokenizer_files(md)

    out = os.path.join(TMP, "a.jpg")
    be = FakeBackend()
    assert P.generate_image("a red fox in snow", 1, 512, 123, md, out, True, be) == out
    img = cv2.imread(out)
    assert img.shape == (512, 512, 3)
    assert be.log == [("load", "te", "basic"), ("unload", "te"), ("load", "unet", "none"), ("unload", "unet"),
                      ("load", "vae", "all"), ("unload", "vae")], be.log
    assert be.runs[0] == "te" and be.runs[1] == "unet" and be.runs.count("vae") == 4  # 2x2 tiles at 512
    st = json.loads(P.get_progress())
    assert st["done"] and st["frac"] == 1.0 and st["seed"] == 123 and st["result"] == out and st["error"] is None
    ok("pipeline: stages, one model resident at a time, dtype casting (fp16/int64/int32), 4 VAE tiles")

    out2 = os.path.join(TMP, "b.jpg")
    P.generate_image("a red fox in snow", 1, 512, 123, md, out2, True, FakeBackend())
    assert open(out, "rb").read() == open(out2, "rb").read()
    out3 = os.path.join(TMP, "c.jpg")
    P.generate_image("a red fox in snow", 1, 512, 124, md, out3, True, FakeBackend())
    assert open(out, "rb").read() != open(out3, "rb").read()
    ok("pipeline: same seed -> identical image, different seed -> different")

    out4 = os.path.join(TMP, "d.jpg")
    be4 = FakeBackend()
    P.generate_image("a red fox in snow", 1, 512, 123, md, out4, False, be4)
    assert be4.runs.count("vae") == 1
    a, b = cv2.imread(out).astype(int), cv2.imread(out4).astype(int)
    assert np.abs(a - b).max() <= 1, np.abs(a - b).max()
    ok("pipeline: tiled VAE decode equals full decode (max diff %d level)" % np.abs(a - b).max())

    be5 = FakeBackend()
    P.generate_image("fox", 4, 384, 5, md, os.path.join(TMP, "e.jpg"), True, be5)
    assert be5.runs.count("unet") == 4 and cv2.imread(os.path.join(TMP, "e.jpg")).shape == (384, 384, 3)
    assert be5.runs.count("vae") == 1  # 384 px decodes in one piece, tiling would only cost time
    ok("pipeline: 4 steps -> 4 UNet runs, 384x384 output decoded without tiles")

    # oracle UNet (eps = its scaled input): one Euler step gives x = x0 * (1 - sigma / sqrt(sigma^2 + 1))
    _, sig, init = scheduler.euler_trailing(1)
    cap = {}
    be6 = FakeBackend(unet_fn=lambda s: cap.setdefault("eps", s))
    pipe = P.SDTurbo(be6, md)
    hid = pipe.encode_prompt("fox")
    x = pipe.denoise(hid, 1, 512, 7)
    expect = init * (1.0 - sig[0] / (sig[0] ** 2 + 1.0) ** 0.5)
    assert abs(float(x.std()) - expect) < 0.01 * expect + 1e-3, (float(x.std()), expect)
    ok("scheduler maths: oracle noise prediction gives x0 std %.4f (expected %.4f)" % (float(x.std()), expect))

    # cancel mid-way
    be7 = FakeBackend()
    be7.on_unet = P.cancel
    r = P.generate_image("fox", 4, 512, 1, md, os.path.join(TMP, "f.jpg"), True, be7)
    assert r == "" and not be7.loaded and be7.runs.count("unet") == 1
    st = json.loads(P.get_progress())
    assert st["done"] and st["error"] == "Cancelled"
    ok("pipeline: cancel stops after the current step and unloads the model")

    for bad, args in (("empty prompt", ("  ", 1, 512)), ("bad size", ("fox", 1, 500)), ("bad steps", ("fox", 0, 512))):
        try:
            P.generate_image(args[0], args[1], args[2], 1, md, os.path.join(TMP, "g.jpg"), True, FakeBackend())
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    try:
        P.generate_image("fox", 1, 512, 1, os.path.join(TMP, "nope"), os.path.join(TMP, "g.jpg"), True, FakeBackend())
    except RuntimeError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("missing files")
    assert json.loads(P.get_progress())["error"]
    assert P.missing_files(md) == [] and len(P.missing_files(os.path.join(TMP, "nope"))) == len(P.REQUIRED_FILES)
    data = os.path.join(md, "unet", "model.onnx_data")
    os.rename(data, data + ".bak")
    assert P.missing_files(md) == ["unet/model.onnx_data"]   # graph present but its weights file is not
    os.rename(data + ".bak", data)
    ok("pipeline: input validation and missing-model errors")

    # a second job while one runs is refused and must not disturb the running job's progress
    P.set_progress(stage="Generating: step 1 of 1", frac=0.5, error=None, done=False)
    assert P._busy.acquire(False)
    try:
        for fn in (lambda: P.generate_image("fox", 1, 512, 1, md, os.path.join(TMP, "h.jpg"), True, FakeBackend()),
                   lambda: motion.make_video(os.path.join(TMP, "nope.jpg"), os.path.join(TMP, "h.mp4"), 1, 12, "zoom_in")):
            try:
                fn()
            except RuntimeError as e:
                assert "already running" in str(e)
            else:
                raise AssertionError("second job was not refused")
    finally:
        P._busy.release()
    st = json.loads(P.get_progress())
    assert st["stage"] == "Generating: step 1 of 1" and st["frac"] == 0.5 and not st["done"]
    ok("pipeline/motion: only one job at a time, refusal leaves progress untouched")


# --------------------------------------------------------------------------- motion
class FakeWriter:
    def __init__(self):
        self.frames, self.size, self.fps, self.closed = [], None, None, False

    def open(self, path, w, h, fps):
        self.size, self.fps, self.path = (w, h), fps, path
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        assert frame.shape == (self.size[1], self.size[0], 3) and frame.dtype == np.uint8
        self.frames.append(frame.copy())
        self.vw.write(frame)

    def close(self):
        self.vw.release()
        self.closed = True


def test_motion():
    rng = np.random.default_rng(0)
    src = cv2.GaussianBlur((rng.random((517, 301, 3)) * 255).astype(np.uint8), (0, 0), 6)
    cv2.circle(src, (150, 250), 60, (0, 0, 255), -1)
    p = os.path.join(TMP, "src.jpg")
    cv2.imwrite(p, src)
    for style in motion.STYLES:
        for t in (0.0, 0.25, 0.5, 1.0):
            x0, y0, cw, ch = motion.crop_rect(style, t, 301, 517)
            assert x0 >= -1e-6 and y0 >= -1e-6 and x0 + cw <= 301 + 1e-6 and y0 + ch <= 517 + 1e-6, (style, t)
        w = FakeWriter()
        out = os.path.join(TMP, "m_%s.mp4" % style)
        assert motion.make_video(p, out, 2, 12, style, 640, w) == out
        assert len(w.frames) == 24 and w.closed and w.size[0] % 2 == 0 and w.size[1] % 2 == 0
        assert max(w.size) <= 640 and abs(w.size[0] / w.size[1] - 301 / 517) < 0.02
        assert np.abs(w.frames[0].astype(int) - w.frames[-1].astype(int)).mean() > 1.0, style
        assert os.path.getsize(out) > 1000
    ok("motion: 7 styles render, crops stay inside the image, even output size, frames differ")
    z = FakeWriter()
    motion.make_video(p, os.path.join(TMP, "z.mp4"), 1, 12, "zoom_in", 480, z)
    mid = z.frames[0][z.size[1] // 2, z.size[0] // 2]
    assert mid[2] > 150 and mid[0] < 80  # red disc is at the centre in frame 0 (zoom starts wide)
    for bad in (("nope", 4, 24), ("zoom_in", 0, 24), ("zoom_in", 4, 60)):
        try:
            motion.make_video(p, os.path.join(TMP, "x.mp4"), bad[1], bad[2], bad[0], 480, FakeWriter())
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    ok("motion: argument validation")
    w = FakeWriter()
    orig = motion.pipeline.check_cancel
    n = {"i": 0}

    def cancelling():
        n["i"] += 1
        if n["i"] == 5:
            motion.pipeline.cancel()
        orig()
    motion.pipeline.check_cancel = cancelling
    try:
        assert motion.make_video(p, os.path.join(TMP, "c.mp4"), 4, 24, "drift", 480, w) == ""
    finally:
        motion.pipeline.check_cancel = orig
    assert w.closed and len(w.frames) == 4
    ok("motion: cancel closes the writer")


# --------------------------------------------------------------------------- onnx_probe + timestep fallback
def _varint_b(n):
    out = b""
    while True:
        c = n & 0x7F
        n >>= 7
        if n:
            out += bytes([c | 0x80])
        else:
            return out + bytes([c])


def _fld(f, payload):
    return _varint_b(f << 3 | 2) + _varint_b(len(payload)) + payload


def _vfld(f, v):
    return _varint_b(f << 3) + _varint_b(v)


def _value_info(name, dims, elem=7):
    shape = b"".join(_fld(1, _fld(2, d.encode()) if isinstance(d, str) else _vfld(1, d)) for d in dims)
    tensor_type = _vfld(1, elem) + _fld(2, shape)
    return _fld(1, name.encode()) + _fld(2, _fld(1, tensor_type))


def _node(op, ins, outs, name, attrs=b""):
    return b"".join(_fld(1, i.encode()) for i in ins) + b"".join(_fld(2, o.encode()) for o in outs) \
        + _fld(3, name.encode()) + _fld(4, op.encode()) + attrs


def _int64_init(name, vals):
    raw = b"".join(int(v).to_bytes(8, "little", signed=True) for v in vals)
    return _vfld(1, len(vals)) + _vfld(2, 7) + _fld(8, name.encode()) + _fld(9, raw)


def synthetic_unet_graph(timestep_dims=("steps",)):
    """A graph shaped like a real unet's time path: timestep -> Unsqueeze(axes=[0]) -> Expand -> ... -> Gemm."""
    nodes = [
        _node("Unsqueeze", ["timestep", "axes0"], ["t1"], "/up_blocks.0/Unsqueeze"),
        _node("Expand", ["t1", "bs"], ["t2"], "/Expand"),
        _node("Cast", ["t2"], ["t3"], "/time_proj/Cast"),
        _node("Gemm", ["t3", "w"], ["t4"], "/time_embedding/linear_1/Gemm"),
    ]
    graph = b"".join(_fld(1, n) for n in nodes)
    graph += _fld(5, _int64_init("axes0", [0])) + _fld(5, _int64_init("bs", [1]))
    graph += _fld(11, _value_info("sample", ["batch", 4, "h", "w"], 1))
    graph += _fld(11, _value_info("timestep", list(timestep_dims)))
    graph += _fld(12, _value_info("out_sample", ["batch", 4, "h", "w"], 1))
    return _vfld(1, 8) + _fld(2, b"test") + _fld(7, graph + _fld(2, b"g"))


def test_onnx_probe():
    from sdgen import onnx_probe as O
    src = os.path.join(TMP, "probe_src.onnx")
    open(src, "wb").write(synthetic_unet_graph())
    g = O.read(src)
    assert [i["name"] for i in g["inputs"]] == ["sample", "timestep"]
    assert g["inputs"][1]["dims"] == ["steps"] and g["inputs"][0]["dims"] == ["batch", 4, "h", "w"]
    assert [n["op"] for n in g["nodes"]] == ["Unsqueeze", "Expand", "Cast", "Gemm"] and g["inits"]["axes0"] == [0]
    assert O.declared_rank(src, "timestep") == 1 and O.declared_rank(src, "nope") is None
    d = O.describe_path(src)
    assert "timestep[steps]" in d and "Unsqueeze([0])" in d and d.endswith("Gemm"), d
    for rank, dims in ((0, []), (2, ["d0", "d1"]), (1, ["d0"])):
        dst = os.path.join(TMP, "probe_%d.onnx" % rank)
        O.patch_input_rank(src, dst, "timestep", rank)
        h = O.read(dst)
        assert h["inputs"][1]["dims"] == dims and h["inputs"][1]["elem"] == 7, (rank, h["inputs"][1])
        assert h["inputs"][0]["dims"] == ["batch", 4, "h", "w"] and [n["op"] for n in h["nodes"]] == [n["op"] for n in g["nodes"]]
        assert h["inits"] == g["inits"] and len(h["inputs"]) == 2
    try:
        O.patch_input_rank(src, os.path.join(TMP, "x.onnx"), "nope", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("patching a missing input must fail")
    open(os.path.join(TMP, "junk.onnx"), "wb").write(b"x")
    try:
        O.read(os.path.join(TMP, "junk.onnx"))
    except Exception:
        pass
    else:
        raise AssertionError("junk must not parse")
    ok("onnx_probe: reads inputs/ops/constants, patches the declared rank, rejects junk")


class RankBackend(FakeBackend):
    """Behaves like a graph that treats `timestep` as a scalar although it declares a vector: it only works once
    the declared rank was patched to 0 and a scalar is fed (anything else fails like the phone did)."""

    GEMM = ("ai.onnxruntime.OrtException: Non-zero status code returned while running Gemm node. "
            "Name:'/time_embedding/linear_1/Gemm' Status Message: left.NumDimensions() == 2 || left.NumDimensions() == 1 was false.")

    def __init__(self, works_at=0, error=None):
        super().__init__()
        self.paths, self.works_at, self.error = [], works_at, error
        self.attempts = []

    def load(self, name, path, opt):
        super().load(name, path, opt)
        self.paths.append(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))
        self.cur = path

    def run(self, name, feeds):
        if name != "unet":
            return super().run(name, feeds)
        from sdgen import onnx_probe as O
        declared = O.declared_rank(self.cur, "timestep")
        ts = feeds["timestep"]
        self.attempts.append((os.path.basename(self.cur), ts.ndim))
        if self.error:
            raise RuntimeError(self.error)
        if ts.ndim != declared:
            raise RuntimeError("Invalid rank for input: timestep Got: %d Expected: %d" % (ts.ndim, declared))
        if ts.ndim != self.works_at:
            raise RuntimeError(self.GEMM)
        return super().run(name, feeds)


def test_unet_timestep_fallback(real_tok_dir):
    md = make_models_dir()
    tiny_tokenizer_files(md) if not real_tok_dir else make_tokenizer_files(md, real_tok_dir)
    unet = os.path.join(md, "unet", "model.onnx")
    open(unet, "wb").write(synthetic_unet_graph())

    # 1) graph acts on timestep as a scalar: first try (rank 1) fails like on the phone, rank 0 works
    be = RankBackend(works_at=0)
    out = os.path.join(TMP, "fb1.jpg")
    assert P.generate_image("fox", 1, 384, 3, md, out, True, be) == out and cv2.imread(out).shape == (384, 384, 3)
    assert be.paths.count("unet/model.onnx") == 1 and "unet/model_ts0.onnx" in be.paths and not be.loaded, be.paths
    assert be.attempts[0] == ("model.onnx", 1) and be.attempts[-1] == ("model_ts0.onnx", 0), be.attempts
    assert open(os.path.join(md, "unet", "timestep_rank.txt")).read() == "0"
    ok("unet timestep: rank mismatch is detected, patched graph used, result produced")

    # 2) the next image goes straight to the patched graph (no failing attempt, no second reload)
    be2 = RankBackend(works_at=0)
    P.generate_image("fox", 2, 384, 3, md, os.path.join(TMP, "fb2.jpg"), True, be2)
    assert be2.paths.count("unet/model.onnx") == 0 and be2.attempts[0] == ("model_ts0.onnx", 0), be2.attempts
    assert len(be2.attempts) == 2 and be2.paths.count("unet/model_ts0.onnx") == 1
    ok("unet timestep: remembered for the next image (2 steps, one load)")

    # 3) nothing works: the error explains how the graph declares/uses timestep
    md3 = make_models_dir()
    tiny_tokenizer_files(md3)
    open(os.path.join(md3, "unet", "model.onnx"), "wb").write(synthetic_unet_graph())
    be3 = RankBackend(works_at=99)
    try:
        P.generate_image("fox", 1, 384, 3, md3, os.path.join(TMP, "fb3.jpg"), True, be3)
    except RuntimeError as e:
        msg = str(e)
        assert "timestep input" in msg and "Gemm" in msg and "timestep[steps]" in msg and "Unsqueeze([0])" in msg, msg
        assert "rank 0" in msg and "rank 2" in msg, msg
    else:
        raise AssertionError("must fail")
    assert not be3.loaded
    st = json.loads(P.get_progress())
    assert st["done"] and "timestep input" in st["error"]
    ok("unet timestep: if no rank works the error text carries the graph diagnostics")

    # 4) other errors (e.g. out of memory) are not mistaken for a shape problem: no reload attempts
    md4 = make_models_dir()
    tiny_tokenizer_files(md4)
    open(os.path.join(md4, "unet", "model.onnx"), "wb").write(synthetic_unet_graph())
    be4 = RankBackend(error="java.lang.OutOfMemoryError: Failed to allocate 3463659136 bytes")
    try:
        P.generate_image("fox", 1, 384, 3, md4, os.path.join(TMP, "fb4.jpg"), True, be4)
    except RuntimeError as e:
        assert "OutOfMemory" in str(e) and "timestep input" not in str(e)
    else:
        raise AssertionError("must fail")
    assert be4.paths.count("unet/model.onnx") == 1 and len(be4.paths) == 2 and len(be4.attempts) == 1, be4.paths  # te + unet once
    ok("unet timestep: out-of-memory style errors are passed through untouched")

    # 5) a graph that already declares what it uses needs no patching
    md5 = make_models_dir()
    tiny_tokenizer_files(md5)
    open(os.path.join(md5, "unet", "model.onnx"), "wb").write(synthetic_unet_graph(timestep_dims=()))
    be5 = RankBackend(works_at=0)
    P.generate_image("fox", 1, 384, 3, md5, os.path.join(TMP, "fb5.jpg"), True, be5)
    assert be5.attempts == [("model.onnx", 0)] and not os.path.exists(os.path.join(md5, "unet", "model_ts0.onnx"))
    ok("unet timestep: a scalar-declared graph is fed a scalar from the start")


if __name__ == "__main__":
    clip_dir = sys.argv[1] if len(sys.argv) > 1 else None
    real = None
    if clip_dir:
        test_tokenizer(clip_dir)
        real = os.path.join(TMP, "tokenizer")
    else:
        print("skip - tokenizer reference test (pass the openai/CLIP clip directory)")
    test_scheduler()
    test_pipeline(real)
    test_onnx_probe()
    test_unet_timestep_fallback(real)
    test_motion()
    print("SDGEN ALL OK")
