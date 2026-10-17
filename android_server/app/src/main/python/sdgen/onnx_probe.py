"""Minimal ONNX protobuf reader / rewriter (no `onnx` package) for the small UNet graph file.

The UNet's weights live in model.onnx_data; model.onnx itself is only ~1 MB of graph. This module can

  * report how the graph declares and uses its `timestep` input (for error messages), and
  * write a copy of the graph in which one input is declared with a different rank. Some exports declare
    `timestep` as a vector but build a graph that treats it as a scalar (or the other way round), which
    makes ONNX Runtime fail inside /time_embedding/linear_1/Gemm.
"""
import os
import struct

MAX_GRAPH_BYTES = 64 * 1024 * 1024   # never parse a file with inline weights (the text encoder is 1.3 GB)


# --------------------------------------------------------------------------- protobuf primitives
def _varint(b, i):
    shift = v = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        if c < 0x80:
            return v, i
        shift += 7


def _enc_varint(n):
    out = bytearray()
    while True:
        c = n & 0x7F
        n >>= 7
        if n:
            out.append(c | 0x80)
        else:
            out.append(c)
            return bytes(out)


def _enc_field(field, payload):
    return _enc_varint(field << 3 | 2) + _enc_varint(len(payload)) + payload


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _walk(b, i, end):
    """Yield (tag_start, field, wire, a, c, field_end). wire 0: a is the value; otherwise payload is b[a:c]."""
    while i < end:
        t0 = i
        tag, i = _varint(b, i)
        f, w = tag >> 3, tag & 7
        if w == 0:
            v, i = _varint(b, i)
            yield t0, f, w, v, None, i
        elif w == 2:
            n, i = _varint(b, i)
            yield t0, f, w, i, i + n, i + n
            i += n
        elif w == 1:
            yield t0, f, w, i, i + 8, i + 8
            i += 8
        elif w == 5:
            yield t0, f, w, i, i + 4, i + 4
            i += 4
        else:
            raise ValueError("unsupported protobuf wire type %d" % w)


def _s(b, a, c):
    return bytes(b[a:c]).decode("utf-8", "replace")


def _packed_varints(b, a, c):
    out = []
    while a < c:
        v, a = _varint(b, a)
        out.append(_signed(v))
    return out


# --------------------------------------------------------------------------- readers
def _parse_value_info(b, p0, p1):
    name, elem, dims, has_shape = "", None, [], False
    for _, f, w, a, c, _e in _walk(b, p0, p1):
        if f == 1 and w == 2:
            name = _s(b, a, c)
        elif f == 2 and w == 2:                                  # TypeProto
            for _, f2, w2, a2, c2, _e2 in _walk(b, a, c):
                if f2 == 1 and w2 == 2:                          # tensor_type
                    for _, f3, w3, a3, c3, _e3 in _walk(b, a2, c2):
                        if f3 == 1 and w3 == 0:
                            elem = a3
                        elif f3 == 2 and w3 == 2:                # shape
                            has_shape = True
                            for _, f4, w4, a4, c4, _e4 in _walk(b, a3, c3):
                                if f4 == 1 and w4 == 2:          # one Dimension
                                    d = None
                                    for _, f5, w5, a5, c5, _e5 in _walk(b, a4, c4):
                                        if f5 == 1 and w5 == 0:
                                            d = a5
                                        elif f5 == 2 and w5 == 2:
                                            d = _s(b, a5, c5)
                                    dims.append(d)
    return {"name": name, "elem": elem, "dims": dims if has_shape else None}


def _parse_small_tensor(b, p0, p1):
    """TensorProto -> {"name", "dims", "vals"} (vals only for small int tensors, else None)."""
    name, dims, dt, raw, i64, i32 = "", [], None, None, [], []
    for _, f, w, a, c, _e in _walk(b, p0, p1):
        if f == 1:
            dims.extend([a] if w == 0 else _packed_varints(b, a, c))
        elif f == 2 and w == 0:
            dt = a
        elif f == 8 and w == 2:
            name = _s(b, a, c)
        elif f == 9 and w == 2 and c - a <= 64:
            raw = bytes(b[a:c])
        elif f == 7 and w == 2:
            i64 = _packed_varints(b, a, c)
        elif f == 5 and w == 2:
            i32 = _packed_varints(b, a, c)
    vals = None
    if raw is not None and dt == 7 and len(raw) % 8 == 0:
        vals = list(struct.unpack("<%dq" % (len(raw) // 8), raw))
    elif raw is not None and dt == 6 and len(raw) % 4 == 0:
        vals = list(struct.unpack("<%di" % (len(raw) // 4), raw))
    elif i64:
        vals = i64
    elif i32:
        vals = i32
    return {"name": name, "dims": dims, "vals": vals}


def _parse_node(b, p0, p1):
    node = {"op": "", "name": "", "in": [], "out": [], "attrs": {}}
    for _, f, w, a, c, _e in _walk(b, p0, p1):
        if f == 1 and w == 2:
            node["in"].append(_s(b, a, c))
        elif f == 2 and w == 2:
            node["out"].append(_s(b, a, c))
        elif f == 3 and w == 2:
            node["name"] = _s(b, a, c)
        elif f == 4 and w == 2:
            node["op"] = _s(b, a, c)
        elif f == 5 and w == 2:                                  # AttributeProto
            aname, ints, scalar, tensor = "", [], None, None
            for _, f2, w2, a2, c2, _e2 in _walk(b, a, c):
                if f2 == 1 and w2 == 2:
                    aname = _s(b, a2, c2)
                elif f2 == 3 and w2 == 0:
                    scalar = _signed(a2)
                elif f2 == 8:
                    ints.extend([_signed(a2)] if w2 == 0 else _packed_varints(b, a2, c2))
                elif f2 == 5 and w2 == 2:
                    tensor = _parse_small_tensor(b, a2, c2)
            if aname:
                node["attrs"][aname] = ints if ints else (scalar if scalar is not None else tensor)
    return node


def read(path):
    """-> {"inputs": [{"name","elem","dims"}], "nodes": [...], "inits": {name: vals}}"""
    if os.path.getsize(path) > MAX_GRAPH_BYTES:
        raise ValueError("graph file too large to inspect")
    with open(path, "rb") as fh:
        b = fh.read()
    graph = None
    for _, f, w, a, c, _e in _walk(b, 0, len(b)):
        if f == 7 and w == 2:
            graph = (a, c)
    if graph is None:
        raise ValueError("no graph in model file")
    nodes, inputs, inits = [], [], {}
    for _, f, w, a, c, _e in _walk(b, graph[0], graph[1]):
        if f == 1 and w == 2:
            nodes.append(_parse_node(b, a, c))
        elif f == 11 and w == 2:
            inputs.append(_parse_value_info(b, a, c))
        elif f == 5 and w == 2:
            t = _parse_small_tensor(b, a, c)
            if t["vals"] is not None:
                inits[t["name"]] = t["vals"]
    return {"inputs": inputs, "nodes": nodes, "inits": inits}


def declared_rank(path, input_name):
    """Number of dimensions the model declares for an input, or None when unknown."""
    for i in read(path)["inputs"]:
        if i["name"] == input_name:
            return None if i["dims"] is None else len(i["dims"])
    return None


def describe_path(path, input_name="timestep", target="time_embedding/linear_1/Gemm", limit=26):
    """One compact line: declared inputs, then the ops between `input_name` and the target node."""
    g = read(path)
    decl = ", ".join("%s%s" % (i["name"], "?" if i["dims"] is None else "[" + ",".join(str(d) for d in i["dims"]) + "]")
                     for i in g["inputs"])
    consts = dict(g["inits"])
    reached, chain = {input_name}, []
    for n in g["nodes"]:
        if n["op"] == "Constant":
            v = n["attrs"].get("value")
            if isinstance(v, dict) and v.get("vals") is not None:
                for o in n["out"]:
                    consts[o] = v["vals"]
        if not any(x in reached for x in n["in"]):
            continue
        reached.update(n["out"])
        extra = [str(consts[x]) for x in n["in"] if x not in reached and x in consts and len(consts[x]) <= 4]
        extra += ["%s=%s" % (k, v) for k, v in n["attrs"].items() if isinstance(v, (int, list)) and k in ("axes", "axis")]
        chain.append(n["op"] + ("(" + ";".join(extra) + ")" if extra else ""))
        if target in n["name"] or len(chain) >= limit:
            break
    return "inputs " + decl + " | " + input_name + " flows through " + ">".join(chain)


# --------------------------------------------------------------------------- rewriter
def _shape_bytes(rank):
    return b"".join(_enc_field(1, _enc_field(2, b"d%d" % i)) for i in range(rank))


def _rewrite_value_info(b, p0, p1, rank):
    out = bytearray()
    for t0, f, w, a, c, fe in _walk(b, p0, p1):
        if f == 2 and w == 2:                                    # TypeProto
            tp = bytearray()
            for t1, f2, w2, a2, c2, fe2 in _walk(b, a, c):
                if f2 == 1 and w2 == 2:                          # tensor_type
                    tt = bytearray()
                    for t2, f3, w3, a3, c3, fe3 in _walk(b, a2, c2):
                        if f3 == 2 and w3 == 2:
                            continue                             # drop the old shape
                        tt += b[t2:fe3]
                    tt += _enc_field(2, _shape_bytes(rank))
                    tp += _enc_field(1, bytes(tt))
                else:
                    tp += b[t1:fe2]
            out += _enc_field(2, bytes(tp))
        else:
            out += b[t0:fe]
    return bytes(out)


def patch_input_rank(src, dst, input_name, rank):
    """Write src to dst (same folder, so external weights still resolve) with `input_name` declared with `rank` dims."""
    if os.path.getsize(src) > MAX_GRAPH_BYTES:
        raise ValueError("graph file too large to rewrite")
    with open(src, "rb") as fh:
        b = fh.read()
    out, done = bytearray(), False
    for t0, f, w, a, c, fe in _walk(b, 0, len(b)):
        if f == 7 and w == 2:
            g = bytearray()
            for g0, gf, gw, ga, gc, gfe in _walk(b, a, c):
                if gf == 11 and gw == 2 and _parse_value_info(b, ga, gc)["name"] == input_name:
                    g += _enc_field(11, _rewrite_value_info(b, ga, gc, rank))
                    done = True
                else:
                    g += b[g0:gfe]
            out += _enc_field(7, bytes(g))
        else:
            out += b[t0:fe]
    if not done:
        raise ValueError("input %r not found in the model" % input_name)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(out)
    os.replace(tmp, dst)
