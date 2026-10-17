"""CLIP byte-pair-encoding tokenizer for Stable Diffusion 2.x (HF vocab.json + merges.txt).

Pure Python, no third-party packages. The pre-tokenizer follows the CLIP pattern
    <|startoftext|> | <|endoftext|> | 's 't 're 've 'm 'll 'd | letters+ | one digit | punctuation+
using unicodedata, so it needs neither the `regex` module nor ftfy.
Difference from the original CLIP code: ftfy's mojibake repair is not applied (only HTML unescape).
"""
import html
import json
import os
import re
import unicodedata

import numpy as np

_SPECIALS = ("<|startoftext|>", "<|endoftext|>")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def bytes_to_unicode():
    """Reversible map from utf-8 bytes to printable unicode characters (GPT-2/CLIP convention)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def pretokenize(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        matched = None
        if ch == "<":
            for s in _SPECIALS:
                if text.startswith(s, i):
                    matched = s
                    break
        elif ch == "'":
            for c in _CONTRACTIONS:
                if text.startswith(c, i):
                    matched = c
                    break
        if matched is None:
            cat = unicodedata.category(ch)[0]
            if cat == "L":
                j = i + 1
                while j < n and unicodedata.category(text[j])[0] == "L":
                    j += 1
                matched = text[i:j]
            elif cat == "N":
                matched = ch
            elif not ch.isspace():
                j = i + 1
                while j < n and not text[j].isspace() and unicodedata.category(text[j])[0] not in ("L", "N"):
                    j += 1
                matched = text[i:j]
        if matched is None:  # whitespace between tokens
            i += 1
            continue
        out.append(matched)
        i += len(matched)
    return out


def _clean(text):
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip().lower()


class ClipTokenizer:
    def __init__(self, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8") as fh:
            self.vocab = json.load(fh)
        ranks = {}
        with open(os.path.join(tokenizer_dir, "merges.txt"), encoding="utf-8") as fh:
            for line in fh.read().split("\n"):
                parts = line.split()
                if len(parts) != 2 or line.startswith("#version"):
                    continue
                ranks[(parts[0], parts[1])] = len(ranks)
        self.ranks = ranks
        self.bos = self.vocab["<|startoftext|>"]
        self.eos = self.vocab["<|endoftext|>"]
        self.pad = self._read_pad(tokenizer_dir)
        self.byte_encoder = bytes_to_unicode()
        self.cache = {s: [s] for s in _SPECIALS}

    def _read_pad(self, tokenizer_dir):
        """SD 2.x pads with '!' (id 0); SD 1.x pads with <|endoftext|>."""
        try:
            with open(os.path.join(tokenizer_dir, "special_tokens_map.json"), encoding="utf-8") as fh:
                pad = json.load(fh).get("pad_token")
            if isinstance(pad, dict):
                pad = pad.get("content")
            if isinstance(pad, str) and pad in self.vocab:
                return self.vocab[pad]
        except (OSError, ValueError):
            pass
        return self.eos

    def _bpe(self, token):
        hit = self.cache.get(token)
        if hit is not None:
            return hit
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            best, best_rank = None, None
            for pair in zip(word, word[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                    merged.append(best[0] + best[1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self.cache[token] = word
        return word

    def tokens(self, text):
        """BPE ids for the text, without BOS/EOS/padding."""
        ids = []
        for piece in pretokenize(_clean(text)):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.vocab[sym] for sym in self._bpe(mapped))
        return ids

    def encode(self, text, max_length=77):
        """(1, max_length) int64 array: BOS + tokens (truncated) + EOS, padded with the pad id."""
        ids = [self.bos] + self.tokens(text)[: max_length - 2] + [self.eos]
        ids += [self.pad] * (max_length - len(ids))
        return np.array([ids], dtype=np.int64)
