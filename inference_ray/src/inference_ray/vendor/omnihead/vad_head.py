#!/usr/bin/env python3
"""
Internal VAD: a handful of attention-head signals of the frozen Qwen3-ASR encoder + logistic regression.

No extra encoder pass and no trained network. The shared pass already returns every layer's input
(feats[l-1] = input of layer l), so for each selected (layer, head) we recompute only that head's
attention inside the encoder's 8 s windows and read one per-frame signal:
  onorm    ||o_j||                      head output norm (before out_proj)
  dist     sum_k A_jk |j-k|             mean attention distance, tokens
  entropy  -sum_k A_jk log A_jk         query entropy
With "norm": true (shipped), entropy is divided by log K and dist by the mean |j-k| of uniform attention,
K = valid keys in the frame's window. Raw entropy/dist shrink in short clips (fewer keys), which moved the
best threshold from -0.02 (15-min audio) to +8.0 (1-3 s clips); normalised, one threshold serves both.
The logit per 13 fps frame is  b + sum_i w_i log(max(s_i, 0) + 1e-6). Weights, bias and threshold live in
the head json (heads/vad/*.json), fitted in OmniHead/vad_heads (VAD_HEADS_REPORT.md §11c).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

FPS = 13.0


class AttnHeadVAD:
    def __init__(self, path: Path, device: torch.device):
        cfg = json.loads(Path(path).read_text())
        self.cfg = cfg
        self.picks = [(int(p["layer"]), int(p["head"]), p["signal"]) for p in cfg["picks"]]
        self.w = torch.tensor(cfg["weights"], dtype=torch.float32, device=device)
        self.b = float(cfg["bias"])
        self.threshold = float(cfg["threshold_logit"])
        self.min_speech = float(cfg.get("min_speech_sec", 0.0))
        self.min_gap = float(cfg.get("min_gap_sec", 0.0))
        self.norm = bool(cfg.get("norm", False))

    @torch.no_grad()
    def signals(self, encoder, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats (25, B, T, D), valid (B, T) -> (B, T, n_picks) float32 (invalid frames are garbage)."""
        B, T = valid.shape
        allowed = encoder._attn_mask(valid)[:, 0]                       # (B, T, T), same mask as the encoder
        idx = torch.arange(T, device=valid.device, dtype=torch.float32)
        absdist = (idx[:, None] - idx[None, :]).abs()                  # only in-window pairs get mass
        dh = encoder.head_dim
        cache, out = {}, []
        for l, h, sig in self.picks:
            key = (l, h)
            if key not in cache:
                at = encoder.layers[l - 1].self_attn
                x = encoder.layers[l - 1].self_attn_layer_norm(feats[l - 1]).float()
                sl = slice(h * dh, (h + 1) * dh)
                q = F.linear(x, at.q_proj.weight[sl].float(), at.q_proj.bias[sl].float())
                k = F.linear(x, at.k_proj.weight[sl].float(), at.k_proj.bias[sl].float())
                s = (q @ k.transpose(-1, -2)) * dh ** -0.5
                A = s.masked_fill(~allowed, float("-inf")).softmax(-1)  # (B, T, T)
                cache[key] = (A, x, sl, at)
            A, x, sl, at = cache[key]
            if sig == "onorm":
                v = F.linear(x, at.v_proj.weight[sl].float(), at.v_proj.bias[sl].float())
                out.append((A @ v).norm(dim=-1))
            elif sig == "dist":
                out.append((A * absdist).sum(-1))
            elif sig == "entropy":
                out.append(-(A * torch.log(A.clamp_min(1e-12))).sum(-1))
            else:
                raise ValueError(sig)
            if self.norm and sig in ("dist", "entropy"):
                K, U = self.window_geometry(encoder.window_tokens, valid)
                out[-1] = out[-1] / (torch.log(K.clamp_min(2)) if sig == "entropy" else U.clamp_min(1e-6))
        return torch.stack(out, -1)

    @staticmethod
    def window_geometry(W: int, valid: torch.Tensor):
        """per frame (B, T): K = valid keys in its window, U = mean |j-k| over them (uniform attention)"""
        T = valid.size(1)
        n = valid.sum(1, keepdim=True).float()                         # (B, 1)
        j = torch.arange(T, device=valid.device, dtype=torch.float32)[None]
        start = torch.div(j, W, rounding_mode="floor") * W
        K = (n - start).clamp(1, W)
        a = j - start
        b = (K - 1 - a).clamp_min(0)
        return K, (a * (a + 1) / 2 + b * (b + 1) / 2) / K

    @torch.no_grad()
    def logits(self, encoder, feats, valid) -> List[np.ndarray]:
        """Per clip: (T_valid,) frame logits at 13 fps."""
        s = self.signals(encoder, feats, valid)
        lg = (torch.log(s.clamp_min(0) + 1e-6) @ self.w + self.b).cpu().numpy()
        n = valid.sum(1).cpu().numpy()
        return [lg[i, :n[i]] for i in range(len(n))]

    def segments(self, logit: np.ndarray) -> List[List[float]]:
        """Thresholded frames -> [[start_sec, end_sec], ...]; merge gaps < min_gap_sec, then drop segments
        < min_speech_sec (shipped json: 0.1 / 0.1; 0 / 0 = raw frame decisions)."""
        on = logit > self.threshold
        segs, start = [], None
        for i, v in enumerate(np.append(on, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                segs.append([start / FPS, i / FPS])
                start = None
        if self.min_gap > 0:
            merged = []
            for s in segs:
                if merged and s[0] - merged[-1][1] < self.min_gap:
                    merged[-1][1] = s[1]
                else:
                    merged.append(s)
            segs = merged
        if self.min_speech > 0:
            segs = [s for s in segs if s[1] - s[0] >= self.min_speech]
        return [[round(a, 3), round(b, 3)] for a, b in segs]

    def decode(self, logit: np.ndarray) -> dict:
        segs = self.segments(logit)
        sp = sum(b - a for a, b in segs)
        dur = len(logit) / FPS
        return {"speech": bool(segs), "speech_sec": round(sp, 2), "speech_ratio": round(sp / max(dur, 1e-9), 3),
                "segments": segs}
