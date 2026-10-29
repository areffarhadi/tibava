#!/usr/bin/env python3
"""
CTC phoneme heads on the frozen Qwen3-ASR-1.7B encoder (25 streams x T x 1024 at ~13 frames/s).

Every head: layer mix over the 25 streams -> optional sequence model at 13 Hz -> x k upsampling
(13 Hz is below the phone rate of fast speech, CTC needs >= 1 frame per phone) -> optional sequence
model at 13k Hz -> linear to the 127-token ZIPA vocabulary (index 0 = CTC blank).

  mix:      "scalar" (one softmax weight per stream) | "qformer" (per-frame learned queries over streams)
  backbone: "none" | "tf" (Transformer) | "conformer" | "bilstm"
  up:       "split" (Linear d -> k*d, reshape) | "tconv" (ConvTranspose1d, stride k)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3asr_frozen_multihead import QFormerLayerMix

VOCAB = 127


class ScalarMix(nn.Module):
    def __init__(self, num_streams):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x):  # (S, B, T, C) layer-normed
        return torch.einsum("s,sbtc->btc", self.w.softmax(0), x)

    def weights(self):
        return self.w.softmax(0).detach()


class TFStack(nn.Module):
    def __init__(self, d, layers, heads, dropout):
        super().__init__()
        self.pos = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, "gelu", batch_first=True,
                                                               norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        m = valid.unsqueeze(-1).to(x.dtype)
        x = x + self.pos((x * m).transpose(1, 2)).transpose(1, 2)
        for l in self.layers:
            x = l(x, src_key_padding_mask=~valid)
        return self.norm(x)


class ConformerStack(nn.Module):
    def __init__(self, d, layers, heads, dropout, kernel=15):
        super().__init__()
        import torchaudio
        self.m = torchaudio.models.Conformer(input_dim=d, num_heads=heads, ffn_dim=4 * d, num_layers=layers,
                                             depthwise_conv_kernel_size=kernel, dropout=dropout)

    def forward(self, x, valid):
        lens = valid.sum(1)
        L = int(lens.max())
        y, _ = self.m(x[:, :L], lens)
        return F.pad(y, (0, 0, 0, x.size(1) - L))


class BiLSTMStack(nn.Module):
    def __init__(self, d, layers, dropout):
        super().__init__()
        self.m = nn.LSTM(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, valid):
        lens = valid.sum(1).cpu()
        p = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, _ = self.m(p)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.size(1))
        return y


def stack(kind, d, layers, heads, dropout):
    if kind == "none" or layers == 0:
        return None
    if kind == "tf":
        return TFStack(d, layers, heads, dropout)
    if kind == "conformer":
        return ConformerStack(d, layers, heads, dropout)
    if kind == "bilstm":
        return BiLSTMStack(d, layers, dropout)
    raise ValueError(kind)


class PhonemeCTCHead(nn.Module):
    def __init__(self, num_streams=25, feature_dim=1024, vocab=VOCAB, mix="qformer", num_queries=8, d=512,
                 backbone="tf", layers=4, heads=8, up="split", k=3, post="none", post_layers=0, dropout=0.1):
        super().__init__()
        self.config = dict(num_streams=num_streams, feature_dim=feature_dim, vocab=vocab, mix=mix,
                           num_queries=num_queries, d=d, backbone=backbone, layers=layers, heads=heads, up=up, k=k,
                           post=post, post_layers=post_layers, dropout=dropout)
        self.k = k
        self.mix = QFormerLayerMix(num_streams, feature_dim, num_queries) if mix == "qformer" else ScalarMix(num_streams)
        if backbone == "none":
            d = feature_dim
            self.inp = nn.Identity()
        else:
            self.inp = nn.Sequential(nn.Linear(feature_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))
        self.body = stack(backbone, d, layers, heads, dropout)
        if up == "split":
            self.up = nn.Linear(d, k * d)
        else:
            self.up = nn.ConvTranspose1d(d, d, kernel_size=k, stride=k)
        self.up_kind = up
        self.post = stack(post, d, post_layers, heads, dropout)
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[0] = -2.0  # mild blank prior; avoids the flat-logit all-blank start

    def stream_weights(self):
        return self.mix.last_layer_attn if hasattr(self.mix, "last_layer_attn") else self.mix.weights()

    def forward(self, feats, valid):
        """feats (S, B, T, C) frozen encoder streams, valid (B, T) -> logits (B, T*k, V), out_valid (B, T*k)"""
        x = F.layer_norm(feats.float(), feats.shape[-1:])
        x = self.inp(self.mix(x))
        if self.body is not None:
            x = self.body(x, valid)
        B, T, D = x.shape
        if self.up_kind == "split":
            x = self.up(x).view(B, T, self.k, D).reshape(B, T * self.k, D)
        else:
            x = self.up(x.transpose(1, 2)).transpose(1, 2)
        ov = valid.repeat_interleave(self.k, dim=1)
        if self.post is not None:
            x = self.post(x, ov)
        return self.out(self.out_norm(x)), ov


def build_head(cfg: dict) -> PhonemeCTCHead:
    return PhonemeCTCHead(**cfg)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    S, B, T = 25, 2, 130
    feats, valid = torch.randn(S, B, T, 1024), torch.ones(B, T, dtype=torch.bool)
    valid[1, 100:] = False
    for cfg in [dict(mix="scalar", backbone="none", up="split", k=3),
                dict(backbone="tf", layers=4, up="split", k=3, post="tf", post_layers=1),
                dict(backbone="conformer", layers=4, up="tconv", k=3),
                dict(backbone="bilstm", layers=2, up="split", k=3),
                dict(backbone="tf", layers=4, up="split", k=2)]:
        h = build_head(cfg)
        lg, ov = h(feats, valid)
        print(cfg, tuple(lg.shape), int(ov[1].sum()), f"{n_params(h) / 1e6:.2f}M")
