"""DiariZen-style diarization head on the frozen Qwen3-ASR-1.7B encoder.

Backbone: Qwen3-ASR-1.7B audio encoder (frozen, bf16), all 25 streams.
Port of the Qwen3-Omni mix87 diarization sidecar (omni-mix87-sidecar-heads/model_sidecar.py)
with these changes:

  * layer fusion: the Omni sidecar read 8 fixed taps through per-tap MLP adapters and a
    *global* softmax layer weighting (its "frame MHFA" only rescaled that one mixture per
    frame). Here N learned queries attend over all 25 streams at every frame (the Q-Former
    mix of the shipped OmniHead heads) plus a learned per-query layer prior, and the N
    query outputs are kept as separate views (e.g. low layers for speech activity, middle
    layers for voice identity) instead of being averaged into one;
  * self-contained (torch only): Conformer and powerset mapping are implemented here, so
    the head loads in the OmniHead inference env without pyannote / diarizen.

    feats (25, B, T, 1024)  frozen streams, T at 13 frames/s
      -> per-stream LayerNorm (no params)
      -> per-frame Q-Former: N queries x 25 streams, N views of 64-d   (B, T, N*64)
      -> Linear -> d_model                                               (B, T, 256)
      -> Conformer x trunk_layers   at 13 Hz (temporal modelling)
      -> transposed conv x upsample  (~19 ms frames for upsample 4)
      -> Conformer x refine_layers
      -> Linear -> log-softmax over powerset classes (4 speakers / chunk, 2 / frame -> 11)
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def powerset_mapping(max_speakers: int, max_per_frame: int) -> torch.Tensor:
    """(C, K) 0/1 matrix, same class order as pyannote.audio.utils.powerset.Powerset."""
    rows = []
    for k in range(max_per_frame + 1):
        for combo in itertools.combinations(range(max_speakers), k):
            row = [0.0] * max_speakers
            for i in combo:
                row[i] = 1.0
            rows.append(row)
    return torch.tensor(rows)


class LayerQueryMix(nn.Module):
    """Per-frame Q-Former over the layer stack; the N query outputs stay separate views."""

    def __init__(self, num_streams: int, feature_dim: int, num_queries: int = 8, view_dim: int = 64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, feature_dim) * 0.02)
        self.layer_prior = nn.Parameter(torch.zeros(num_queries, num_streams))
        self.proj = nn.Parameter(torch.randn(num_queries, feature_dim, view_dim) * feature_dim ** -0.5)
        self.proj_bias = nn.Parameter(torch.zeros(num_queries, view_dim))
        self.register_buffer("last_layer_attn", torch.full((num_queries, num_streams), 1.0 / num_streams),
                             persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (S, B, T, D) -> (B, T, N * view_dim)"""
        D = feats.shape[-1]
        x = F.layer_norm(feats.float(), (D,)).permute(1, 2, 0, 3)            # (B, T, S, D)
        logits = torch.einsum("btsd,nd->btns", x, self.queries.to(x.dtype)) * D ** -0.5
        w = (logits + self.layer_prior.to(logits.dtype)).softmax(dim=-1)       # (B, T, N, S)
        if not torch.jit.is_scripting():
            self.last_layer_attn.copy_(w.detach().float().mean(dim=(0, 1)))
        mixed = torch.matmul(w.to(x.dtype), x)                                  # (B, T, N, D)
        out = torch.einsum("btnd,ndk->btnk", mixed, self.proj.to(mixed.dtype)) + self.proj_bias.to(mixed.dtype)
        return out.flatten(2)


class _FeedForward(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _ConvModule(nn.Module):
    def __init__(self, d: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = F.silu(self.bn(self.dw(y)))
        return self.drop(self.pw2(y)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Macaron Conformer block (FFN/2, MHSA, conv, FFN/2, LayerNorm), as in DiariZen."""

    def __init__(self, d: int, ffn: int, heads: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ff1 = _FeedForward(d, ffn, dropout)
        self.att_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.att_out = nn.Linear(d, d)
        self.heads = heads
        self.att_drop = dropout
        self.drop = nn.Dropout(dropout)
        self.conv = _ConvModule(d, kernel_size, dropout)
        self.ff2 = _FeedForward(d, ffn, dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, key_padding: Optional[torch.Tensor] = None):
        x = x + 0.5 * self.ff1(x)
        B, T, D = x.shape
        q, k, v = self.qkv(self.att_norm(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        mask = None if key_padding is None else (~key_padding)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.att_drop if self.training else 0.0)
        x = x + self.drop(self.att_out(a.transpose(1, 2).reshape(B, T, D)))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class TemporalUpsampler(nn.Module):
    """Exact integer upsampling in time with a transposed convolution."""

    def __init__(self, d: int, factor: int):
        super().__init__()
        self.factor = factor
        self.conv = nn.ConvTranspose1d(d, d, kernel_size=2 * factor, stride=factor, padding=factor // 2)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        T = x.shape[1]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(y[:, : T * self.factor])


class QwenDiarHead(nn.Module):
    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_queries: int = 8,
        view_dim: int = 64,
        d_model: int = 256,
        ffn_hidden: int = 1024,
        conformer_heads: int = 4,
        trunk_layers: int = 4,
        refine_layers: int = 2,
        kernel_size: int = 31,
        refine_kernel_size: int = 31,
        dropout: float = 0.1,
        upsample: int = 4,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
    ):
        super().__init__()
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_queries=num_queries, view_dim=view_dim,
            d_model=d_model, ffn_hidden=ffn_hidden, conformer_heads=conformer_heads,
            trunk_layers=trunk_layers, refine_layers=refine_layers, kernel_size=kernel_size,
            refine_kernel_size=refine_kernel_size, dropout=dropout, upsample=upsample,
            max_speakers_per_chunk=max_speakers_per_chunk, max_speakers_per_frame=max_speakers_per_frame,
        )
        self.upsample = upsample
        self.register_buffer("mapping", powerset_mapping(max_speakers_per_chunk, max_speakers_per_frame),
                             persistent=False)
        self.num_classes = self.mapping.shape[0]

        self.mix = LayerQueryMix(num_streams, feature_dim, num_queries, view_dim)
        self.inp = nn.Sequential(nn.Linear(num_queries * view_dim, d_model), nn.LayerNorm(d_model),
                                 nn.Dropout(dropout))
        self.trunk = nn.ModuleList(
            ConformerBlock(d_model, ffn_hidden, conformer_heads, kernel_size, dropout) for _ in range(trunk_layers))
        self.upsampler = TemporalUpsampler(d_model, upsample) if upsample > 1 else nn.Identity()
        self.refine = nn.ModuleList(
            ConformerBlock(d_model, ffn_hidden, conformer_heads, refine_kernel_size, dropout)
            for _ in range(refine_layers))
        self.classifier = nn.Linear(d_model, self.num_classes)

    def forward(self, feats: torch.Tensor, return_trunk: bool = False, valid: Optional[torch.Tensor] = None):
        """feats (S, B, T, D) -> log-probs (B, T * upsample, num_classes) [, trunk (B, T, d_model)].

        ``valid`` (B, T) bool optionally masks padded frames out of self-attention (only needed
        for inputs shorter than a chunk; training never pads)."""
        pad = None if valid is None else ~valid
        x = self.inp(self.mix(feats))
        for blk in self.trunk:
            x = blk(x, pad)
        trunk = x
        x = self.upsampler(x)
        pad_up = None if pad is None else pad.repeat_interleave(self.upsample, dim=1)
        for blk in self.refine:
            x = blk(x, pad_up)
        scores = F.log_softmax(self.classifier(x).float(), dim=-1)
        return (scores, trunk) if return_trunk else scores

    def to_multilabel(self, log_probs: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """(B, F, C) powerset log-probs -> (B, F, K) speaker activities (hard by default)."""
        if soft:
            return torch.matmul(log_probs.exp(), self.mapping)
        return self.mapping[log_probs.argmax(-1)]

    def to_powerset_index(self, multilabel: torch.Tensor) -> torch.Tensor:
        """(B, F, K) 0/1 targets -> (B, F) nearest powerset class (frames with >2 speakers
        map to the closest allowed class, as in pyannote)."""
        dist = (multilabel.float()[..., None, :] - self.mapping).abs().sum(-1)
        return dist.argmin(-1)

    def stream_weights(self) -> torch.Tensor:
        return self.mix.last_layer_attn

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_from_checkpoint(ckpt: dict) -> QwenDiarHead:
    """Shipped file (``state_dict``) or a raw training checkpoint (``model``)."""
    head = QwenDiarHead(**ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model"])
    return head
