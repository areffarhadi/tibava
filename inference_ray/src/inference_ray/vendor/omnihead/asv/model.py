"""OmniASV: speaker-embedding head on the frozen Qwen3-ASR-1.7B encoder (MHFA + Q-Former).

Backbone: Qwen3-ASR-1.7B audio encoder (frozen, bf16), all 25 streams.
Port of the Omni mix87 MHFA speaker head (omni-mix87-sidecar-heads/model_speaker.py: 8 taps, one global
softmax layer weighting for the Key view and one for the Value view) with these changes:

  * Key and Value views come from per-frame Q-Formers over all 25 streams: N learned queries attend over
    the layers of every frame (+ a learned layer prior per query) and each query keeps its own projected
    view, so "where to attend in time" (Key) and "what to pool" (Value) each draw on several layer mixes;
  * attentive statistics: every head pools the weighted mean AND the weighted standard deviation of
    the Value view (MHFA pools the mean only);
  * frame masks: ``forward(feats, valid)`` pools only the valid frames, so the same head embeds a
    padded utterance or one local speaker's frames of a diarization chunk (shared encoder pass);
  * trained with sub-center ArcFace (K centres per speaker), robust to the clustered (noisy) speaker
    labels of VoxLingua.

    feats (25, B, T, 1024) -> LayerNorm per stream
      Key   Q-Former: 8 queries x 16-d views -> (B, T, 128) -> Linear -> H attention heads over time
      Value Q-Former: 8 queries x 32-d views -> (B, T, 256)
      per head: masked softmax over T, weighted mean + std of Value -> (B, H * 512)
      -> Linear 256 + LayerNorm -> residual adapter -> MLP -> 256-d embedding
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryViews(nn.Module):
    """N queries attend over the S layer vectors of each frame; each query output is projected to its own view."""

    def __init__(self, num_streams: int, feature_dim: int, num_queries: int, view_dim: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, feature_dim) * 0.02)
        self.layer_prior = nn.Parameter(torch.zeros(num_queries, num_streams))
        self.proj = nn.Parameter(torch.randn(num_queries, feature_dim, view_dim) * feature_dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(num_queries, view_dim))
        self.register_buffer("last_layer_attn", torch.full((num_queries, num_streams), 1.0 / num_streams),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, S, D) layer-normed -> (B, T, N * view_dim)"""
        logits = torch.einsum("btsd,nd->btns", x, self.queries.to(x.dtype)) * x.shape[-1] ** -0.5
        w = (logits + self.layer_prior.to(logits.dtype)).softmax(-1)
        self.last_layer_attn.copy_(w.detach().float().mean(dim=(0, 1)))
        mixed = torch.matmul(w.to(x.dtype), x)                                   # (B, T, N, D)
        out = torch.einsum("btnd,ndk->btnk", mixed, self.proj.to(mixed.dtype)) + self.bias.to(mixed.dtype)
        return out.flatten(2)


class OmniASVHead(nn.Module):
    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        key_queries: int = 8,
        key_view: int = 16,
        value_queries: int = 8,
        value_view: int = 32,
        num_heads: int = 16,
        stats: bool = True,
        embedding_dim: int = 256,
        adapter_bottleneck: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(num_streams=num_streams, feature_dim=feature_dim, key_queries=key_queries,
                           key_view=key_view, value_queries=value_queries, value_view=value_view,
                           num_heads=num_heads, stats=stats, embedding_dim=embedding_dim,
                           adapter_bottleneck=adapter_bottleneck, dropout=dropout)
        self.embedding_dim, self.num_heads, self.stats = embedding_dim, num_heads, stats
        self.kviews = QueryViews(num_streams, feature_dim, key_queries, key_view)
        self.vviews = QueryViews(num_streams, feature_dim, value_queries, value_view)
        kd, vd = key_queries * key_view, value_queries * value_view
        self.attention = nn.Linear(kd, num_heads)
        self.drop = nn.Dropout(dropout)
        pooled = num_heads * vd * (2 if stats else 1)
        self.embedding_projection = nn.Sequential(nn.Linear(pooled, embedding_dim), nn.LayerNorm(embedding_dim),
                                                  nn.Dropout(dropout))
        self.adapter = nn.Sequential(nn.Linear(embedding_dim, adapter_bottleneck), nn.BatchNorm1d(adapter_bottleneck),
                                     nn.ReLU(inplace=True), nn.Dropout(dropout),
                                     nn.Linear(adapter_bottleneck, embedding_dim))
        self.adapter_norm = nn.LayerNorm(embedding_dim)
        self.embedding_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.BatchNorm1d(embedding_dim),
                                             nn.PReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim))

    def forward(self, feats: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """feats (S, B, T, D) frozen streams, valid (B, T) bool frames to pool -> (B, embedding_dim)."""
        x = F.layer_norm(feats.float(), (feats.shape[-1],)).permute(1, 2, 0, 3)  # (B, T, S, D)
        K = self.drop(self.kviews(x))
        V = self.drop(self.vviews(x)).float()
        logits = self.attention(K).float()                                          # (B, T, H)
        if valid is not None:
            logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
        a = logits.softmax(dim=1)
        mean = torch.einsum("bth,btc->bhc", a, V)
        if self.stats:
            var = torch.einsum("bth,btc->bhc", a, V * V) - mean * mean
            pooled = torch.cat([mean, var.clamp(min=1e-5).sqrt()], dim=-1)
        else:
            pooled = mean
        e = self.embedding_projection(pooled.flatten(1))
        e = self.adapter_norm(e + self.adapter(e))
        return self.embedding_layer(e)

    def stream_weights(self):
        return {"key": self.kviews.last_layer_attn, "value": self.vviews.last_layer_attn}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SubCenterArcFace(nn.Module):
    """Additive angular margin softmax with K sub-centres per class (training only)."""

    def __init__(self, embedding_dim: int, num_classes: int, k: int = 3, scale: float = 32.0, margin: float = 0.2):
        super().__init__()
        self.k, self.num_classes, self.scale = k, num_classes, scale
        self.weight = nn.Parameter(torch.empty(num_classes * k, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.set_margin(margin)

    def set_margin(self, margin: float) -> None:
        self.margin = margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th, self.mm = math.cos(math.pi - margin), math.sin(math.pi - margin) * margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        cos = F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1))
        cos = cos.view(-1, self.num_classes, self.k).max(dim=2).values.clamp(-1 + 1e-7, 1 - 1e-7)
        sin = (1.0 - cos * cos).clamp(0, 1).sqrt()
        phi = torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m, cos - self.mm)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.scale * (onehot * phi + (1 - onehot) * cos)
        return F.cross_entropy(logits, labels), cos


def build_from_checkpoint(ckpt: dict) -> OmniASVHead:
    """Shipped file (``state_dict``) or a raw training checkpoint (``head``)."""
    head = OmniASVHead(**ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt["head"])
    return head
