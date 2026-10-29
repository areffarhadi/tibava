#!/usr/bin/env python3
"""
Frozen Qwen3-ASR-1.7B audio tower shared by any number of independently trained heads.

The encoder is never modified (no WPT, no LoRA), so every head reads exactly the same
features and new heads can be trained later without touching existing ones. One encoder
pass serves all heads at inference.

Features per clip: 25 streams of (T, 1024)
  index 0      : input embedding (conv front end + sinusoidal positions, input to layer 1)
  index 1..24  : output of encoder layers 1..24

The forward pass follows the native Qwen3ASRAudioEncoder (FA2 path):
  * log-mel is split into chunks of 2*n_window frames (1 s); each chunk goes through the
    conv front end on its own and positions restart at 0 for every chunk (13 tokens/s);
  * self-attention is block-diagonal over windows of 13 * n_window_infer / (2*n_window)
    tokens (8 s), and never crosses clips in a batch.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional source checkout of the qwen_asr package (pip "qwen-asr==0.0.6" otherwise). Read at encoder build time.
QWEN3_ASR_SRC = os.environ.get("QWEN3_ASR_SRC", "")
SAMPLE_RATE = 16000


# =========================================================================
# Whisper log-mel on GPU (matches WhisperFeatureExtractor, padding="longest")
# =========================================================================
class WhisperLogMel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.n_fft = fe.n_fft
        self.hop = fe.hop_length
        self.register_buffer(
            "mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False
        )
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, wave_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """wave (B, N) float32, wave_lens (B,) -> mel (B, 128, T), mel_lens (B,)"""
        stft = torch.stft(
            wave.float(), self.n_fft, self.hop, window=self.window, center=True,
            pad_mode="reflect", return_complex=True,
        )
        power = stft[..., :-1].abs() ** 2
        mel = torch.einsum("fm,bft->bmt", self.mel_filters, power)
        log_spec = torch.clamp(mel, min=1e-10).log10()
        mel_lens = torch.div(wave_lens, self.hop, rounding_mode="floor").clamp(min=1)
        valid = torch.arange(log_spec.size(-1), device=wave.device)[None, :] < mel_lens[:, None]
        peak = log_spec.masked_fill(~valid[:, None, :], float("-inf")).amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, peak - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec * valid[:, None, :]
        return log_spec, mel_lens


def qwen3asr_output_lengths(mel_lens: torch.Tensor) -> torch.Tensor:
    leave = mel_lens % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lens // 100) * 13


# =========================================================================
# Frozen encoder returning every layer
# =========================================================================
class FrozenQwen3ASRAllLayers(nn.Module):
    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if QWEN3_ASR_SRC and QWEN3_ASR_SRC not in sys.path:
            sys.path.insert(0, QWEN3_ASR_SRC)
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
        )

        cfg = Qwen3ASRConfig.from_pretrained(model_id)
        full = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, config=cfg, dtype=torch.float32)
        enc = full.thinker.audio_tower
        del full

        self.model_id = model_id
        self.dtype = dtype
        self.conv2d1, self.conv2d2, self.conv2d3 = enc.conv2d1, enc.conv2d2, enc.conv2d3
        self.conv_out = enc.conv_out
        self.layers = enc.layers
        self.register_buffer(
            "pos_emb", enc.positional_embedding.positional_embedding.clone(), persistent=False
        )
        self.chunk = enc.n_window * 2
        self.tokens_per_chunk = int(qwen3asr_output_lengths(torch.tensor([self.chunk])).item())
        self.window_tokens = self.tokens_per_chunk * (enc.n_window_infer // self.chunk)
        self.d_model = enc.layers[0].embed_dim
        self.num_heads = enc.layers[0].self_attn.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_streams = len(self.layers) + 1
        del enc

        self.melspec = WhisperLogMel(model_id)
        for p in self.parameters():
            p.requires_grad = False
        self.to(dtype)
        self.melspec.float()
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def spec(self) -> dict:
        spec = {
            "model_id": self.model_id,
            "streams": "input_embedding+layers1-24",
            "num_streams": self.num_streams,
            "d_model": self.d_model,
            "chunk_mel_frames": self.chunk,
            "window_tokens": self.window_tokens,
        }
        if getattr(self, "capture_heads", None):
            spec["capture_heads"] = [[int(layer), int(head)] for layer, head in self.capture_heads]
        return spec

    def _front_end(self, mel: torch.Tensor, mel_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, T = mel.shape
        n_chunks = math.ceil(T / self.chunk)
        mel = F.pad(mel, (0, n_chunks * self.chunk - T))
        x = mel.view(B, M, n_chunks, self.chunk).permute(0, 2, 1, 3).reshape(B * n_chunks, 1, M, self.chunk)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        bc, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(bc, t, c * f))
        x = x + self.pos_emb[:t].unsqueeze(0).to(x.dtype)
        x = x.view(B, n_chunks * t, self.d_model)
        tok_lens = qwen3asr_output_lengths(mel_lens)
        valid = torch.arange(x.size(1), device=x.device)[None, :] < tok_lens[:, None]
        return x, valid

    def _attn_mask(self, valid: torch.Tensor) -> torch.Tensor:
        T = valid.size(1)
        idx = torch.arange(T, device=valid.device)
        same_window = (idx[:, None] // self.window_tokens) == (idx[None, :] // self.window_tokens)
        allowed = same_window[None] & valid[:, None, :]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=valid.device)[None]
        return allowed[:, None]

    def _layer(self, layer: nn.Module, h: torch.Tensor, mask: torch.Tensor,
               keep_heads: Sequence[int] = ()) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = h.shape
        attn = layer.self_attn
        x = layer.self_attn_layer_norm(h)
        q = attn.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        picked = None
        if keep_heads:
            idx = torch.tensor(list(keep_heads), device=o.device, dtype=torch.long)
            picked = o.index_select(1, idx).permute(1, 0, 2, 3).contiguous()
        h = h + attn.out_proj(o.transpose(1, 2).reshape(B, T, D))
        x = layer.final_layer_norm(h)
        return h + layer.fc2(layer.activation_fn(layer.fc1(x))), picked

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, wave_lens: Optional[torch.Tensor] = None):
        """
        wave (B, N) at 16 kHz. Returns
          feats (S=25, B, T, D) in self.dtype
          valid (B, T) bool token mask
        When capture_heads is set, also returns head_feats (H, B, T, head_dim)
        for those (layer, head) pairs, read before out_proj. The encoder stays frozen.
        """
        if wave_lens is None:
            wave_lens = torch.full((wave.size(0),), wave.size(1), dtype=torch.long, device=wave.device)
        mel, mel_lens = self.melspec(wave, wave_lens)
        h, valid = self._front_end(mel.to(self.dtype), mel_lens)
        mask = self._attn_mask(valid)
        capture = [(int(layer), int(head)) for layer, head in (getattr(self, "capture_heads", None) or [])]
        by_layer: Dict[int, List[int]] = {}
        for layer_no, head_idx in capture:
            by_layer.setdefault(layer_no, []).append(head_idx)
        streams = [h]
        picked: Dict[Tuple[int, int], torch.Tensor] = {}
        for li, layer in enumerate(self.layers):
            layer_no = li + 1
            keep = by_layer.get(layer_no, [])
            h, got = self._layer(layer, h, mask, keep)
            streams.append(h)
            if got is not None:
                for j, head_idx in enumerate(keep):
                    picked[(layer_no, head_idx)] = got[j]
        feats = torch.stack(streams, dim=0)
        if not capture:
            return feats, valid
        missing = [pair for pair in capture if pair not in picked]
        if missing:
            raise RuntimeError(f"attention heads not found in the encoder: {missing}")
        head_feats = torch.stack([picked[pair] for pair in capture], dim=0)
        return feats, head_feats, valid


# =========================================================================
# Frozen w2v-BERT 2.0 returning every layer (same 25 x 1024 interface)
# =========================================================================
class FrozenW2VBertAllLayers(nn.Module):
    def __init__(self, model_dir: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import Wav2Vec2BertModel, AutoFeatureExtractor

        self.model_dir = model_dir
        self.dtype = dtype
        # fp32: the model's internal layer norms keep fp32 params, so a bf16 cast breaks them
        self.encoder = Wav2Vec2BertModel.from_pretrained(model_dir, dtype=torch.float32)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.d_model = self.encoder.config.hidden_size
        self.num_streams = self.encoder.config.num_hidden_layers + 1
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.autocast = dtype == torch.bfloat16
        print(f"  w2v-BERT weights fp32, compute {'bf16 autocast' if self.autocast else 'fp32'}")

    def train(self, mode: bool = True):
        return super().train(False)

    def spec(self) -> dict:
        return {"model_id": self.model_dir, "streams": "input_embedding+layers1-24",
                "num_streams": self.num_streams, "d_model": self.d_model, "backbone": "w2v-bert-2.0",
                "compute": "bf16_autocast" if self.autocast else "fp32"}

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, wave_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if wave_lens is None:
            wave_lens = torch.full((wave.size(0),), wave.size(1), dtype=torch.long, device=wave.device)
        feats = self.feature_extractor(
            [w[:n].float().cpu().numpy() for w, n in zip(wave, wave_lens)],
            sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        feats = {k: v.to(wave.device) for k, v in feats.items()}
        # Weights stay fp32; autocast runs the matmuls in bf16. Layer norms are unchanged.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            out = self.encoder(**feats, output_hidden_states=True)
        streams = torch.stack(out.hidden_states, dim=0).float()
        valid = torch.ones(streams.shape[1:3], dtype=torch.bool, device=wave.device)
        if "attention_mask" in feats:
            valid = feats["attention_mask"].to(torch.bool)
        return streams, valid


def build_frozen_encoder(backbone: str, model_id: str, dtype: torch.dtype) -> nn.Module:
    if backbone == "qwen3asr":
        return FrozenQwen3ASRAllLayers(model_id, dtype=dtype)
    if backbone == "w2vbert":
        return FrozenW2VBertAllLayers(model_id, dtype=dtype)
    raise ValueError(f"unknown backbone {backbone}")


# =========================================================================
# Heads
# =========================================================================
class MaskedAttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        w = self.attention(x).squeeze(-1).masked_fill(~valid, float("-inf"))
        w = F.softmax(w, dim=1).unsqueeze(-1)
        mean = (w * x).sum(dim=1)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, var.clamp(min=1e-7).sqrt()], dim=1)


class AllLayerAttentiveHead(nn.Module):
    """Softmax layer weights over all streams -> attentive stats pooling -> 2-layer MLP."""

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        pooling_bottleneck: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            pooling_bottleneck=pooling_bottleneck, projector_dim=projector_dim,
            head_dropout=head_dropout, layer_norm=layer_norm,
        )
        self.layer_norm = layer_norm
        self.layer_weights = nn.Parameter(torch.zeros(num_streams))
        self.asp = MaskedAttentiveStatsPooling(feature_dim, pooling_bottleneck)
        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(projector_dim, num_classes),
        )
        for i in (0, 4):
            nn.init.xavier_uniform_(self.classifier[i].weight)
            nn.init.zeros_(self.classifier[i].bias)

    def stream_weights(self) -> torch.Tensor:
        return F.softmax(self.layer_weights, dim=0)

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        x = (self.stream_weights()[:, None, None, None] * x).sum(dim=0)
        return self.classifier(self.asp(x, valid))


class QueryPoolHead(nn.Module):
    """Layer mix, then learned queries cross-attend over the frozen frames."""

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        projector_dim: int = 256,
        head_dropout: float = 0.2,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
    ):
        super().__init__()
        self.head_type = "query"
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, projector_dim=projector_dim,
            head_dropout=head_dropout, layer_norm=layer_norm, pooling_bottleneck=pooling_bottleneck,
        )
        self.layer_norm = layer_norm
        self.layer_weights = nn.Parameter(torch.zeros(num_streams))
        self.queries = nn.Parameter(torch.empty(num_queries, feature_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=head_dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(num_queries * feature_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(projector_dim, num_classes),
        )
        for i in (0, 4):
            nn.init.xavier_uniform_(self.classifier[i].weight)
            nn.init.zeros_(self.classifier[i].bias)

    def stream_weights(self) -> torch.Tensor:
        return F.softmax(self.layer_weights, dim=0)

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        x = (self.stream_weights()[:, None, None, None] * x).sum(dim=0)
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=~valid, need_weights=False)
        out = self.norm(out)
        return self.classifier(out.flatten(1))


class MHFAClassHead(nn.Module):
    """
    Multi-head factorized attention, same pooling as the speaker-verification MHFA head.
    Separate softmax weights build a key stream and a value stream. Each head pools time
    from the value stream. A linear layer maps the embedding to classes.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
    ):
        super().__init__()
        self.head_type = "mhfa"
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_heads=num_heads, compression_dim=compression_dim, projector_dim=projector_dim,
            head_dropout=head_dropout, layer_norm=layer_norm, pooling_bottleneck=pooling_bottleneck,
        )
        self.layer_norm = layer_norm
        self.layer_weights_key = nn.Parameter(torch.zeros(num_streams))
        self.layer_weights_value = nn.Parameter(torch.zeros(num_streams))
        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def stream_weights(self) -> torch.Tensor:
        return F.softmax(self.layer_weights_value, dim=0)

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        w_k = F.softmax(self.layer_weights_key, dim=0).view(-1, 1, 1, 1)
        w_v = F.softmax(self.layer_weights_value, dim=0).view(-1, 1, 1, 1)
        key = self.key_projection((x * w_k).sum(dim=0))
        value = self.value_projection((x * w_v).sum(dim=0))
        attn = self.attention_projection(key).masked_fill(~valid.unsqueeze(-1), float("-inf"))
        attn = F.softmax(attn, dim=1).transpose(1, 2).unsqueeze(-1)
        pooled = (attn * value.unsqueeze(1)).sum(dim=2)
        pooled = pooled.reshape(pooled.size(0), self.num_heads * self.compression_dim)
        return self.classifier(self.embedding_projection(pooled))


class MHFAQueryHead(nn.Module):
    """
    One head. Learned queries cross-attend over the frozen layer stack (Q-Former on layers).
    Separate key and value queries are the MHFA factorization. Each query is one head and
    pools time with its own attention score. The encoder is only read.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
    ):
        super().__init__()
        self.head_type = "mhfa_query"
        self.num_queries = num_queries
        self.compression_dim = compression_dim
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_queries, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck,
        )
        self.layer_norm = layer_norm
        self.proj = nn.Linear(feature_dim, compression_dim)
        self.key_queries = nn.Parameter(torch.empty(num_queries, compression_dim))
        self.value_queries = nn.Parameter(torch.empty(num_queries, compression_dim))
        nn.init.normal_(self.key_queries, std=0.02)
        nn.init.normal_(self.value_queries, std=0.02)
        self.time_score = nn.Linear(compression_dim, 1)
        self.dropout = nn.Dropout(head_dropout)
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_queries * compression_dim, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer("last_layer_attn", torch.full((num_streams,), 1.0 / num_streams), persistent=False)

    def stream_weights(self) -> torch.Tensor:
        return self.last_layer_attn

    def _attend_layers(self, h: torch.Tensor, queries: torch.Tensor):
        scale = h.size(-1) ** -0.5
        logits = torch.einsum("nc,btlc->btnl", queries, h) * scale
        weights = logits.softmax(dim=-1)
        return torch.einsum("btnl,btlc->btnc", weights, h), weights

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        h = self.proj(x.permute(1, 2, 0, 3))
        key, _ = self._attend_layers(h, self.key_queries)
        value, layer_w = self._attend_layers(h, self.value_queries)
        self.last_layer_attn.copy_(layer_w.detach().mean(dim=(0, 1, 2)))
        key = self.dropout(key)
        value = self.dropout(value)
        score = self.time_score(key).squeeze(-1).masked_fill(~valid.unsqueeze(-1), float("-inf"))
        attn = score.softmax(dim=1)
        pooled = torch.einsum("btn,btnc->bnc", attn, value)
        pooled = pooled.reshape(pooled.size(0), self.num_queries * self.compression_dim)
        return self.classifier(self.embedding_projection(pooled))


class QFormerMHFAHead(nn.Module):
    """
    Q-Former over the frozen layers, then the VoxLingua MHFA time pool.
    Learned key and value queries replace that model's scalar layer weights.
    The time pool matches its head: project to 128, eight heads over time,
    embed to 256, then a linear classifier. The encoder is only read.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
    ):
        super().__init__()
        self.head_type = "qformer_mhfa"
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck,
        )
        self.layer_norm = layer_norm
        self.key_queries = nn.Parameter(torch.empty(num_queries, feature_dim))
        self.value_queries = nn.Parameter(torch.empty(num_queries, feature_dim))
        nn.init.normal_(self.key_queries, std=0.02)
        nn.init.normal_(self.value_queries, std=0.02)
        self.key_mix = nn.Parameter(torch.zeros(num_queries))
        self.value_mix = nn.Parameter(torch.zeros(num_queries))
        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        self.dropout = nn.Dropout(head_dropout)
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer("last_layer_attn", torch.full((num_streams,), 1.0 / num_streams), persistent=False)

    def stream_weights(self) -> torch.Tensor:
        return self.last_layer_attn

    def _attend_layers(self, h: torch.Tensor, queries: torch.Tensor):
        scale = h.size(-1) ** -0.5
        logits = torch.einsum("nc,btlc->btnl", queries, h) * scale
        weights = logits.softmax(dim=-1)
        return torch.einsum("btnl,btlc->btnc", weights, h), weights

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        h = x.permute(1, 2, 0, 3)
        key, _ = self._attend_layers(h, self.key_queries)
        value, layer_w = self._attend_layers(h, self.value_queries)
        self.last_layer_attn.copy_(layer_w.detach().mean(dim=(0, 1, 2)))
        key = (key * F.softmax(self.key_mix, dim=0).view(1, 1, -1, 1)).sum(dim=2)
        value = (value * F.softmax(self.value_mix, dim=0).view(1, 1, -1, 1)).sum(dim=2)
        key = self.dropout(self.key_projection(key))
        value = self.dropout(self.value_projection(value))
        attn = self.attention_projection(key).masked_fill(~valid.unsqueeze(-1), float("-inf"))
        attn = F.softmax(attn, dim=1).transpose(1, 2).unsqueeze(-1)
        pooled = (attn * value.unsqueeze(1)).sum(dim=2)
        pooled = pooled.reshape(pooled.size(0), self.num_heads * self.compression_dim)
        return self.classifier(self.embedding_projection(pooled))


# Best 7-language probe heads: (encoder layer, attention head), 1-based layer index.
LID_ATTN_HEADS = ((17, 10), (19, 2), (16, 4), (18, 10), (23, 14))


class QFormerMHFAAttnHead(QFormerMHFAHead):
    """
    Same Q-Former + MHFA head, plus one learned connection from each
    language-specialist attention head. Those heads are read before out_proj
    (64-d) and projected to 1024 so the Q-Former can attend to them beside
    the 25 layer streams. The encoder is only read.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
        attn_heads: Optional[Sequence[Sequence[int]]] = None,
        head_dim: int = 64,
    ):
        chosen = [tuple(int(x) for x in pair) for pair in (attn_heads or LID_ATTN_HEADS)]
        if num_streams < 25 + len(chosen):
            num_streams = num_streams + len(chosen)
        super().__init__(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck,
        )
        self.head_type = "qformer_mhfa_attn"
        self.attn_heads = chosen
        self.head_projections = nn.ModuleList(nn.Linear(head_dim, feature_dim) for _ in chosen)
        self.config = dict(self.config)
        self.config["attn_heads"] = chosen
        self.config["head_dim"] = head_dim

    def forward(self, feats: torch.Tensor, valid: torch.Tensor, head_feats: torch.Tensor) -> torch.Tensor:
        extra = torch.stack(
            [proj(head_feats[i].float()) for i, proj in enumerate(self.head_projections)], dim=0,
        )
        return super().forward(torch.cat([feats, extra.to(dtype=feats.dtype)], dim=0), valid)


class PaperLayerHead(nn.Module):
    """
    Frozen-layer head drawn from the TidyVoice speaker papers.
    Each layer has its own adapter. The top layer queries every layer
    (hierarchical cross-attention). Frames then query a shared latent
    array. Attentive statistics pooling makes the utterance vector.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 32,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
    ):
        super().__init__()
        self.head_type = "paper_layer"
        self.compression_dim = compression_dim
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck,
        )
        self.layer_norm = layer_norm
        self.adapters = nn.ModuleList(
            nn.Linear(feature_dim, compression_dim) for _ in range(num_streams)
        )
        self.latent = nn.Parameter(torch.empty(num_queries, compression_dim))
        nn.init.normal_(self.latent, std=0.02)
        self.latent_key = nn.Linear(compression_dim, compression_dim, bias=False)
        self.latent_value = nn.Linear(compression_dim, compression_dim, bias=False)
        self.conv = nn.Conv1d(compression_dim, compression_dim, kernel_size=3, padding=1)
        self.self_attn = nn.MultiheadAttention(
            compression_dim, num_heads=4, dropout=head_dropout, batch_first=True,
        )
        self.asp = MaskedAttentiveStatsPooling(compression_dim, pooling_bottleneck)
        self.embedding = nn.Sequential(
            nn.Linear(2 * compression_dim, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer("last_layer_attn", torch.full((num_streams,), 1.0 / num_streams), persistent=False)

    def stream_weights(self) -> torch.Tensor:
        return self.last_layer_attn

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        h = torch.stack([adapter(x[i]) for i, adapter in enumerate(self.adapters)], dim=2)
        query = h[:, :, -1, :]
        scale = self.compression_dim ** -0.5
        layer_logits = torch.einsum("btc,btsc->bts", query, h) * scale
        layer_w = layer_logits.softmax(dim=-1)
        self.last_layer_attn.copy_(layer_w.detach().mean(dim=(0, 1)))
        mixed = torch.einsum("bts,btsc->btc", layer_w, h)
        keys = self.latent_key(self.latent)
        values = self.latent_value(self.latent)
        lat_logits = torch.einsum("btc,nc->btn", mixed, keys) * scale
        lat_w = lat_logits.softmax(dim=-1)
        latent = torch.einsum("btn,nc->btc", lat_w, values)
        frames = mixed + latent
        frames = frames + self.conv(frames.transpose(1, 2)).transpose(1, 2)
        refined, _ = self.self_attn(frames, frames, frames, key_padding_mask=~valid)
        frames = frames + refined
        return self.classifier(self.embedding(self.asp(frames, valid)))


class QFormerLayerMix(nn.Module):
    """
    Per-frame Q-Former over the layer stack (the value side of QFormerMHFAHead):
    learned queries attend over the streams at every frame, a softmax mixes the
    queries, output (B, T, C). The encoder is only read.
    """

    def __init__(self, num_streams: int, feature_dim: int, num_queries: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, feature_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.mix = nn.Parameter(torch.zeros(num_queries))
        self.register_buffer("last_layer_attn", torch.full((num_streams,), 1.0 / num_streams), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (S, B, T, C), already layer-normed -> (B, T, C)"""
        h = x.permute(1, 2, 0, 3)
        logits = torch.einsum("nc,btlc->btnl", self.queries, h) * h.size(-1) ** -0.5
        w = logits.softmax(dim=-1)
        w = (w * F.softmax(self.mix, dim=0).view(1, 1, -1, 1)).sum(dim=2)
        self.last_layer_attn.copy_(w.detach().mean(dim=(0, 1)))
        return torch.einsum("btl,btlc->btc", w, h)


class MaskedMultiHeadPool(nn.Module):
    """H attention heads over time, each pools the value frames; masked; returns (B, H*D)."""

    def __init__(self, dim: int, heads: int = 8, bottleneck: int = 128):
        super().__init__()
        self.heads = heads
        self.score = nn.Sequential(nn.Linear(dim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, heads))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        a = self.score(x).masked_fill(~valid.unsqueeze(-1), float("-inf")).softmax(dim=1)
        return torch.einsum("bth,btd->bhd", a, x).flatten(1)


class QFormerTransformerHead(nn.Module):
    """
    Q-Former layer mix -> linear to d_model -> conv positional mixing ->
    small Transformer over time (masked) -> multi-head attentive pool -> embedding -> classes.
    Adds temporal context modeling that the 0.5M MHFA pool lacks. Encoder only read.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
        d_model: int = 384,
        tf_layers: int = 2,
        tf_heads: int = 6,
        pool_heads: int = 8,
    ):
        super().__init__()
        self.head_type = "qformer_tf"
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck, d_model=d_model, tf_layers=tf_layers,
            tf_heads=tf_heads, pool_heads=pool_heads,
        )
        self.layer_norm = layer_norm
        self.mixer = QFormerLayerMix(num_streams, feature_dim, num_queries)
        self.inp = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(head_dropout))
        self.pos_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.tf = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, tf_heads, 4 * d_model, dropout=head_dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(tf_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.pool = MaskedMultiHeadPool(d_model, pool_heads, pooling_bottleneck)
        self.embedding = nn.Sequential(
            nn.Linear(pool_heads * d_model, projector_dim), nn.LayerNorm(projector_dim), nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def stream_weights(self) -> torch.Tensor:
        return self.mixer.last_layer_attn

    def embed(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        x = self.inp(self.mixer(x))
        m = valid.unsqueeze(-1).to(x.dtype)
        x = x + self.pos_conv((x * m).transpose(1, 2)).transpose(1, 2)
        for layer in self.tf:
            x = layer(x, src_key_padding_mask=~valid)
        x = self.out_norm(x)
        return self.embedding(self.pool(x, valid))

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(feats, valid))


class _ChanLN(nn.Module):
    """LayerNorm over channels of a (B, C, T) tensor; frame-local, so padding never leaks in."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _MaskedSE(nn.Module):
    def __init__(self, c: int, bottleneck: int = 128):
        super().__init__()
        self.fc1 = nn.Conv1d(c, bottleneck, 1)
        self.fc2 = nn.Conv1d(bottleneck, c, 1)

    def forward(self, x, m):
        s = (x * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class _SERes2Block(nn.Module):
    def __init__(self, c: int, dilation: int, scale: int = 8, k: int = 3):
        super().__init__()
        self.scale, self.width = scale, c // scale
        self.conv1 = nn.Conv1d(c, c, 1)
        self.n1 = _ChanLN(c)
        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, k, dilation=dilation, padding=dilation * (k - 1) // 2)
            for _ in range(scale - 1)
        )
        self.norms = nn.ModuleList(_ChanLN(self.width) for _ in range(scale - 1))
        self.conv3 = nn.Conv1d(c, c, 1)
        self.n3 = _ChanLN(c)
        self.se = _MaskedSE(c)

    def forward(self, x, m):
        r = x
        x = self.n1(F.relu(self.conv1(x))) * m
        xs = torch.split(x, self.width, dim=1)
        outs, sp = [], None
        for i in range(self.scale - 1):
            sp = xs[i] if i == 0 else sp + xs[i]
            sp = self.norms[i](F.relu(self.convs[i](sp))) * m
            outs.append(sp)
        outs.append(xs[-1])
        x = self.n3(F.relu(self.conv3(torch.cat(outs, dim=1))))
        return (self.se(x, m) + r) * m


class QFormerECAPAHead(nn.Module):
    """
    Q-Former layer mix -> ECAPA-TDNN back end (SE-Res2 blocks, multi-layer aggregation,
    masked attentive statistics pooling) -> embedding -> classes. Channel LayerNorm instead of
    BatchNorm so zero-padded frames never enter the statistics. Encoder only read.
    """

    def __init__(
        self,
        num_streams: int = 25,
        feature_dim: int = 1024,
        num_classes: int = 7,
        num_queries: int = 8,
        num_heads: int = 8,
        compression_dim: int = 128,
        projector_dim: int = 256,
        head_dropout: float = 0.1,
        layer_norm: bool = True,
        pooling_bottleneck: int = 128,
        channels: int = 512,
    ):
        super().__init__()
        self.head_type = "qformer_ecapa"
        self.config = dict(
            num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes,
            num_queries=num_queries, num_heads=num_heads, compression_dim=compression_dim,
            projector_dim=projector_dim, head_dropout=head_dropout, layer_norm=layer_norm,
            pooling_bottleneck=pooling_bottleneck, channels=channels,
        )
        self.layer_norm = layer_norm
        self.mixer = QFormerLayerMix(num_streams, feature_dim, num_queries)
        self.conv0 = nn.Conv1d(feature_dim, channels, 5, padding=2)
        self.n0 = _ChanLN(channels)
        self.blocks = nn.ModuleList(_SERes2Block(channels, d) for d in (2, 3, 4))
        self.mfa = nn.Conv1d(3 * channels, 3 * channels, 1)
        self.asp = MaskedAttentiveStatsPooling(3 * channels, pooling_bottleneck)
        self.embedding = nn.Sequential(
            nn.Linear(6 * channels, projector_dim), nn.LayerNorm(projector_dim), nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(projector_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def stream_weights(self) -> torch.Tensor:
        return self.mixer.last_layer_attn

    def embed(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = feats.float()
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])
        x = self.mixer(x).transpose(1, 2)
        m = valid.unsqueeze(1).to(x.dtype)
        x = self.n0(F.relu(self.conv0(x * m))) * m
        outs = []
        for blk in self.blocks:
            x = blk(x, m)
            outs.append(x)
        x = F.relu(self.mfa(torch.cat(outs, dim=1))) * m
        return self.embedding(self.asp(x.transpose(1, 2), valid))

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(feats, valid))


AllLayerAttentiveHead.head_type = "all_layer_attentive"

HEAD_TYPES = {
    "all_layer_attentive": AllLayerAttentiveHead,
    "query": QueryPoolHead,
    "mhfa": MHFAClassHead,
    "mhfa_query": MHFAQueryHead,
    "qformer_mhfa": QFormerMHFAHead,
    "qformer_mhfa_attn": QFormerMHFAAttnHead,
    "paper_layer": PaperLayerHead,
    "qformer_tf": QFormerTransformerHead,
    "qformer_ecapa": QFormerECAPAHead,
}


def build_classification_head(head_type: str, num_streams: int, feature_dim: int, num_classes: int, **kwargs) -> nn.Module:
    cls = HEAD_TYPES[head_type]
    return cls(num_streams=num_streams, feature_dim=feature_dim, num_classes=num_classes, **kwargs)


def save_head(path: str | Path, head: nn.Module, name: str, task: str, labels: Sequence[str],
              encoder_spec: dict, extra: Optional[dict] = None) -> None:
    torch.save(
        {
            "name": name,
            "task": task,
            "labels": list(labels),
            "head_type": head.head_type,
            "head_config": head.config,
            "state_dict": head.state_dict(),
            "encoder_spec": encoder_spec,
            **(extra or {}),
        },
        path,
    )


def load_head(path: str | Path, map_location="cpu") -> Tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    head = HEAD_TYPES[ckpt["head_type"]](**ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt


# =========================================================================
# Shared encoder + N heads, one forward
# =========================================================================
class Qwen3ASRMultiHead(nn.Module):
    def __init__(self, encoder: FrozenQwen3ASRAllLayers, heads: Optional[Dict[str, nn.Module]] = None):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads or {})

    def add_head(self, name: str, head: nn.Module) -> None:
        self.heads[name] = head

    def forward(self, wave: torch.Tensor, wave_lens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feats, valid = self.encoder(wave, wave_lens)
        return {name: head(feats, valid) for name, head in self.heads.items()}
