"""OmniASV speaker embeddings on the frozen Qwen3-ASR-1.7B encoder (backbone ``qwen3-asr-1.7b``).

    from omnihead.asv import SpeakerEmbedder
    e = SpeakerEmbedder("var")                         # or "4s", or a path to a head file
    emb = e.embed_files(["a.wav", "b.wav"])            # (2, 256) L2-normalised
    score = float(emb[0] @ emb[1])                     # cosine; same speaker if >= e.threshold

Full utterances of any length are embedded in one encoder pass each (batched by total duration, padding
masked out of the pooling). ``embed_features`` reads an existing encoder pass, so diarization (or any other
OmniHead head) and speaker embeddings can share it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from ..infer import load_audio
from ..paths import models_root, resolve_backbone_weights
from ..qwen3asr_frozen_multihead import build_frozen_encoder
from .model import OmniASVHead, build_from_checkpoint

BACKBONE = "qwen3-asr-1.7b"
ENCODER_CFG = {"model_id": "Qwen/Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5"}
HEADS = {"4s": "asv/omniasv_4s_v1.pt", "var": "asv/omniasv_var_v1.pt"}


def head_path(name: str) -> Path:
    if name in HEADS:
        return models_root() / BACKBONE / HEADS[name]
    return Path(name)


def load_head(name: str = "var", device="cpu") -> OmniASVHead:
    ck = torch.load(head_path(name), map_location="cpu", weights_only=False)
    head = build_from_checkpoint(ck).to(device).eval()
    head.meta = {k: ck.get(k) for k in ("name", "vox1o", "protocol", "training")}
    return head


class SpeakerEmbedder:
    def __init__(self, head: str = "var", device: Optional[str] = None, backbone: Optional[str] = None,
                 encoder=None, budget_sec: float = 300.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.head = load_head(head, self.device)
        if encoder is None:
            weights, _ = resolve_backbone_weights(ENCODER_CFG, backbone)
            encoder = build_frozen_encoder("qwen3asr", weights, torch.bfloat16).to(self.device).eval()
        self.encoder = encoder
        self.budget = budget_sec
        self.threshold = (self.head.meta.get("vox1o") or {}).get("eer_threshold")

    @torch.inference_mode()
    def embed_features(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats (25, B, T, 1024) from the shared encoder pass, valid (B, T) frames to pool -> (B, 256) unit vectors."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            e = self.head(feats, valid).float()
        return torch.nn.functional.normalize(e, dim=1)

    @torch.inference_mode()
    def embed(self, waves: Sequence[torch.Tensor]) -> np.ndarray:
        """List of 1-D 16 kHz waveforms (any length) -> (N, 256) L2-normalised embeddings."""
        order = np.argsort([len(w) for w in waves])
        out = np.zeros((len(waves), self.head.embedding_dim), np.float32)
        batch, tot = [], 0.0
        for i in list(order) + [None]:
            if i is not None and (not batch or tot + len(waves[i]) / 16000 <= self.budget) and len(batch) < 128:
                batch.append(int(i)); tot += len(waves[i]) / 16000
                continue
            if batch:
                lens = torch.tensor([max(len(waves[j]), 400) for j in batch])
                x = torch.zeros(len(batch), int(lens.max()))
                for b, j in enumerate(batch):
                    x[b, : len(waves[j])] = waves[j]
                feats, valid = self.encoder(x.to(self.device), lens.to(self.device))
                out[batch] = self.embed_features(feats, valid).cpu().numpy()
            if i is not None:
                batch, tot = [int(i)], len(waves[i]) / 16000
        return out

    def embed_files(self, paths: Sequence[Union[str, Path]]) -> np.ndarray:
        return self.embed([load_audio(str(p)) for p in paths])

    def score(self, a: Union[str, Path], b: Union[str, Path]) -> float:
        e = self.embed_files([a, b])
        return float(e[0] @ e[1])
