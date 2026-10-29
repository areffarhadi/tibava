"""Full-recording speaker diarization with the OmniHead diarization head.

Backbone: Qwen3-ASR-1.7B audio encoder (frozen, bf16).

    recording -> frozen encoder ONCE (120 s segments = 15 attention windows, exact)
              -> 12 s chunks every `step_sec` s cut from that single pass
              -> diarization head -> up to 6 local speakers per chunk (powerset, hard decisions)
              -> one speaker embedding per active (chunk, local speaker)
              -> agglomerative clustering -> global speakers
              -> overlap-aware reconstruction (pyannote / DiariZen) -> pyannote Annotation / RTTM

Everything after the segmentation is the DiariZen fork of pyannote.audio's ``SpeakerDiarization``.
Only two steps are replaced: ``get_segmentations`` (the encoder runs once per recording instead of
once per sliding window) and ``get_embeddings`` (inactive local speakers are skipped).

Speaker embeddings: any pyannote-compatible embedder. The default is WeSpeaker SimAM-ResNet34
(VoxBlink2) as ONNX; set ``OMNIHEAD_DIAR_EMBEDDING`` or pass ``embedding=``.
The path must contain "wespeaker" (pyannote selects its ONNX backend by substring).

Frame grid: the head emits 52 frames/s (13 encoder frames x 4). Their true centres follow the
encoder's per-second grid; pyannote treats them as a uniform 1/52 s grid, which moves any frame by at
most 13 ms (less than one 19.2 ms frame).

Requires the DiariZen fork of pyannote.audio (``SpeakerDiarization(seg_duration=...)``), see
requirements-diar.txt.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..paths import models_root, resolve_backbone_weights
from ..qwen3asr_frozen_multihead import build_frozen_encoder
from .frame_grid import FRAMES_PER_SECOND, num_encoder_frames
from .model import QwenDiarHead, build_from_checkpoint

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pyannote loads the ONNX embedder via torch.load

BACKBONE = "qwen3-asr-1.7b"
ENCODER_CFG = {"model_id": "Qwen/Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5"}
CHECKPOINT_NAME = "diarization/diar_qwen3asr_v1.pt"
WINDOW_SEC = 8

DEFAULTS = dict(
    # unified setting: best average DER over the validation splits of AMI, AliMeeting, NOTSOFAR-1 and
    # VoxConverse with OmniASV-var embeddings (docs/DIARIZATION.md); VoxConverse-only optimum: t 0.70, mcs 8
    step_sec=1,                       # chunk hop in whole seconds (chunks stay on the encoder's 1 s grid)
    clustering="AgglomerativeClustering",
    method="centroid",
    threshold=0.55,
    min_cluster_size=20,              # counted in embeddings: with a 1 s hop, ~20 s of a speaker
    embedding_exclude_overlap=True,   # embed the frames where the speaker talks alone (all frames if < 0.3 s)
    min_duration_off=0.0,
    embedding_batch_size=64,
    head_batch_size=32,
    encoder_segment_sec=120,
    asv_min_frames=4,                 # OmniASV embeddings need >= 4 encoder frames (0.3 s) of a speaker
)


def default_checkpoint() -> Path:
    return models_root() / BACKBONE / CHECKPOINT_NAME


def default_embedding() -> Optional[str]:
    return os.environ.get("OMNIHEAD_DIAR_EMBEDDING")


@torch.no_grad()
def encode_long(encoder, wave: torch.Tensor, segment_sec: int = 120) -> torch.Tensor:
    """(N,) 16 kHz waveform -> (25, T, 1024) on the waveform's device, in ``segment_sec`` pieces.

    Segment boundaries fall on 8 s attention-window and 1 s conv-chunk boundaries, so this equals one
    pass over the recording except for the log-mel peak floor, taken per segment (as ship_v2 does)."""
    if segment_sec % WINDOW_SEC:
        raise ValueError(f"segment_sec must be a multiple of {WINDOW_SEC}")
    seg = segment_sec * 16000
    parts = []
    for start in range(0, wave.shape[0], seg):
        piece = wave[start:start + seg]
        if piece.shape[0] < 400:
            piece = torch.nn.functional.pad(piece, (0, 400 - piece.shape[0]))
        feats, valid = encoder(piece[None].float())
        parts.append(feats[:, 0, : int(valid[0].sum())])
    return torch.cat(parts, dim=1)


def _segmentation_model_cls():
    from pyannote.audio.core.model import Model
    from pyannote.core import SlidingWindow

    class QwenDiarSegmentation(Model):
        """pyannote-facing wrapper: frozen encoder + diarization head on one chunk."""

        def __init__(self, head: QwenDiarHead, encoder, chunk_sec: int = 12):
            cfg = head.config
            super().__init__(sample_rate=16000, num_channels=1,
                             max_speakers_per_chunk=cfg["max_speakers_per_chunk"],
                             max_speakers_per_frame=cfg["max_speakers_per_frame"],
                             duration=chunk_sec, min_duration=chunk_sec)
            self.head, self.encoder, self.chunk_sec = head, encoder, chunk_sec
            self.upsample = cfg["upsample"]

        @property
        def _receptive_field(self):
            step = 1.0 / (FRAMES_PER_SECOND * self.upsample)
            return SlidingWindow(start=0.0, duration=step, step=step)

        def num_frames(self, num_samples: int) -> int:
            return num_encoder_frames(num_samples) * self.upsample

        @property
        def dimension(self) -> int:
            return self.head.num_classes

        @torch.no_grad()
        def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
            """(B, 1, N) chunks, each encoded on its own -> powerset log-probs (B, F, C)."""
            feats, valid = self.encoder(waveforms[:, 0].float())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=waveforms.is_cuda):
                return self.head(feats, valid=valid).float()

    return QwenDiarSegmentation


def _pipeline_cls():
    from pyannote.audio import Audio
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    class QwenDiarizationPipeline(SpeakerDiarization):
        def __init__(self, seg_model, step_sec: int, head_batch_size: int = 32, encoder_segment_sec: int = 120,
                     asv_head=None, asv_min_frames: int = 4, **kw):
            if asv_head is not None:  # OmniASV on the shared encoder pass: no external embedder is loaded
                kw.pop("embedding", None)
                clustering = kw.pop("clustering", "AgglomerativeClustering")
                super().__init__(seg_duration=seg_model.chunk_sec, segmentation=seg_model,
                                 segmentation_step=step_sec / seg_model.chunk_sec, clustering="OracleClustering", **kw)
                from pyannote.audio.pipelines.clustering import Clustering
                self.klustering = clustering
                self.clustering = Clustering[clustering].value(metric="cosine")
            else:
                super().__init__(seg_duration=seg_model.chunk_sec, segmentation=seg_model,
                                 segmentation_step=step_sec / seg_model.chunk_sec, **kw)
            self.asv_head, self.asv_min_frames = asv_head, asv_min_frames
            self.seg_model, self.step_sec = seg_model, int(step_sec)
            self.head_batch_size, self.encoder_segment_sec = head_batch_size, encoder_segment_sec
            self._loader = Audio(sample_rate=16000, mono="downmix")

        @torch.no_grad()
        def get_segmentations(self, file, hook=None, soft=False):
            """Same chunks as pyannote's sliding-window Inference, cut from one encoder pass.
            Returns hard multilabel activities (num_chunks, 624, 6)."""
            if "omnihead_segmentations" in file:
                return file["omnihead_segmentations"]
            m = self.seg_model
            dev = next(m.head.parameters()).device
            wave, _ = self._loader(file)
            n_samples = wave.shape[1]
            feats = encode_long(m.encoder, wave[0].to(dev), self.encoder_segment_sec)  # (25, T, D)
            T = feats.shape[1]
            chunk, step, up = m.chunk_sec, self.step_sec, m.upsample
            cf = chunk * FRAMES_PER_SECOND
            W, S = chunk * 16000, step * 16000
            n_full = (n_samples - W) // S + 1 if n_samples >= W else 0
            has_last = n_samples < W or (n_samples - W) % S > 0
            starts = [k * step for k in range(n_full + int(has_last))]
            # a chunk that runs past the end is read from the last whole-second chunk inside the file
            src_max = max(0, (T - cf) // FRAMES_PER_SECOND)
            srcs = sorted({min(s, src_max) for s in starts})
            out_src = {}
            for i in range(0, len(srcs), self.head_batch_size):
                part = srcs[i:i + self.head_batch_size]
                x = feats.new_zeros(feats.shape[0], len(part), cf, feats.shape[2])
                valid = torch.zeros(len(part), cf, dtype=torch.bool, device=dev)
                for b, s in enumerate(part):
                    seg = feats[:, s * FRAMES_PER_SECOND: s * FRAMES_PER_SECOND + cf]
                    x[:, b, : seg.shape[1]] = seg
                    valid[b, : seg.shape[1]] = True
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lp = m.head(x, valid=None if bool(valid.all()) else valid).float()
                ml = m.head.to_multilabel(lp).cpu().numpy()
                for b, s in enumerate(part):
                    ml[b, int(valid[b].sum()) * up:] = 0
                    out_src[s] = ml[b]
            K, F = m.head.mapping.shape[1], cf * up
            data = np.zeros((len(starts), F, K), np.float32)
            for c, s in enumerate(starts):
                src = min(s, src_max)
                off = (s - src) * FRAMES_PER_SECOND * up
                n_in = max(0, min(F, (T - s * FRAMES_PER_SECOND) * up))
                take = min(n_in, F - off)
                if take > 0:
                    data[c, :take] = out_src[src][off:off + take]
            segs = SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=float(chunk), step=float(step)))
            file["omnihead_segmentations"] = segs
            if self.asv_head is not None:
                file["omnihead_feats"] = feats
            return segs

        @torch.no_grad()
        def _omniasv_embeddings(self, file, binary_segmentations, exclude_overlap: bool):
            """One OmniASV embedding per (chunk, local speaker) from the SAME encoder pass as the segmentation.
            A local speaker's 52 Hz activity is mapped to the 13 Hz encoder frames (>= 2 of 4 sub-frames active) and
            the ASV head pools only those frames; fewer than ``asv_min_frames`` frames -> NaN (not clustered)."""
            feats = file.pop("omnihead_feats", None)
            dev = next(self.asv_head.parameters()).device
            if feats is None:
                wave, _ = self._loader(file)
                feats = encode_long(self.seg_model.encoder, wave[0].to(dev), self.encoder_segment_sec)
            up = self.seg_model.upsample
            C, F, K = binary_segmentations.data.shape
            cf = F // up
            act = np.nan_to_num(binary_segmentations.data, nan=0.0).reshape(C, cf, up, K).sum(2) >= 2
            clean = act & (act.sum(2, keepdims=True) < 2)
            out = np.full((C, K, self.asv_head.embedding_dim), np.nan, np.float32)
            jobs = []
            for c in range(C):
                for k in range(K):
                    m = clean[c, :, k] if exclude_overlap and clean[c, :, k].sum() >= self.asv_min_frames else act[c, :, k]
                    if m.sum() >= self.asv_min_frames:
                        jobs.append((c, k, m))
            bs = max(8, self.embedding_batch_size)
            for i in range(0, len(jobs), bs):
                part = jobs[i:i + bs]
                x = feats.new_zeros(feats.shape[0], len(part), cf, feats.shape[2])
                valid = torch.zeros(len(part), cf, dtype=torch.bool, device=dev)
                for b, (c, k, m) in enumerate(part):
                    f0 = c * self.step_sec * FRAMES_PER_SECOND
                    seg = feats[:, f0: f0 + cf]
                    x[:, b, : seg.shape[1]] = seg
                    valid[b, : seg.shape[1]] = torch.from_numpy(m[: seg.shape[1]]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    e = self.asv_head(x, valid).float()
                e = torch.nn.functional.normalize(e, dim=1).cpu().numpy()
                for (c, k, _), v in zip(part, e):
                    out[c, k] = v
            return out

        def get_embeddings(self, file, binary_segmentations, exclude_overlap: bool = False, hook=None):
            """pyannote's embedding step, but inactive (chunk, speaker) pairs are skipped (left NaN)."""
            if self.asv_head is not None:
                return self._omniasv_embeddings(file, binary_segmentations, exclude_overlap)
            duration = binary_segmentations.sliding_window.duration
            C, F, K = binary_segmentations.data.shape
            data = np.nan_to_num(binary_segmentations.data, nan=0.0)
            if exclude_overlap:
                min_frames = math.ceil(F * self._embedding.min_num_samples / (duration * self._embedding.sample_rate))
                clean = data * (data.sum(2, keepdims=True) < 2)
            else:
                min_frames, clean = -1, data
            out = np.full((C, K, self._embedding.dimension), np.nan, np.float32)
            jobs = [(c, k) for c in range(C) for k in range(K) if data[c, :, k].sum() > 0]
            sw = binary_segmentations.sliding_window
            bs, cache = self.embedding_batch_size, {}
            for i in range(0, len(jobs), bs):
                part = jobs[i:i + bs]
                waves, masks = [], []
                for c, k in part:
                    if c not in cache:
                        cache = {c: self._audio.crop(file, sw[c], duration=duration, mode="pad")[0]}
                    waves.append(cache[c][None])
                    msk = clean[c, :, k] if clean[c, :, k].sum() > min_frames else data[c, :, k]
                    masks.append(torch.from_numpy(msk.astype(np.float32))[None])
                emb = self._embedding(torch.vstack(waves), masks=torch.vstack(masks))
                for (c, k), e in zip(part, emb):
                    out[c, k] = e
            return out

    return QwenDiarizationPipeline


class Diarizer:
    """Full-recording diarization; one encoder pass serves the diarization head AND the speaker embeddings.

        from omnihead.diarization import Diarizer
        d = Diarizer()                         # embeddings: OmniASV-var head on the shared encoder pass
        ann = d("meeting.wav")                 # pyannote Annotation
        open("meeting.rttm", "w").write(ann.to_rttm())

    ``embedding``: "omniasv-var" (default), "omniasv-4s", or the path of an external pyannote-compatible
    embedder (e.g. WeSpeaker SimAM-ResNet34 ONNX; the path must contain "wespeaker").
    """

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None,
                 embedding: Optional[str] = None, backbone: Optional[str] = None, encoder=None, **cfg):
        self.cfg = dict(DEFAULTS, **cfg)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint or default_checkpoint(), map_location="cpu", weights_only=False)
        head = build_from_checkpoint(ckpt).to(self.device).eval()
        if encoder is None:
            weights, _ = resolve_backbone_weights(ENCODER_CFG, backbone)
            encoder = build_frozen_encoder("qwen3asr", weights, torch.bfloat16).to(self.device).eval()
        embedding = embedding or default_embedding() or "omniasv-var"
        asv_head = None
        if embedding.startswith("omniasv"):
            from ..asv.embedder import load_head as load_asv_head
            asv_head = load_asv_head(embedding.split("-", 1)[1] if "-" in embedding else "var", self.device)
        elif not Path(embedding).is_file():
            raise FileNotFoundError(f"speaker embedder not found: {embedding} (use omniasv-var / omniasv-4s or an ONNX path)")
        self.embedding = embedding
        chunk = int(ckpt.get("protocol", {}).get("chunk_sec", 12))
        self.seg_model = _segmentation_model_cls()(head, encoder, chunk).to(self.device).eval()
        self.pipeline = _pipeline_cls()(
            self.seg_model, self.cfg["step_sec"], head_batch_size=self.cfg["head_batch_size"],
            encoder_segment_sec=self.cfg["encoder_segment_sec"], asv_head=asv_head,
            asv_min_frames=self.cfg["asv_min_frames"], embedding=embedding,
            embedding_exclude_overlap=self.cfg["embedding_exclude_overlap"], clustering=self.cfg["clustering"],
            embedding_batch_size=self.cfg["embedding_batch_size"], segmentation_batch_size=8, device=self.device)
        self.set_clustering(self.cfg["threshold"], self.cfg["min_cluster_size"], self.cfg["method"])

    def set_clustering(self, threshold: float, min_cluster_size: int, method: str = "centroid",
                       min_duration_off: Optional[float] = None):
        self.pipeline.instantiate({
            "segmentation": {"min_duration_off": self.cfg["min_duration_off"] if min_duration_off is None
                             else min_duration_off},
            "clustering": {"method": method, "min_cluster_size": int(min_cluster_size), "threshold": float(threshold)},
        })

    def __call__(self, audio: str, uri: Optional[str] = None, num_speakers: Optional[int] = None):
        uri = uri or Path(audio).stem
        return self.pipeline({"audio": str(audio), "uri": uri}, num_speakers=num_speakers)

    @torch.no_grad()
    def local_activity(self, audio: str):
        """Head output only (no embeddings / clustering): pyannote SlidingWindowFeature of hard
        per-chunk activities (num_chunks, 624, 6), chunk = 12 s, hop = step_sec."""
        return self.pipeline.get_segmentations({"audio": str(audio), "uri": Path(audio).stem})
