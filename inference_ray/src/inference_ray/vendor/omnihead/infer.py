"""
OmniHead reference inference: frozen Qwen3-ASR-1.7B audio encoder (backbone ``qwen3-asr-1.7b``) + task heads.

One encoder pass per clip serves every head. Each task lists its head checkpoints in the config;
a classification task's ensemble is the mean of the heads' log-softmax. Clips longer than a task's max_sec are
center-cropped to it (LID 20 s, SER 15 s, phoneme / CTC ASR 60 s, the same rule as the evals); the turn task reads
the last 8 s. A clip longer than the smallest max_sec is encoded once per distinct crop it needs; shorter clips are
encoded once.

LID returns the 114-way language plus the 7-class decision (en fr es it de pt others): the softmax mass of the
108 non-target languages is summed into `others`, a fixed log-bias is added to `others` (UA-optimal), then argmax.
`lid7_probs` are the bias-adjusted 7-class posteriors, consistent with that argmax.

VAD (every config, "kind": "vad"): 6 encoder attention heads + logistic regression (vad_head.py), read from
the same pass; returns speech segments. Clips longer than its max_sec (120 s) are encoded in 120 s chunks so
the VAD covers the whole clip ("crop": "chunks"); skip it with --tasks to keep the cost low on long files.

Configs (omnihead/configs): default, max, with_phoneme, with_phoneme_light, with_phoneme_noboundary, with_turn,
full, full_asr_ctc. Head paths are relative to <models root>/<backbone>/ (env OMNIHEAD_MODELS); backbone weights
come from env OMNIHEAD_BACKBONE (local dir or HF id, default Qwen/Qwen3-ASR-1.7B).

CLI
  omnihead a.wav b.flac some_dir/ [--config default|max|full|...|path.json] [--out preds.jsonl]
  omnihead --manifest eval.jsonl --tasks lid --save_logits lid_logits.npz
  python -m omnihead ...                       (same CLI)
Python
  from omnihead import OmniHead
  m = OmniHead("default", device="cuda:0")
  m.predict_files(["a.wav"])          # list of dicts
  m.predict([wave_16k_tensor])        # 1-D float tensors at 16 kHz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    import soxr
except ImportError:  # torchaudio for every rate
    soxr = None

from .paths import backbone_dir, config_path, list_configs, resolve_backbone_weights
from .qwen3asr_frozen_multihead import SAMPLE_RATE, build_frozen_encoder, load_head

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}
DEFAULT_BACKBONE = "qwen3-asr-1.7b"
# config "backbone" id -> encoder builder name in qwen3asr_frozen_multihead.build_frozen_encoder
ENCODER_BUILDERS = {"qwen3-asr-1.7b": "qwen3asr"}


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_audio(path: str) -> torch.Tensor:
    """Mono float32 at 16 kHz. Same torchaudio resampler as training, except for rates whose
    reduced ratio to 16 kHz is awkward (e.g. 24414 Hz: ~3 s per clip in torchaudio), which go to soxr."""
    wave, sr = torchaudio.load(path)
    wave = wave.mean(0)
    if sr != SAMPLE_RATE:
        if sr // math.gcd(sr, SAMPLE_RATE) > 1000 and soxr is not None:
            wave = torch.from_numpy(soxr.resample(wave.numpy(), sr, SAMPLE_RATE, quality="HQ").astype(np.float32))
        else:
            wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    return wave


def crop(wave: torch.Tensor, n: int, mode: str = "center") -> torch.Tensor:
    """Clip to n samples: centre crop (LID/SER/phoneme/CTC ASR, as in their evals), the LAST n samples (turn), or
    the chunk starting at sample s (mode "seg<s>", VAD over long clips)."""
    if mode.startswith("seg"):
        s = int(mode[3:])
        return wave[s:s + n]
    if wave.numel() <= n:
        return wave
    return wave[-n:] if mode == "last" else center_crop(wave, n)


def center_crop(wave: torch.Tensor, n: int) -> torch.Tensor:
    if wave.numel() <= n:
        return wave
    s = (wave.numel() - n) // 2
    return wave[s:s + n]


class Task:
    """One config task. ``head_dir`` = <models root>/<backbone> (heads, tokenizer); ``config_dir`` = folder of the
    config json (phoneme token list)."""

    def __init__(self, name: str, cfg: dict, head_dir: Path, config_dir: Path, device: torch.device):
        self.name = name
        self.max_len = int(cfg["max_sec"] * SAMPLE_RATE)
        self.head_files = list(cfg["heads"])
        self.heads, self.labels = [], None
        self.kind = cfg.get("kind", "class")
        self.crop = cfg.get("crop", "center")
        self.lid7 = None
        if self.kind == "vad":  # attention-head VAD (vad_head.py): frame logits -> speech segments
            from .vad_head import AttnHeadVAD
            self.vad = AttnHeadVAD(head_dir / self.head_files[0], device)
            self.heads = [self.vad]
            return
        if self.kind == "turn":  # end-of-turn head (Smart Turn design): P(turn complete)
            from .model_smart_turn import load_turn_head
            head, ck = load_turn_head(str(head_dir / self.head_files[0]), device)
            self.heads = [head]
            self.threshold = float(ck.get("threshold_logit", 0.0))
            return
        if self.kind == "ctc":  # frame-level CTC heads (phoneme_heads.py), greedy decoding to ZIPA tokens
            from .phoneme_heads import build_head
            tokens = config_dir / cfg.get("tokens", "zipa_tokens.txt")
            self.labels = [l.split()[0] for l in open(tokens, encoding="utf-8")]
            for rel in self.head_files:
                st = torch.load(head_dir / rel, map_location="cpu", weights_only=False)
                h = build_head(st["config"])
                h.load_state_dict(st["state_dict"])
                self.heads.append(h.to(device).eval())
            return
        if self.kind == "ctc_asr":  # CTC ASR head (ctc_asr.py), greedy decoding to SentencePiece text
            from .ctc_asr import CtcAsrTask
            self.asr = CtcAsrTask(cfg, head_dir, device)
            self.heads = self.asr.heads
            return
        for rel in self.head_files:
            head, ckpt = load_head(head_dir / rel)
            if self.labels is None:
                self.labels = list(ckpt["labels"])
            elif list(ckpt["labels"]) != self.labels:
                raise ValueError(f"{rel}: label order differs from the task's first head")
            self.heads.append(head.to(device).eval())
        # optional merged label set (e.g. 7 emotions -> anger / neutral / sadness / others): the predicted label
        # (argmax) is mapped; merged probabilities are the sums of their members
        self.merge = cfg.get("merge")
        if self.merge:
            classes = list(self.merge["classes"])
            m = self.merge.get("map", {})
            self.merge_classes = classes
            self.merge_idx = np.array([classes.index(m.get(l, self.merge.get("default", "others"))) for l in self.labels])
        self.lid7 = cfg.get("lid7")
        if self.lid7:
            self.targets = list(self.lid7["targets"])
            self.classes7 = self.targets + ["others"]
            self.t_idx = [self.labels.index(t) for t in self.targets]
            self.bias7 = np.zeros(7)
            self.bias7[6] = float(self.lid7["others_log_bias"])

    def ctc(self, feats, valid) -> List[str]:
        """Greedy CTC over the mean log-softmax of the task's heads (they must share the upsampling factor)."""
        outs = [h(feats, valid) for h in self.heads]
        lp = torch.stack([torch.log_softmax(o[0].float(), -1) for o in outs]).mean(0)
        pred, lens = lp.argmax(-1).cpu().numpy(), outs[0][1].sum(1).cpu().numpy()
        hyps = []
        for p, n in zip(pred, lens):
            p = p[:n]
            keep = np.ones(len(p), bool)
            keep[1:] = p[1:] != p[:-1]
            # 0-2: blank, sos/eos, unk; 3 = ZIPA word boundary "▁" -> "|" (only heads trained with boundaries emit it)
            hyps.append(" ".join("|" if t == 3 else self.labels[t] for t in p[keep] if t >= 3).strip("| ").replace("| |", "|"))
        return hyps

    def turn(self, feats, valid) -> np.ndarray:
        """Shared-pass adapter of the Smart Turn head: packed frames of the head's encoder taps -> (B,) logit of
        P(turn complete). On CUDA the head runs under bf16 autocast, as in training. Elsewhere autocast is off, so
        the bf16 encoder taps are cast to fp32 to match the fp32 head weights."""
        h = self.heads[0]
        x = feats[h.taps][:, valid].transpose(0, 1)
        if not x.is_cuda:
            x = x.float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return h(x, valid.sum(1)).float().cpu().numpy()

    def logits(self, feats, valid) -> List[torch.Tensor]:
        return [h(feats, valid).float() for h in self.heads]

    @staticmethod
    def ensemble(logits: Sequence[torch.Tensor]) -> torch.Tensor:
        lp = torch.stack([torch.log_softmax(l.double(), -1) for l in logits]).mean(0)
        return lp - torch.logsumexp(lp, -1, keepdim=True)

    def decode(self, logp, topk: int) -> dict:
        if self.kind == "vad":
            return self.vad.decode(logp)
        if self.kind == "ctc":
            return {"phones": logp, "ipa": logp.replace(" ", "").replace("|", " ")}
        if self.kind == "ctc_asr":
            return {"text": logp}
        if self.kind == "turn":
            return {"complete": bool(logp > self.threshold), "prob": round(float(1 / (1 + np.exp(-logp))), 4)}
        p = np.exp(logp)
        order = np.argsort(-p)
        out = {"label": self.labels[order[0]], "prob": round(float(p[order[0]]), 4)}
        if self.merge:
            pm = np.zeros(len(self.merge_classes))
            np.add.at(pm, self.merge_idx, p)
            out["merged"] = {"label": self.merge_classes[int(self.merge_idx[order[0]])],
                             "probs": {c: round(float(v), 4) for c, v in zip(self.merge_classes, pm)}}
        if not self.lid7:
            out["probs"] = {l: round(float(v), 4) for l, v in zip(self.labels, p)}
        else:
            out["top"] = [[self.labels[j], round(float(p[j]), 4)] for j in order[:topk]]
            p7 = np.append(p[self.t_idx], max(0.0, 1.0 - p[self.t_idx].sum()))
            s7 = np.log(np.clip(p7, 1e-12, None)) + self.bias7
            q7 = np.exp(s7 - s7.max())
            q7 /= q7.sum()
            out["lid7"] = self.classes7[int(s7.argmax())]
            out["lid7_probs"] = {c: round(float(v), 4) for c, v in zip(self.classes7, q7)}
        return out


class OmniHead:
    """Frozen backbone + the heads of one config.

    config:      shipped config name (see ``omnihead.list_configs()``) or path to a config json
    device:      torch device string (default: cuda:0 if available, else cpu)
    tasks:       subset of the config's tasks
    models_root: head weights root (default: $OMNIHEAD_MODELS, else the repository's models/ folder)
    backbone:    backbone weights, local dir or HF id (default: $OMNIHEAD_BACKBONE, else the config's model_id)
    dtype:       encoder dtype override (default: the config's, bfloat16, which the heads were trained on)
    """

    def __init__(self, config: str = "default", device: Optional[str] = None, tasks: Optional[Sequence[str]] = None,
                 budget_sec: float = 320.0, max_bs: int = 64, models_root: Optional[str] = None,
                 backbone: Optional[str] = None, dtype: Optional[str] = None):
        path = config_path(config)
        self.cfg = json.loads(path.read_text())
        self.device = torch.device(device or default_device())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.backbone = self.cfg.get("backbone", DEFAULT_BACKBONE)
        if self.backbone not in ENCODER_BUILDERS:
            raise ValueError(f"backbone {self.backbone!r} has no encoder builder; known: {list(ENCODER_BUILDERS)}")
        head_dir = backbone_dir(self.backbone, models_root)
        enc = self.cfg["encoder"]
        self.backbone_weights, _ = resolve_backbone_weights(enc, backbone)
        self.encoder = build_frozen_encoder(ENCODER_BUILDERS[self.backbone], self.backbone_weights,
                                            getattr(torch, dtype or enc.get("dtype", "bfloat16")))
        self.encoder.to(self.device).eval()
        names = list(tasks) if tasks else list(self.cfg["tasks"])
        unknown = [n for n in names if n not in self.cfg["tasks"]]
        if unknown:
            raise ValueError(f"tasks {unknown} not in config {self.cfg['name']!r}: {list(self.cfg['tasks'])}")
        self.tasks = {n: Task(n, self.cfg["tasks"][n], head_dir, path.parent, self.device) for n in names}
        self.max_len = max(t.max_len for t in self.tasks.values())
        # a "last"-crop task (turn) needs the clip's end and a "chunks" task (VAD) all of it, so inputs are
        # not pre-cropped from the centre
        self.precrop = all(t.crop not in ("last", "chunks") for t in self.tasks.values())
        self.budget = int(budget_sec * SAMPLE_RATE)
        self.max_bs = max_bs

    @torch.inference_mode()
    def forward(self, waves: Sequence[torch.Tensor], keep_head_logits: bool = False):
        """waves: 1-D 16 kHz tensors. Returns {task: per-clip output} (classification: (N, C) ensemble log-probs;
        ctc / ctc_asr: strings; turn: logits; vad: 13 fps frame logits) and, if asked,
        {task: [(N, C) raw logits per head]}."""
        N = len(waves)
        items = []  # (clip, crop length, crop mode, tasks served by that crop)
        for i, w in enumerate(waves):
            crops: Dict[tuple, List[str]] = {}
            for name, t in self.tasks.items():
                if t.crop == "chunks" and w.numel() > t.max_len:   # consecutive max_len chunks cover the clip
                    for s in range(0, w.numel(), t.max_len):
                        crops.setdefault((min(w.numel() - s, t.max_len), f"seg{s}"), []).append(name)
                    continue
                mode = t.crop if w.numel() > t.max_len else "full"
                crops.setdefault((min(w.numel(), t.max_len), mode), []).append(name)
            items += [(i, n, mode, names) for (n, mode), names in crops.items()]
        items.sort(key=lambda x: x[1])
        batches, cur, longest = [], [], 0
        for it in items:
            if cur and ((len(cur) + 1) * max(longest, it[1]) > self.budget or len(cur) >= self.max_bs):
                batches.append(cur)
                cur, longest = [], 0
            cur.append(it)
            longest = max(longest, it[1])
        if cur:
            batches.append(cur)

        logp = {n: None for n in self.tasks}
        raw = {n: [None] * len(t.heads) for n, t in self.tasks.items()}
        for batch in batches:
            ws = [crop(waves[i], n, mode) for i, n, mode, _ in batch]
            lens = torch.tensor([w.numel() for w in ws])
            x = torch.zeros(len(ws), int(lens.max()))
            for j, w in enumerate(ws):
                x[j, :w.numel()] = w
            feats, valid = self.encoder(x.to(self.device, non_blocking=True), lens.to(self.device))
            for name, t in self.tasks.items():
                rows = [j for j, (_, _, _, names) in enumerate(batch) if name in names]
                if not rows:
                    continue
                clips = np.array([batch[j][0] for j in rows])
                if len(rows) == len(batch):
                    f, v = feats, valid
                else:
                    sel = torch.tensor(rows, device=self.device)
                    f, v = feats.index_select(1, sel), valid.index_select(0, sel)
                if t.kind == "vad":  # per clip {chunk start: frame logits}, joined in order below
                    if logp[name] is None:
                        logp[name] = [dict() for _ in range(N)]
                    for j, lg in zip(rows, t.vad.logits(self.encoder, f, v)):
                        mode = batch[j][2]
                        logp[name][batch[j][0]][int(mode[3:]) if mode.startswith("seg") else 0] = lg
                    continue
                if t.kind == "turn":
                    if logp[name] is None:
                        logp[name] = np.zeros(N)
                    logp[name][clips] = t.turn(f, v)
                    continue
                if t.kind in ("ctc", "ctc_asr"):
                    if logp[name] is None:
                        logp[name] = np.empty(N, dtype=object)
                    hyps = t.ctc(f, v) if t.kind == "ctc" else t.asr.ctc_asr(f, v)
                    for c, hyp in zip(clips, hyps):
                        logp[name][c] = hyp
                    continue
                lg = t.logits(f, v)
                lp = Task.ensemble(lg).cpu().numpy()
                if logp[name] is None:
                    logp[name] = np.zeros((N, lp.shape[1]))
                logp[name][clips] = lp
                if keep_head_logits:
                    for k, l in enumerate(lg):
                        if raw[name][k] is None:
                            raw[name][k] = np.zeros((N, l.shape[1]), np.float32)
                        raw[name][k][clips] = l.cpu().numpy()
        for name, t in self.tasks.items():
            if t.kind == "vad" and logp[name] is not None:
                logp[name] = [np.concatenate([d[k] for k in sorted(d)]) for d in logp[name]]
        return (logp, raw) if keep_head_logits else logp

    def predict(self, waves: Sequence[torch.Tensor], topk: int = 5) -> List[dict]:
        """waves: 1-D float tensors (or arrays) at 16 kHz, mono. Returns one dict per clip, keyed by task."""
        waves = [torch.as_tensor(w).float().flatten() for w in waves]
        if self.precrop:
            waves = [center_crop(w, self.max_len) for w in waves]
        logp = self.forward(waves)
        return [{name: t.decode(logp[name][i], topk) for name, t in self.tasks.items()} for i in range(len(waves))]

    def predict_files(self, paths: Sequence[str], topk: int = 5) -> List[dict]:
        out = []
        for p in paths:
            w = load_audio(p)
            r = self.predict([w], topk)[0]
            out.append({"audio": str(p), "duration_sec": round(w.numel() / SAMPLE_RATE, 2), **r})
        return out


class Clips(Dataset):
    def __init__(self, rows, max_len):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            w = load_audio(self.rows[i]["audio_path"])
            n, err = w.numel(), None if w.numel() > 0 else "empty audio"
        except Exception as e:  # noqa: BLE001
            w, n, err = torch.zeros(0), 0, f"{type(e).__name__}: {e}"
        # center crop to the longest task crop here (nested center crops equal direct ones); max_len 0 = keep all
        return i, (center_crop(w, self.max_len) if self.max_len else w), n, err


def collect_inputs(a) -> List[dict]:
    rows = []
    if a.manifest:
        for l in open(a.manifest, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                rows.append({"id": r.get("utt_id", r["audio_path"]), "audio_path": r["audio_path"]})
    if a.list:
        rows += [{"id": l.strip(), "audio_path": l.strip()} for l in open(a.list, encoding="utf-8") if l.strip()]
    for p in a.inputs:
        p = Path(p)
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT) if p.is_dir() else [p]
        rows += [{"id": str(f), "audio_path": str(f)} for f in files]
    return rows


def one_thread(_):
    torch.set_num_threads(1)


def chunks(loader, size):
    """Small parallel loader batches -> chunks of `size` clips for one length-sorted forward."""
    buf = []
    for b in loader:
        buf += b
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="omnihead", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="audio files or directories")
    ap.add_argument("--manifest", default="", help="jsonl with audio_path (and utt_id)")
    ap.add_argument("--list", default="", help="text file, one audio path per line")
    ap.add_argument("--config", default="default", help=f"{' | '.join(list_configs())} | path to a config json")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of the config's tasks")
    ap.add_argument("--out", default="", help="output jsonl (default: stdout)")
    ap.add_argument("--save_logits", default="", help="npz with per-head raw logits and ensemble log-probs")
    ap.add_argument("--device", default=None, help="torch device (default: cuda:0 if available, else cpu)")
    ap.add_argument("--models", default=None, help="head weights root (default: $OMNIHEAD_MODELS or repo models/)")
    ap.add_argument("--backbone", default=None,
                    help="backbone weights, local dir or HF id (default: $OMNIHEAD_BACKBONE or the config's model_id)")
    ap.add_argument("--dtype", default=None, help="encoder dtype override (default: config, bfloat16)")
    ap.add_argument("--budget_sec", type=float, default=320.0, help="audio seconds per encoder batch")
    ap.add_argument("--chunk", type=int, default=1024, help="clips loaded before each length-sorted forward")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--list_configs", action="store_true", help="print the shipped config names and exit")
    return ap


def main(argv: Optional[Sequence[str]] = None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.list_configs:
        print("\n".join(list_configs()))
        return

    rows = collect_inputs(a)
    if not rows:
        ap.error("no inputs")
    t0 = time.time()
    model = OmniHead(a.config, a.device, a.tasks, a.budget_sec, models_root=a.models, backbone=a.backbone,
                     dtype=a.dtype)
    print(f"[omnihead] config {model.cfg['name']} | backbone {model.backbone} | tasks {list(model.tasks)} | "
          f"heads {sum(len(t.heads) for t in model.tasks.values())} | {len(rows)} clips | device {model.device} | "
          f"load {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    loader = DataLoader(Clips(rows, model.max_len if model.precrop else 0), batch_size=16, shuffle=False,
                        num_workers=a.num_workers, collate_fn=list, worker_init_fn=one_thread,
                        prefetch_factor=4 if a.num_workers else None)
    fo = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    keep = bool(a.save_logits)
    saved = {}
    t1, done, audio_sec = time.time(), 0, 0.0
    for chunk in chunks(loader, a.chunk):
        good = [(i, w, n) for i, w, n, err in chunk if err is None]
        res = model.forward([w for _, w, _ in good], keep) if good else None
        logp, raw = (res if keep else (res, None)) if good else (None, None)
        k = 0
        for i, w, n, err in chunk:
            rec = {"id": rows[i]["id"], "audio": rows[i]["audio_path"], "duration_sec": round(n / SAMPLE_RATE, 2)}
            if err is not None:
                rec["error"] = err
            else:
                rec.update({name: t.decode(logp[name][k], a.topk) for name, t in model.tasks.items()})
                if keep:
                    for name, t in model.tasks.items():
                        if t.kind in ("ctc", "ctc_asr", "turn", "vad"):
                            continue
                        if name not in saved:
                            saved[name] = {"logp": np.full((len(rows), len(t.labels)), np.nan, np.float32),
                                           "heads": [np.full((len(rows), len(t.labels)), np.nan, np.float32) for _ in t.heads]}
                        saved[name]["logp"][i] = logp[name][k]
                        for h in range(len(t.heads)):
                            saved[name]["heads"][h][i] = raw[name][h][k]
                k += 1
                audio_sec += n / SAMPLE_RATE
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(chunk)
        print(f"[omnihead] {done}/{len(rows)} clips, {done / (time.time() - t1):.1f} clips/s", file=sys.stderr, flush=True)
    if a.out:
        fo.close()
    if keep:
        arrays = {}
        for name, t in model.tasks.items():
            if name not in saved:
                continue
            arrays[f"{name}__logp"] = saved[name]["logp"]
            for f, h in zip(t.head_files, saved[name]["heads"]):
                arrays[f"{name}__{Path(f).stem}"] = h
        np.savez_compressed(a.save_logits, ids=np.array([r["id"] for r in rows]), **arrays)
    el = time.time() - t1
    print(f"[omnihead] done: {len(rows)} clips, {audio_sec / 3600:.2f} h audio in {el:.0f}s "
          f"({len(rows) / el:.1f} clips/s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
