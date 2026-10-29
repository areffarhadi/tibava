"""CLI: python -m omnihead.diarization a.wav b.wav --out_dir rttm/ [--embedding X.onnx] [--json]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="OmniHead speaker diarization (backbone qwen3-asr-1.7b)")
    ap.add_argument("audio", nargs="+", help="audio files (any rate / channels; downmixed and resampled to 16 kHz)")
    ap.add_argument("--out_dir", default=".", help="one <stem>.rttm per input")
    ap.add_argument("--checkpoint", default=None, help="default: models/qwen3-asr-1.7b/diarization/diar_qwen3asr_v1.pt")
    ap.add_argument("--embedding", default=None, help="omniasv-var (default), omniasv-4s, or an external embedder ONNX")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_speakers", type=int, default=None)
    ap.add_argument("--step_sec", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min_cluster_size", type=int, default=20)
    ap.add_argument("--keep_overlap", action="store_true", help="embed overlapped frames too (default: excluded)")
    ap.add_argument("--json", action="store_true", help="also write <stem>.json with time-ordered segments")
    a = ap.parse_args(argv)

    from .pipeline import Diarizer
    d = Diarizer(a.checkpoint, a.device, a.embedding, step_sec=a.step_sec, threshold=a.threshold,
                 min_cluster_size=a.min_cluster_size, embedding_exclude_overlap=not a.keep_overlap)
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.audio:
        stem = Path(path).stem
        ann = d(path, uri=stem, num_speakers=a.num_speakers)
        Path(a.out_dir, f"{stem}.rttm").write_text(ann.to_rttm())
        if a.json:
            segs = sorted(({"speaker": s, "start": round(t.start, 3), "end": round(t.end, 3)}
                           for t, _, s in ann.itertracks(yield_label=True)), key=lambda x: (x["start"], x["speaker"]))
            Path(a.out_dir, f"{stem}.json").write_text(json.dumps(
                {"uri": stem, "num_speakers": len(ann.labels()), "segments": segs}, indent=1))
        print(f"{path}: {len(ann.labels())} speakers -> {Path(a.out_dir, stem + '.rttm')}", flush=True)


if __name__ == "__main__":
    main()
