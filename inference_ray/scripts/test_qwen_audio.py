"""Run Qwen3-ASR-1.7B, the 0.6B forced aligner, and the speech-emotion head.

Usage:
  CUDA_VISIBLE_DEVICES=1 python inference_ray/scripts/test_qwen_audio.py
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference_ray" / "src"))

from inference_ray.qwen_audio import (  # noqa: E402
    load_asr,
    transcribe,
    turns_from_transcription,
)

SAMPLE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
SAMPLE_PATH = Path("/tmp/qwen3_asr_en.wav")
MODELS_ROOT = Path("/models/omnihead")
if not MODELS_ROOT.exists():
    MODELS_ROOT = ROOT / "data" / "omnihead"


def main():
    import torch

    if not SAMPLE_PATH.exists():
        print(f"downloading {SAMPLE_URL}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)

    import soundfile as sf

    samples, sr = sf.read(SAMPLE_PATH, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu'}")
    model = load_asr(device)
    result = transcribe(model, samples, sr, None)
    duration = len(samples) / float(sr)
    turns = turns_from_transcription(result, duration)
    print("language:", result.language)
    print("text:", result.text)
    print("turns:")
    for start, end, text in turns:
        print(f"  {start:.2f}-{end:.2f} {text}")
    if not result.text.strip():
        raise SystemExit("ASR returned empty text")
    if not turns:
        raise SystemExit("forced alignment produced no turns")

    head = MODELS_ROOT / "qwen3-asr-1.7b" / "ser" / "s1_qformer_tf.pt"
    if not head.is_file() or head.stat().st_size < 1024:
        print(f"emotion head not available at {head}; ASR and alignment only")
        return
    with head.open("rb") as handle:
        if handle.read(40).startswith(b"version https://git-lfs.github.com"):
            print(f"emotion head at {head} is a Git LFS pointer, not weights")
            return

    from inference_ray.plugins.audio_emotion import AudioEmotion

    plugin = AudioEmotion(config={"models_root": str(MODELS_ROOT), "save_dir": "/tmp/audio_emotion"})
    emotion = plugin._load_model(device)
    wave = torch.tensor(np.asarray(samples, dtype=np.float32))
    prediction = emotion.predict([wave])[0]["ser"]
    print("emotion:", prediction["label"], prediction["prob"])
    print("probs:", prediction["probs"])


if __name__ == "__main__":
    main()
