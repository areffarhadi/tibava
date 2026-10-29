"""Qwen3-ASR-1.7B transcription and Qwen3-ForcedAligner-0.6B timestamps.

The speech-emotion heads were trained on the Qwen3-ASR-1.7B audio
encoder, so ASR uses that same checkpoint. Forced alignment stays on
Qwen3-ForcedAligner-0.6B, which is the only aligner Qwen ships.
"""

from typing import List, Optional, Sequence, Tuple

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

# Qwen3-ASR language names. Keys are the codes the old WhisperX plugin accepted.
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

NO_SPACE_LANGUAGES = {"Chinese", "Cantonese", "Japanese", "Thai"}


def resolve_language(language_code: Optional[str]) -> Optional[str]:
    if language_code is None:
        return None
    code = str(language_code).strip()
    if not code or code.lower() in {"none", "auto", "null"}:
        return None
    if code.lower() in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code.lower()]
    if code in LANGUAGE_NAMES.values():
        return code
    titled = code[:1].upper() + code[1:]
    if titled in LANGUAGE_NAMES.values():
        return titled
    return None


def load_asr(device: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=2048,
        forced_aligner=ALIGNER_MODEL,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
    )


def _align_items(transcription) -> list:
    stamps = getattr(transcription, "time_stamps", None)
    if stamps is None:
        return []
    items = getattr(stamps, "items", None)
    if items is None and isinstance(stamps, Sequence) and not isinstance(stamps, (str, bytes)):
        items = stamps
    return list(items or [])


def _join_words(words: Sequence[str], language: str) -> str:
    if language in NO_SPACE_LANGUAGES:
        return "".join(words).strip()
    return " ".join(w for w in words if w).strip()


def turns_from_transcription(
    transcription,
    duration: float,
    gap: float = 0.6,
    max_duration: float = 15.0,
) -> List[Tuple[float, float, str]]:
    """Group aligned words into short turns. One speaker: Qwen3-ASR does not diarize."""
    language = getattr(transcription, "language", "") or ""
    items = _align_items(transcription)
    text = (getattr(transcription, "text", "") or "").strip()
    if not items:
        if not text:
            return []
        return [(0.0, max(duration, 0.01), text)]

    groups: List[list] = []
    current: list = []
    for item in items:
        if not current:
            current = [item]
            continue
        pause = float(item.start_time) - float(current[-1].end_time)
        span = float(item.end_time) - float(current[0].start_time)
        if pause >= gap or span >= max_duration:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        groups.append(current)

    turns = []
    for group in groups:
        words = [str(item.text).strip() for item in group if str(item.text).strip()]
        joined = _join_words(words, language)
        if not joined:
            continue
        start = max(0.0, float(group[0].start_time))
        end = min(duration, float(group[-1].end_time)) if duration else float(group[-1].end_time)
        if end <= start:
            end = start + 0.01
        turns.append((start, end, joined))
    if len(turns) == 1 and text:
        start, end, _ = turns[0]
        turns = [(start, end, text)]
    return turns


def transcribe(model, samples, sample_rate: int, language: Optional[str] = None):
    try:
        return model.transcribe(
            audio=(samples, sample_rate),
            language=language,
            return_time_stamps=True,
        )[0]
    except Exception:
        return model.transcribe(
            audio=(samples, sample_rate),
            language=language,
            return_time_stamps=False,
        )[0]
