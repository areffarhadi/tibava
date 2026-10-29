# 2026-10-30 — Qwen3-ASR speech, alignment, and emotion

## Speech recognition

Speech recognition uses Qwen3-ASR-1.7B. The plugin id stays `whisper`, so the transcript tab and word cloud still call it. The default timeline name is “Qwen3-ASR Transcript”.

## Forced alignment

Word times come from Qwen3-ForcedAligner-0.6B, loaded with the speech model. Words are grouped into turns: a new turn starts after a pause of at least 0.6 seconds, or when a turn would run longer than 15 seconds. The plugin id stays `whisper_x`. Turns use one speaker label. Speaker diarization is not included. Emotion, gender, named entities, part of speech, and sentiment still read these turns.

## Emotion

Emotion recognition uses the speech-emotion heads trained on the Qwen3-ASR-1.7B audio encoder. Labels, in order: anger, disgust, fear, happiness, neutral, sadness, surprise. The head weight files are not in this tree, so this path still needs those files before it can run.

Gender, sound classification, and active-speaker detection are unchanged.

## Check

Jennifer Cearns, “How people are using AI to grieve, love and connect” (14:01). Speech recognition finished in English with 168 segments on the timeline “Qwen3-ASR Transcript”.
