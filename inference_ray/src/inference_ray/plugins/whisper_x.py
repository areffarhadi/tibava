from inference_ray.plugin import AnalyserPlugin, AnalyserPluginManager
from inference_ray.qwen_audio import load_asr, resolve_language, transcribe, turns_from_transcription
from tibava_data import AudioData, AnnotationData, ListData, Annotation
from tibava_data import DataManager, Data

from typing import Callable, Dict
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model": "Qwen/Qwen3-ASR-1.7B",
    "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
}

default_parameters = {"language_code": None}

requires = {
    "audio": AudioData,
}

provides = {
    "annotations": ListData,
}


@AnalyserPluginManager.export("whisper_x")
class WhisperX(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.2",
    requires=requires,
    provides=provides,
):
    """Same plugin id as before so emotion, gender, NER, POS and sentiment keep calling it.

    Qwen3-ASR does not diarize. Turns are pause-separated slices of one speaker,
    with word times from Qwen3-ForcedAligner-0.6B.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.model = None

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import librosa
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model = load_asr(device)

        language = resolve_language((parameters or {}).get("language_code"))

        with (
            inputs["audio"] as input_data,
            data_manager.create_data("ListData") as output_data,
        ):
            with input_data.open_audio("r") as f_audio:
                samples, sr = librosa.load(f_audio, sr=16000, mono=True)
                result = transcribe(self.model, samples, sr, language)
                duration = len(samples) / float(sr) if sr else 0.0
                turns = turns_from_transcription(result, duration)

                with output_data.create_data("AnnotationData") as ann_data:
                    for start, end, text in turns:
                        ann_data.annotations.append(
                            Annotation(start=start, end=end, labels=[text])
                        )
                    ann_data.name = "speaker"

                logging.info(
                    "Qwen3-ASR language=%s turns=%s",
                    getattr(result, "language", ""),
                    len(turns),
                )
                self.update_callbacks(callbacks, progress=1.0)
                return {"annotations": output_data}
