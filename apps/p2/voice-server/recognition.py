"""Streaming Parakeet Unified English recognition for one microphone at a time per connection."""

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sherpa_onnx

from segmenter import Segmenter

SAMPLE_RATE = 16000


def load_recognizer(model: Path) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(model / "encoder.int8.onnx"),
        decoder=str(model / "decoder.int8.onnx"),
        joiner=str(model / "joiner.int8.onnx"),
        tokens=str(model / "tokens.txt"),
        num_threads=int(os.environ.get("VOICE_ASR_THREADS", "4")),
        sample_rate=SAMPLE_RATE,
        feature_dim=128,
        model_type="nemo_transducer",
    )


class Microphone:
    """Silero decides turns; Parakeet decodes each audio chunk just once, with its encoder state carried forward."""

    def __init__(
        self,
        recognizer: sherpa_onnx.OnlineRecognizer,
        config: Callable[[float], sherpa_onnx.VadModelConfig],
        preroll_ms: float,
    ):
        self.recognizer = recognizer
        self.config = config
        self.preroll_ms = preroll_ms
        self.silence = 0.8
        self.next_silence = self.silence
        self.segmenter = Segmenter(config(self.silence), self.preroll_ms)
        self.stream = None
        self.last_text = ""

    def set_pause(self, seconds: float) -> None:
        self.next_silence = seconds
        if not self.segmenter.speaking:
            self.apply_pause()

    def reset(self) -> None:
        self.segmenter.reset()
        self.stream = None
        self.last_text = ""
        self.apply_pause()

    def feed(self, pcm: bytes) -> list[dict]:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        events = []
        started = time.monotonic()
        for event, audio in self.segmenter.stream(samples):
            if event == "speech_start":
                self.stream = self.recognizer.create_stream()
                self.last_text = ""
                self.stream.accept_waveform(SAMPLE_RATE, audio)
                events.append({"type": "speech_start"})
            elif event == "speech_audio" and self.stream is not None:
                self.stream.accept_waveform(SAMPLE_RATE, audio)
            elif event == "speech_end":
                if self.stream is None:
                    # The VAD may split on its maximum duration without another speech_start.
                    self.stream = self.recognizer.create_stream()
                    self.stream.accept_waveform(SAMPLE_RATE, audio)
                    events.append({"type": "speech_start"})
                # The encoder needs trailing samples to emit its last chunk, even after input_finished.
                self.stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
                self.stream.input_finished()
                text = self.decode()
                self.stream = None
                events.append({"type": "speech_end"})
                events.append({"type": "final", "text": text, "ms": elapsed_ms(started)})
        if self.stream is not None:
            text = self.decode()
            if text and text != self.last_text:
                self.last_text = text
                events.append({"type": "interim", "text": text, "ms": elapsed_ms(started)})
        if not self.segmenter.speaking:
            self.apply_pause()
        return events

    def decode(self) -> str:
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return self.recognizer.get_result(self.stream)

    def apply_pause(self) -> None:
        if self.silence != self.next_silence:
            self.silence = self.next_silence
            self.segmenter = Segmenter(self.config(self.silence), self.preroll_ms)


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
