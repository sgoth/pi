"""Run against the packaged streaming model with `python test_recognition.py` (or set VOICE_MODELS)."""

import os
import unittest
import wave
from pathlib import Path

import sherpa_onnx
from recognition import SAMPLE_RATE, Microphone, load_recognizer

MODELS = Path(os.environ.get("VOICE_MODELS", "/models"))
MODEL = Path(os.environ.get("VOICE_PARAKEET_STREAMING", str(MODELS / "parakeet-streaming")))
FRAME = 320 * 2


def config(pause: float) -> sherpa_onnx.VadModelConfig:
    vad = sherpa_onnx.VadModelConfig()
    vad.silero_vad.model = str(MODELS / "silero_vad.onnx")
    vad.silero_vad.min_silence_duration = pause
    vad.num_threads = 1
    vad.sample_rate = SAMPLE_RATE
    return vad


def feed(mic: Microphone, pcm: bytes) -> list[dict]:
    return [event for offset in range(0, len(pcm), FRAME) for event in mic.feed(pcm[offset : offset + FRAME])]


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recognizer = load_recognizer(MODEL)
        with wave.open(str(MODEL / "test_wavs/0.wav"), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, SAMPLE_RATE)
            cls.speech = wav.readframes(wav.getnframes())

    def test_streams_words_before_sending_one_final(self) -> None:
        mic = Microphone(self.recognizer, config, 750)
        mic.set_pause(2)
        events = feed(mic, b"\0" * FRAME * 10 + self.speech)
        self.assertEqual(events[0]["type"], "speech_start")
        self.assertTrue(any(event["type"] == "interim" and event["text"] for event in events))
        events += feed(mic, b"\0" * FRAME * 150)
        self.assertEqual([event["type"] for event in events if event["type"] != "interim"],
                         ["speech_start", "speech_end", "final"])
        self.assertGreater(len(events[-1]["text"].split()), 4)
        self.assertEqual(feed(mic, b"\0" * FRAME * 100), [])

    def test_reset_discards_in_progress_transcript(self) -> None:
        mic = Microphone(self.recognizer, config, 750)
        mic.set_pause(2)
        events = feed(mic, self.speech)
        self.assertNotIn("final", [event["type"] for event in events])
        mic.reset()
        self.assertEqual(feed(mic, b"\0" * FRAME * 150), [])


if __name__ == "__main__":
    unittest.main()
