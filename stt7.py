"""stt7 — ਲੋਕਲ ਆਵਾਜ਼ → ਲਿਖਤ (local speech-to-text, nothing leaves the Mac).

The browser's Web Speech API sends audio to Apple/Google and Safari has no
Punjabi (pa-IN) recogniser, so the mic silently failed. Instead the page
records audio and posts it here:

  ਪੰਜਾਬੀ  -> Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10 (Gurmukhi out, <1 s)
  English -> faster-whisper "small" (int8, CPU)

The language is chosen explicitly by the user (no auto-detect): Whisper's
language-ID mislabels Punjabi as Urdu/Hindi and English as Punjabi often
enough to be useless for routing. Models load lazily on first use.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading

import numpy as np

PUNJABI_W2V = "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10"
WHISPER_SIZE = "small"
MAX_AUDIO_BYTES = 15 * 1024 * 1024       # ~ a few minutes of opus; refuse anything bigger
SAMPLE_RATE = 16000


class STTError(Exception):
    pass


def decode_audio(data: bytes) -> np.ndarray:
    """Any browser recording (webm/opus, mp4/aac, wav) -> float32 mono 16 kHz."""
    if not data:
        raise STTError("ਖਾਲੀ ਆਡੀਓ")
    if len(data) > MAX_AUDIO_BYTES:
        raise STTError("ਆਡੀਓ ਬਹੁਤ ਲੰਮਾ")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise STTError("ffmpeg ਨਹੀਂ ਮਿਲਿਆ (brew install ffmpeg)")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        raise STTError("ਆਡੀਓ decode ਨਹੀਂ ਹੋਇਆ: " + proc.stderr.decode(errors="ignore")[-200:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE // 4:
        raise STTError("ਆਡੀਓ ਬਹੁਤ ਛੋਟਾ — ਥੋੜ੍ਹਾ ਲੰਮਾ ਬੋਲੋ")
    return audio


class LocalSTT:
    def __init__(self):
        self._lock = threading.Lock()
        self._whisper = None
        self._w2v = None

    def transcribe_pcm(self, pcm16_bytes: bytes, lang: str) -> str:
        """Raw PCM16 mono 16 kHz bytes -> text. No ffmpeg (already decoded).

        Same engines and serialisation as transcribe(); the browser's live
        mic path hands us PCM directly (7 Live offline mode, SPEC E8).
        """
        if not pcm16_bytes:
            raise STTError("ਖਾਲੀ ਆਡੀਓ")
        lang = "pa" if lang.lower().startswith("pa") else "en"
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < SAMPLE_RATE // 4:
            raise STTError("ਆਡੀਓ ਬਹੁਤ ਛੋਟਾ — ਥੋੜ੍ਹਾ ਲੰਮਾ ਬੋਲੋ")
        with self._lock:
            return self._punjabi(audio) if lang == "pa" else self._english(audio)

    def _punjabi(self, audio: np.ndarray) -> str:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if self._w2v is None:
            self._w2v = (Wav2Vec2Processor.from_pretrained(PUNJABI_W2V),
                         Wav2Vec2ForCTC.from_pretrained(PUNJABI_W2V).eval())
        proc, model = self._w2v
        with torch.no_grad():
            logits = model(proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values).logits
        text = proc.batch_decode(logits.argmax(-1))[0].replace("<s>", "")
        return re.sub(r"\s+", " ", text).strip()

    def _english(self, audio: np.ndarray) -> str:
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
        segments, _ = self._whisper.transcribe(audio, language="en", vad_filter=True, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe(self, data: bytes, lang: str) -> dict:
        lang = "pa" if lang.lower().startswith("pa") else "en"
        audio = decode_audio(data)
        with self._lock:                 # one utterance at a time; models aren't thread-safe
            text = self._punjabi(audio) if lang == "pa" else self._english(audio)
        return {"text": text, "lang": lang, "seconds": round(audio.size / SAMPLE_RATE, 2),
                "engine": "wav2vec2-pa" if lang == "pa" else f"whisper-{WHISPER_SIZE}"}
