"""Energy VAD that emits speech segments for chunked transcription."""

from __future__ import annotations

import audioop


class Segmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        threshold: int = 300,
        start_frames: int = 3,
        silence_ms: int = 450,
        min_speech_ms: int = 200,
        max_speech_ms: int = 8000,
        preroll_ms: int = 240,
        tail_ms: int = 120,
        overlap_ms: int = 200,
    ):
        if frame_ms <= 0 or sample_rate <= 0:
            raise ValueError("invalid segmenter timing")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.threshold = threshold
        self.start_frames = max(1, start_frames)
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.min_speech_ms = min_speech_ms
        self.max_speech_frames = max(1, int(max_speech_ms / frame_ms))
        self.preroll_frames = max(1, int(preroll_ms / frame_ms))
        self.tail_frames = max(0, int(tail_ms / frame_ms))
        self.overlap_frames = max(0, int(overlap_ms / frame_ms))
        self.pending = bytearray()
        self.preroll: list[bytes] = []
        self.speech = bytearray()
        self.in_speech = False
        self.voiced_run = 0
        self.silence_run = 0
        self.voiced_ms = 0

    def push(self, pcm: bytes) -> list[bytes]:
        self.pending.extend(pcm)
        emitted: list[bytes] = []
        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[: self.frame_bytes])
            del self.pending[: self.frame_bytes]
            emitted.extend(self._on_frame(frame))
        return emitted

    def snapshot(self) -> bytes:
        if not self.in_speech:
            return b""
        return bytes(self.speech)

    def flush(self) -> list[bytes]:
        if not self.in_speech:
            self.preroll.clear()
            self.voiced_run = 0
            return []
        return self._finish(trim_silence=False)

    def _voiced(self, frame: bytes) -> bool:
        if not frame:
            return False
        return audioop.rms(frame, 2) >= self.threshold

    def _on_frame(self, frame: bytes) -> list[bytes]:
        voiced = self._voiced(frame)
        if not self.in_speech:
            self.preroll.append(frame)
            if len(self.preroll) > self.preroll_frames:
                self.preroll.pop(0)
            self.voiced_run = self.voiced_run + 1 if voiced else 0
            if self.voiced_run >= self.start_frames:
                self.in_speech = True
                self.speech = bytearray(b"".join(self.preroll))
                self.silence_run = 0
                self.voiced_ms = self.voiced_run * self.frame_ms
            return []

        self.speech.extend(frame)
        if voiced:
            self.silence_run = 0
            self.voiced_ms += self.frame_ms
        else:
            self.silence_run += 1
        speech_frames = len(self.speech) // self.frame_bytes
        if self.silence_run >= self.silence_frames:
            return self._finish(trim_silence=True)
        if speech_frames >= self.max_speech_frames:
            return self._split()
        return []

    def _finish(self, trim_silence: bool) -> list[bytes]:
        data = bytes(self.speech)
        voiced_ms = self.voiced_ms
        if trim_silence and self.silence_run > self.tail_frames:
            drop = (self.silence_run - self.tail_frames) * self.frame_bytes
            if drop < len(data):
                data = data[:-drop]
            else:
                data = b""
        self._reset_speech()
        if voiced_ms < self.min_speech_ms or not data:
            return []
        return [data]

    def _split(self) -> list[bytes]:
        data = bytes(self.speech)
        overlap = self.overlap_frames * self.frame_bytes
        if overlap and overlap < len(data):
            kept = data[-overlap:]
        else:
            kept = b""
        voiced_ms = self.voiced_ms
        self.speech = bytearray(kept)
        self.silence_run = 0
        self.voiced_run = self.overlap_frames
        self.voiced_ms = min(voiced_ms, self.overlap_frames * self.frame_ms)
        self.in_speech = True
        if voiced_ms < self.min_speech_ms or not data:
            return []
        return [data]

    def _reset_speech(self) -> None:
        self.in_speech = False
        self.speech = bytearray()
        self.preroll.clear()
        self.voiced_run = 0
        self.silence_run = 0
        self.voiced_ms = 0
