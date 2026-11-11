"""16-bit PCM helpers. Uses stdlib audioop, so Python 3.13+ is not supported."""

from __future__ import annotations

import array
import audioop
import io
import wave


def downmix_s16(pcm: bytes, channels: int) -> bytes:
    if channels < 1:
        raise ValueError("channels must be >= 1")
    width = 2 * channels
    usable = len(pcm) - (len(pcm) % width)
    pcm = pcm[:usable]
    if not pcm:
        return b""
    if channels == 1:
        return pcm
    if channels == 2:
        return audioop.tomono(pcm, 2, 0.5, 0.5)
    samples = array.array("h")
    samples.frombytes(pcm)
    frames = len(samples) // channels
    out = array.array("h", [0]) * frames
    for i in range(frames):
        base = i * channels
        acc = 0
        for ch in range(channels):
            acc += int(samples[base + ch])
        out[i] = int(acc / channels)
    return out.tobytes()


class Resampler:
    """Convert interleaved s16 PCM to 16 kHz mono, keeping ratecv state."""

    def __init__(self, in_rate: int, out_rate: int = 16000):
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("sample rate must be positive")
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._state = None

    def push(self, pcm_s16: bytes, channels: int) -> bytes:
        mono = downmix_s16(pcm_s16, channels)
        if not mono or self.in_rate == self.out_rate:
            return mono
        converted, self._state = audioop.ratecv(
            mono, 2, 1, self.in_rate, self.out_rate, self._state
        )
        return converted


def pcm16_to_wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buf.getvalue()


def rms_s16(pcm: bytes) -> int:
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    if len(usable) < 2:
        return 0
    return int(audioop.rms(usable, 2))


def peak_s16(pcm: bytes) -> int:
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    if len(usable) < 2:
        return 0
    return int(audioop.max(usable, 2))


def strip_overlap(previous: str, current: str, max_chars: int = 80) -> str:
    prev = (previous or "").strip()
    cur = (current or "").strip()
    if not prev or not cur:
        return cur
    limit = min(len(prev), len(cur), max_chars)
    for size in range(limit, 3, -1):
        if prev.endswith(cur[:size]):
            return cur[size:].lstrip(" ，,。.")
    return cur
