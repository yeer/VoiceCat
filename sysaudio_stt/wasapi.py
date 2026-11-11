"""Windows system-audio capture through WASAPI loopback.

The macOS helper is bin/sysaudio-capture. On Windows the same session
interface is provided by pyaudiowpatch, which must be installed there.
"""

from __future__ import annotations

import queue
import sys

from sysaudio_stt.capture import CaptureError


def _audio():
    if sys.platform != "win32":
        raise CaptureError("WASAPI 只在 Windows 上可用")
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise CaptureError("Windows 需要先安装 pyaudiowpatch：python -m pip install pyaudiowpatch") from exc
    return pyaudio


def loopback_devices():
    pyaudio = _audio()
    audio = pyaudio.PyAudio()
    try:
        rows = []
        for item in audio.get_loopback_device_info_generator():
            rows.append({
                "name": str(item.get("name") or ""),
                "uid": str(item.get("index")),
                "index": int(item["index"]),
                "rate": int(item.get("defaultSampleRate") or 48000),
                "channels": int(item.get("maxInputChannels") or 2),
                "rms": 0,
                "peak": 0,
                "type": "output",
            })
        return rows
    finally:
        audio.terminate()


def scan_loopback():
    return loopback_devices()


class WasapiLoopback:
    def __init__(self, device=""):
        self.requested = device or ""
        self.device_label = self.requested or "系统输出"
        self.rate = 48000
        self.channels = 2
        self._audio = None
        self._stream = None
        self._queue = queue.Queue(maxsize=64)
        self._closed = False

    def start(self):
        pyaudio = _audio()
        audio = pyaudio.PyAudio()
        chosen = None
        for item in audio.get_loopback_device_info_generator():
            name = str(item.get("name") or "")
            if not self.requested or self.requested in name or self.requested == str(item.get("index")):
                chosen = item
                break
        if chosen is None:
            audio.terminate()
            raise CaptureError("没有找到 Windows 环回输出设备")
        self.rate = int(chosen.get("defaultSampleRate") or 48000)
        self.channels = max(1, int(chosen.get("maxInputChannels") or 2))
        self.device_label = str(chosen.get("name") or self.device_label)

        def callback(in_data, frame_count, time_info, status):
            del frame_count, time_info, status
            if not self._closed:
                try:
                    self._queue.put_nowait(bytes(in_data))
                except queue.Full:
                    pass
            return (None, pyaudio.paContinue)

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=int(chosen["index"]),
            frames_per_buffer=1024,
            stream_callback=callback,
        )
        stream.start_stream()
        self._audio = audio
        self._stream = stream
        return self

    def read(self, nbytes, timeout=0.3):
        if self._closed:
            return b""
        chunks = bytearray()
        try:
            chunks.extend(self._queue.get(timeout=timeout))
        except queue.Empty:
            return None
        while len(chunks) < nbytes:
            try:
                chunks.extend(self._queue.get_nowait())
            except queue.Empty:
                break
        return bytes(chunks[:nbytes])

    def close(self):
        self._closed = True
        stream = self._stream
        audio = self._audio
        self._stream = None
        self._audio = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if audio is not None:
            try:
                audio.terminate()
            except Exception:
                pass
