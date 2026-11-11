"""Spawn the native capture helper and read s16le PCM."""

from __future__ import annotations

import json
import logging
import select
import subprocess
import threading
import time
import wave
from pathlib import Path


class CaptureError(RuntimeError):
    pass


def parse_header(raw: bytes) -> dict[str, str]:
    text = raw.decode("utf-8", "replace").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "SYSAUDIO1":
        raise CaptureError("音频头无效")
    fields: dict[str, str] = {"magic": lines[0]}
    for line in lines[1:]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    if "rate" not in fields or "channels" not in fields:
        raise CaptureError("音频头缺少采样率或声道")
    return fields


def list_devices(binary: Path) -> list[dict]:
    proc = subprocess.run(
        [str(binary), "list"],
        check=False,
        capture_output=True,
        timeout=8,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise CaptureError(err or "列出音频设备失败")
    devices = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("type") == "device":
            devices.append(item)
    return devices


class CaptureSession:
    def __init__(self, binary: Path, source: str, device: str | None = None, uid: str | None = None, seconds: float = 0):
        self.binary = binary
        self.source = source
        self.device = device
        self.uid = uid
        self.seconds = seconds
        self.proc: subprocess.Popen | None = None
        self.rate = 0
        self.channels = 0
        self.encoding = ""
        self.device_label = ""
        self.logs: list[str] = []
        self._pending = bytearray()
        self._stderr_thread: threading.Thread | None = None

    def start(self, header_timeout: float = 20) -> "CaptureSession":
        cmd = [str(self.binary), "capture", "--source", self.source]
        if self.device:
            cmd.extend(["--device", self.device])
        if self.uid:
            cmd.extend(["--uid", self.uid])
        if self.seconds and self.seconds > 0:
            cmd.extend(["--seconds", str(self.seconds)])
        logging.info("启动采集: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._read_header(header_timeout)
        logging.info(
            "采集已连接 source=%s device=%s rate=%s channels=%s",
            self.source,
            self.device_label,
            self.rate,
            self.channels,
        )
        return self

    def _drain_stderr(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            text = line.decode("utf-8", "replace").rstrip()
            self.logs.append(text)
            logging.info("%s", text)

    def _readable(self, timeout: float) -> bool:
        if not self.proc or not self.proc.stdout:
            return False
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        return bool(ready)

    def _read_header(self, timeout: float) -> None:
        assert self.proc and self.proc.stdout
        buf = bytearray()
        deadline = time.monotonic() + timeout
        warned = False
        started = time.monotonic()
        while b"\n\n" not in buf:
            if time.monotonic() - started > 3 and not warned:
                logging.info("仍在等待采集启动。如果系统弹出权限，请允许后保持本程序运行。")
                warned = True
            if time.monotonic() > deadline:
                detail = "；".join(self.logs[-4:])
                self.close()
                raise CaptureError("等待采集音频头超时。%s" % detail)
            if self.proc.poll() is not None and not self._readable(0):
                detail = "；".join(self.logs[-6:]) or "采集进程已退出"
                self.close()
                raise CaptureError(detail)
            if not self._readable(0.2):
                continue
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                detail = "；".join(self.logs[-6:]) or "采集进程已关闭输出"
                self.close()
                raise CaptureError(detail)
            buf.extend(chunk)
        raw, _, rest = bytes(buf).partition(b"\n\n")
        fields = parse_header(raw)
        self.rate = int(float(fields["rate"]))
        self.channels = int(fields["channels"])
        self.encoding = fields.get("encoding", "")
        self.device_label = fields.get("device", "")
        if self.encoding != "s16le":
            self.close()
            raise CaptureError("不支持的编码: " + self.encoding)
        self._pending.extend(rest)

    def read(self, nbytes: int, timeout: float = 0.3) -> bytes | None:
        if not self.proc or not self.proc.stdout:
            return b""
        deadline = time.monotonic() + timeout
        while len(self._pending) < nbytes:
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            if not self._readable(min(remain, 0.2)):
                if self.proc.poll() is not None:
                    break
                continue
            chunk = self.proc.stdout.read(max(1, nbytes - len(self._pending)))
            if not chunk:
                break
            self._pending.extend(chunk)
        if not self._pending:
            if self.proc.poll() is not None:
                return b""
            return None
        take = min(nbytes, len(self._pending))
        out = bytes(self._pending[:take])
        del self._pending[:take]
        return out

    def close(self) -> None:
        proc = self.proc
        if not proc:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        self.proc = None


class WavFileSource:
    def __init__(self, path: str, realtime: bool = True):
        self.wf = wave.open(path, "rb")
        if self.wf.getsampwidth() != 2:
            self.wf.close()
            raise CaptureError("文件模式只支持 16-bit WAV")
        self.rate = self.wf.getframerate()
        self.channels = self.wf.getnchannels()
        self.device_label = path
        self.realtime = realtime
        self._next = time.monotonic()

    def read(self, nbytes: int, timeout: float = 0.3) -> bytes | None:
        del timeout
        frames = nbytes // (2 * self.channels)
        if frames <= 0:
            return None
        data = self.wf.readframes(frames)
        if not data:
            return b""
        if self.realtime:
            duration = frames / float(self.rate)
            self._next += duration
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return data

    def close(self) -> None:
        self.wf.close()
