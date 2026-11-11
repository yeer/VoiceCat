"""Speech-to-text backends. API keys are read from arguments and never logged."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

CRLF = chr(13) + chr(10)
_BEARER = re.compile(r"(?i)(bearer\s+)\S+")
_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


class STTError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def redact(text: str) -> str:
    cleaned = text or ""
    cleaned = _BEARER.sub(r"\1<redacted>", cleaned)
    cleaned = _SECRET.sub("sk-<redacted>", cleaned)
    return cleaned


def needs_api_key(backend: str, base_url: str) -> bool:
    if backend in ("local", "command", "transformers", "sensevoice", "offline-paraformer", "online-paraformer", "online-transducer"):
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return host not in ("localhost", "127.0.0.1", "::1")


def build_transcription_body(
    *,
    wav: bytes,
    model: str,
    language: str | None,
    prompt: str | None,
    task: str | None,
    include_optional: bool,
) -> tuple[bytes, str]:
    boundary = "----sysaudio-stt-" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.append(
            (
                "--"
                + boundary
                + CRLF
                + 'Content-Disposition: form-data; name="'
                + name
                + '"'
                + CRLF
                + CRLF
                + value
                + CRLF
            ).encode("utf-8")
        )

    file_header = (
        "--"
        + boundary
        + CRLF
        + 'Content-Disposition: form-data; name="file"; filename="segment.wav"'
        + CRLF
        + "Content-Type: audio/wav"
        + CRLF
        + CRLF
    ).encode("utf-8")
    chunks.append(file_header + wav + CRLF.encode("ascii"))
    field("model", model)
    field("response_format", "json")
    if include_optional:
        if language and language != "auto":
            field("language", language)
        if prompt:
            field("prompt", prompt)
        if task and task != "transcribe":
            field("task", task)
        field("temperature", "0")
    chunks.append(("--" + boundary + "--" + CRLF).encode("utf-8"))
    return b"".join(chunks), boundary


def extract_chat_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "output_text") and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts).strip()
    return ""


def parse_transcription_response(body: bytes, content_type: str = "") -> str:
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return ""
    looks_json = text.startswith("{") or "json" in content_type.lower()
    if not looks_json:
        return text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        if isinstance(data.get("text"), str):
            return data["text"].strip()
        chat = extract_chat_text(data)
        if chat:
            return chat
    return text


def build_chat_payload(wav: bytes, model: str, instruction: str) -> dict:
    return {
        "model": model,
        "modalities": ["text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
    }


def _request(url: str, data: bytes, headers: dict[str, str], timeout: float) -> tuple[bytes, str]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = redact(exc.read().decode("utf-8", "replace")[:500])
        if exc.code == 401:
            raise STTError("鉴权失败（HTTP 401）。请检查 STT_API_KEY，密钥不会被打印。", status=401) from exc
        raise STTError("转写接口失败 HTTP %s: %s" % (exc.code, detail), status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise STTError("无法连接转写服务: %s" % redact(str(exc.reason))) from exc


class OpenAITranscriber:
    def __init__(self, api_key: str, base_url: str, model: str, task: str | None = None, timeout: float = 45):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.task = task
        self.timeout = timeout

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        try:
            return self._post(wav, language, prompt, include_optional=True)
        except STTError as exc:
            if exc.status == 400:
                return self._post(wav, language, prompt, include_optional=False)
            raise

    def _post(self, wav: bytes, language: str | None, prompt: str | None, include_optional: bool) -> str:
        body, boundary = build_transcription_body(
            wav=wav,
            model=self.model,
            language=language,
            prompt=prompt,
            task=self.task,
            include_optional=include_optional,
        )
        headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        raw, content_type = _request(
            self.base_url + "/audio/transcriptions",
            body,
            headers,
            self.timeout,
        )
        return parse_transcription_response(raw, content_type)


class ChatTranscriber:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.instruction = "请把这段音频逐字转写成文本。只输出转写内容，不要解释，不要加引号。如果没有清晰人声，输出空字符串。"

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        instruction = self.instruction
        if language and language not in ("auto", ""):
            instruction += " 目标语言: " + language + "。"
        if prompt:
            instruction += " 上下文: " + prompt
        payload = build_chat_payload(wav, self.model, instruction)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        raw, content_type = _request(
            self.base_url + "/chat/completions",
            json.dumps(payload).encode("utf-8"),
            headers,
            self.timeout,
        )
        return parse_transcription_response(raw, content_type)


class CommandTranscriber:
    def __init__(self, command: str, timeout: float = 120):
        if not command.strip():
            raise STTError("command 后端需要 --command")
        self.command = command
        self.timeout = timeout

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        del language, prompt
        path = None
        try:
            command = self.command
            if "{wav}" in command:
                handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                path = handle.name
                handle.write(wav)
                handle.close()
                command = command.replace("{wav}", path)
                proc = subprocess.run(
                    shlex.split(command),
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
            else:
                proc = subprocess.run(
                    shlex.split(command),
                    input=wav,
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise STTError("外部转写命令超时") from exc
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        if proc.returncode != 0:
            err = redact(proc.stderr.decode("utf-8", "replace")[:500])
            raise STTError("外部转写命令失败: " + err)
        return proc.stdout.decode("utf-8", "replace").strip()


class FasterWhisperTranscriber:
    def __init__(self, model: str, device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError("未安装 faster-whisper。可执行: python3.12 -m pip install faster-whisper") from exc
        self.model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=4)

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        import io
        import wave

        import numpy as np

        with wave.open(io.BytesIO(wav), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise STTError("faster-whisper 只接受 16-bit 音频")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if rate != 16000 and len(audio):
            target = max(1, int(len(audio) * 16000 / rate))
            index = np.linspace(0, len(audio) - 1, target).astype(np.int64)
            audio = audio[index]
        del prompt
        lang = None if not language or language == "auto" else language
        segments, _info = self.model.transcribe(
            audio,
            language=lang,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            without_timestamps=True,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.5,
            hallucination_silence_threshold=1.5,
        )
        kept = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            if getattr(segment, "no_speech_prob", 0) >= 0.5:
                continue
            if getattr(segment, "avg_logprob", 0) <= -0.8:
                continue
            if getattr(segment, "compression_ratio", 0) >= 2.2:
                continue
            compact = "".join(ch for ch in text if ch.strip() and ch not in "，。、！？,.!?;；：:")
            if len(compact) < 6:
                continue
            kept.append(text)
        return drop_hallucination("".join(kept).strip())


_TRANSFORMERS_MODELS = {
    "small": "openai/whisper-small",
    "whisper-small": "openai/whisper-small",
    "turbo": "openai/whisper-large-v3-turbo",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
}


def resolve_transformers_model(name: str) -> str:
    key = (name or "").strip()
    if not key:
        return "openai/whisper-small"
    return _TRANSFORMERS_MODELS.get(key, key)


_HALLUCINATIONS = (
    "字幕由",
    "字幕志愿者",
    "谢谢观看",
    "感谢观看",
    "请不吝点赞",
    "明镜与点点",
    "Amara.org",
)


def normalize_audio(audio):
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 0.01:
        return audio
    gain = min(0.9 / peak, 6.0)
    if gain <= 1.05:
        return audio
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def drop_hallucination(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for phrase in _HALLUCINATIONS:
        if phrase in cleaned and len(cleaned) <= 28:
            return ""
    return cleaned


def wav_pcm16_to_float32(wav: bytes):
    """Decode a 16-bit WAV into a float32 numpy array in [-1, 1]."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(wav), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise STTError("transformers 后端只接受 16-bit WAV")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        pcm = handle.readframes(handle.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio), rate


class TransformersTranscriber:
    """Run the cached Hugging Face Whisper weights with transformers."""

    def __init__(self, model: str, device: str = "cpu", task: str = "transcribe"):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise STTError(
                "未安装 transformers。请在项目虚拟环境执行: .venv/bin/python -m pip install -r requirements.txt"
            ) from exc
        import logging

        self.model_id = resolve_transformers_model(model)
        self.task = task or "transcribe"
        self._torch = torch
        if device not in ("cpu", "", "auto"):
            logging.info("transformers 在这台机器上使用 CPU，忽略设备 %s", device)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        logging.info("正在用 transformers 加载本地模型 %s", self.model_id)
        try:
            loaded = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, local_files_only=True)
            self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
        except OSError as exc:
            raise STTError(
                "本地没有可用的 transformers 权重: %s。已缓存的是 openai/whisper-small、openai/whisper-large-v3-turbo 和 openai/whisper-large-v3。"
                % self.model_id
            ) from exc
        loaded.eval()
        if getattr(loaded, "generation_config", None) is not None:
            loaded.generation_config.forced_decoder_ids = None
        if getattr(loaded, "config", None) is not None:
            loaded.config.forced_decoder_ids = None
        self.model = loaded

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        audio, rate = wav_pcm16_to_float32(wav)
        if rate != 16000:
            import numpy as np

            target = 16000
            positions = np.linspace(0, len(audio) - 1, int(len(audio) * target / rate))
            audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
            rate = target
        audio = normalize_audio(audio)
        features = self.processor(audio, sampling_rate=rate, return_tensors="pt").input_features
        kwargs = {
            "input_features": features,
            "attention_mask": self._torch.ones((features.shape[0], features.shape[-1]), dtype=self._torch.long),
            "task": self.task,
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": 128,
            "no_repeat_ngram_size": 4,
            "return_timestamps": False,
        }
        if language and language != "auto":
            kwargs["language"] = language
        if prompt:
            prompt_ids = self.processor.get_prompt_ids(prompt)
            if len(prompt_ids):
                kwargs["prompt_ids"] = self._torch.as_tensor(prompt_ids, dtype=self._torch.long)
        with self._torch.inference_mode():
            ids = self.model.generate(**kwargs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return drop_hallucination(str(text).strip())


def sensevoice_model_dir():
    from sysaudio_stt.polish import app_data_dir

    path = app_data_dir() / "models" / "sense-voice-small"
    tokens = path / "tokens.txt"
    model = path / "model.int8.onnx"
    if not model.is_file():
        model = path / "model.onnx"
    if not tokens.is_file() or not model.is_file():
        raise STTError("找不到 SenseVoice-Small 模型。它应在本机应用数据的 models/sense-voice-small 里。")
    return path, model


class SenseVoiceTranscriber:
    def __init__(self, language: str = "zh"):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise STTError("未安装 sherpa-onnx。可执行: python3.12 -m pip install sherpa-onnx==1.12.40") from exc
        lang = language if language in ("zh", "en", "ja", "ko", "yue", "auto") else "auto"
        if lang == "auto":
            lang = ""
        folder, model = sensevoice_model_dir()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(folder / "tokens.txt"),
            num_threads=4,
            language=lang,
            use_itn=True,
            provider="cpu",
        )

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        import io
        import re
        import wave

        import numpy as np

        del prompt
        with wave.open(io.BytesIO(wav), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise STTError("SenseVoice 只接受 16-bit 音频")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if rate != 16000 and len(audio):
            target = max(1, int(len(audio) * 16000 / rate))
            index = np.linspace(0, len(audio) - 1, target).astype(np.int64)
            audio = audio[index]
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        text = getattr(stream.result, "text", "") or ""
        return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _wav_to_float(wav: bytes):
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(wav), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise STTError("只接受 16-bit 音频")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and len(audio):
        target = max(1, int(len(audio) * 16000 / rate))
        index = np.linspace(0, len(audio) - 1, target).astype(np.int64)
        audio = audio[index]
    return audio


class SherpaTranscriber:
    def __init__(self, model_id: str, language: str = "zh"):
        import sherpa_onnx

        from sysaudio_stt.models_catalog import _find_role, get_model, sherpa_dir

        item = get_model(model_id)
        if item is None:
            raise STTError("没有这个 sherpa 模型: " + model_id)
        root = sherpa_dir(model_id)
        tokens = next(root.rglob("tokens.txt"))
        kind = item["kind"]
        self.mode = "online" if kind.startswith("online") else "offline"
        if kind == "sensevoice":
            lang = language if language in ("zh", "en", "ja", "ko", "yue", "auto") else "auto"
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(_find_role(root, "model")),
                tokens=str(tokens),
                num_threads=4,
                language="" if lang == "auto" else lang,
                use_itn=True,
                provider="cpu",
            )
        elif kind == "offline-paraformer":
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(_find_role(root, "model")),
                tokens=str(tokens),
                num_threads=4,
                provider="cpu",
            )
        elif kind == "online-paraformer":
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=str(tokens),
                encoder=str(_find_role(root, "encoder")),
                decoder=str(_find_role(root, "decoder")),
                num_threads=4,
                provider="cpu",
            )
        elif kind == "online-transducer":
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(_find_role(root, "encoder")),
                decoder=str(_find_role(root, "decoder")),
                joiner=str(_find_role(root, "joiner")),
                num_threads=4,
                provider="cpu",
            )
        else:
            raise STTError("不支持的 sherpa 模型: " + kind)

    def transcribe(self, wav: bytes, language: str | None, prompt: str | None) -> str:
        import re

        del language, prompt
        audio = _wav_to_float(wav)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        if self.mode == "online":
            stream.input_finished()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            text = self.recognizer.get_result(stream)
        else:
            self.recognizer.decode_stream(stream)
            text = getattr(stream.result, "text", "") or ""
        return self._clean(text)

    def _clean(self, text) -> str:
        import re

        return re.sub(r"<\|[^|]+\|>", "", str(text or "")).strip()

    def open_stream(self):
        if self.mode != "online":
            raise STTError("这个模型不是流式模型")
        self._stream = self.recognizer.create_stream()

    def push_pcm(self, pcm_s16: bytes) -> str:
        import numpy as np

        usable = pcm_s16[: len(pcm_s16) - (len(pcm_s16) % 2)]
        if usable:
            audio = np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0
            self._stream.accept_waveform(16000, np.ascontiguousarray(audio))
            while self.recognizer.is_ready(self._stream):
                self.recognizer.decode_stream(self._stream)
        return self._clean(self.recognizer.get_result(self._stream))

    def at_endpoint(self) -> bool:
        return bool(self.recognizer.is_endpoint(self._stream))

    def finish_utterance(self) -> str:
        text = self._clean(self.recognizer.get_result(self._stream))
        self.recognizer.reset(self._stream)
        return text


_TRANSCRIBERS = {}
_TRANSCRIBER_LOCK = threading.Lock()
_TRANSCRIBER_INFLIGHT = {}


def preload_transcriber(args):
    key = (args.backend, getattr(args, "catalog_id", "") or args.model, getattr(args, "language", ""))
    with _TRANSCRIBER_LOCK:
        cached = _TRANSCRIBERS.get(key)
        if cached is not None:
            return cached
        inflight = _TRANSCRIBER_INFLIGHT.get(key)
        if inflight is None:
            inflight = threading.Event()
            _TRANSCRIBER_INFLIGHT[key] = inflight
            owner = True
        else:
            owner = False
    if not owner:
        inflight.wait()
        with _TRANSCRIBER_LOCK:
            return _TRANSCRIBERS.get(key) or make_transcriber(args)
    try:
        made = make_transcriber(args)
        with _TRANSCRIBER_LOCK:
            _TRANSCRIBERS[key] = made
        return made
    finally:
        inflight.set()
        with _TRANSCRIBER_LOCK:
            _TRANSCRIBER_INFLIGHT.pop(key, None)


def make_transcriber(args):
    backend = args.backend
    if backend == "openai":
        return OpenAITranscriber(args.api_key, args.base_url, args.model, task=args.task)
    if backend == "chat":
        return ChatTranscriber(args.api_key, args.base_url, args.model)
    if backend == "command":
        return CommandTranscriber(args.command)
    if backend == "local":
        return FasterWhisperTranscriber(args.model, device=args.local_device, compute_type=args.compute_type)
    if backend in ("sensevoice", "offline-paraformer", "online-paraformer", "online-transducer"):
        return SherpaTranscriber(getattr(args, "catalog_id", "") or args.model, language=args.language)
    if backend == "transformers":
        return TransformersTranscriber(args.model, device=args.local_device, task=args.task)
    raise STTError("未知后端: " + backend)
