"""Command-line entry for system-audio transcription."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sysaudio_stt.audio import Resampler, pcm16_to_wav, peak_s16, rms_s16, strip_overlap
from sysaudio_stt.capture import CaptureError, CaptureSession, WavFileSource, list_devices
from sysaudio_stt.devices import blackhole_routed_name, routing_advice
from sysaudio_stt.polish import llm_credentials, load_glossary, polish_text
from sysaudio_stt.stt import STTError, make_transcriber, needs_api_key, preload_transcriber, redact
from sysaudio_stt.vad import Segmenter


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def parse_env_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def apply_env_file(path: str) -> None:
    data = parse_env_assignments(Path(path).read_text(encoding="utf-8"))
    for key, value in data.items():
        os.environ.setdefault(key, value)


def default_binary() -> Path:
    return Path(__file__).resolve().parent.parent / "bin" / "sysaudio-capture"


def missing_credential_text(base_url: str) -> str:
    return "\n".join(
        [
            "受阻环节: 把语音片段送给在线转写服务。",
            "所需凭据: STT_API_KEY 或 OPENAI_API_KEY，对应服务 " + base_url + "。",
            "未执行的动作: 未读取配置文件、命令历史或密钥文件。",
            "建议: 在当前环境注入变量后重试，例如 export STT_API_KEY='<凭据>'；也可以改用 --backend local，或先运行 --monitor。不要把密钥贴到对话里。",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sysaudio-stt",
        description="从本机系统声音输出采集音频，并实时送给 Whisper 类或在线语音模型转写。没有界面。",
    )
    parser.add_argument("--list", action="store_true", help="列出音频设备并给出采集建议")
    parser.add_argument("--monitor", action="store_true", help="只打印电平，不调用模型，也不保存音频")
    parser.add_argument("--source", default="auto", choices=["auto", "system", "device", "sckit", "file"], help="采集来源，默认 auto")
    parser.add_argument("--choose", action="store_true", help="单独入口：测量输出设备并选择模型，然后开始听")
    parser.add_argument("--web", action="store_true", help="打开桌面窗口")
    parser.add_argument("--serve", action="store_true", help="只启动本机服务，由桌面壳打开窗口")
    parser.add_argument("--device", default="", help="要监听的输出设备名称。交互选择请用 ./choose.sh")
    parser.add_argument("--uid", default="", help="输入设备 UID")
    parser.add_argument("--input", default="", help="file 模式的 WAV 路径")
    parser.add_argument("--seconds", type=float, default=0, help="采集时长，0 表示直到 Ctrl+C")
    parser.add_argument("--fast", action="store_true", help="file 模式不等实时时钟")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "openai", "chat", "local", "command"])
    parser.add_argument("--base-url", default="", help="在线后端读取 STT_BASE_URL / OPENAI_BASE_URL")
    parser.add_argument("--model", default="", help="transformers 默认 openai/whisper-small；也可用 turbo 或 large-v3")
    parser.add_argument("--api-key", default="", help="不建议使用。优先用环境变量 STT_API_KEY")
    parser.add_argument("--language", default="zh", help="zh / en / auto。auto 表示让模型检测")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--prompt", default="", help="初始提示词，用来放专有名词")
    parser.add_argument("--polish", action=argparse.BooleanOptionalAction, default=True, help="识别后交给大模型校对标点、重复和明显错词")
    parser.add_argument("--polish-model", default="", help="覆盖 llm.toml 里的润色模型")
    parser.add_argument("--glossary", default="", help="润色时优先保留的说法，也可用环境变量 STT_GLOSSARY")
    parser.add_argument("--command", default="", help="command 后端。可用 {wav} 占位临时文件")
    parser.add_argument("--threshold", type=int, default=40, help="VAD 能量阈值，越大越不容易触发。耳机音量往往数字电平很低")
    parser.add_argument("--silence", type=float, default=0.45, help="静音多少秒后送出一段")
    parser.add_argument("--max-segment", type=float, default=8, help="一段语音最长秒数，到时也会送出")
    parser.add_argument("--min-speech", type=float, default=0.2, help="短于此时长的声音丢弃")
    parser.add_argument("--partial-every", type=float, default=0, help="说话过程中每隔多少秒先出一次进行中文本，0 关闭")
    parser.add_argument("--output", default="", help="把转写追加写入这个文本文件")
    parser.add_argument("--jsonl", action="store_true", help="用 JSONL 输出")
    parser.add_argument("--no-context", action="store_true", help="不把上一段文本作为下一段提示")
    parser.add_argument("--env-file", default="", help="只在显式传入时读取 KEY=VALUE，不打印其中的值")
    parser.add_argument("--local-device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--binary", default="", help="采集程序路径，默认使用项目内 bin/sysaudio-capture")
    return parser


def configure_runtime(args: argparse.Namespace) -> None:
    if args.env_file:
        apply_env_file(args.env_file)
    args.api_key = args.api_key or os.environ.get("STT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    args.base_url = args.base_url or os.environ.get("STT_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not args.model:
        args.model = os.environ.get("STT_MODEL") or ""
    if not args.model:
        if args.backend == "transformers":
            args.model = "openai/whisper-small"
        elif args.backend == "chat":
            args.model = "gpt-4o-audio-preview"
        elif args.backend == "local":
            args.model = "small"
        else:
            args.model = "whisper-1"
    args.binary_path = Path(args.binary) if args.binary else default_binary()
    args.polish_model = args.polish_model or os.environ.get("STT_POLISH_MODEL") or ""
    if not args.glossary:
        args.glossary = os.environ.get("STT_GLOSSARY") or load_glossary()
    if not getattr(args, "script", ""):
        args.script = "zh-cn" if args.language == "zh" else ""
    if args.script == "zh-cn" and not getattr(args, "style_prompt", ""):
        args.style_prompt = "以下是简体中文。"


def require_binary(path: Path) -> None:
    if not path.is_file():
        raise SystemExit("找不到采集程序: %s\n请先在项目目录执行 ./build.sh" % path)


LANGUAGE_CHOICES = (
    {"id": "zh-Hans", "whisper": "zh", "script": "zh-cn", "label": "简体中文", "prompt": "以下是简体中文。"},
    {"id": "zh-Hant", "whisper": "zh", "script": "zh-tw", "label": "繁体中文", "prompt": "以下是繁體中文。"},
    {"id": "en", "whisper": "en", "script": "", "label": "English", "prompt": ""},
    {"id": "auto", "whisper": "auto", "script": "", "label": "自动检测", "prompt": ""},
)


def resolve_language(choice: str) -> dict:
    text = (choice or "").strip()
    if not text:
        return LANGUAGE_CHOICES[0]
    for item in LANGUAGE_CHOICES:
        if text in (item["id"], item["whisper"], item["label"]):
            return item
    raise ValueError("没有这个语言: %s" % text)


def repeated_fragment(previous: str, current: str) -> bool:
    import re

    def norm(value: str) -> str:
        return re.sub(r"[\\s，。、！？,.!?;；：:]+", "", value or "")

    cur = norm(current)
    prev = norm(previous)
    return bool(cur) and len(cur) <= 6 and cur in prev[-16:]


def apply_script(text: str, script: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned or not script:
        return cleaned
    import zhconv
    return zhconv.convert(cleaned, script)


MODEL_CHOICES = (
    {"id": "openai/whisper-small", "label": "small", "note": "默认可跟上播放，一段大约 7 秒"},
    {"id": "openai/whisper-large-v3-turbo", "label": "turbo", "note": "更准，但这台 CPU 一段大约 20 秒"},
    {"id": "openai/whisper-large-v3", "label": "large-v3", "note": "最慢，一段通常要一两分钟"},
)
FASTER_WHISPER_CHOICES = (
    {"id": "small", "label": "small", "note": "int8，通常比原来的 small 快"},
    {"id": "large-v3-turbo", "label": "turbo", "note": "int8，同一套 turbo 权重"},
    {"id": "large-v3", "label": "large-v3", "note": "int8，仍慢于 turbo"},
)
SENSEVOICE_CHOICES = (
    {"id": "sense-voice-small", "label": "SenseVoice-Small", "note": "准流式，短段很快，中文通常比 small 准"},
)
ENGINE_CHOICES = (
    {"id": "transformers", "label": "Whisper", "models": MODEL_CHOICES},
    {"id": "faster-whisper", "label": "faster-whisper", "models": FASTER_WHISPER_CHOICES},
    {"id": "sensevoice", "label": "SenseVoice", "models": SENSEVOICE_CHOICES},
)


def select_model(choice: str) -> str:
    text = (choice or "").strip()
    if not text:
        return MODEL_CHOICES[0]["id"]
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(MODEL_CHOICES):
            return MODEL_CHOICES[index - 1]["id"]
    for item in MODEL_CHOICES:
        if text in (item["id"], item["label"]):
            return item["id"]
    raise ValueError("没有编号或模型名 %s" % text)


def choose_model() -> str:
    print("选择模型：", flush=True)
    for index, item in enumerate(MODEL_CHOICES, 1):
        print("  %d  %-10s %s  %s" % (index, item["label"], item["id"], item["note"]), flush=True)
    if not sys.stdin.isatty():
        print("当前不是交互终端，使用 %s。" % MODEL_CHOICES[0]["label"], flush=True)
        return MODEL_CHOICES[0]["id"]
    print("输入编号后回车。直接回车使用 1。", flush=True)
    while True:
        try:
            choice = input("> ")
        except EOFError:
            return MODEL_CHOICES[0]["id"]
        try:
            return select_model(choice)
        except ValueError as exc:
            print("%s，请重新输入。" % exc, flush=True)


def select_output(items: list[dict], choice: str) -> dict:
    if not items:
        raise ValueError("没有输出设备")
    text = (choice or "").strip()
    if not text:
        return max(
            items,
            key=lambda item: (int(item.get("rms") or 0), bool(item.get("is_default_system_output"))),
        )
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(items):
            return items[index - 1]
    for item in items:
        if text == str(item.get("name") or "") or text == str(item.get("uid") or ""):
            return item
    raise ValueError("没有编号或设备名 %s" % text)


def scan_outputs(binary: Path, seconds: float = 0.6) -> list[dict]:
    if sys.platform == "win32":
        from sysaudio_stt.wasapi import scan_loopback
        return scan_loopback()
    proc = subprocess.Popen(
        [str(binary), "scan", "--seconds", "%.2f" % seconds],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    items: list[dict] = []
    hint = ""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("type") == "hint":
            hint = str(item.get("message") or "")
            continue
        if item.get("type") != "output":
            continue
        items.append(item)
        marks = []
        if item.get("is_default_system_output"):
            marks.append("当前系统输出")
        if item.get("error"):
            marks.append("测量失败")
        suffix = ("  [" + ", ".join(marks) + "]") if marks else ""
        try:
            print(
                "  %d  %s  rms=%s peak=%s%s"
                % (len(items), item.get("name") or "", item.get("rms", 0), item.get("peak", 0), suffix),
                flush=True,
            )
        except BrokenPipeError:
            logging.info(
                "输出 %s rms=%s peak=%s",
                item.get("name") or "",
                item.get("rms", 0),
                item.get("peak", 0),
            )
    code = proc.wait(timeout=60)
    if code != 0 and not items:
        raise CaptureError("测量输出设备失败")
    loudest = max((int(item.get("rms") or 0) for item in items), default=0)
    if hint and loudest == 0:
        items.append({"type": "hint", "message": hint})
    return items


def sckit_level(binary: Path, seconds: float = 2.0) -> tuple[int, int, str]:
    proc = subprocess.run(
        [str(binary), "probe", "--source", "sckit", "--seconds", "%.1f" % seconds],
        capture_output=True,
        timeout=30,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    if not lines:
        return 0, 0, err[-1] if err else "系统混音没有输出"
    try:
        item = json.loads(lines[-1])
    except json.JSONDecodeError:
        return 0, 0, lines[-1]
    if not item.get("ok", True) and item.get("error"):
        return 0, 0, str(item.get("error"))
    return int(round(float(item.get("rms_mean") or 0))), int(item.get("rms_peak") or 0), ""


def choose_output(binary: Path) -> dict:
    print("正在测量各路输出的电平。请保持要听的声音播放，大约需要几秒。", flush=True)
    scanned = scan_outputs(binary)
    hint = next((str(item.get("message") or "") for item in scanned if item.get("type") == "hint"), "")
    items = [item for item in scanned if item.get("type") != "hint"]
    if not items:
        raise SystemExit("没有找到带输出声道的设备")
    best = select_output(items, "")
    if hint and int(best.get("rms") or 0) == 0:
        print("进程采集全是 0。改用已允许的录屏权限听系统混音，约 2 秒。", flush=True)
        print("菜单栏可能出现录制指示。不改默认设备，不采集麦克风。", flush=True)
        rms, peak, err = sckit_level(binary)
        print("系统混音 rms=%d peak=%d" % (rms, peak), flush=True)
        if peak > 0 or rms > 0:
            return {"name": "系统混音", "uid": "", "source": "sckit", "rms": rms, "peak": peak}
        extra = ("系统混音也没有波形：%s。" % err) if err else "系统混音也没有波形。"
        raise SystemExit(hint + "\n" + extra + "请往下滚到「仅系统录音」，把 Kaku 加进去并打开，然后完全退出 Kaku 再打开。")
    if not sys.stdin.isatty():
        print("当前不是交互终端，改为监听电平最高的「%s」。" % best.get("name"), flush=True)
        return best
    print("输入编号后回车。直接回车则听电平最高的「%s」。" % best.get("name"), flush=True)
    while True:
        try:
            choice = input("> ")
        except EOFError:
            return best
        try:
            return select_output(items, choice)
        except ValueError as exc:
            print("%s，请重新输入。" % exc, flush=True)


def cmd_list(binary: Path) -> int:
    require_binary(binary)
    devices = list_devices(binary)
    by_uid = {item.get("uid"): item.get("name") for item in devices}
    for item in devices:
        marks = []
        if item.get("is_default_input"):
            marks.append("默认输入")
        if item.get("is_default_output"):
            marks.append("默认输出")
        if item.get("is_default_system_output"):
            marks.append("默认系统输出")
        suffix = (" [" + ", ".join(marks) + "]") if marks else ""
        print(
            "%s  in=%s out=%s %.0fHz%s"
            % (item.get("name"), item.get("input_channels"), item.get("output_channels"), float(item.get("nominal_rate") or 0), suffix)
        )
        subs = item.get("subdevice_uids") or []
        if subs:
            print("  子设备: " + "、".join(str(by_uid.get(uid, uid)) for uid in subs))
    print("")
    for line in routing_advice(devices):
        print(line)
    return 0


def open_source(args: argparse.Namespace):
    if args.source == "file":
        if not args.input:
            raise SystemExit("file 模式需要 --input sample.wav")
        logging.info("读取文件 %s realtime=%s", args.input, not args.fast)
        return WavFileSource(args.input, realtime=not args.fast)
    if sys.platform == "win32":
        from sysaudio_stt.wasapi import WasapiLoopback
        return WasapiLoopback(args.device).start()
    require_binary(args.binary_path)
    if args.source == "auto":
        devices = list_devices(args.binary_path)
        try:
            return CaptureSession(args.binary_path, "system", seconds=args.seconds).start()
        except CaptureError as exc:
            logging.warning("系统 tap 不可用: %s", exc)
            name = args.device or blackhole_routed_name(devices)
            if not name:
                raise SystemExit(
                    "系统音频 tap 失败，且没有发现已接入系统输出的 BlackHole。\n"
                    "不会改用麦克风。请运行 --list 查看路由，或在授予权限后使用 --source system / --source sckit。"
                ) from exc
            logging.info("改用已接入系统输出的设备: %s", name)
            return CaptureSession(args.binary_path, "device", device=name, seconds=args.seconds).start()
    if args.source == "device" and not args.device and not args.uid:
        devices = list_devices(args.binary_path)
        name = blackhole_routed_name(devices)
        if not name:
            raise SystemExit("device 模式需要 --device 或 --uid。未发现已接入系统输出的 BlackHole，不会改用麦克风。")
        args.device = name
        logging.info("使用已接入系统输出的设备: %s", name)
    return CaptureSession(
        args.binary_path,
        args.source,
        device=args.device or None,
        uid=args.uid or None,
        seconds=args.seconds,
    ).start()


@dataclass
class Job:
    pcm: bytes
    final: bool
    queued_at: float


def emit_text(text: str, partial: bool, args: argparse.Namespace, out_fp) -> None:
    stamp = time.strftime("%H:%M:%S")
    if args.jsonl:
        line = json.dumps({"ts": stamp, "text": text, "partial": partial}, ensure_ascii=False)
    else:
        prefix = "（进行中）" if partial else ""
        line = "[%s] %s%s" % (stamp, prefix, text)
    print(line, flush=True)
    if out_fp:
        out_fp.write(line + "\n")
        out_fp.flush()


def run_monitor(args: argparse.Namespace) -> int:
    source = open_source(args)
    stop = threading.Event()

    def handle(signum, _frame):
        del signum
        stop.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    chunk = max(2, int(source.rate * 0.1) * source.channels * 2)
    window = bytearray()
    started = time.monotonic()
    last_log = 0.0
    logging.info("监听电平中。没有播放时 rms 接近 0 是正常的。音频不落盘、不外发。")
    try:
        while not stop.is_set():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            raw = source.read(chunk, timeout=0.3)
            if raw is None:
                continue
            if raw == b"":
                break
            window.extend(raw)
            now = time.monotonic()
            if now - last_log >= 0.5:
                level = rms_s16(bytes(window))
                peak = peak_s16(bytes(window))
                logging.info("电平 rms=%d peak=%d", level, peak)
                print("rms=%d peak=%d" % (level, peak), flush=True)
                window.clear()
                last_log = now
    finally:
        source.close()
    logging.info("监听结束")
    return 0


def _with_corrections(text):
    from sysaudio_stt.polish import apply_corrections
    return apply_corrections(text)


def _polish_later(args, text, batch, notify):
    if not getattr(args, "polish", False) or not text:
        return

    def run():
        try:
            chosen = getattr(args, "polish_model", "") or "local"
            if chosen == "local":
                from sysaudio_stt.local_polish import polish_locally
                polished = polish_locally(text, script=getattr(args, "script", ""), glossary=getattr(args, "glossary", ""))
            else:
                polished = polish_text(
                    text,
                    script=getattr(args, "script", ""),
                    glossary=getattr(args, "glossary", ""),
                    model=chosen,
                    instruction=getattr(args, "polish_prompt", ""),
                )
        except Exception as exc:
            logging.error("润色失败: %s", redact(str(exc)))
            return
        if polished:
            notify("polish", batch=batch, text=polished)

    threading.Thread(target=run, daemon=True).start()


def run_online_transcribe(args, transcriber, source, on_event, stop_event, install_signals) -> int:
    stop = stop_event or threading.Event()

    def notify(kind: str, **payload) -> None:
        if not on_event:
            return
        try:
            on_event({"type": kind, "ts": time.strftime("%H:%M:%S"), **payload})
        except Exception:
            logging.exception("网页事件发送失败")

    if install_signals:
        def handle(signum, _frame):
            del signum, _frame
            stop.set()
            try:
                source.close()
            except Exception:
                pass
            os._exit(130)

        signal.signal(signal.SIGINT, handle)
        signal.signal(signal.SIGTERM, handle)

    resampler = Resampler(source.rate)
    transcriber.open_stream()
    draft = 1
    batch = 1
    last = ""
    full = ""
    committed = ""
    line_started = time.monotonic()
    silence_since = None
    punct_box = {"pending": None, "busy": False}

    def request_punct(text, draft_id, final):
        punct_box["pending"] = (text, draft_id, final)
        if punct_box["busy"]:
            return
        punct_box["busy"] = True

        def run():
            while punct_box["pending"]:
                snapshot, draft_id, final = punct_box["pending"]
                punct_box["pending"] = None
                try:
                    from sysaudio_stt.local_polish import restore_punctuation
                    done = _with_corrections(restore_punctuation(snapshot) or snapshot)
                except Exception:
                    logging.exception("补标点失败")
                    continue
                if not final and punct_box["pending"] and punct_box["pending"][0] != snapshot:
                    continue
                if done:
                    notify("text", text=done, partial=not final, draft=draft_id)
            punct_box["busy"] = False

        threading.Thread(target=run, daemon=True).start()

    chunk = max(2, int(source.rate * 0.1) * source.channels * 2)
    notify("status", message="流式模型会边听边出字。")
    print("流式模型会边听边出字。", flush=True)
    level_buf = bytearray()
    last_level = time.monotonic()

    def piece_of(full_text):
        full_text = full_text or ""
        if not committed:
            return full_text.strip()
        limit = min(len(committed), len(full_text))
        index = 0
        while index < limit and committed[index] == full_text[index]:
            index += 1
        return full_text[index:].strip()

    def close_line(full_text, reset):
        nonlocal draft, batch, committed, line_started, last, silence_since
        piece = piece_of(full_text)
        if piece:
            notify("text", text=piece, partial=False, draft=draft, ts=time.strftime("%H:%M:%S"))
            request_punct(piece, draft, True)
            _polish_later(args, piece, batch, notify)
            notify("status", message="已按时间拆开这一句，正在润色。")
            draft += 1
            batch += 1
        if reset:
            transcriber.finish_utterance()
            committed = ""
        else:
            committed = full_text
        last = ""
        line_started = time.monotonic()
        silence_since = None

    try:
        while not stop.is_set():
            raw = source.read(chunk, timeout=0.3)
            if raw is None:
                continue
            if raw == b"":
                break
            level_buf.extend(raw)
            now = time.monotonic()
            level = rms_s16(raw)
            if now - last_level >= 0.12:
                notify("level", rms=rms_s16(bytes(level_buf)), peak=peak_s16(bytes(level_buf)))
                level_buf.clear()
                last_level = now
            pcm = resampler.push(raw, source.channels)
            if pcm:
                full = _with_corrections(transcriber.push_pcm(pcm))
                piece = piece_of(full)
                if piece and piece != last:
                    last = piece
                    notify("text", text=piece, partial=True, draft=draft)
            else:
                piece = last
            if level < 180:
                if silence_since is None:
                    silence_since = now
            else:
                silence_since = None
            ended = bool(last) and (transcriber.at_endpoint() or (silence_since and now - silence_since >= 0.7))
            if ended:
                close_line(full if pcm else committed + last, True)
            elif last and now - line_started >= 8:
                close_line(full if pcm else committed + last, False)
        if last:
            close_line(full if full else committed + last, True)
    finally:
        stop.set()
        source.close()
    return 0


def run_transcribe(args: argparse.Namespace, on_event=None, stop_event=None, install_signals=True) -> int:
    if needs_api_key(args.backend, args.base_url) and not args.api_key:
        print(missing_credential_text(args.base_url), file=sys.stderr)
        return 2
    try:
        transcriber = preload_transcriber(args)
    except STTError as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 2
    source = open_source(args)
    if getattr(transcriber, "mode", "") == "online":
        return run_online_transcribe(args, transcriber, source, on_event, stop_event, install_signals)
    segmenter = Segmenter(
        threshold=args.threshold,
        silence_ms=int(args.silence * 1000),
        min_speech_ms=int(args.min_speech * 1000),
        max_speech_ms=int(args.max_segment * 1000),
    )
    resampler = Resampler(source.rate)
    job_queue = __import__("queue").Queue(maxsize=1)
    stop = stop_event or threading.Event()
    context = {"text": args.prompt or ""}

    def notify(kind: str, **payload) -> None:
        if not on_event:
            return
        try:
            on_event({"type": kind, "ts": time.strftime("%H:%M:%S"), **payload})
        except Exception:
            logging.exception("网页事件发送失败")
    state = {"busy": False, "latest": None, "batch": 1, "line": 1, "buf": [], "buf_len": 0}
    polish_chars = 80

    def flush_polish(force=False):
        if not getattr(args, "polish", False) or not state["buf"]:
            return
        if not force and state["buf_len"] < polish_chars:
            return
        chosen = getattr(args, "polish_model", "") or "local"
        if chosen != "local":
            _config, key_name, ready = llm_credentials()
        else:
            ready = True
            key_name = ""
        if chosen != "local" and not ready:
            if not state.get("polish_warned"):
                state["polish_warned"] = True
                message = "未配置 " + key_name + "，润色已跳过，先显示识别原文。"
                print(message, flush=True)
                notify("status", message=message)
            state["buf"] = []
            state["buf_len"] = 0
            return
        batch = state["batch"]
        chunk = "\n".join(state["buf"])
        state["batch"] += 1
        state["buf"] = []
        state["buf_len"] = 0
        notify("status", message="已积累 %d 字，开始一次%s。" % (len(chunk), "本地纠错" if chosen == "local" else "润色"))

        def run_polish():
            try:
                if chosen == "local":
                    from sysaudio_stt.local_polish import polish_locally
                    polished = polish_locally(
                        chunk,
                        script=getattr(args, "script", ""),
                        glossary=getattr(args, "glossary", ""),
                    )
                else:
                    polished = polish_text(
                        chunk,
                        script=getattr(args, "script", ""),
                        glossary=getattr(args, "glossary", ""),
                        model=chosen,
                        instruction=getattr(args, "polish_prompt", ""),
                    )
            except Exception as exc:
                logging.error("润色失败: %s", redact(str(exc)))
                notify("status", message="润色失败，页面保留识别原文。")
                return
            if not polished:
                return
            print("润色: " + polished, flush=True)
            notify("polish", batch=batch, text=polished)

        threading.Thread(target=run_polish, daemon=True).start()
    out_fp = open(args.output, "a", encoding="utf-8") if args.output else None

    def worker() -> None:
        while not stop.is_set():
            job = state["latest"]
            state["latest"] = None
            if job is None:
                try:
                    job = job_queue.get(timeout=0.2)
                except __import__("queue").Empty:
                    continue
            if job is None or stop.is_set():
                flush_polish(force=True)
                return
            state["busy"] = True
            try:
                wav = pcm16_to_wav(job.pcm)
                started = time.monotonic()
                prompt_parts = [
                    getattr(args, "style_prompt", "") or "",
                    getattr(args, "glossary", "") or "",
                ]
                prompt = " ".join(part.strip() for part in prompt_parts if part and part.strip()) or None
                try:
                    text = transcriber.transcribe(wav, args.language, prompt)
                except Exception as exc:
                    logging.error("转写失败: %s", redact(str(exc)))
                    continue
                text = apply_script(text, getattr(args, "script", ""))
                latency = int((time.monotonic() - started) * 1000)
                logging.info("转写完成 partial=%s latency_ms=%d chars=%d", not job.final, latency, len(text))
                if stop.is_set():
                    continue
                if not text:
                    notify("status", message="这段没有识别出文字。")
                    continue
                if job.final and not args.no_context:
                    text = strip_overlap(context["text"], text)
                    if text and repeated_fragment(context["text"], text):
                        text = ""
                    if not text:
                        print("这段和上一句重复，已跳过。", flush=True)
                        notify("status", message="这段和上一句重复，已跳过。")
                        continue
                    context["text"] = (context["text"] + " " + text).strip()[-240:]
                text = _with_corrections(text)
                emit_text(text, partial=not job.final, args=args, out_fp=out_fp)
                notify("text", text=text, partial=not job.final, batch=state["batch"], draft=state["line"])
                if job.final:
                    state["line"] += 1
                state["buf"].append(text)
                state["buf_len"] += len(text)
                flush_polish()
            finally:
                state["busy"] = False
        flush_polish(force=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def handle(signum, _frame):
        del signum
        stop.set()
        print("\n已退出。", flush=True)
        try:
            source.close()
        except Exception:
            pass
        os._exit(130)

    if install_signals:
        signal.signal(signal.SIGINT, handle)
        signal.signal(signal.SIGTERM, handle)
    chunk = max(2, int(source.rate * 0.1) * source.channels * 2)
    started = time.monotonic()
    last_level = started
    last_print = started
    last_partial = started
    level_buf = bytearray()
    logging.info("开始转写 backend=%s model=%s language=%s", args.backend, args.model, args.language)
    print(
        "正在听「%s」。文字不会马上出现：要先检测到人声，停顿约 %.1f 秒后才送去转写；这台 CPU 转写一段通常还要 20 秒以上。"
        % (getattr(source, "device_label", "") or "系统输出", args.silence),
        flush=True,
    )
    print("如果下面的 rms 一直接近 0，就是没采到声音，不会有文本。按一次 Ctrl+C 会立刻退出。", flush=True)
    notify("status", message="正在听「%s」。" % (getattr(source, "device_label", "") or "系统输出"))
    queued = 0

    def submit(segment: bytes, final: bool) -> None:
        nonlocal queued
        if rms_s16(segment) < 180:
            return
        duration = len(segment) / 2 / 16000
        job = Job(segment, final, time.monotonic())
        queued += 1
        if state["busy"] or not job_queue.empty():
            state["latest"] = job
            print("上一段还在转写，只保留最新 %.1f 秒，避免越积越慢。" % duration, flush=True)
            notify("status", message="上一段还在转写，只保留最新 %.1f 秒。" % duration)
            return
        logging.info("语音片段 %.2fs rms=%d 已入队", duration, rms_s16(segment))
        print("正在转写 %.1f 秒语音，请稍等..." % duration, flush=True)
        notify("status", message="正在转写 %.1f 秒语音。" % duration)
        job_queue.put(job)

    try:
        while not stop.is_set():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            raw = source.read(chunk, timeout=0.3)
            if raw is None:
                continue
            if raw == b"":
                break
            level_buf.extend(raw)
            now = time.monotonic()
            if now - last_level >= 0.12:
                level = rms_s16(bytes(level_buf))
                peak = peak_s16(bytes(level_buf))
                notify("level", rms=level, peak=peak)
                if now - last_print >= 2:
                    logging.info("采集电平 rms=%d peak=%d", level, peak)
                    print("rms=%d peak=%d" % (level, peak), flush=True)
                    last_print = now
                level_buf.clear()
                last_level = now
            pcm = resampler.push(raw, source.channels)
            for segment in segmenter.push(pcm):
                submit(segment, True)
            if args.partial_every and now - last_partial >= args.partial_every and job_queue.empty() and not state["busy"]:
                snap = segmenter.snapshot()
                if len(snap) > 16000 * 2:
                    submit(snap, False)
                last_partial = now
        if not stop.is_set():
            for segment in segmenter.flush():
                submit(segment, True)
        if queued == 0:
            print("没有检测到足够响的声音，所以没有文本。请先播放视频或语音，看到 rms 升到几百以上再等转写。", flush=True)
    finally:
        stop.set()
        source.close()
        thread.join(timeout=0.5)
        if out_fp:
            out_fp.close()
    logging.info("转写结束")
    return 0


def apply_chosen_output(args: argparse.Namespace, chosen: dict) -> None:
    if chosen.get("source") == "sckit":
        args.source = "sckit"
        args.uid = ""
        args.device = ""
        print("将用系统混音监听。菜单栏会出现录制指示。", flush=True)
        return
    args.source = "system"
    args.uid = str(chosen.get("uid") or "")
    args.device = str(chosen.get("name") or "")
    print("将监听「%s」。" % (args.device or args.uid), flush=True)


def run_choose(args: argparse.Namespace) -> int:
    require_binary(args.binary_path)
    chosen = choose_output(args.binary_path)
    apply_chosen_output(args, chosen)
    args.model = choose_model()
    print("将使用模型 %s。" % args.model, flush=True)
    if args.monitor:
        return run_monitor(args)
    return run_transcribe(args)


def main(argv: list[str] | None = None) -> int:
    if sys.version_info >= (3, 13):
        print("需要 Python 3.10 到 3.12。请使用 python3.12。", file=sys.stderr)
        return 2
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_runtime(args)
        if args.list:
            return cmd_list(args.binary_path)
        if args.choose:
            return run_choose(args)
        if args.serve or args.web:
            from sysaudio_stt.web import serve
            return serve(args.binary_path, headless=args.serve)
        if args.monitor:
            return run_monitor(args)
        return run_transcribe(args)
    except (CaptureError, STTError, OSError, ValueError) as exc:
        logging.error("%s", redact(str(exc)))
        return 1


queue_type = object
