"""Local models that this Intel Mac can run, plus download status."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

CATALOG = (
    {"id": "openai/whisper-small", "family": "transformers", "family_label": "Whisper", "label": "small", "note": "原来的识别", "size": "约 480MB", "kind": "transformers", "runtime": "openai/whisper-small", "repo": "openai/whisper-small"},
    {"id": "openai/whisper-large-v3-turbo", "family": "transformers", "family_label": "Whisper", "label": "turbo", "note": "更准，CPU 上更慢", "size": "约 1.6GB", "kind": "transformers", "runtime": "openai/whisper-large-v3-turbo", "repo": "openai/whisper-large-v3-turbo"},
    {"id": "openai/whisper-large-v3", "family": "transformers", "family_label": "Whisper", "label": "large-v3", "note": "最准也最慢", "size": "约 3GB", "kind": "transformers", "runtime": "openai/whisper-large-v3", "repo": "openai/whisper-large-v3"},
    {"id": "fw-small", "family": "faster-whisper", "family_label": "faster-whisper", "label": "small", "note": "CPU int8", "size": "约 480MB", "kind": "faster", "runtime": "small", "repo": "Systran/faster-whisper-small"},
    {"id": "fw-turbo", "family": "faster-whisper", "family_label": "faster-whisper", "label": "turbo", "note": "CPU int8", "size": "约 1.6GB", "kind": "faster", "runtime": "large-v3-turbo", "repo": "Systran/faster-whisper-large-v3-turbo"},
    {"id": "fw-large-v3", "family": "faster-whisper", "family_label": "faster-whisper", "label": "large-v3", "note": "CPU int8，仍慢于 turbo", "size": "约 3GB", "kind": "faster", "runtime": "large-v3", "repo": "Systran/faster-whisper-large-v3"},
    {"id": "sv-int8-2025", "family": "sherpa", "family_label": "sherpa", "label": "SenseVoice-Small 2025", "note": "准流式，短段很快", "size": "158MB", "kind": "sensevoice", "url": BASE + "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"},
    {"id": "sv-int8-2024", "family": "sherpa", "family_label": "sherpa", "label": "SenseVoice-Small 2024", "note": "上一版 int8", "size": "155MB", "kind": "sensevoice", "url": BASE + "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"},
    {"id": "para-zh-int8", "family": "sherpa", "family_label": "sherpa", "label": "Paraformer 中文 int8", "note": "非流式，干净普通话通常更准", "size": "218MB", "kind": "offline-paraformer", "url": BASE + "sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2"},
    {"id": "para-zh-small", "family": "sherpa", "family_label": "sherpa", "label": "Paraformer 中文 small", "note": "更小更快", "size": "74MB", "kind": "offline-paraformer", "url": BASE + "sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2"},
    {"id": "stream-para-zh-en", "family": "sherpa", "family_label": "sherpa", "label": "流式 Paraformer 中英", "note": "边听边出字", "size": "999MB", "kind": "online-paraformer", "url": BASE + "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2"},
    {"id": "stream-zip-zh-en", "family": "sherpa", "family_label": "sherpa", "label": "流式 Zipformer 中英", "note": "Transducer，更快，准确度通常低一截", "size": "488MB", "kind": "online-transducer", "url": BASE + "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"},
    {"id": "stream-zip-zh-int8", "family": "sherpa", "family_label": "sherpa", "label": "流式 Zipformer 中文 int8", "note": "小而快", "size": "126MB", "kind": "online-transducer", "url": BASE + "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2"},
    {"id": "stream-xasr-480", "family": "sherpa", "family_label": "sherpa", "label": "流式 Zipformer 480ms", "note": "约 480ms 一块，带标点", "size": "128MB", "kind": "online-transducer", "url": BASE + "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2"},
)

JOBS = {}
LOCK = threading.Lock()

def get_model(model_id):
    for item in CATALOG:
        if item["id"] == model_id:
            return item
    return None

def project_root():
    env = os.environ.get("SYSAUDIO_ROOT") or os.environ.get("VOICECAT_ROOT")
    if env and (Path(env) / "sysaudio_stt").is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sysaudio_stt" / "__main__.py").is_file():
            return parent
    return here.parents[1]

def models_root():
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"
    path = base / "VoiceCat" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def configure_model_cache():
    hub = models_root() / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hub.parent)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_CACHE"] = str(hub)
    _migrate_old_models(hub)
    return hub

def _hf_repo_dir(repo):
    return configure_model_cache() / ("models--" + repo.replace("/", "--"))

def _hf_installed(repo):
    snaps = _hf_repo_dir(repo) / "snapshots"
    if not snaps.is_dir():
        return False
    for child in snaps.iterdir():
        if child.is_dir() and any(child.iterdir()):
            return True
    return False

def sherpa_dir(model_id):
    return models_root() / "sherpa" / model_id

def _migrate_old_models(hub):
    old_hub = Path.home() / ".cache" / "huggingface" / "hub"
    repos = [item["repo"] for item in CATALOG if item.get("repo")]
    repos.extend((
        "shibing624/macbert4csc-base-chinese",
        "p208p2002/zh-wiki-punctuation-restore",
    ))
    for repo in repos:
        name = "models--" + repo.replace("/", "--")
        src = old_hub / name
        dest = hub / name
        if src.is_dir() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
    old_models = Path.home() / "Library" / "Application Support" / "实时识别" / "models"
    if not old_models.is_dir():
        return
    for item in CATALOG:
        if item.get("repo"):
            continue
        src = old_models / item["id"]
        dest = sherpa_dir(item["id"])
        if src.is_dir() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

def _find_role(root, role):
    hits = []
    for path in root.rglob("*.onnx"):
        name = path.name.lower()
        if role == "model" and name.startswith("model"):
            hits.append(path)
        elif role != "model" and role in name:
            hits.append(path)
    if not hits:
        return None
    hits.sort(key=lambda path: (0 if "int8" in path.name.lower() else 1, len(path.name)))
    return hits[0]

def _has_tokens(root):
    return any(path.name == "tokens.txt" for path in root.rglob("tokens.txt"))

def sherpa_installed(item):
    root = sherpa_dir(item["id"])
    if not root.is_dir() or not _has_tokens(root):
        return False
    kind = item["kind"]
    if kind in ("sensevoice", "offline-paraformer"):
        return _find_role(root, "model") is not None
    if kind == "online-paraformer":
        return _find_role(root, "encoder") is not None and _find_role(root, "decoder") is not None
    if kind == "online-transducer":
        return all(_find_role(root, role) is not None for role in ("encoder", "decoder", "joiner"))
    return False

def is_installed(item):
    if item["kind"] in ("transformers", "faster"):
        return _hf_installed(item["repo"])
    return sherpa_installed(item)

def public_rows():
    with LOCK:
        jobs = {key: dict(value) for key, value in JOBS.items()}
    rows = []
    for item in CATALOG:
        job = jobs.get(item["id"], {})
        rows.append({
            "id": item["id"],
            "family": item["family"],
            "family_label": item["family_label"],
            "label": item["label"],
            "note": item["note"],
            "size": item["size"],
            "installed": is_installed(item),
            "status": job.get("status") or "",
            "received": int(job.get("received") or 0),
            "total": int(job.get("total") or 0),
            "error": job.get("error") or "",
        })
    return rows

def installed_engines():
    groups = []
    for item in CATALOG:
        if not is_installed(item):
            continue
        if not groups or groups[-1]["id"] != item["family"]:
            groups.append({"id": item["family"], "label": item["family_label"], "models": []})
        groups[-1]["models"].append({"id": item["id"], "label": item["label"], "note": item["note"]})
    return groups

def _set_job(model_id, **updates):
    with LOCK:
        JOBS.setdefault(model_id, {}).update(updates)

def _download_hf(item):
    from huggingface_hub import snapshot_download
    model_id = item["id"]

    class Hook:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total") or 0
            self.n = 0
            _set_job(model_id, total=self.total)

        def update(self, n=1):
            self.n += n
            _set_job(model_id, received=self.n, total=self.total or 0)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    snapshot_download(item["repo"], cache_dir=str(configure_model_cache()), tqdm_class=Hook)

def _download_sherpa(item):
    dest_dir = sherpa_dir(item["id"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir.parent / (item["id"] + ".tar.bz2")
    request = urllib.request.Request(item["url"], headers={"User-Agent": "sysaudio-stt"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        _set_job(item["id"], total=total, received=0)
        with archive.open("wb") as handle:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                _set_job(item["id"], received=received, total=total)
    with tarfile.open(archive, "r:bz2") as handle:
        handle.extractall(dest_dir)
    archive.unlink(missing_ok=True)

def _run(item):
    try:
        if item["kind"] in ("transformers", "faster"):
            _download_hf(item)
        else:
            _download_sherpa(item)
        if not is_installed(item):
            raise RuntimeError("下载结束但没有找到模型文件")
        _set_job(item["id"], status="done", error="")
    except Exception as exc:
        _set_job(item["id"], status="error", error=str(exc))

def start_download(model_id):
    item = get_model(model_id)
    if item is None:
        raise ValueError("没有这个模型")
    if is_installed(item):
        _set_job(model_id, status="done", error="")
        return "已下载"
    with LOCK:
        active = [key for key, job in JOBS.items() if job.get("status") == "downloading"]
        if active and model_id not in active:
            raise RuntimeError("已有模型在下载，请等它完成")
        if JOBS.get(model_id, {}).get("status") == "downloading":
            return "正在下载"
        JOBS[model_id] = {"status": "downloading", "received": 0, "total": 0, "error": ""}
    threading.Thread(target=_run, args=(item,), daemon=True).start()
    return "开始下载"

def open_models_dir():
    path = models_root()
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
    return str(path)

def delete_model(model_id):
    item = get_model(model_id)
    if item is None:
        raise ValueError("没有这个模型")
    with LOCK:
        if JOBS.get(model_id, {}).get("status") == "downloading":
            raise RuntimeError("正在下载，不能删除")
    path = (_hf_repo_dir(item["repo"]) if item.get("repo") else sherpa_dir(item["id"])).resolve()
    root = models_root().resolve()
    if path != root and root not in path.parents:
        raise RuntimeError("拒绝删除 VoiceCat 模型目录以外的文件")
    if path.exists():
        shutil.rmtree(path)
    _set_job(model_id, status="", error="", received=0, total=0)
    return "已删除"

configure_model_cache()
