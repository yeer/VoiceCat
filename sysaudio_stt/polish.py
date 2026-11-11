"""Polish recognized text through the configured LLM gateway."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sysaudio_stt.stt import STTError, _request


def app_data_dir():
    override = os.environ.get('SYSAUDIO_DATA')
    if override:
        path = Path(override)
    elif sys.platform == 'darwin':
        path = Path.home() / 'Library' / 'Application Support' / 'VoiceCat'
    elif sys.platform == 'win32':
        base = os.environ.get('APPDATA') or str(Path.home() / 'AppData' / 'Roaming')
        path = Path(base) / 'VoiceCat'
    else:
        path = Path.home() / '.local' / 'share' / 'VoiceCat'
    path.mkdir(parents=True, exist_ok=True)
    old = Path.home() / 'Library' / 'Application Support' / '实时识别'
    if old.is_dir() and old != path:
        for name in ('settings.json', 'llm.toml', 'glossary.txt', 'polish_prompt.txt'):
            src = old / name
            dest = path / name
            if src.is_file() and not dest.exists():
                dest.write_bytes(src.read_bytes())
    return path


def _legacy_project_file(name):
    return Path(__file__).resolve().parents[1] / name


def _data_file(name):
    dest = app_data_dir() / name
    src = _legacy_project_file(name)
    if not dest.exists() and src.is_file():
        dest.write_bytes(src.read_bytes())
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
    if dest.is_file() and src.is_file() and src.resolve() != dest.resolve():
        src.unlink()
    return dest


def default_llm_config_path():
    return _data_file('llm.toml')


def load_llm_config(path=None):
    import tomllib

    config_path = Path(path) if path else default_llm_config_path()
    if not config_path.is_file():
        raise STTError('找不到大模型配置文件: ' + str(config_path))
    with config_path.open('rb') as handle:
        data = tomllib.load(handle)
    required = ('base_url', 'wire_api', 'model')
    missing = [key for key in required if not str(data.get(key) or '').strip()]
    if missing:
        raise STTError('大模型配置缺少字段: ' + ', '.join(missing))
    return {
        'base_url': str(data['base_url']).rstrip('/'),
        'env_key': str(data.get('env_key') or '').strip(),
        'api_key': str(data.get('api_key') or '').strip(),
        'wire_api': str(data['wire_api']).strip(),
        'model': str(data['model']).strip(),
        'path': str(config_path),
    }


def resolve_api_key(config):
    file_key = str(config.get('api_key') or '').strip()
    if file_key:
        return file_key
    key_name = str(config.get('env_key') or '').strip()
    if key_name:
        return os.environ.get(key_name) or ''
    return ''


def llm_credentials():
    config = load_llm_config()
    ready = bool(resolve_api_key(config))
    return config, 'llm.toml 的 api_key', ready


def public_llm_config():
    try:
        config = load_llm_config()
    except STTError:
        return {'base_url': '', 'wire_api': 'responses', 'model': '', 'has_api_key': False}
    return {
        'base_url': config['base_url'],
        'wire_api': config['wire_api'],
        'model': config['model'],
        'has_api_key': bool(resolve_api_key(config)),
    }


def save_llm_config(base_url, wire_api, model, api_key=None):
    path = default_llm_config_path()
    current_key = ''
    if path.is_file():
        try:
            current_key = load_llm_config(path).get('api_key') or ''
        except STTError:
            current_key = ''
    if api_key is not None and str(api_key).strip():
        current_key = str(api_key).strip()
    def quote(value):
        return json.dumps(str(value or ''), ensure_ascii=False)
    rows = [
        'base_url = ' + quote(base_url),
        'env_key = "LLM_API_KEY"',
        'wire_api = ' + quote(wire_api or 'responses'),
        'model = ' + quote(model),
        'api_key = ' + quote(current_key),
    ]
    path.write_text(chr(10).join(rows) + chr(10), encoding='utf-8')
    return public_llm_config()


def settings_path():
    return _data_file('settings.json')


def load_settings():
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(updates):
    data = load_settings()
    for key, value in (updates or {}).items():
        if key == 'api_key':
            continue
        data[key] = value
    settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return data


def parse_model_ids(body):
    text = body.decode('utf-8', 'replace').strip()
    if not text:
        return []
    data = json.loads(text)
    rows = []
    if isinstance(data, dict):
        rows = data.get('data') or data.get('models') or []
    elif isinstance(data, list):
        rows = data
    ids = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            ids.append(row.strip())
        elif isinstance(row, dict) and str(row.get('id') or '').strip():
            ids.append(str(row['id']).strip())
    return ids


def list_llm_models(timeout=20):
    config, key_name, ready = llm_credentials()
    api_key = resolve_api_key(config)
    if not ready or not api_key:
        raise STTError('缺少润色所需的 ' + key_name)
    url = config['base_url'] + '/models'
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + api_key}, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise STTError('获取模型列表失败 HTTP %s' % exc.code) from exc
    except urllib.error.URLError as exc:
        raise STTError('无法连接模型列表') from exc
    return parse_model_ids(body)


def polish_prompt_path():
    return _data_file('polish_prompt.txt')


def load_polish_prompt(path=None):
    target = Path(path) if path else polish_prompt_path()
    if target.is_file():
        return target.read_text(encoding='utf-8')
    return default_polish_prompt()


def save_polish_prompt(text, path=None):
    target = Path(path) if path else polish_prompt_path()
    target.write_text(text if text is not None else '', encoding='utf-8')
    return str(target)


def glossary_path():
    return _data_file('glossary.txt')


def load_glossary(path=None):
    target = Path(path) if path else glossary_path()
    if target.is_file():
        return target.read_text(encoding='utf-8').strip()
    return '换峰'


def save_glossary(text, path=None):
    target = Path(path) if path else glossary_path()
    target.write_text((text or '').strip(), encoding='utf-8')
    return str(target)


def corrections_path():
    return _data_file('corrections.json')


def load_corrections():
    path = corrections_path()
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get('wrong') and row.get('right')]


def apply_corrections(text):
    cleaned = text or ''
    rows = sorted(load_corrections(), key=lambda row: len(row['wrong']), reverse=True)
    for row in rows:
        cleaned = cleaned.replace(row['wrong'], row['right'])
    return cleaned


def add_correction(wrong, right):
    source = (wrong or '').strip()
    target = (right or '').strip()
    if not source or not target or source == target:
        raise ValueError('请选择识别错的文字，并填写正确写法')
    glossary = load_glossary()
    terms = [part for part in glossary.replace('、', ' ').replace('，', ' ').replace(',', ' ').split() if part]
    if target not in terms:
        terms.append(target)
    save_glossary(' '.join(terms))
    rows = [row for row in load_corrections() if row['wrong'] != source]
    rows.append({'wrong': source, 'right': target})
    corrections_path().write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'glossary': ' '.join(terms), 'corrections': rows}


def default_polish_prompt():
    return (
        '你是语音识别校对员。只输出校对后的文本，不要解释，不要加引号。'
        '补上自然的标点。'
        '合并口语里无意义的重复，例如「然后然后然后」改为「然后」。'
        '修正明显听错的同音词，例如「幻风下来」改为「换峰下来」，「你看这个8」改为「你看这个八」。'
        '口语句子里的数字写成汉字，不要留阿拉伯数字。'
        '不要补充原文没有的信息，不要总结。原文已经清楚时只做最小修改。'
    )


def build_polish_messages(text, script='', glossary='', instruction=''):
    system = (instruction or '').strip() or load_polish_prompt().strip() or default_polish_prompt()
    if script == 'zh-cn':
        system += '输出使用简体中文。'
    elif script == 'zh-tw':
        system += '输出使用繁体中文。'
    terms = (glossary or '').strip()
    if terms:
        system += '这些说法优先保留：' + terms
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': text},
    ]


def accept_polish(original, polished):
    cleaned = (polished or '').strip()
    for mark in ('"', "'", '「', '」'):
        cleaned = cleaned.strip(mark)
    source = (original or '').strip()
    if not cleaned:
        return source
    if len(cleaned) > max(80, len(source) * 3):
        return source
    return cleaned


def parse_responses_text(body):
    text = body.decode('utf-8', 'replace').strip()
    if not text:
        return ''
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text
    if isinstance(data.get('error'), dict) and data['error'].get('message'):
        raise STTError('润色接口失败: ' + str(data['error']['message'])[:300])
    if isinstance(data.get('output_text'), str) and data['output_text'].strip():
        return data['output_text'].strip()
    parts = []
    for item in data.get('output') or []:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        content = item.get('content')
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get('text') or ''))
    return ''.join(parts).strip()


def polish_text(text, script='', glossary='', model='', instruction='', timeout=20):
    source = (text or '').strip()
    if not source:
        return ''
    config, key_name, ready = llm_credentials()
    api_key = resolve_api_key(config)
    if not ready or not api_key:
        raise STTError('缺少润色所需的 ' + key_name)
    if config['wire_api'] != 'responses':
        raise STTError('不支持的 wire_api: ' + config['wire_api'])
    messages = build_polish_messages(source, script, glossary, instruction)
    payload = {
        'model': model or config['model'],
        'instructions': messages[0]['content'],
        'input': messages[1]['content'],
        'temperature': 0,
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
    }
    raw, _content_type = _request(
        config['base_url'] + '/responses',
        json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers,
        timeout,
    )
    return accept_polish(source, parse_responses_text(raw))
