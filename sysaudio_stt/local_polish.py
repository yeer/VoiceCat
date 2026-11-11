"""Fast local polish with Hugging Face Chinese correction models."""

from __future__ import annotations

import re
import threading

CSC_MODEL = 'shibing624/macbert4csc-base-chinese'
PUNCT_MODEL = 'p208p2002/zh-wiki-punctuation-restore'
_LOCK = threading.Lock()
_MODELS = {}
_DIGITS = '零一二三四五六七八九'
_KEEP = set(' \n“”‘’…—')


def collapse_repeats(text):
    return re.sub(r'(.{1,4})\1{2,}', r'\1', text or '')


def digits_to_hanzi(text):
    return re.sub(
        r'(?<![0-9])([0-9])(?![0-9])',
        lambda match: _DIGITS[int(match.group(1))],
        text or '',
    )


def apply_glossary(text, glossary):
    cleaned = text or ''
    terms = glossary or ''
    if '换峰' in terms:
        cleaned = cleaned.replace('幻风', '换峰')
    return cleaned


def _load(name):
    cached = _MODELS.get(name)
    if cached:
        return cached
    with _LOCK:
        cached = _MODELS.get(name)
        if cached:
            return cached
        from sysaudio_stt.models_catalog import configure_model_cache
        from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

        configure_model_cache()
        tokenizer = AutoTokenizer.from_pretrained(name)
        if name == PUNCT_MODEL:
            model = AutoModelForTokenClassification.from_pretrained(name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(name)
        model.eval()
        _MODELS[name] = (tokenizer, model)
        return _MODELS[name]


def correct_spelling(text, protected=()):
    source = text or ''
    if not source.strip():
        return source
    import torch

    tokenizer, model = _load(CSC_MODEL)
    encoded = tokenizer(source, return_tensors='pt', truncation=True, max_length=128)
    with torch.inference_mode():
        pred = model(**encoded).logits.argmax(-1)[0]
    decoded = tokenizer.decode(pred, skip_special_tokens=True).replace(' ', '')
    chars = []
    for index, char in enumerate(source):
        if char in _KEEP or index >= len(decoded):
            chars.append(char)
        elif char.isascii() and char.lower() == decoded[index].lower():
            chars.append(char)
        else:
            chars.append(decoded[index])
    corrected = ''.join(chars)
    for term in protected:
        start = 0
        while term and start < len(source):
            found = source.find(term, start)
            if found < 0:
                break
            corrected = corrected[:found] + term + corrected[found + len(term):]
            start = found + len(term)
    return corrected


def restore_punctuation(text):
    source = (text or '').replace(' ', '')
    if not source:
        return text or ''
    import torch

    tokenizer, model = _load(PUNCT_MODEL)
    encoded = tokenizer(source, return_tensors='pt', truncation=True, max_length=128)
    with torch.inference_mode():
        labels = model(**encoded).logits.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    id2label = model.config.id2label
    parts = []
    for token, label_id in zip(tokens, labels):
        if token in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        char = token[2:] if token.startswith('##') else token
        if char.startswith('[') and char.endswith(']'):
            continue
        parts.append(char)
        name = id2label.get(label_id, id2label.get(str(label_id), 'O'))
        if isinstance(name, str) and name.startswith('S-') and len(name) > 2:
            parts.append(name[2:])
    return ''.join(parts)


def warmup():
    _load(CSC_MODEL)
    _load(PUNCT_MODEL)


def polish_locally(text, script='', glossary=''):
    cleaned = apply_glossary(collapse_repeats(text or ''), glossary)
    protected = []
    if glossary:
        for term in glossary.replace('，', ' ').replace(',', ' ').split():
            if term and term in cleaned:
                protected.append(term)
    if script != 'zh-tw':
        cleaned = digits_to_hanzi(cleaned)
        for digit in _DIGITS:
            if digit in cleaned and digit not in protected:
                protected.append(digit)
    cleaned = correct_spelling(cleaned, protected)
    cleaned = restore_punctuation(cleaned)
    cleaned = apply_glossary(cleaned, glossary)
    return cleaned.strip()
