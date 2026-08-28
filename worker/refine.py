#!/usr/bin/env python3
"""Причёсывание расшифровки через DeepSeek API (OpenAI-совместимый chat/completions).

refine_text(text) -> str: нормализует текст расшифровки, сохраняя роли
(Собеседник N) и смысл. Длинный текст режется на куски по границам реплик.
Бросает RuntimeError, если API недоступен или не задан DEEPSEEK_API_KEY.
"""
import json
import os
import urllib.request

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
CHUNK_CHARS = int(os.environ.get("REFINE_CHUNK_CHARS", "9000"))

SYSTEM_PROMPT = """Роль: Ты расшифровщик голосовых записей.

Нормализация текста:
— исправляй паузы, повторы и слова-паразиты; проясняй местоимения и время;
— сохраняй смысл без выдумывания фактов;
— не придумывай никакие дополнительные факты. Просто хорошо перескажи расшифрованный текст записи;
— если в транскрипте есть метки говорящих ("Собеседник 1", "Собеседник 2" и т.п.), сохраняй разбиение по ролям: каждую реплику начинай с новой строки в формате "Собеседник N: текст". Метки времени можно опустить;
— если меток говорящих нет, верни связный текст без ролей.

Язык: русский.
Вывод: только готовая запись, без пояснений и меток."""


def _chunks(text, limit):
    """Режет текст на куски <= limit по границам пустых строк (реплик)."""
    if len(text) <= limit:
        return [text]
    blocks = text.split("\n\n")
    chunks, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > limit:
            chunks.append(cur)
            cur = b
        else:
            cur = cur + "\n\n" + b if cur else b
    if cur:
        chunks.append(cur)
    return chunks


def _complete(text):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "— Транскрипт записи: " + text},
        ],
        "temperature": 1.0,
        "max_tokens": 8000,
        "stream": False,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def refine_text(text):
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY не задан")
    if not text.strip():
        raise RuntimeError("пустой транскрипт")
    parts = [_complete(c) for c in _chunks(text.strip(), CHUNK_CHARS)]
    return "\n\n".join(p for p in parts if p) + "\n"


if __name__ == "__main__":
    import sys
    src = sys.argv[1]
    with open(src, encoding="utf-8") as f:
        print(refine_text(f.read()))
