#!/usr/bin/env python3
"""Очередь распознавания: следит за папкой входящих записей, транскрибирует
каждую (recognize.py: faster-whisper + sherpa-диаризация), причёсывает текст
через DeepSeek (refine.py) и складывает в папку транскриптов два файла:
*_source.txt (сырой диалог) и *_fine.txt (нормализованный). При наличии
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID шлёт уведомления и файлы в Telegram.
Также принимает аудио и видео, присланные боту в Telegram, и гонит их по
тому же циклу.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import refine

INCOMING = os.environ.get("INCOMING_DIR", "/data/incoming")
TRANSCRIPTS = os.environ.get("TRANSCRIPTS_DIR", "/data/transcripts")
STATE_FILE = os.path.join(TRANSCRIPTS, ".processed.json")
MODEL = os.environ.get("WHISPER_MODEL", "medium")
LANGUAGE = os.environ.get("LANGUAGE", "ru")
THREADS = os.environ.get("THREADS", "8")
CLUSTER_THRESHOLD = os.environ.get("DIAR_CLUSTER_THRESHOLD", "auto")
POLL_SEC = 5
# Пауза после появления файла: пара wav+m4a одной записи должна успеть доехать
SETTLE_SEC = 20
# Аудио храним ограниченно, расшифровки — всегда
AUDIO_RETENTION_DAYS = int(os.environ.get("AUDIO_RETENTION_DAYS", "60"))

TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m",
     "n", "o", "p", "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch",
     "", "y", "", "e", "yu", "ya"]))

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "всё", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы",
    "за", "бы", "по", "ее", "её", "мне", "было", "вот", "от", "меня", "еще",
    "ещё", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас",
    "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",
    "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы",
    "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "неё", "сейчас", "были", "куда", "зачем", "всех", "можно",
    "при", "об", "хотя", "это", "эта", "эти",
}


def translit(word):
    return "".join(TRANSLIT.get(ch, ch if ch.isalnum() else "") for ch in word.lower())


def topic_slug(dialog_path, max_words=3):
    """2-3 значимых слова из начала расшифровки, транслитом через дефис."""
    words = []
    try:
        with open(dialog_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("[") or not line.strip():
                    continue
                for w in re.findall(r"[а-яёa-z0-9]+", line.lower()):
                    if w in STOPWORDS or len(w) < 3:
                        continue
                    words.append(translit(w))
                    if len(words) >= max_words:
                        return "-".join(words)
    except OSError:
        pass
    return "-".join(words) if words else "zapis"


def nice_name(rec_id, dialog_path):
    """2026-08-27_13-05_tema-zapisi из yyyyMMddHHmmss и текста (без расширения)."""
    base = os.path.basename(rec_id)
    m = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", base)
    stamp = f"{m[1]}-{m[2]}-{m[3]}_{m[4]}-{m[5]}" if m else base
    return f"{stamp}_{topic_slug(dialog_path)}"

AUDIO_EXT = {".wav", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".flac"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp"}
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), msg, flush=True)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def save_state(state):
    os.makedirs(TRANSCRIPTS, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(state), f, ensure_ascii=False, indent=0)


def recording_id(path):
    """sn/имя записи без расширения и суффикса _compressed."""
    rel = os.path.relpath(path, INCOMING).replace("\\", "/")
    base, _ = os.path.splitext(rel)
    if base.endswith("_compressed"):
        base = base[: -len("_compressed")]
    return base


def scan_recordings():
    """Группирует входящие медиафайлы по записям: id -> список путей."""
    recordings = {}
    for root, _, files in os.walk(INCOMING):
        for name in files:
            path = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() not in MEDIA_EXT:
                continue
            recordings.setdefault(recording_id(path), []).append(path)
    return recordings


def pick_source(paths):
    """Из файлов одной записи предпочитает WAV, затем самый большой."""
    paths = sorted(paths, key=lambda p: (
        0 if p.lower().endswith(".wav") else 1, -os.path.getsize(p)))
    return paths[0]


def is_settled(paths):
    now = time.time()
    return all(now - os.path.getmtime(p) >= SETTLE_SEC for p in paths)


def telegram_send(text, file_path=None, chat_id=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        if file_path:
            boundary = "----workerform"
            with open(file_path, "rb") as f:
                content = f.read()
            parts = []
            for name, value in (("chat_id", chat_id), ("caption", text[:1024])):
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                             f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; "
                f"filename=\"{os.path.basename(file_path)}\"\r\n"
                f"Content-Type: text/plain\r\n\r\n".encode() + content + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode())
            body = b"".join(parts)
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendDocument", data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        else:
            body = json.dumps({"chat_id": chat_id, "text": text}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage", data=body,
                headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        log("[telegram] отправлено")
    except Exception as e:
        log(f"[telegram] ошибка отправки: {e}")


def reply_chat_id(rec_id):
    """chat_id для ответа: из меты записи (телеграм-загрузка) или None (общий)."""
    meta_path = os.path.join(INCOMING, rec_id + ".meta.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("chat_id")
    except (OSError, ValueError):
        return None


def make_fine(source_path, fine_path):
    """Строит причёсанную версию расшифровки. Возвращает True при успехе."""
    try:
        with open(source_path, encoding="utf-8") as f:
            text = f.read()
        fine = refine.refine_text(text)
        with open(fine_path, "w", encoding="utf-8") as f:
            f.write(fine)
        return True
    except Exception as e:
        log(f"[refine] ошибка: {e}")
        return False


def process(rec_id, paths):
    src = pick_source(paths)
    chat = reply_chat_id(rec_id)
    out_dir = os.path.join(TRANSCRIPTS, os.path.dirname(rec_id))
    os.makedirs(out_dir, exist_ok=True)
    tmp_out = os.path.join(TRANSCRIPTS, rec_id + ".dialog.txt")
    log(f"[queue] распознаю {src}")
    telegram_send(f"🎙 Новая запись: {rec_id}\nРаспознаю...", chat_id=chat)
    t0 = time.time()
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "recognize.py"),
           src, "--model", MODEL, "--language", LANGUAGE,
           "--threads", THREADS, "--cluster-threshold", CLUSTER_THRESHOLD,
           "--out", tmp_out]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        log(f"[queue] ОШИБКА распознавания {rec_id}:\n{result.stdout}\n{result.stderr}")
        telegram_send(f"❌ Ошибка распознавания {rec_id}:\n{result.stderr[-500:]}",
                      chat_id=chat)
        return False
    try:
        empty = not open(tmp_out, encoding="utf-8").read().strip()
    except OSError:
        empty = True
    if empty:
        log(f"[queue] пустая расшифровка {rec_id} — речи не найдено")
        telegram_send(f"⚠️ {os.path.basename(rec_id)}: речи не распознано, "
                      "файлы не создаю.", chat_id=chat)
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return True
    base = os.path.join(out_dir or TRANSCRIPTS, nice_name(rec_id, tmp_out))
    source_out = base + "_source.txt"
    fine_out = base + "_fine.txt"
    os.replace(tmp_out, source_out)
    has_fine = make_fine(source_out, fine_out)
    mins = (time.time() - t0) / 60
    log(f"[queue] готово за {mins:.1f} мин: {source_out}"
        + (f" + {fine_out}" if has_fine else " (без fine)"))
    telegram_send(f"✅ Готово: {os.path.basename(source_out)} (за {mins:.1f} мин)",
                  file_path=source_out, chat_id=chat)
    if has_fine:
        telegram_send(f"✨ Причёсанная версия: {os.path.basename(fine_out)}",
                      file_path=fine_out, chat_id=chat)
    else:
        telegram_send("⚠️ Причёсанную версию сделать не удалось (ошибка DeepSeek)",
                      chat_id=chat)
    return True


TG_OFFSET_FILE = os.path.join(TRANSCRIPTS, ".tg_offset")
TG_MAX_FILE_BYTES = 20 * 1024 * 1024  # лимит Bot API на скачивание файлов
# Типы вложений, на которые бот отвечает, даже если взять из них нечего
ATTACHMENT_KEYS = ("voice", "audio", "video", "video_note", "document",
                   "animation", "photo", "sticker")


def tg_api(method, params=None, timeout=30):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=json.dumps(params or {}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_tg_offset():
    try:
        with open(TG_OFFSET_FILE, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def save_tg_offset(offset):
    os.makedirs(TRANSCRIPTS, exist_ok=True)
    with open(TG_OFFSET_FILE, "w", encoding="utf-8") as f:
        f.write(str(offset))


def extract_tg_media(msg):
    """(file_id, имя, размер) из voice/audio/video/video_note или документа
    с аудио- либо видеодорожкой, иначе None."""
    voice = msg.get("voice")
    if voice:
        return voice["file_id"], "voice.ogg", voice.get("file_size", 0)
    audio = msg.get("audio")
    if audio:
        return (audio["file_id"], audio.get("file_name") or "audio.m4a",
                audio.get("file_size", 0))
    video = msg.get("video")
    if video:
        return (video["file_id"], video.get("file_name") or "video.mp4",
                video.get("file_size", 0))
    note = msg.get("video_note")
    if note:
        return note["file_id"], "video_note.mp4", note.get("file_size", 0)
    doc = msg.get("document")
    if doc:
        name = doc.get("file_name") or ""
        mime = doc.get("mime_type") or ""
        if (mime.startswith(("audio/", "video/"))
                or os.path.splitext(name)[1].lower() in MEDIA_EXT):
            return doc["file_id"], name or "media.bin", doc.get("file_size", 0)
    return None


def download_tg_file(file_id, dest):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    info = tg_api("getFile", {"file_id": file_id})
    file_path = info["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{token}/{urllib.parse.quote(file_path)}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def sanitize_name(name):
    return re.sub(r"[^\w.\-]", "_", name)[:100]


def handle_tg_message(msg):
    chat_id = msg.get("chat", {}).get("id")
    allowed = os.environ.get("TELEGRAM_CHAT_ID")
    if allowed and str(chat_id) != str(allowed):
        log(f"[telegram] игнорирую сообщение из чужого чата {chat_id}")
        return
    found = extract_tg_media(msg)
    if not found:
        attachment = next((k for k in ATTACHMENT_KEYS if msg.get(k)), None)
        if attachment:
            log(f"[telegram] вложение без звуковой дорожки ({attachment}) "
                f"из чата {chat_id}")
            telegram_send(
                "🤔 Не вижу здесь записи со звуком. Пришлите голосовое, кружок "
                "или файл с аудио/видео (wav, m4a, mp3, ogg, mp4, mov, mkv…).",
                chat_id=chat_id)
        return
    file_id, name, size = found
    if size and size > TG_MAX_FILE_BYTES:
        log(f"[telegram] файл {name} слишком велик ({size} байт)")
        telegram_send(
            f"❌ «{name}» весит {size / 1024 / 1024:.0f} МБ, а Telegram Bot API "
            "отдаёт боту только до 20 МБ. Загрузите запись через диктофон/сервер.",
            chat_id=chat_id)
        return
    stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(msg.get("date", time.time())))
    base, ext = os.path.splitext(sanitize_name(name))
    if ext.lower() == ".oga":
        ext = ".ogg"
    rec_name = f"{stamp}_{base}" if base else stamp
    dest = os.path.join(INCOMING, "telegram", rec_name + (ext.lower() or ".ogg"))
    log(f"[telegram] скачиваю из чата {chat_id}: {name}")
    try:
        download_tg_file(file_id, dest)
    except Exception as e:
        log(f"[telegram] не удалось скачать файл: {e}")
        telegram_send(f"❌ Не удалось скачать файл: {e}", chat_id=chat_id)
        return
    meta = os.path.join(INCOMING, "telegram", rec_name + ".meta.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"chat_id": chat_id}, f)
    telegram_send("📥 Принял, поставил в очередь на распознавание.", chat_id=chat_id)


def poll_telegram():
    """Забирает новые сообщения бота и складывает аудио во входящую очередь."""
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        return
    offset = load_tg_offset()
    try:
        resp = tg_api("getUpdates", {"offset": offset, "timeout": 0,
                                     "allowed_updates": ["message"]})
    except Exception as e:
        log(f"[telegram] ошибка getUpdates: {e}")
        return
    if not resp or not resp.get("ok"):
        return
    for upd in resp["result"]:
        offset = max(offset, upd["update_id"] + 1)
        try:
            msg = upd.get("message")
            if msg:
                handle_tg_message(msg)
        except Exception as e:
            log(f"[telegram] ошибка обработки сообщения: {e}")
    save_tg_offset(offset)


def cleanup_old_audio(processed):
    """Удаляет обработанные аудио и кэши старше AUDIO_RETENTION_DAYS."""
    cutoff = time.time() - AUDIO_RETENTION_DAYS * 86400
    for root, _, files in os.walk(INCOMING):
        for name in files:
            path = os.path.join(root, name)
            if os.path.getmtime(path) > cutoff:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in MEDIA_EXT and recording_id(path) not in processed:
                continue
            if ext not in MEDIA_EXT and ext != ".json":
                continue
            try:
                os.remove(path)
                log(f"[cleanup] удалён (старше {AUDIO_RETENTION_DAYS} дн.): {path}")
            except OSError as e:
                log(f"[cleanup] не удалось удалить {path}: {e}")


def main():
    log(f"worker запущен: {INCOMING} -> {TRANSCRIPTS}, model={MODEL}, "
        f"хранение аудио {AUDIO_RETENTION_DAYS} дн.")
    processed = load_state()
    last_cleanup = 0
    while True:
        try:
            if time.time() - last_cleanup > 86400:
                cleanup_old_audio(processed)
                last_cleanup = time.time()
            poll_telegram()
            for rec_id, paths in sorted(scan_recordings().items()):
                if rec_id in processed or not is_settled(paths):
                    continue
                if process(rec_id, paths):
                    processed.add(rec_id)
                    save_state(processed)
                else:
                    # не ретраим бесконечно: помечаем, чтобы не зациклиться
                    processed.add(rec_id)
                    save_state(processed)
        except Exception as e:
            log(f"[queue] ошибка цикла: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
