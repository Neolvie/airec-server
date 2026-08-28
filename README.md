# AIREC Upload Server

Сервер приёма аудиозаписей от диктофона AIREC с автоматической расшифровкой:
faster-whisper + диаризация sherpa-onnx (кто что сказал), «причёсывание»
текста через DeepSeek и доставка результатов в Telegram. Работает целиком
локально/на своём VPS, в облако уходит только текст на нормализацию.

## Как это работает

```
диктофон AIREC ──POST /api/upload──▶ server (Node) ──▶ data/incoming/
Telegram-бот  ──voice/видео/файл──▶ worker (poll)  ──▶ data/incoming/telegram/
                                                          │
                                          worker: whisper + диаризация
                                                          │
                                        data/transcripts/<sn>/
                                          ├─ <дата>_<тема>_source.txt  (сырой диалог)
                                          └─ <дата>_<тема>_fine.txt    (причёсанный DeepSeek)
                                                          │
                                                оба файла → Telegram
```

- `*_source.txt` — расшифровка как есть: таймкоды, роли «Собеседник N».
- `*_fine.txt` — тот же текст без повторов и слов-паразитов, роли сохранены.
- Запись, присланная боту напрямую, проходит тот же цикл; ответ уходит
  отправителю. Принимаются голосовые, кружки, аудио и видео — файлом или
  вложением: wav/m4a/mp3/ogg/opus/aac/flac и mp4/m4v/mov/mkv/webm/avi/3gp
  (из видео берётся звуковая дорожка), до 20 МБ — лимит Telegram Bot API.
  На вложение, в котором звука нет, бот отвечает подсказкой. Принимаются
  только сообщения из чата `TELEGRAM_CHAT_ID`, остальные игнорируются.
- Исходное аудио хранится `AUDIO_RETENTION_DAYS` дней, расшифровки — всегда.

## API

- `POST /api/upload` — multipart/form-data с полями `file`, `fileName`, `sn`.
  200 при успехе, дедупликация по `sn + fileName`.
- `GET /api/files` — JSON-список принятых файлов.
- `GET /` — проверка, что сервер жив.

## Установка на VPS (Docker)

Нужен любой Linux-VPS с 4+ ГБ RAM (whisper `medium` на CPU) и Docker.
Для модели `small` хватит и 2 ГБ.

### 1. Docker

```bash
curl -fsSL https://get.docker.com | sh
```

### 2. Клонировать и настроить

```bash
git clone https://github.com/Neolvie/airec-server.git
cd airec-server
cp .env.example .env
nano .env   # вписать TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY
```

Где взять значения:

- `TELEGRAM_BOT_TOKEN` — создать бота у [@BotFather](https://t.me/BotFather);
- `TELEGRAM_CHAT_ID` — написать своему боту что-нибудь, затем узнать свой id
  у [@userinfobot](https://t.me/userinfobot);
- `DEEPSEEK_API_KEY` — [platform.deepseek.com](https://platform.deepseek.com).
  Можно оставить пустым — тогда будет приходить только `*_source.txt`.

### 3. Скачать модели диаризации (~46 МБ)

```bash
sh scripts/download-models.sh
```

Модель whisper скачается сама при первом запуске (кэшируется в docker-volume).

### 4. Запустить

```bash
docker compose up -d --build
```

Проверка:

```bash
curl http://localhost:8080/
docker compose logs -f worker
```

Первая расшифровка стартует медленно — worker скачивает модель whisper.

### 5. HTTPS для диктофона

Приложению AIREC нужен внешний HTTPS-URL. Самый простой способ — Caddy
(автоматический Let's Encrypt). На VPS с доменом, направленным на его IP:

```bash
sudo apt install -y caddy
```

`/etc/caddy/Caddyfile`:

```
rec.example.com {
    reverse_proxy localhost:8080
}
```

```bash
sudo systemctl reload caddy
```

В настройках AIREC указать `https://rec.example.com/api/upload`.

Альтернатива без своего домена — постоянный туннель
(`cloudflared tunnel`, ngrok с зарезервированным доменом и т.п.).

### Обновление

```bash
cd airec-server
git pull
docker compose up -d --build
```

## Проверка вручную

```bash
curl -X POST http://localhost:8080/api/upload \
  -F "file=@recording.wav;type=audio/wav" \
  -F "fileName=20260827104300.wav" \
  -F "sn=DEVICE_SN_123456"
```

Или просто отправить голосовое сообщение боту.

## Переменные окружения

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Модель faster-whisper (`small` — быстрее, `large-v3` — точнее) |
| `LANGUAGE` | `ru` | Язык распознавания |
| `THREADS` | `8` | Потоки CPU на распознавание |
| `TELEGRAM_BOT_TOKEN` | — | Токен бота (уведомления + приём записей) |
| `TELEGRAM_CHAT_ID` | — | Чат для уведомлений; только из него принимаются записи |
| `AUDIO_RETENTION_DAYS` | `60` | Сколько дней хранить исходное аудио |
| `DEEPSEEK_API_KEY` | — | Ключ DeepSeek для `*_fine.txt` (пусто — без причёсывания) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Модель DeepSeek |
| `DIAR_CLUSTER_THRESHOLD` | `auto` | `auto` — адаптивный подбор числа говорящих; число — фиксированный порог кластеризации (меньше = больше говорящих) |
