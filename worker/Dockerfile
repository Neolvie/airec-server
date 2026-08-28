FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir faster-whisper sherpa-onnx numpy

WORKDIR /app
COPY recognize.py worker.py refine.py ./

ENV MODELS_DIR=/models \
    INCOMING_DIR=/data/incoming \
    TRANSCRIPTS_DIR=/data/transcripts \
    HF_HOME=/cache/huggingface

CMD ["python", "-u", "worker.py"]
