#!/usr/bin/env python3
"""Транскрибация аудио (faster-whisper) с диаризацией (pyannote) в формат диалога.

Использование:
    recognize.py <audio> [--model base|small|medium|large-v3] [--speakers N]
                 [--language ru] [--out <txt>] [--threads N] [--remerge]

Выход: текстовый файл-диалог:
    [00:12:34] Собеседник 1:
    текст реплики...

Промежуточные результаты кэшируются рядом с выходным файлом:
    <out>.words.json  — слова whisper с таймкодами
    <out>.diar.json   — интервалы говорящих
--remerge пересобирает диалог из кэшей без повторного распознавания.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def read_token(args):
    if args.hf_token:
        return args.hf_token
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    token_file = os.path.join(PROJECT_DIR, ".hf_token")
    if os.path.exists(token_file):
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()
    return None


def to_wav16k(src, normalize=False):
    """Декодирует любой входной формат в моно WAV 16 кГц во временный файл.

    normalize=True добавляет speechnorm — адаптивно подтягивает тихую речь
    (для whisper; диаризации отдаётся оригинальный уровень).
    """
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if normalize:
        cmd += ["-af", "speechnorm=e=12.5:r=0.0001:l=1"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav]
    subprocess.run(cmd, check=True)
    return wav


def transcribe(wav, model_name, language, threads, batch_size):
    import inspect
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    t0 = time.time()
    log(f"[whisper] загрузка модели '{model_name}' (CPU, int8, threads={threads})")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
    pipe = BatchedInferencePipeline(model=model)
    log(f"[whisper] модель готова за {time.time() - t0:.0f}s, распознаю (batch={batch_size})...")
    kwargs = dict(language=language, vad_filter=True, word_timestamps=True,
                  condition_on_previous_text=False, no_speech_threshold=0.9,
                  log_prob_threshold=-2.0, batch_size=batch_size)
    accepted = set(inspect.signature(pipe.transcribe).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    segments, info = pipe.transcribe(wav, **kwargs)
    words = []
    last_logged = 0
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"w": w.word.strip(), "start": w.start, "end": w.end})
        else:
            words.append({"w": seg.text.strip(), "start": seg.start, "end": seg.end})
        if seg.end - last_logged >= 60:
            last_logged = seg.end
            log(f"[whisper] {seg.end / info.duration * 100:3.0f}% "
                f"({fmt_ts(seg.end)} из {fmt_ts(info.duration)})")
    log(f"[whisper] готово: {len(words)} слов за {time.time() - t0:.0f}s")
    return words


def load_wav16k(path):
    """Читает моно PCM16 WAV в массив (time,) float32."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        sr = f.getframerate()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return pcm.astype("float32") / 32768.0, sr


MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_DIR, "models"))
SHERPA_SEG = os.path.join(MODELS_DIR,
                          "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
SHERPA_EMB = os.path.join(MODELS_DIR,
                          "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")


def diarize_sherpa(wav, num_speakers, threads, cluster_threshold=1.3):
    import sherpa_onnx
    for p in (SHERPA_SEG, SHERPA_EMB):
        if not os.path.exists(p):
            sys.exit(f"ОШИБКА: нет модели {p} — см. README, раздел про модели sherpa")
    t0 = time.time()
    log("[diar] sherpa-onnx: pyannote segmentation-3.0 + 3D-Speaker eres2net")
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=SHERPA_SEG),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SHERPA_EMB, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=cluster_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    waveform, sr = load_wav16k(wav)
    assert sr == sd.sample_rate
    segments = sd.process(waveform).sort_by_start_time()
    turns = [{"start": s.start, "end": s.end, "speaker": f"SPK{s.speaker:02d}"}
             for s in segments]
    speakers = sorted({t["speaker"] for t in turns})
    log(f"[diar] готово за {time.time() - t0:.0f}s: "
        f"{len(turns)} интервалов, {len(speakers)} говорящих")
    return turns


def diarize(wav, token, num_speakers, min_speakers=None, max_speakers=None):
    import torch
    from pyannote.audio import Pipeline
    t0 = time.time()
    log("[diar] загрузка пайплайна pyannote/speaker-diarization-3.1")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=token)
    pipeline.to(torch.device("cpu"))
    log(f"[diar] пайплайн готов за {time.time() - t0:.0f}s, диаризация...")
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
    waveform, sr = load_wav16k(wav)
    audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            annotation = pipeline(audio, hook=hook, **kwargs)
    except ImportError:
        annotation = pipeline(audio, **kwargs)
    annotation = getattr(annotation, "speaker_diarization", annotation)
    turns = [{"start": t.start, "end": t.end, "speaker": s}
             for t, _, s in annotation.itertracks(yield_label=True)]
    speakers = sorted({t["speaker"] for t in turns})
    log(f"[diar] готово за {time.time() - t0:.0f}s: "
        f"{len(turns)} интервалов, {len(speakers)} говорящих")
    return turns


def assign_speakers(words, turns):
    """Каждому слову — говорящий с максимальным перекрытием по времени.

    Слова без перекрытия получают говорящего ближайшего интервала.
    """
    labels = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None:
            nearest = min(turns, default=None,
                          key=lambda t: max(t["start"] - w["end"], w["start"] - t["end"]))
            best = nearest["speaker"] if nearest else None
        labels.append(best)
    return labels


def build_dialog(words, labels):
    """Группирует слова в реплики; говорящие нумеруются по первому появлению."""
    order = {}
    for lab in labels:
        if lab is not None and lab not in order:
            order[lab] = len(order) + 1
    replicas = []
    cur = None
    for w, lab in zip(words, labels):
        n = order.get(lab)
        if cur is None or n != cur["speaker"]:
            cur = {"speaker": n, "start": w["start"], "words": []}
            replicas.append(cur)
        cur["words"].append(w["w"])
    return replicas


def fmt_ts(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write_dialog(replicas, dst):
    with open(dst, "w", encoding="utf-8") as f:
        for r in replicas:
            name = f"Собеседник {r['speaker']}" if r["speaker"] else "Собеседник ?"
            f.write(f"[{fmt_ts(r['start'])}] {name}:\n")
            f.write(" ".join(r["words"]) + "\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="путь к аудиофайлу (m4a/mp3/wav/ogg...)")
    ap.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium", "large-v3",
                             "large-v3-turbo", "turbo"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cluster-threshold", type=float, default=1.3,
                    help="sherpa, авторежим: порог слияния голосов "
                         "(больше = меньше говорящих; 1.2 -> 3, 1.3 -> 2, 1.4 -> 1)")
    ap.add_argument("--engine", default="sherpa", choices=["sherpa", "pyannote"],
                    help="движок диаризации: sherpa (onnx, без токена и torch) "
                         "или pyannote (нужны torch и HF-токен)")
    ap.add_argument("--speakers", type=int, default=None,
                    help="число говорящих, если известно заранее")
    ap.add_argument("--min-speakers", type=int, default=None,
                    help="нижняя граница числа говорящих (только pyannote)")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="верхняя граница числа говорящих (только pyannote)")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--out", default=None,
                    help="путь к txt (по умолчанию: <аудио>.dialog.txt)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--remerge", action="store_true",
                    help="пересобрать диалог из кэшей без распознавания")
    args = ap.parse_args()

    src = os.path.abspath(args.audio)
    if not os.path.exists(src):
        sys.exit(f"ОШИБКА: файл не найден: {src}")
    dst = args.out or os.path.splitext(src)[0] + ".dialog.txt"
    base = os.path.splitext(src)[0]
    words_cache = f"{base}.words.{args.model}.json"
    if args.speakers:
        spk_spec = f"n{args.speakers}"
    elif args.engine == "sherpa":
        spk_spec = f"auto-t{args.cluster_threshold}"
    else:
        spk_spec = f"auto{args.min_speakers or ''}-{args.max_speakers or ''}"
    diar_cache = f"{base}.diar.{args.engine}.{spk_spec}.json"

    t0 = time.time()
    if args.remerge:
        with open(words_cache, encoding="utf-8") as f:
            words = json.load(f)
        with open(diar_cache, encoding="utf-8") as f:
            turns = json.load(f)
        log(f"[кэш] слов: {len(words)}, интервалов: {len(turns)}")
    else:
        token = read_token(args)
        if args.engine == "pyannote" and not token:
            sys.exit("ОШИБКА: нет HF-токена (--hf-token, $HF_TOKEN или .hf_token)")

        def words_job():
            log(f"[ffmpeg] конвертация + speechnorm: {src}")
            wav = to_wav16k(src, normalize=True)
            try:
                result = transcribe(wav, args.model, args.language,
                                    args.threads, args.batch_size)
            finally:
                os.remove(wav)
            with open(words_cache, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            return result

        def diar_job():
            log(f"[ffmpeg] конвертация (оригинальный уровень): {src}")
            wav = to_wav16k(src)
            try:
                if args.engine == "sherpa":
                    result = diarize_sherpa(wav, args.speakers, args.threads,
                                            args.cluster_threshold)
                else:
                    result = diarize(wav, token, args.speakers,
                                     args.min_speakers, args.max_speakers)
            finally:
                os.remove(wav)
            with open(diar_cache, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            return result

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            fw = None if os.path.exists(words_cache) else ex.submit(words_job)
            fd = None if os.path.exists(diar_cache) else ex.submit(diar_job)
            if fw:
                words = fw.result()
            else:
                with open(words_cache, encoding="utf-8") as f:
                    words = json.load(f)
                log(f"[кэш] слова: {len(words)} из {words_cache}")
            if fd:
                turns = fd.result()
            else:
                with open(diar_cache, encoding="utf-8") as f:
                    turns = json.load(f)
                log(f"[кэш] интервалы: {len(turns)} из {diar_cache}")

    labels = assign_speakers(words, turns)
    replicas = build_dialog(words, labels)
    write_dialog(replicas, dst)
    n_spk = len({r["speaker"] for r in replicas})
    log(f"Готово за {time.time() - t0:.0f}s: {len(replicas)} реплик, "
        f"{n_spk} говорящих -> {dst}")


if __name__ == "__main__":
    main()
