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


def transcribe(wav, model_name, language, threads):
    from faster_whisper import WhisperModel
    t0 = time.time()
    log(f"[whisper] загрузка модели '{model_name}' (CPU, int8, threads={threads})")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
    log(f"[whisper] модель готова за {time.time() - t0:.0f}s, распознаю...")
    segments, info = model.transcribe(
        wav, language=language, vad_filter=True, word_timestamps=True,
        condition_on_previous_text=False, no_speech_threshold=0.9,
        log_prob_threshold=-2.0,
        vad_parameters=dict(threshold=0.25, min_silence_duration_ms=1000,
                            speech_pad_ms=200))
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


def _sherpa_process(waveform, num_speakers, threads, cluster_threshold):
    """Один проход sherpa-диаризации, возвращает интервалы с метками."""
    import sherpa_onnx
    for p in (SHERPA_SEG, SHERPA_EMB):
        if not os.path.exists(p):
            sys.exit(f"ОШИБКА: нет модели {p} — см. README, раздел про модели sherpa")
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
    assert sd.sample_rate == 16000
    segments = sd.process(waveform).sort_by_start_time()
    return [{"start": s.start, "end": s.end, "speaker": f"SPK{s.speaker:02d}"}
            for s in segments]


def diarize_sherpa(wav, num_speakers, threads, cluster_threshold=1.2):
    t0 = time.time()
    log("[diar] sherpa-onnx: pyannote segmentation-3.0 + 3D-Speaker eres2net")
    waveform, sr = load_wav16k(wav)
    turns = _sherpa_process(waveform, num_speakers, threads, cluster_threshold)
    speakers = sorted({t["speaker"] for t in turns})
    log(f"[diar] готово за {time.time() - t0:.0f}s: "
        f"{len(turns)} интервалов, {len(speakers)} говорящих")
    return turns


MIN_EMB_SEC = 0.6      # интервалы короче не эмбеддятся (метка от соседей)
MAX_EMB_SEC = 20.0     # длинные интервалы режутся: эмбеддингу хватает
ADAPTIVE_SIL_MIN = 0.12  # ниже — считаем, что говорящий один
ADAPTIVE_SIL_MARGIN = 0.05  # берём наименьшее k в этой марже от максимума
MIN_CLUSTER_SEC = 3.0    # кластер с меньшей суммарной речью — ложный
MIN_CLUSTER_SHARE = 0.05  # ...или с меньшей долей от всей речи


def _embed_turns(waveform, turns, threads):
    """Эмбеддинг голоса (512-мерный) для каждого достаточно длинного интервала."""
    import numpy as np
    import sherpa_onnx
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SHERPA_EMB, num_threads=threads))
    idx, embs = [], []
    for i, t in enumerate(turns):
        if t["end"] - t["start"] < MIN_EMB_SEC:
            continue
        a = int(t["start"] * 16000)
        b = int(min(t["end"], t["start"] + MAX_EMB_SEC) * 16000)
        stream = extractor.create_stream()
        stream.accept_waveform(16000, waveform[a:b])
        stream.input_finished()
        v = np.array(extractor.compute(stream), dtype="float32")
        n = np.linalg.norm(v)
        if not n:
            continue
        idx.append(i)
        embs.append(v / n)
    return idx, np.array(embs)


def _agglomerative_levels(dist, max_k):
    """Агломеративная кластеризация (average linkage).

    Возвращает {k: метки} для k = 1..max_k.
    """
    import numpy as np
    n = len(dist)
    d = dist.copy().astype("float64")
    np.fill_diagonal(d, np.inf)
    members = {i: [i] for i in range(n)}
    levels = {}

    def snapshot():
        labels = np.empty(n, dtype=int)
        for j, items in enumerate(members.values()):
            labels[items] = j
        return labels

    if n <= max_k:
        levels[n] = snapshot()
    while len(members) > 1:
        keys = list(members.keys())
        sub = d[np.ix_(keys, keys)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = keys[i], keys[j]
        na, nb = len(members[a]), len(members[b])
        for k in members:
            if k not in (a, b):
                d[a, k] = d[k, a] = (na * d[a, k] + nb * d[b, k]) / (na + nb)
        members[a].extend(members.pop(b))
        d[b, :] = d[:, b] = np.inf
        if len(members) <= max_k:
            levels[len(members)] = snapshot()
    return levels


def _absorb_minor_clusters(labels, durs, dist, min_dur):
    """Вливает кластеры с малой суммарной речью в ближайший крупный.

    Возвращает (новые метки 0..k-1, k) или (None, 0), если крупных нет.
    """
    import numpy as np
    labels = labels.copy()
    clusters = sorted(set(labels))
    share = {c: durs[labels == c].sum() for c in clusters}
    major = [c for c in clusters if share[c] >= min_dur]
    if not major:
        return None, 0
    major_masks = {c: labels == c for c in major}
    for c in clusters:
        if c in major:
            continue
        pts = labels == c
        best = min(major, key=lambda m: dist[np.ix_(pts, major_masks[m])].mean())
        labels[pts] = best
    remap = {c: i for i, c in enumerate(sorted(set(labels)))}
    return np.array([remap[c] for c in labels]), len(remap)


def _silhouette(dist, labels):
    """Средний silhouette-коэффициент; одиночные кластеры дают 0."""
    import numpy as np
    scores = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i][same].mean()
        b = min(dist[i][labels == other].mean()
                for other in set(labels) if other != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def diarize_sherpa_adaptive(wav, threads, max_speakers=8):
    """Диаризация с автоподбором числа говорящих.

    Сегментация с низким порогом даёт мелкие чистые интервалы; на каждый
    считается эмбеддинг голоса; число кластеров выбирается по максимуму
    silhouette-коэффициента (при слабой кластерной структуре — один голос).
    """
    import numpy as np
    t0 = time.time()
    log("[diar] sherpa-onnx (adaptive): сегментация + автоподбор числа говорящих")
    waveform, sr = load_wav16k(wav)
    turns = _sherpa_process(waveform, None, threads, cluster_threshold=0.6)
    log(f"[diar] сегментация: {len(turns)} интервалов")
    if not turns:
        return []
    idx, embs = _embed_turns(waveform, turns, threads)
    if len(embs) < 2:
        for t in turns:
            t["speaker"] = "SPK00"
        return turns
    dist = 1.0 - embs @ embs.T
    np.clip(dist, 0.0, None, out=dist)
    levels = _agglomerative_levels(dist, min(max_speakers, len(embs)))
    durs = np.array([turns[i]["end"] - turns[i]["start"] for i in idx])
    min_dur = max(MIN_CLUSTER_SEC, MIN_CLUSTER_SHARE * durs.sum())
    candidates = []
    seen = set()
    for k, labels in sorted(levels.items()):
        if k < 2:
            continue
        eff_labels, eff_k = _absorb_minor_clusters(labels, durs, dist, min_dur)
        if eff_k < 2 or (eff_k, tuple(eff_labels)) in seen:
            continue
        seen.add((eff_k, tuple(eff_labels)))
        sil = _silhouette(dist, eff_labels)
        log(f"[diar] k={k} -> {eff_k} говорящих: silhouette={sil:.3f}")
        candidates.append((eff_k, sil, eff_labels))
    best_k, best_sil, final = 1, ADAPTIVE_SIL_MIN, np.zeros(len(embs), dtype=int)
    if candidates:
        top = max(c[1] for c in candidates)
        if top > ADAPTIVE_SIL_MIN:
            best_k, best_sil, final = min(
                (c for c in candidates
                 if c[1] >= top - ADAPTIVE_SIL_MARGIN and c[1] > ADAPTIVE_SIL_MIN),
                key=lambda c: c[0])
    for j, i in enumerate(idx):
        turns[i]["speaker"] = f"SPK{final[j]:02d}"
    embedded_idx = set(idx)
    embedded = [turns[i] for i in idx]
    for i, t in enumerate(turns):
        if i not in embedded_idx:
            nearest = min(embedded, key=lambda m: max(
                m["start"] - t["end"], t["start"] - m["end"], 0.0))
            t["speaker"] = nearest["speaker"]
    log(f"[diar] готово за {time.time() - t0:.0f}s: {len(turns)} интервалов, "
        f"{best_k} говорящих (silhouette={best_sil:.3f})")
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


def merge_minor_speakers(turns, min_sec=MIN_CLUSTER_SEC, min_share=MIN_CLUSTER_SHARE):
    """Интервалы редких говорящих отдаёт ближайшему по времени основному.

    Убирает ложные кластеры из нескольких секунд речи (шум, смех,
    наложение голосов); основные говорящие не трогаются.
    """
    dur = {}
    for t in turns:
        dur[t["speaker"]] = dur.get(t["speaker"], 0.0) + t["end"] - t["start"]
    total = sum(dur.values())
    if not total:
        return turns
    major = {s for s, d in dur.items() if d >= min_sec and d / total >= min_share}
    if not major or len(major) == len(dur):
        return turns
    major_turns = [t for t in turns if t["speaker"] in major]
    for t in turns:
        if t["speaker"] not in major:
            nearest = min(major_turns, key=lambda m: max(
                m["start"] - t["end"], t["start"] - m["end"], 0.0))
            t["speaker"] = nearest["speaker"]
    log(f"[diar] ложные кластеры слиты: {len(dur)} -> {len(major)} говорящих")
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
    ap.add_argument("--cluster-threshold", default="auto",
                    help="sherpa: 'auto' — адаптивный подбор числа говорящих, "
                         "либо число — порог слияния голосов "
                         "(больше = меньше говорящих)")
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
        spk_spec = ("adaptive" if args.cluster_threshold == "auto"
                    else f"auto-t{args.cluster_threshold}")
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
                result = transcribe(wav, args.model, args.language, args.threads)
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
                    if not args.speakers and args.cluster_threshold == "auto":
                        result = diarize_sherpa_adaptive(wav, args.threads)
                    else:
                        thr = (1.2 if args.cluster_threshold == "auto"
                               else float(args.cluster_threshold))
                        result = diarize_sherpa(wav, args.speakers,
                                                args.threads, thr)
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

    turns = merge_minor_speakers(turns)
    labels = assign_speakers(words, turns)
    replicas = build_dialog(words, labels)
    write_dialog(replicas, dst)
    n_spk = len({r["speaker"] for r in replicas})
    log(f"Готово за {time.time() - t0:.0f}s: {len(replicas)} реплик, "
        f"{n_spk} говорящих -> {dst}")


if __name__ == "__main__":
    main()
