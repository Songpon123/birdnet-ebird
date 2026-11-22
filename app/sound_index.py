"""ดัชนีเสียงของไฟล์ต้นฉบับ: เวกเตอร์เสียง (embedding 1,280 มิติ) ของ BirdNET 3.0
ทุกช่วง 3 วินาที เลื่อนทีละ 1.5 วินาที + ชนิดที่ BirdNET ทาย 5 อันดับแรกของแต่ละช่วง (คะแนนตั้งแต่ 0.01)
+ ความดังเทียบพื้นหลัง — ทำครั้งเดียวต่อไฟล์ (~1.5 นาทีต่อเสียง 1 ชม. บน M-series)

ใช้กับ
  1. Find similar sounds: เสียงที่คล้ายคลิปหรือช่วงที่เลือก ในไฟล์นี้ ผลชุดนี้ หรือทุกผลชุด
  2. Sound groups: จัดกลุ่มเสียงที่ดังเด่นแต่ BirdNET ยังไม่ได้ตัดเป็นคลิป แล้วตั้งชื่อทีละกลุ่ม
  3. Your recognizers (recognizers.py): ตัวจำแนกต่อชนิดที่ฝึกจากคลิปที่คุณตรวจแล้ว

data/cache/sound-index/<Source ID>.npz   starts, vectors (float16 ยาว 1), labels/scores (5 อันดับ), bird (คะแนนนก
                                          สูงสุดของช่วง), activity (dB เหนือพื้นหลัง), duration
data/cache/sound-index/labels-*.json      ป้ายชื่อของโมเดล (labels ในดัชนีเป็นเลขลำดับในรายการนี้)
data/cache/sound-index/groups-*.json      ผลจัดกลุ่มล่าสุดของแต่ละผลชุด
ค้นหาใช้ numpy อย่างเดียว (หน้าจอทำเองได้); สร้างดัชนีและจัดกลุ่มรันเป็น process แยก:
  field_audio_to_ebird.py --sound-index build SUMMARY.xlsx ...
  field_audio_to_ebird.py --sound-index groups SUMMARY.xlsx [--include-clipped]
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import paths

CACHE = paths.CACHE / "sound-index"
VERSION = 2                    # 2 = เก็บชนิดที่ BirdNET ทายต่อช่วงด้วย
WINDOW = 3.0                   # BirdNET ฟังทีละ 3 วินาที
HOP = 1.5
CHUNK = 60.0                   # วัดความดังทีละ 60 วินาที (พื้นหลังของแต่ละนาที)
BAND = (150.0, 12000.0)
TOP_LABELS = 5
PREDICT_FLOOR = 0.01
BIRD_MIN = 0.05                # ช่วงที่ BirdNET ว่าเป็นนกได้อย่างน้อยเท่านี้ (ชนิดไหนก็ได้) = มีเสียงนก (ใช้จัดกลุ่ม)
LOUD_DB = 25.0                 # หรือดังกว่าพื้นหลังเกินนี้ (เสียงที่ BirdNET ไม่รู้จักเลย) — ทดลองกับ REC-001/002:
                               # ความดังอย่างเดียวแยกเสียงนกเบา ๆ ออกจากเสียงรบกวนไม่ได้ จึงใช้คู่กับคะแนนของ BirdNET
QUERY_WINDOWS = 8              # ตัวอย่างยาว: ใช้ช่วงที่ดังที่สุดเท่านี้
STRONG, POSSIBLE = 0.8, 0.7    # ความคล้าย: ทดลองกับ REC-001 — เสียงเดียวกัน 0.95, เสียงอื่นที่ใกล้ที่สุด 0.72
GROUP_SIMILARITY = 0.3         # ช่วงในกลุ่มเดียวกันคล้ายกันเฉลี่ยอย่างน้อยเท่านี้ หลังตัดเสียงพื้นหลังเฉลี่ยของไฟล์ออก
                               # (ทดลองกับ REC-001: ไม่ตัด = กลุ่มใหญ่ 1,295 ช่วงปนกันหมดเพราะพื้นหลังเหมือนกัน;
                               #  ตัดแล้ว 0.3 = กลุ่มใหญ่ส่วนมากเป็นชนิดเดียว และเสียงนกหัวขวานอยู่กลุ่มเดียวกัน)
MAX_GROUP_WINDOWS = 3000       # มากกว่านี้ใช้เฉพาะช่วงที่ดังที่สุด (จัดกลุ่มใช้หน่วยความจำ n²)
MAX_GROUPS = 120               # กลุ่มที่ให้ BirdNET ทายชนิด (ใหญ่สุดก่อน แล้วเสียงที่ได้ยินครั้งเดียว)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


# ----------------------------- ไฟล์ต้นฉบับในผลชุด -----------------------------
def sources(rows, summary=None):
    """{Source ID: {"path", "name", "summary", "rows"}} จากแถวใน summary (คลิปของ BirdNET + ที่เพิ่มเอง)"""
    found = {}
    for row in rows:
        sid = str(row.get("Source ID") or "").strip()
        path = str(row.get("Source path") or "").strip()
        if not sid or sid == "nan" or not path:
            continue
        entry = found.setdefault(sid, {"path": Path(path), "name": str(row.get("Source file") or Path(path).name),
                                       "summary": Path(summary) if summary else row.get("_summary"), "rows": []})
        entry["rows"].append(row)
    return found


def index_file(sid):
    return CACHE / f"{sid}.npz"


def has_index(sid):
    return load(sid) is not None


def missing(found):
    """ไฟล์ต้นฉบับที่ยังไม่มีดัชนีและยังอยู่ที่เดิม -> [(sid, path)]"""
    return [(sid, entry["path"]) for sid, entry in found.items()
            if not has_index(sid) and Path(entry["path"]).is_file()]


_loaded = {}


def load(sid):
    """ดัชนีของไฟล์นี้ -> {"starts", "vectors" (float16), "activity", "duration"} หรือ None"""
    path = index_file(sid)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _loaded.get(sid)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        with np.load(path) as data:
            if int(data["version"]) != VERSION:
                return None
            index = {"starts": data["starts"].astype(np.float32), "vectors": data["vectors"],
                     "activity": data["activity"].astype(np.float32), "duration": float(data["duration"]),
                     "labels": data["labels"], "scores": data["scores"].astype(np.float32),
                     "bird": data["bird"].astype(np.float32), "label_set": str(data["label_set"])}
    except (OSError, ValueError, KeyError):
        return None
    if len(_loaded) > 40:                       # ค้นทุกผลชุด: ไม่เก็บทุกไฟล์ไว้ในหน่วยความจำ
        _loaded.clear()
    _loaded[sid] = (stamp, index)
    return index


def _label_file(label_set):
    return CACHE / f"labels-{label_set}.json"


_label_lists = {}


def label_list(label_set):
    """ป้ายชื่อของโมเดล ('Scientific_Common') ตามลำดับที่ใช้ในดัชนี"""
    if label_set not in _label_lists:
        try:
            _label_lists[label_set] = json.loads(_label_file(label_set).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _label_lists[label_set] = []
    return _label_lists[label_set]


def _remember_labels(species_list):
    labels = [str(label) for label in species_list]
    label_set = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()[:12]
    path = _label_file(label_set)
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(labels), encoding="utf-8")
        temporary.replace(path)
    return label_set, labels


def window_guesses(index, i, top=TOP_LABELS):
    """ชนิดที่ BirdNET ทายในช่วงที่ i ของดัชนี -> [(scientific, common, คะแนน)] (ชื่อของ BirdNET)"""
    labels = label_list(index["label_set"])
    found = []
    for label, score in zip(index["labels"][i][:top], index["scores"][i][:top]):
        if label >= 0 and label < len(labels):
            scientific, _, common = labels[label].partition("_")
            found.append((scientific, common or scientific, round(float(score), 3)))
    return found


def _save(sid, starts, vectors, activity, duration, labels, scores, bird, label_set):
    CACHE.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, version=VERSION, starts=np.asarray(starts, np.float32),
                        vectors=np.asarray(vectors, np.float16), activity=np.asarray(activity, np.float16),
                        duration=np.float32(duration), labels=np.asarray(labels, np.int32),
                        scores=np.asarray(scores, np.float16), bird=np.asarray(bird, np.float16),
                        label_set=label_set)
    path = index_file(sid)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


# ----------------------------- สร้างดัชนี (process แยก) -----------------------------
def frame_activity(samples, sr):
    """ความดังของแต่ละเฟรมเทียบพื้นหลังของความถี่นั้นในช่วงนี้ (dB) -> (เวลากลางเฟรม, ค่า)
    ค่าของเฟรม = ความถี่ที่ดังเด่นที่สุดราว 3% (เสียงนกกินแค่บางความถี่ เสียงลม/น้ำกินทั้งช่วง)"""
    n_fft = 1024 if sr <= 32000 else 2048
    hop = n_fft // 2
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < n_fft:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    count = 1 + (samples.size - n_fft) // hop
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= BAND[0]) & (freqs <= min(BAND[1], sr / 2))
    window = np.hanning(n_fft).astype(np.float32)
    parts = []
    for first in range(0, count, 512):
        index = np.arange(first, min(count, first + 512))[:, None] * hop + np.arange(n_fft)
        power = np.abs(np.fft.rfft(samples[index] * window, axis=1))[:, band] ** 2
        parts.append(10 * np.log10(power.astype(np.float32) + 1e-12))
    db = np.concatenate(parts)
    excess = db - np.median(db, axis=0, keepdims=True)
    times = (np.arange(count) * hop + n_fft / 2) / sr
    return times.astype(np.float32), np.percentile(excess, 97, axis=1).astype(np.float32)


def window_activity(source, duration, starts):
    """ความดังของแต่ละช่วง 3 วินาที (dB เหนือพื้นหลังของนาทีนั้น; ใช้เฟรมที่ดังอันดับ 97% ในช่วง)"""
    from audio_io import read_mono
    starts = np.asarray(starts, dtype=np.float32)
    activity = np.zeros(len(starts), np.float32)
    begin = 0.0
    while begin < max(duration, float(starts[-1]) + WINDOW if len(starts) else 0.0):
        mine = np.where((starts >= begin) & (starts < begin + CHUNK))[0]
        if len(mine):
            samples, sr = read_mono(source, begin, begin + CHUNK + WINDOW)
            times, scores = frame_activity(samples, sr)
            times = times + begin
            for i in mine:
                inside = scores[(times >= starts[i]) & (times < starts[i] + WINDOW)]
                activity[i] = float(np.percentile(inside, 97)) if inside.size else 0.0
        begin += CHUNK
    return activity


def _analyze(model, source):
    """BirdNET 3.0 ฟังทั้งไฟล์ -> (starts, vectors ยาว 1, labels, scores, bird, label_set)
    ไฟล์ที่อ่านตรงไม่ได้ (m4a) แปลงเป็น WAV ชั่วคราวก่อน"""
    import species_names

    def run(path):
        encoded = model.encode([str(path)], n_workers=1, overlap_duration_s=WINDOW - HOP,
                               show_stats=None).to_structured_array()
        predicted = model.predict([str(path)], top_k=TOP_LABELS, n_workers=1, overlap_duration_s=WINDOW - HOP,
                                  default_confidence_threshold=PREDICT_FLOOR, show_stats=None).to_structured_array()
        return encoded, predicted
    try:
        encoded, predicted = run(source)
    except Exception:  # noqa: BLE001  libsndfile อ่านไม่ได้
        encoded, predicted = [], []
    if len(encoded) == 0:
        import platform_tools
        ffmpeg = platform_tools.bundled_tool("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"cannot read {Path(source).name} (ffmpeg not found)")
        with tempfile.TemporaryDirectory() as temp:
            wav = Path(temp) / "source.wav"
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "48000", str(wav)],
                           check=True, creationflags=platform_tools.NO_WINDOW)
            encoded, predicted = run(wav)
    order = np.argsort([float(r["start_time"]) for r in encoded])
    starts = np.array([float(encoded[i]["start_time"]) for i in order], np.float32)
    vectors = np.array([encoded[i]["embedding"] for i in order], np.float32)
    if len(vectors):
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    label_set, labels = _remember_labels(model.species_list)
    position = {label: n for n, label in enumerate(labels)}
    row_of = {round(float(start), 2): n for n, start in enumerate(starts)}
    found = [[] for _start in starts]
    for r in predicted:
        n = row_of.get(round(float(r["start_time"]), 2))
        label = position.get(str(r["species_name"]))
        if n is not None and label is not None:
            found[n].append((float(r["confidence"]), label))
    top_labels = np.full((len(starts), TOP_LABELS), -1, np.int32)
    top_scores = np.zeros((len(starts), TOP_LABELS), np.float32)
    bird = np.zeros(len(starts), np.float32)
    is_bird = {}
    for n, items in enumerate(found):
        for slot, (score, label) in enumerate(sorted(items, reverse=True)[:TOP_LABELS]):
            top_labels[n, slot], top_scores[n, slot] = label, score
            if label not in is_bird:
                scientific, _, common = labels[label].partition("_")
                is_bird[label] = species_names.is_bird_label(scientific, common)
            if is_bird[label]:
                bird[n] = max(bird[n], score)
    return starts, vectors, top_labels, top_scores, bird, label_set


def build(source, sid, model, log=print):
    """ดัชนีของไฟล์ต้นฉบับหนึ่งไฟล์ -> จำนวนช่วง"""
    from audio_io import audio_duration
    import time
    began = time.monotonic()
    duration = audio_duration(source)
    starts, vectors, labels, scores, bird, label_set = _analyze(model, source)
    activity = window_activity(source, duration, starts) if len(starts) else np.zeros(0, np.float32)
    _save(sid, starts, vectors, activity, duration or (float(starts[-1]) + WINDOW if len(starts) else 0.0),
          labels, scores, bird, label_set)
    log(f"  {len(starts)} windows, {int(duration // 60)}:{int(duration % 60):02d} of audio "
        f"({time.monotonic() - began:.0f} s)")
    return len(starts)


def _load_birdnet():
    import birdnet
    return birdnet.load("acoustic", "3.0", "onnx", precision="fp16")


def build_for(summaries, log=print, model=None):
    """ดัชนีของทุกไฟล์ต้นฉบับในผลชุดเหล่านี้ (หรือไฟล์เสียงที่ระบุตรง ๆ) ที่ยังไม่มี -> จำนวนไฟล์ที่ทำ"""
    from review_store import load_reviews
    todo = {}
    for item in map(Path, summaries):
        if item.suffix.lower() != ".xlsx":            # ไฟล์เสียงที่ยังไม่อยู่ในผลชุดไหน
            from field_audio_to_ebird import source_id
            sid = source_id(item)
            if sid not in todo and not has_index(sid):
                todo[sid] = {"path": item, "name": item.name}
            continue
        for sid, entry in sources(load_reviews(item), item).items():
            if sid not in todo and not has_index(sid):
                todo[sid] = entry
    todo = {sid: entry for sid, entry in todo.items() if Path(entry["path"]).is_file()}
    if not todo:
        log("sound index: nothing to do")
        return 0
    log(f"Processing {len(todo)} file(s) for the sound index")
    model = model or _load_birdnet()
    for sid, entry in todo.items():
        log(f"=== {entry['name']} ===")
        build(entry["path"], sid, model, log)
    return len(todo)


# ----------------------------- ค้นเสียงที่คล้ายกัน -----------------------------
def query_vectors(sid, t0, t1):
    """เวกเตอร์ของช่วง [t0, t1] วินาทีในไฟล์ต้นฉบับ (ช่วงยาว: ใช้ส่วนที่ดังที่สุด QUERY_WINDOWS ช่วง)"""
    index = load(sid)
    if index is None or not len(index["starts"]):
        return np.zeros((0, 1), np.float32)
    starts = index["starts"]
    overlap = np.minimum(starts + WINDOW, t1) - np.maximum(starts, t0)
    # ช่วงที่อยู่ในตัวอย่างอย่างน้อย 2/3 (ช่วงขอบที่ทับแค่ครึ่งอาจเป็นแต่พื้นหลัง -> ไปตรงกับพื้นหลังที่อื่น)
    chosen = np.where(overlap >= min(WINDOW * 2 / 3, max(0.1, (t1 - t0) * 2 / 3)))[0]
    if not len(chosen):
        chosen = np.array([int(np.argmin(np.abs(starts + WINDOW / 2 - (t0 + t1) / 2)))])
    if len(chosen) > QUERY_WINDOWS:
        chosen = chosen[np.argsort(-index["activity"][chosen])[:QUERY_WINDOWS]]
    return index["vectors"][np.sort(chosen)].astype(np.float32)


def similarity(index, queries):
    """ความคล้ายของทุกช่วงในดัชนีกับตัวอย่าง (cosine สูงสุดเทียบกับช่วงไหนของตัวอย่างก็ได้)"""
    if index is None or not len(index["starts"]) or not len(queries):
        return np.zeros(0, np.float32)
    return (index["vectors"].astype(np.float32) @ np.asarray(queries, np.float32).T).max(axis=1)


def merge_hits(starts, scores, threshold, exclude=()):
    """ช่วงที่คะแนนถึงเกณฑ์ติดกัน = เสียงครั้งเดียว -> [{"start", "end", "score", "best"}] (best = ช่วงที่คะแนนสูงสุด)
    exclude = [(a, b)] ช่วงที่ไม่เอา (เช่น ตัวอย่างเอง) — ช่วงที่ทับกันตั้งแต่ครึ่งหน้าต่างถูกข้าม"""
    events, current = [], None
    for i in np.argsort(starts):
        start, score = float(starts[i]), float(scores[i])
        if score < threshold or any(min(start + WINDOW, b) - max(start, a) >= WINDOW / 2 - 1e-6 for a, b in exclude):
            current = None
            continue
        if current is not None and start - current["last"] <= HOP + 1e-6:
            current["end"], current["last"] = start + WINDOW, start
            if score > current["score"]:
                current["score"], current["best"] = score, start
        else:
            current = {"start": start, "end": start + WINDOW, "score": score, "best": start, "last": start}
            events.append(current)
    for event in events:
        del event["last"]
        event["score"] = round(event["score"], 3)
    return events


def find_similar(queries, targets, threshold=POSSIBLE, exclude=None, limit=300):
    """ตัวอย่าง (เวกเตอร์) เทียบกับไฟล์ targets = [sid] -> เสียงที่คล้าย เรียงจากคล้ายที่สุด
    exclude = {sid: [(a, b)]} ช่วงที่ไม่เอา -> [{"sid", "start", "end", "score", "best"}]"""
    exclude = exclude or {}
    found = []
    for sid in targets:
        index = load(sid)
        if index is None:
            continue
        for event in merge_hits(index["starts"], similarity(index, queries), threshold, exclude.get(sid, ())):
            found.append(dict(event, sid=sid))
    found.sort(key=lambda event: -event["score"])
    return found[:limit]


def clip_spans(rows):
    """{sid: [(start, end, แถว)]} ช่วงที่ได้ยินของคลิปทุกอันที่มีอยู่แล้ว (ใช้บอกว่าเสียงที่เจอมีคลิปแล้วไหม)"""
    spans = {}
    for row in rows:
        sid = str(row.get("Source ID") or "").strip()
        first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
        if sid and first is not None:
            spans.setdefault(sid, []).append((first, max(last if last is not None else first, first + 0.5), row))
    return spans


def clips_at(spans, sid, start, end):
    """คลิปที่ทับช่วงนี้อย่างน้อย 1 วินาที (หรือครึ่งหนึ่งของช่วงสั้น)"""
    need = min(1.0, (end - start) / 2)
    return [row for a, b, row in spans.get(sid, ()) if min(b, end) - max(a, start) >= need]


# ----------------------------- จัดกลุ่มเสียง (process แยก) -----------------------------
def centered(index):
    """เวกเตอร์ที่ตัดเสียงพื้นหลังเฉลี่ยของไฟล์นี้ออกแล้ว (ยาว 1) — ช่วงที่มีแต่พื้นหลังไม่คล้ายกันเองอีกต่อไป"""
    vectors = index["vectors"].astype(np.float32)
    vectors = vectors - vectors.mean(axis=0, keepdims=True)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def group_labels(vectors, similarity_floor=GROUP_SIMILARITY):
    """จัดกลุ่มแบบ average linkage (cosine): ช่วงในกลุ่มเดียวกันคล้ายกันเฉลี่ย >= similarity_floor
    เสียงที่ไม่คล้ายอะไรเลยได้กลุ่มของตัวเอง -> label ต่อช่วง"""
    if len(vectors) < 2:
        return np.zeros(len(vectors), int)
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - similarity_floor).fit_predict(np.asarray(vectors, np.float32))


def find_groups(summary, include_clipped=False, analyzer=None, log=print):
    """เสียงที่ดังเด่นในผลชุดนี้ (ยกเว้นช่วงที่มีคลิปแล้ว ถ้าไม่ include_clipped) จัดกลุ่มตามความคล้าย
    แล้วให้ BirdNET ทายชนิดของแต่ละกลุ่มจากช่วงตัวแทน -> dict (เก็บเป็น JSON ด้วย)"""
    from review_store import load_reviews
    summary = Path(summary)
    rows = load_reviews(summary)
    found = sources(rows, summary)
    spans = clip_spans(rows)
    candidates = []                                  # (sid, index ของช่วง)
    vectors, activity = [], []
    for sid in found:
        index = load(sid)
        if index is None:
            log(f"  no sound index for {found[sid]['name']} — skipped")
            continue
        background_free = centered(index)
        for i, start in enumerate(index["starts"]):
            if index["bird"][i] < BIRD_MIN and index["activity"][i] < LOUD_DB:
                continue
            if not include_clipped and clips_at(spans, sid, float(start), float(start) + WINDOW):
                continue
            candidates.append((sid, i))
            vectors.append(background_free[i])
            activity.append(float(index["activity"][i]))
    log(f"sound groups: {len(candidates)} bird-like or loud 3-s windows"
        + ("" if include_clipped else " without a clip yet"))
    if len(candidates) > MAX_GROUP_WINDOWS:
        keep = np.sort(np.argsort(-np.asarray(activity))[:MAX_GROUP_WINDOWS])
        candidates = [candidates[i] for i in keep]
        vectors = [vectors[i] for i in keep]
        activity = [activity[i] for i in keep]
        log(f"  using the {MAX_GROUP_WINDOWS} loudest")
    vectors = np.asarray(vectors, np.float32)
    labels = group_labels(vectors)
    groups = []
    for label in sorted(set(labels.tolist())):
        members = np.where(labels == label)[0]
        center = vectors[members].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-9)
        events = []
        by_source = {}
        for m in members:
            by_source.setdefault(candidates[m][0], []).append(m)
        for sid, items in by_source.items():
            index = load(sid)
            starts = np.array([index["starts"][candidates[m][1]] for m in items])
            scores = vectors[items] @ center
            for event in merge_hits(starts, scores, -1.0):
                events.append(dict(event, sid=sid, name=found[sid]["name"],
                                   loud=round(max(activity[m] for m in items
                                                  if event["start"] <= index["starts"][candidates[m][1]] < event["end"]), 1)))
        events.sort(key=lambda event: -event["score"])
        groups.append({"events": events, "windows": len(members), "loud": round(max(activity[m] for m in members), 1),
                       "seconds": round(sum(e["end"] - e["start"] for e in events), 1)})
    # ใหญ่สุดก่อน (เสียงที่ร้องซ้ำ) แล้วเสียงที่ได้ยินครั้งเดียว เรียงตามความดัง
    groups.sort(key=lambda group: (-len(group["events"]), -group["loud"]))
    for number, group in enumerate(groups, start=1):
        group["id"] = number
    log("  BirdNET guesses for each group …")
    _guess_groups(groups[:MAX_GROUPS], found, analyzer, log)
    result = {"summary": str(summary.resolve()), "made": datetime.now().isoformat(timespec="seconds"),
              "include_clipped": bool(include_clipped), "windows": len(candidates), "groups": groups,
              "sources": {sid: {"path": str(entry["path"]), "name": entry["name"]} for sid, entry in found.items()}}
    path = groups_file(summary, include_clipped)
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1), encoding="utf-8")
    temporary.replace(path)
    log(f"sound groups: {sum(len(g['events']) > 1 for g in groups)} groups of repeated sounds, "
        f"{sum(len(g['events']) == 1 for g in groups)} sounds heard once")
    return result


def groups_file(summary, include_clipped=False):
    digest = hashlib.sha256(str(Path(summary).resolve()).encode("utf-8")).hexdigest()[:12]
    return CACHE / f"groups-{digest}{'-all' if include_clipped else ''}.json"


def load_groups(summary, include_clipped=False):
    try:
        return json.loads(groups_file(summary, include_clipped).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def event_guesses(index, event, top=3):
    """ชนิดที่ BirdNET ทายในช่วงของเสียงนี้ (จากดัชนี) -> [{"common", "scientific", "confidence"}] ชื่อตาม eBird
    คะแนน = สูงสุดในช่วงนั้น (คำใบ้ ไม่ใช่การยืนยัน)"""
    import species_names
    if index is None:
        return []
    starts = index["starts"]
    overlap = np.minimum(starts + WINDOW, event["end"]) - np.maximum(starts, event["start"])
    inside = np.where(overlap >= min(HOP, (event["end"] - event["start"]) / 2))[0]
    best = {}
    for i in inside:
        for scientific, common, score in window_guesses(index, i):
            names = species_names.to_ebird(scientific, common) or {"common": common, "scientific": scientific}
            key = names["scientific"]
            if key not in best or score > best[key]["confidence"]:
                best[key] = {"common": names["common"], "scientific": key, "confidence": score,
                             "birdnet_scientific": scientific}
    return sorted(best.values(), key=lambda item: -item["confidence"])[:top]


class _Geo:
    """โมเดลพื้นที่ BirdNET 3.0 (โหลดครั้งเดียว): ชนิดนี้ควรพบที่จุด/สัปดาห์นั้นไหม"""
    def __init__(self):
        self.model = None

    def expected(self, scientific, birdnet_scientific, common, lat, lon, when):
        try:
            from v3_model import location_check
            if self.model is None:
                import birdnet
                self.model = birdnet.load("geo", "3.0", "onnx", precision="fp16")
            found = location_check(birdnet_scientific, common, lat, lon, when, geo_model=self.model)
            return found if found is not None else location_check(scientific, common, lat, lon, when,
                                                                  geo_model=self.model)
        except Exception:  # noqa: BLE001  ไม่มีโมเดล: ให้คนตรวจเอง
            return None


def describe_guesses(guesses, context, geo=None):
    """เติมผลโมเดลพื้นที่ (expected) + checklist ของประเทศ ให้คำทาย (context = เวลาเริ่ม/พิกัดของไฟล์)"""
    import country_checklist
    lat, lon, when = context.get("lat"), context.get("lon"), context.get("when")
    for item in guesses:
        item["expected"] = (geo.expected(item["scientific"], item.get("birdnet_scientific", item["scientific"]),
                                         item["common"], lat, lon, when)
                            if geo is not None and lat is not None and when is not None else None)
        item["checklist"] = country_checklist.describe(
            country_checklist.check(item["scientific"], lat, lon, item["common"])) if lat is not None else ""
    return guesses


def _guess_groups(groups, found, analyzer, log):
    """ชนิดที่ BirdNET ทายในช่วงตัวแทนของแต่ละกลุ่ม (คล้ายจุดกลางกลุ่มที่สุด ≤ 3 เสียง) -> group["guess"]"""
    import manual_clips
    geo, contexts = _Geo(), {}
    for group in groups:
        merged = {}
        for event in group["events"][:3]:
            index = load(event["sid"])
            for item in event_guesses(index, event):
                if item["scientific"] not in merged or item["confidence"] > merged[item["scientific"]]["confidence"]:
                    merged[item["scientific"]] = item
        first = group["events"][0]
        if first["sid"] not in contexts:
            contexts[first["sid"]] = manual_clips.source_context(found[first["sid"]]["rows"],
                                                                 found[first["sid"]]["path"]) or {}
        context = contexts[first["sid"]]
        when = context["start"] + timedelta(seconds=first["best"]) if context.get("start") else None
        group["guess"] = describe_guesses(sorted(merged.values(), key=lambda item: -item["confidence"])[:3],
                                          dict(context, when=when), geo)


# ----------------------------- CLI -----------------------------
def _rows_of(summary):
    from review_store import load_reviews
    return load_reviews(Path(summary))


def main(argv):
    """GUI อ่านบรรทัด INDEX done N / GROUPS <path> / NEAR <path> / EXPECTED <path> / ERROR ..."""
    if not argv or argv[0] not in ("build", "groups", "near", "expected"):
        print("usage: --sound-index build SUMMARY.xlsx ... | groups SUMMARY.xlsx [--include-clipped] "
              "| near SUMMARY.xlsx | expected SUMMARY.xlsx")
        return 2
    try:
        if argv[0] == "build":
            done = build_for([Path(p) for p in argv[1:]], log=lambda text: print(text, flush=True))
            print(f"INDEX done {done}", flush=True)
        elif argv[0] in ("near", "expected"):
            import near_misses
            if len(argv) != 2:
                print(f"usage: --sound-index {argv[0]} SUMMARY.xlsx")
                return 2
            log = lambda text: print(text, flush=True)  # noqa: E731
            if argv[0] == "near":
                model = _load_birdnet() if missing(sources(_rows_of(argv[1]), argv[1])) else None
                build_for([argv[1]], log=log, model=model)
                near_misses.find_near_misses(argv[1], model=model, log=log)
                print(f"NEAR {near_misses.near_file(argv[1])}", flush=True)
            else:
                near_misses.find_expected(argv[1], log=log)
                print(f"EXPECTED {near_misses.expected_file(argv[1])}", flush=True)
        else:
            include = "--include-clipped" in argv
            event_mode = "--events" in argv
            summaries = [p for p in argv[1:] if p not in ("--include-clipped", "--events")]
            if len(summaries) != 1:
                print("usage: --sound-index groups SUMMARY.xlsx [--include-clipped]")
                return 2
            if event_mode:
                import event_groups
                event_groups.find_groups(summaries[0], include, log=lambda text: print(text, flush=True))
                print(f"GROUPS {event_groups.groups_file(summaries[0], include)}", flush=True)
            else:
                build_for(summaries, log=lambda text: print(text, flush=True))
                find_groups(summaries[0], include, log=lambda text: print(text, flush=True))
                print(f"GROUPS {groups_file(summaries[0], include)}", flush=True)
    except KeyboardInterrupt:
        print("sound index stopped; finished files are kept", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
