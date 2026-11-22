"""Second opinion for one reviewed clip: is it closer to xeno-canto recordings of the
detected species or of BirdNET's other plausible candidates?

The clip's loudest BirdNET 3.0 segments and segments of nearby, good-quality xeno-canto
recordings (only where BirdNET also hears that species) are turned into BirdNET 3.0
embeddings; each candidate's score is the cosine similarity to its nearest reference
segments, averaged over the clip segments. A hint for the reviewer, not an identification.

Run through the CLI: field_audio_to_ebird.py --second-opinion SUMMARY.xlsx ROW
"""

import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import paths
import species_names
import xeno_canto
from app_settings import load_settings
from review_store import load_reviews

CACHE = paths.CACHE / "second-opinion"
MAX_CANDIDATES = 4             # ชนิดที่ตรวจเจอ + ตัวเลือกรองอีก 3
TOP_PER_SEGMENT = 10           # ตัวเลือกดูแค่ชนิดอันดับต้น ๆ ของแต่ละช่วง (auto-checks ฟังมา 25 อันดับ ต้องตัดให้ผลเหมือนกัน)
CANDIDATE_MIN_SCORE = 0.02     # คะแนน BirdNET ขั้นต่ำของตัวเลือกรองในคลิป
QUERY_SEGMENTS = 3             # ช่วง 3 วิของคลิปที่ใช้เทียบ (ที่ BirdNET ให้คะแนนสูงสุด)
REFS_PER_SPECIES = 6
MAX_REF_SECONDS = 180          # ข้ามไฟล์อ้างอิงยาวเกิน (โหลดนาน)
REF_MIN_SCORE = 0.2            # ใช้เฉพาะช่วงที่ BirdNET ได้ยินชนิดนั้นในไฟล์อ้างอิงจริง
REF_SEGMENTS = 3
OVERLAP = 1.5
SEARCH_DAYS = 30               # จำรายการไฟล์อ้างอิงของชนิดหนึ่งไว้เท่านี้วัน (ค้นครั้งละ ~3 วิ ถ้าไม่จำต้องค้นซ้ำทุกคลิป)
DOWNLOADS = 3                  # โหลดไฟล์อ้างอิงพร้อมกันเท่านี้ไฟล์ (xeno-canto เริ่มส่งช้า ~3 วิต่อไฟล์)
ONE_CLIP = {"n_workers": 1}
# ฟังหลายคลิปรวดเดียว: วัดจริง 40 คลิปบน Mac 18 คอร์ 8.3 -> 5.9 วิ (n_workers=4 กลับช้าลง), คะแนนต่างกันไม่เกิน 0.001
MANY_CLIPS = {"n_workers": 2, "batch_size": 16}


def _write_atomic(path, data: bytes):
    """เขียนไฟล์ชั่วคราวชื่อเฉพาะ process แล้วแทนที่ — งานสองงานเขียนไฟล์เดียวกันพร้อมกันได้โดยไม่เสีย"""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def result_path(clip_sha):
    return CACHE / f"{clip_sha}.json"


def cached_result(clip_sha):
    try:
        return json.loads(result_path(clip_sha).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def hear(model, clips, top_k=TOP_PER_SEGMENT, speed=ONE_CLIP):
    """คะแนน BirdNET ทุกช่วงของคลิปเหล่านี้ (ฟังรวมกันรอบเดียว) -> {path: [(start, end, ป้าย, score)]}"""
    found = {str(Path(clip).resolve()): [] for clip in clips}
    for r in model.predict([str(clip) for clip in clips], top_k=top_k, overlap_duration_s=OVERLAP,
                           default_confidence_threshold=0.01, show_stats=None, **speed).to_structured_array():
        found[str(Path(str(r["input"])).resolve())].append(
            (float(r["start_time"]), float(r["end_time"]), str(r["species_name"]), float(r["confidence"])))
    return found


def vectors(model, clips, speed=ONE_CLIP):
    """เวกเตอร์ BirdNET ทุกช่วงของคลิปเหล่านี้ (ฟังรวมกันรอบเดียว) -> {path: {start: vector}}"""
    found = {str(Path(clip).resolve()): {} for clip in clips}
    for r in model.encode([str(clip) for clip in clips], overlap_duration_s=OVERLAP, show_stats=None,
                          **speed).to_structured_array():
        found[str(Path(str(r["input"])).resolve())][round(float(r["start_time"]), 2)] = \
            np.asarray(r["embedding"], dtype=np.float32)
    return found


def run(summary, row_number, key, log=print, analyzer=None, row=None, heard=None, encoded=None):
    """row, heard, encoded = ผลที่มีอยู่แล้วของคลิปนี้ (auto-checks ฟังทีละชุด) ไม่ส่งมา = อ่าน/ฟังตอนนี้"""
    summary = Path(summary)
    row = row or {r["_row"]: r for r in load_reviews(summary)}.get(int(row_number))
    if row is None:
        raise ValueError("Selected clip is no longer in the summary")
    clip = (summary.parent / str(row["File"])).resolve()
    if not clip.is_file():
        raise FileNotFoundError(clip)
    clip_sha = str(row.get("Generated sha256") or "") or _sha256(clip)

    if analyzer is None:
        log("Loading BirdNET 3.0 …")
        from v3_model import V3Analyzer
        analyzer = V3Analyzer()
    model = analyzer.model
    labels = {str(label).partition("_")[0]: str(label) for label in model.species_list}
    spans = _spans(row)

    log("Scoring the clip …")
    segments = defaultdict(list)
    for start, end, label, score in hear(model, [clip])[str(clip)] if heard is None else heard:
        if spans is None or any(start < b and end > a for a, b in spans):
            segments[round(start, 2)].append((score, label))
    scores = defaultdict(dict)                    # segment start -> {scientific: score}
    for start, found in segments.items():
        for score, label in sorted(found, reverse=True)[:TOP_PER_SEGMENT]:
            scores[start][label.partition("_")[0]] = score
    embeddings = vectors(model, [clip])[str(clip)] if encoded is None else encoded

    current = str(row.get("Reviewed scientific") or row.get("Species (scientific)") or "").strip()
    names = species_names.to_ebird(current, row.get("Reviewed common") or row.get("Species (common)") or "")
    current = names["scientific"] if names else " ".join(current.split()[:2])   # ผลรุ่นเก่าเก็บชื่อของ BirdNET
    current_label = birdnet_label(current, labels) or current
    candidates = _candidates(scores, current_label, labels, analyzer, row)
    if not candidates:
        raise ValueError("BirdNET found no candidate species in this clip")
    top_segments = sorted(scores, key=lambda s: -max(scores[s].get(c, 0.0) for c in candidates))
    query = [_unit(embeddings[s]) for s in top_segments[:QUERY_SEGMENTS] if s in embeddings]
    if not query:
        raise ValueError("No BirdNET segments found in this clip")

    lat, lon = _number(row.get("Latitude")), _number(row.get("Longitude"))
    ranking = []
    for scientific in candidates:
        log(f"References for {scientific} …")
        refs, used = reference_embeddings(model, scientific, labels.get(scientific, scientific),
                                          key, lat, lon, log)
        best = max((s.get(scientific, 0.0) for s in scores.values()), default=0.0)
        similarity = (float(np.mean([float((refs @ q).max()) for q in query])) if len(refs) else None)
        names = species_names.to_ebird(scientific, _common(labels.get(scientific, scientific))) or {}
        ranking.append({"scientific": names.get("scientific", scientific),
                        "common": names.get("common", _common(labels.get(scientific, scientific))),
                        "similarity": None if similarity is None else round(similarity, 3),
                        "birdnet": round(best, 3), "references": used})
    ranking.sort(key=lambda item: -(item["similarity"] if item["similarity"] is not None else -1))
    result = {"clip_sha": clip_sha, "current": current, "ranking": ranking,
              "query_segments": len(query), "created": datetime.now().isoformat(timespec="seconds")}
    CACHE.mkdir(parents=True, exist_ok=True)
    _write_atomic(result_path(clip_sha), json.dumps(result, indent=1).encode("utf-8"))
    return result


def birdnet_label(scientific, labels):
    """ชื่อตาม eBird -> ชื่อวิทยาศาสตร์ในป้ายของ BirdNET (Thinornis dubius = Charadrius dubius) หรือ None"""
    if scientific in labels:
        return scientific
    return next((sci for sci, label in labels.items()
                 if (species_names.to_ebird(sci, _common(label)) or {}).get("scientific") == scientific), None)


def _candidates(scores, current, labels, analyzer, row):
    best = defaultdict(float)
    for segment in scores.values():
        for scientific, score in segment.items():
            best[scientific] = max(best[scientific], score)
    plausible = _area_filter(analyzer, row, labels)
    others = [s for s, score in sorted(best.items(), key=lambda kv: -kv[1])
              if s != current and score >= CANDIDATE_MIN_SCORE and plausible(s)]
    picks = ([current] if current in labels else []) + others
    return picks[:MAX_CANDIDATES]


def _area_filter(analyzer, row, labels):
    """ตัวเลือกรองต้องผ่านตัวกรองพื้นที่/ฤดูเดียวกับตอนวิเคราะห์ (ถ้ารู้พิกัดและวันที่)"""
    lat, lon = _number(row.get("Latitude")), _number(row.get("Longitude"))
    try:
        date = datetime.fromisoformat(json.loads(row.get("Analysis settings") or "{}").get("date"))
    except (TypeError, ValueError):
        date = None
    if lat is None or lon is None or date is None:
        return lambda _scientific: True
    from v3_model import GEO_MIN_CONFIDENCE
    by_scientific, by_common = analyzer._geo_probabilities(lat, lon, date)

    def plausible(scientific):
        probability = by_scientific.get(scientific)
        if probability is None:
            probability = by_common.get(_common(labels.get(scientific, "")).casefold(), (None,))[0]
        return probability is not None and probability >= GEO_MIN_CONFIDENCE
    return plausible


def reference_embeddings(model, scientific, label, key, lat, lon, log=print):
    """เวกเตอร์ (n, dim) ของช่วงที่ BirdNET ได้ยินชนิดนี้ในไฟล์ xeno-canto ใกล้จุดบันทึก"""
    return embed_references(model, scientific, label, chosen_references(scientific, label, key, lat, lon), log)


def chosen_references(scientific, label, key, lat, lon):
    """ไฟล์ xeno-canto ที่ใช้เทียบชนิดนี้แถวนี้ — จำผลค้นตามองศาพิกัด (จุดใกล้กันใช้ไฟล์ชุดเดียวกันอยู่แล้ว)"""
    where = "world" if lat is None or lon is None else f"{round(lat)}_{round(lon)}"
    path = CACHE / "searches" / f"{scientific.replace(' ', '_')}_{where}.json"
    try:
        if time.time() - path.stat().st_mtime < SEARCH_DAYS * 86400:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    recordings, _total = xeno_canto.search(scientific, key, lat=lat, lon=lon, limit=30,
                                           common=str(label).partition("_")[2])
    chosen = [r for r in recordings if r["file"] and _seconds(r["length"]) <= MAX_REF_SECONDS][:REFS_PER_SPECIES]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(chosen).encode("utf-8"))
    return chosen


def embed_references(model, scientific, label, recordings, log=print):
    """เวกเตอร์ของไฟล์ xeno-canto เหล่านี้ (โหลด + ให้ BirdNET ฟังเฉพาะไฟล์ที่ยังไม่มีใน cache) -> (array, จำนวนไฟล์ที่ใช้ได้)"""
    found, used, missing = [], 0, []
    for recording in recordings:
        cached = _ref_cache(recording["id"], scientific)
        if cached.is_file():
            stored = np.load(cached)
            found += list(stored)
            used += bool(len(stored))
        else:
            missing.append(recording)

    def fetch(recording):
        try:
            return recording, _prepare(xeno_canto.download(recording))
        except xeno_canto.XenoCantoSlow:          # เน็ตช้า: ไม่ใช่ไฟล์เสีย — อย่าคิดผลจากไฟล์อ้างอิงไม่ครบ
            raise
        except (xeno_canto.XenoCantoError, RuntimeError, OSError) as exc:
            log(f"  skipped {recording['id']}: {exc}")
            return None

    pending = []
    if missing:
        with ThreadPoolExecutor(DOWNLOADS) as pool:
            pending = [item for item in pool.map(fetch, missing) if item]
    if pending:
        log(f"  analyzing {len(pending)} new reference recording(s) …")
        scores, encoded = _analyze(model, label, [str(path) for _recording, path in pending], log)
        for recording, path in pending:
            file_scores = scores.get(str(path), {})
            best = sorted((s for s, c in file_scores.items() if c >= REF_MIN_SCORE),
                          key=lambda s: -file_scores[s])[:REF_SEGMENTS]
            picked = np.array([_unit(encoded[str(path)][s]) for s in best if s in encoded.get(str(path), {})],
                              dtype=np.float32).reshape(-1, model_dim(encoded))
            cache = _ref_cache(recording["id"], scientific)
            cache.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            np.save(buffer, picked)                     # ว่าง = ไฟล์นี้ใช้ไม่ได้ (จำไว้ ไม่ต้องลองซ้ำ)
            _write_atomic(cache, buffer.getvalue())
            Path(path).unlink(missing_ok=True)          # WAV ที่เตรียมให้ BirdNET ไม่ต้องใช้อีก (มีเวกเตอร์แล้ว)
            found += list(picked)
            used += bool(len(picked))
    return (np.array(found, dtype=np.float32) if found else np.zeros((0, 1), np.float32)), used


def tidy_prepared():
    """ลบ WAV ที่เตรียมให้ BirdNET ของไฟล์อ้างอิงที่มีเวกเตอร์แล้ว (รุ่นก่อนเก็บไว้ ~4 MB ต่อไฟล์) -> จำนวนที่ลบ"""
    done = {f.stem.partition("_")[0] for f in (CACHE / "references").glob("*.npy")}
    removed = 0
    for wav in (CACHE / "prepared").glob("*.wav"):
        if wav.stem in done:
            wav.unlink(missing_ok=True)
            removed += 1
    return removed


def _prepare(path):
    """WAV โมโนสำหรับ BirdNET (ตัวโหลดของ BirdNET 3.0 ล้มกับ MP3 สเตอริโอบางไฟล์)"""
    import soundfile as sf
    target = CACHE / "prepared" / (Path(path).stem + ".wav")
    if target.is_file():
        return target
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001  ไฟล์เสีย/อ่านไม่ได้
        raise RuntimeError(f"cannot read audio: {exc}") from exc
    if len(data) < rate:                     # สั้นกว่า 1 วินาที
        raise RuntimeError("recording too short")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.{os.getpid()}.tmp.wav")   # แอปกับงานอื่นอาจทำพร้อมกัน
    sf.write(str(temporary), data.mean(axis=1), rate, subtype="PCM_16")
    temporary.replace(target)
    return target


def _analyze(model, label, paths, log=print):
    """คะแนนของชนิดนี้ + เวกเตอร์ ทุกช่วง 3 วิ ของแต่ละไฟล์; ถ้าทั้งชุดล้ม ลองทีละไฟล์แล้วข้ามไฟล์ที่เสีย"""
    scores, encoded = defaultdict(dict), defaultdict(dict)

    def one_batch(batch):
        for r in model.predict(batch, top_k=None, overlap_duration_s=OVERLAP, default_confidence_threshold=0.0,
                               custom_species_list=[label], show_stats=None, **MANY_CLIPS).to_structured_array():
            scores[str(r["input"])][round(float(r["start_time"]), 2)] = float(r["confidence"])
        for r in model.encode(batch, overlap_duration_s=OVERLAP, show_stats=None,
                              **MANY_CLIPS).to_structured_array():
            encoded[str(r["input"])][round(float(r["start_time"]), 2)] = np.asarray(r["embedding"], np.float32)

    try:
        one_batch(paths)
    except RuntimeError:
        for path in paths:
            try:
                one_batch([path])
            except RuntimeError:
                log(f"  skipped {Path(path).stem}: BirdNET could not read it")
                scores.pop(path, None)
                encoded.pop(path, None)
    return scores, encoded


def model_dim(encoded):
    for segments in encoded.values():
        for vector in segments.values():
            return len(vector)
    return 1


def _ref_cache(xc_id, scientific):
    return CACHE / "references" / f"{xc_id}_{scientific.replace(' ', '_')}.npy"


def _spans(row):
    try:
        windows = json.loads(row.get("Detection boxes") or "")["windows"]
        return [(float(a), float(b)) for a, b, *_rest in windows] or None
    except (TypeError, ValueError, KeyError):
        return None                                   # summary รุ่นเก่า: ใช้ทั้งคลิป


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def disagrees(result):
    """ผลความเห็นที่สอง: True = คลิปคล้ายชนิดอื่นมากกว่าชนิดปัจจุบัน, False = เห็นด้วย,
    None = บอกไม่ได้ (ไม่มีเสียงอ้างอิงของชนิดปัจจุบัน หรือมีตัวเลือกเดียว)"""
    if not result:
        return None
    ranking = [r for r in result.get("ranking", []) if r.get("similarity") is not None]
    if len(ranking) < 2 or not any(r["scientific"] == result.get("current") for r in ranking):
        return None
    return ranking[0]["scientific"] != result["current"]


def _common(label):
    from v3_model import EBIRD_COMMON_NAMES
    scientific, _, common = str(label).partition("_")
    return EBIRD_COMMON_NAMES.get(scientific, common or scientific)


def _seconds(length):
    try:
        parts = [int(p) for p in str(length).split(":")]
    except ValueError:
        return 0
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    if len(argv) != 2:
        print("usage: --second-opinion SUMMARY.xlsx ROW")
        return 2
    key = load_settings().get("xeno_canto_key", "")
    try:
        result = run(argv[0], argv[1], key, log=lambda text: print(text, flush=True))
    except (xeno_canto.XenoCantoError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}", flush=True)
        return 1
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
