"""เสียงที่ BirdNET "เกือบเจอ" (คะแนนต่ำกว่า min_conf) + ฟังซ้ำแบบวางหน้าต่างตรงเสียง (second look)
+ ชนิดที่ควรพบที่จุด/สัปดาห์นั้นตามโมเดลพื้นที่

second look: BirdNET ฟังทีละ 3 วินาทีตามเวลาตายตัว (0, 1.5, 3 …) เสียงสั้นที่คร่อมรอยต่อจึงถูกแบ่งครึ่ง
ช่วงที่ชนิดหนึ่งได้คะแนน NEAR_FLOOR–min_conf และโมเดลพื้นที่ว่าควรพบ ให้ BirdNET ฟังอีกรอบ 2 แบบ ต่อเสียงที่เด่นที่สุด
EVENTS_PER_SPAN เสียงในช่วงนั้น: (1) หน้าต่าง 3 วินาทีที่มีเสียงนั้นอยู่ตรงกลาง (2) ตัวเสียงเดี่ยว ๆ เติมเงียบให้ครบ 3 วินาที
คะแนนดีที่สุดถึง min_conf = ผ่าน
ทดลองกับ REC-001 18 ต.ค. (2026-10-24):
  - ช่วงเกือบเจอที่ควรพบที่นี่ ~200 ช่วง ผ่านราว 13 — รวม Grey Wagtail 0.48→0.58, Eurasian Coot 0.45→0.53,
    Little Owl 0.32→0.76 (3 ชนิดที่โปรแกรมอ้างอิงรายงานแต่แอปไม่ได้ตัด)
  - ฟังตัวเสียงเดี่ยว ๆ ทั้งไฟล์ (ไม่จำกัดเฉพาะช่วงเกือบเจอ) ได้ 78 ชนิดที่คะแนน ≥ 0.5 ส่วนใหญ่ผิด (นกอเมริกา)
    จึงใช้เฉพาะชนิดที่ BirdNET ทายอยู่แล้วในช่วงนั้นและโมเดลพื้นที่ว่าควรพบ
  - คลิปที่ตรวจแล้ว: ฟังซ้ำไม่ช่วยแยก ID ผิด (Reject 2 ใน 3 ยังได้ ≥ 0.6) จึงใช้หาเสียงที่พลาด ไม่ใช่ตัวกรอง

data/cache/sound-index/near-<hash>.json      ผลของหน้าต่าง Near misses (จากดัชนีเสียง) ต่อผลชุด
data/cache/sound-index/expected-<hash>.json  ชนิดที่ควรพบ (โมเดลพื้นที่) ต่อผลชุด
CLI (process แยก):
  field_audio_to_ebird.py --sound-index near SUMMARY.xlsx
  field_audio_to_ebird.py --sound-index expected SUMMARY.xlsx
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import sound_index

NEAR_FLOOR = 0.2               # คะแนนต่ำสุดที่นับว่า "เกือบเจอ" (ต่ำกว่านี้ BirdNET แทบเดา)
EVENTS_PER_SPAN = 2            # ฟังซ้ำเสียงที่เด่นที่สุดกี่เสียงต่อช่วง
MAX_SPANS_PER_SPECIES = 25     # หน้าต่าง Near misses: ฟังซ้ำช่วงที่คะแนนสูงสุดเท่านี้ต่อชนิดต่อไฟล์
WINDOW = 3.0
GEO_EXPECTED = 0.03            # เท่ากับตัวกรองพื้นที่ของการวิเคราะห์ (v3_model.GEO_MIN_CONFIDENCE)


def _detect_params():
    import signal_events
    # ตัดตัวเสียงก่อนให้ BirdNET ฟัง (250–10000 Hz, ≤ 7.5 วิ, เว้น 0.35 วิ) แต่ไวกว่า 3 dB: เสียงไกล/เบา
    return signal_events.DetectionParams(low=250.0, high=10000.0, minimum=0.05, maximum=7.5, gap=0.35, excess_db=9.0)


def loud_events(samples, sr, top=EVENTS_PER_SPAN):
    """เสียงที่เด่นที่สุดในช่วงนี้ -> [(กึ่งกลาง, ความยาว)] วินาทีจากต้นช่วง; ไม่มีเลย = กึ่งกลางช่วง"""
    import signal_events
    events = signal_events.detect_samples(samples, sr, _detect_params()) if len(samples) else []
    events = sorted(events, key=lambda e: -e["score"])[:top]
    if not events:
        middle = len(samples) / sr / 2 if sr else 0.0
        return [(middle, min(1.0, middle * 2))]
    return [((e["start"] + e["end"]) / 2, e["end"] - e["start"]) for e in events]


def _fit(samples, sr, seconds=WINDOW):
    """ยาวเกิน: เอากลาง ๆ, สั้นกว่า: เติมเงียบสองข้าง (BirdNET ฟังทีละ 3 วินาทีพอดี)"""
    n = int(round(seconds * sr))
    if len(samples) >= n:
        first = (len(samples) - n) // 2
        return np.ascontiguousarray(samples[first:first + n], np.float32)
    pad = n - len(samples)
    return np.pad(np.asarray(samples, np.float32), (pad // 2, pad - pad // 2))


def look_inputs(source, duration, start, end, top=EVENTS_PER_SPAN):
    """เสียงที่จะให้ BirdNET ฟังซ้ำสำหรับช่วง [start, end] -> [(samples, sr, เวลาของเสียงในไฟล์, "centred"/"isolated")]"""
    from audio_io import read_mono
    start, end = max(0.0, start), min(duration, end) if duration else end
    samples, sr = read_mono(source, start, end)
    found = []
    for centre, length in loud_events(samples, sr, top):
        at = start + centre
        first = min(max(0.0, at - WINDOW / 2), max(0.0, duration - WINDOW)) if duration else max(0.0, at - WINDOW / 2)
        window, rate = read_mono(source, first, first + WINDOW)
        found.append((_fit(window, rate), rate, at, "centred"))
        a, b = int(max(0.0, centre - length / 2) * sr), int(min(len(samples) / sr, centre + length / 2) * sr)
        found.append((_fit(samples[a:max(b, a + 1)], sr), sr, at, "isolated"))
    return found


def second_look(model, source, spans, duration=None, log=print):
    """spans = [{"start", "end", "label"}] (label = ป้ายชื่อของ BirdNET 'Scientific_Common')
    -> เติม "look" (คะแนนดีที่สุด), "look_how" (centred/isolated), "look_at" (เวลาของเสียงในไฟล์) ให้แต่ละช่วง"""
    if not spans:
        return spans
    from audio_io import audio_duration
    duration = duration if duration is not None else audio_duration(source)
    arrays, owners = [], []
    for number, span in enumerate(spans):
        span.update(look=0.0, look_how="", look_at=None)
        try:
            items = look_inputs(source, duration, span["start"], span["end"])
        except Exception as exc:  # noqa: BLE001  อ่านช่วงนี้ไม่ได้: ข้าม ไม่ล้มทั้งไฟล์
            log(f"  second look: cannot read {span['start']:.1f}–{span['end']:.1f} s ({exc})")
            continue
        for samples, rate, at, how in items:
            arrays.append((samples, rate))
            owners.append((number, at, how))
    if not arrays:
        return spans
    labels = sorted({str(span["label"]) for span in spans})
    result = model.predict_arrays(arrays, top_k=None, n_workers=1, default_confidence_threshold=0.01,
                                  custom_species_list=labels, show_stats=None).to_structured_array()
    for row in result:
        number, at, how = owners[int(row["input"])]
        span = spans[number]
        score = float(row["confidence"])
        if str(row["species_name"]) == str(span["label"]) and score > span["look"]:
            span.update(look=round(score, 3), look_how=how, look_at=round(at, 2))
    return spans


def describe_look(grid, look, how):
    """ข้อความใน summary/Review: ทำไมเสียงนี้ถึงถูกตัด"""
    way = "a 3-s window centred on the sound" if how == "centred" else "the sound on its own"
    return f"Second look: BirdNET {grid:.2f} in its usual 3-s windows, {look:.2f} for {way}"


# ----------------------------- โมเดลพื้นที่ -----------------------------
class Geo:
    """ความน่าจะเป็นที่แต่ละชนิดจะพบที่พิกัด/สัปดาห์นั้น (BirdNET geo 3.0) — โหลดโมเดลครั้งเดียว"""

    def __init__(self, model=None):
        self.model, self._cache = model, {}

    def probabilities(self, lat, lon, when=None):
        """{scientific: prob} ของสัปดาห์นั้น; when = None -> สูงสุดของทั้งปี (นกที่มาบางฤดู)"""
        from v3_model import birdnet_week
        weeks = (birdnet_week(when),) if when is not None else tuple(range(1, 49))
        key = (round(float(lat), 4), round(float(lon), 4), weeks)
        if key not in self._cache:
            if self.model is None:
                import birdnet
                self.model = birdnet.load("geo", "3.0", "onnx", precision="fp16")
            found = {}
            for week in weeks:
                prediction = self.model.predict(float(lat), float(lon), week=week, min_confidence=0.0)
                for label, probability in zip(prediction.species_list, prediction.species_probs):
                    scientific, _, common = str(label).partition("_")
                    if float(probability) > found.get(scientific, (0.0, ""))[0]:
                        found[scientific] = (float(probability), common)
            self._cache[key] = found
        return self._cache[key]

    def probability(self, scientific, common, lat, lon, when):
        """ชื่อของโมเดลเสียงกับโมเดลพื้นที่ต่างกัน ~900 ชื่อ: ไม่เจอชื่อวิทยาศาสตร์ -> ลองชื่อสามัญ; ไม่รู้จัก = None"""
        found = self.probabilities(lat, lon, when)
        if scientific in found:
            return found[scientific][0]
        common = str(common or "").casefold()
        return next((p for p, name in found.values() if name.casefold() == common), None) if common else None


# ----------------------------- Near misses จากดัชนีเสียง -----------------------------
def _cache_file(kind, summary):
    digest = hashlib.sha256(str(Path(summary).resolve()).encode("utf-8")).hexdigest()[:12]
    return sound_index.CACHE / f"{kind}-{digest}.json"


def near_file(summary):
    return _cache_file("near", summary)


def expected_file(summary):
    return _cache_file("expected", summary)


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_near(summary):
    return _load(near_file(summary))


def load_expected(summary):
    return _load(expected_file(summary))


def _write(path, data):
    sound_index.CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    temporary.replace(path)


def min_conf_of(rows, default=0.5):
    """min_conf ที่ใช้วิเคราะห์ผลชุดนี้ (จาก Analysis settings ของคลิป BirdNET)"""
    for row in rows:
        try:
            return float(json.loads(str(row.get("Analysis settings") or "{}")).get("min_conf", default))
        except (TypeError, ValueError):
            continue
    return default


def species_clips(rows):
    """{scientific (eBird): {"clips", "approved"}} ของผลชุดนี้ (ชื่อที่ผู้ตรวจแก้แล้วมาก่อน)
    คลิปที่ Reject ไม่นับ (Reject = ชนิดผิด: ชนิดนั้นไม่ได้อยู่ตรงนั้น)"""
    import species_names
    found = {}
    for row in rows:
        if str(row.get("Species (common)") or "") == "_Unknown" or str(row.get("Review status")) == "Rejected":
            continue
        scientific = " ".join(str(row.get("Reviewed scientific") or "").split()[:2]) or species_names.row_names(row)[1]
        item = found.setdefault(scientific, {"clips": 0, "approved": 0})
        item["clips"] += 1
        item["approved"] += str(row.get("Review status")) == "Approved"
    return found


def candidates(index, floor=NEAR_FLOOR):
    """ช่วงในดัชนีที่ชนิดนกได้คะแนน >= floor -> {label (BirdNET): [(start, score)]}"""
    import species_names
    labels = sound_index.label_list(index["label_set"])
    found, is_bird = {}, {}
    for i, start in enumerate(index["starts"]):
        for label, score in zip(index["labels"][i], index["scores"][i]):
            if label < 0 or label >= len(labels) or score < floor:
                continue
            if label not in is_bird:
                scientific, _, common = labels[label].partition("_")
                is_bird[label] = species_names.is_bird_label(scientific, common)
            if is_bird[label]:
                found.setdefault(labels[label], []).append((float(start), float(score)))
    return found


def find_near_misses(summary, model=None, geo=None, log=print, second=True):
    """ทุกไฟล์ในผลชุด: ชนิดนกที่ BirdNET ได้ยิน (คะแนน >= NEAR_FLOOR) แต่ไม่มีคลิปของชนิดนั้นตรงนั้น
    + โมเดลพื้นที่ + checklist + ฟังซ้ำ (เฉพาะชนิดที่ควรพบ) -> dict (เก็บเป็น JSON ด้วย)"""
    import country_checklist
    import manual_clips
    import species_names
    from review_store import load_reviews
    summary = Path(summary)
    rows = load_reviews(summary)
    found = sound_index.sources(rows, summary)
    spans = sound_index.clip_spans(rows)
    min_conf = min_conf_of(rows)
    geo = geo or Geo()
    species = {}
    looked = 0
    for sid, entry in found.items():
        index = sound_index.load(sid)
        if index is None:
            log(f"  no sound index for {entry['name']} — skipped")
            continue
        context = manual_clips.source_context(entry["rows"], entry["path"]) or {}
        lat, lon, began = context.get("lat"), context.get("lon"), context.get("start")
        todo = []
        for label, items in candidates(index).items():
            scientific, _, common = label.partition("_")
            names = species_names.to_ebird(scientific, common) or {"common": common, "scientific": scientific}
            events = []
            for event in sound_index.merge_hits(np.array([s for s, _ in items]), np.array([c for _, c in items]), 0.0):
                same = [row for row in sound_index.clips_at(spans, sid, event["start"], event["end"])
                        if names["scientific"] in {" ".join(str(row.get("Reviewed scientific") or "").split()[:2]),
                                                   species_names.row_names(row)[1]}]
                if not same:
                    events.append(dict(event, sid=sid, label=label))
            if not events:
                continue
            when = began + timedelta(seconds=events[0]["best"]) if began else None
            probability = (geo.probability(scientific, common, lat, lon, when)
                           if lat is not None and when is not None else None)
            item = species.setdefault(names["scientific"], {
                "common": names["common"], "scientific": names["scientific"], "birdnet": label,
                "geo": probability, "events": [],
                "checklist": country_checklist.describe(country_checklist.check(names["scientific"], lat, lon,
                                                                                names["common"]))
                if lat is not None else ""})
            if probability is not None and (item["geo"] is None or probability > item["geo"]):
                item["geo"] = probability
            item["events"] += events
            if second and probability is not None and probability >= GEO_EXPECTED:
                todo += sorted((e for e in events if e["score"] < min_conf), key=lambda e: -e["score"])[:MAX_SPANS_PER_SPECIES]
        if second and todo:
            model = model or sound_index._load_birdnet()
            log(f"  {entry['name']}: second look at {len(todo)} near miss(es) …")
            second_look(model, entry["path"], todo, index.get("duration") or None, log)
            looked += len(todo)
    for item in species.values():
        item["events"].sort(key=lambda e: -max(e["score"], e.get("look") or 0.0))
        item["best"] = max(e["score"] for e in item["events"])
        item["look"] = max((e.get("look") or 0.0 for e in item["events"]), default=0.0)
        item["passed"] = sum((e.get("look") or 0.0) >= min_conf for e in item["events"])
    result = {"summary": str(summary.resolve()), "made": datetime.now().isoformat(timespec="seconds"),
              "min_conf": min_conf, "floor": NEAR_FLOOR, "looked": looked,
              "species": sorted(species.values(), key=lambda item: (-item["passed"], -item["best"])),
              "clips": species_clips(rows),
              "sources": {sid: {"path": str(entry["path"]), "name": entry["name"]} for sid, entry in found.items()}}
    _write(near_file(summary), result)
    log(f"near misses: {len(species)} species without a clip at those moments; second look at {looked} sound(s), "
        f"{sum(item['passed'] for item in species.values())} reached {min_conf:g}")
    return result


def find_expected(summary, geo=None, log=print):
    """ชนิดที่โมเดลพื้นที่ว่าควรพบที่จุด/สัปดาห์ของผลชุดนี้ (และทั้งปี) + checklist + คลิปที่มี + คะแนนสูงสุดในดัชนี"""
    import country_checklist
    import manual_clips
    import species_names
    from review_store import load_reviews
    summary = Path(summary)
    rows = load_reviews(summary)
    found = sound_index.sources(rows, summary)
    geo = geo or Geo()
    context = {}
    heard = {}
    for sid, entry in found.items():
        context = context or manual_clips.source_context(entry["rows"], entry["path"]) or {}
        index = sound_index.load(sid)
        if index is None:
            continue
        for label, items in candidates(index, floor=0.05).items():
            scientific, _, common = label.partition("_")
            names = species_names.to_ebird(scientific, common) or {"scientific": scientific}
            heard[names["scientific"]] = max(heard.get(names["scientific"], 0.0), max(s for _, s in items))
    lat, lon, began = context.get("lat"), context.get("lon"), context.get("start")
    if lat is None or began is None:
        raise ValueError("this result has no recording date or location")
    week = geo.probabilities(lat, lon, began)
    year = geo.probabilities(lat, lon, None)
    species = []
    for scientific, (probability, common) in year.items():
        # ชนิดที่แทบไม่มีโอกาสพบที่นี่ทั้งปี (ส่วนใหญ่ของ 14,000 ชนิดในโมเดล) ไม่ต้องเก็บ
        if probability < 0.005 or not species_names.is_bird_label(scientific, common):
            continue
        names = species_names.to_ebird(scientific, common) or {"common": common, "scientific": scientific}
        species.append({"common": names["common"], "scientific": names["scientific"], "geo_name": scientific,
                        "week": round(week.get(scientific, (0.0, ""))[0], 4), "year": round(probability, 4),
                        "heard": round(heard.get(names["scientific"], 0.0), 3),
                        "checklist": country_checklist.describe(country_checklist.check(
                            names["scientific"], lat, lon, names["common"]))})
    species.sort(key=lambda item: (-item["week"], -item["year"]))
    result = {"summary": str(summary.resolve()), "made": datetime.now().isoformat(timespec="seconds"),
              "lat": lat, "lon": lon, "start": began.isoformat(), "species": species, "clips": species_clips(rows),
              "indexed": sum(sound_index.has_index(sid) for sid in found), "recordings": len(found)}
    _write(expected_file(summary), result)
    log(f"expected here: {sum(item['week'] >= GEO_EXPECTED for item in species)} species this week, "
        f"{sum(item['year'] >= GEO_EXPECTED for item in species)} at some time of year")
    return result
