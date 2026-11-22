"""ตัวจำแนกของคุณเอง: ต่อชนิด ฝึกจากคลิปที่คุณตรวจแล้ว
แล้วใช้สแกนไฟล์ต้นฉบับหาเสียงชนิดนั้น รวมเสียงที่ BirdNET ไม่เจอ (เช่น ชนิดจีน/ไทยที่ BirdNET ยังไม่แม่น)

  ตัวอย่างบวก  = คลิปที่ Approve เป็นชนิดนั้น (ของ BirdNET หรือที่เพิ่มเอง)
  ตัวอย่างลบ   = คลิปที่ BirdNET ทายเป็นชนิดนั้นแต่ผิด (Reject หรือ Approve เป็นชนิดอื่น) — ยากและสำคัญที่สุด
                 + คลิปที่ Approve เป็นชนิดอื่น + ช่วงสุ่มจากไฟล์เดียวกันที่ไม่มีชนิดนั้น (เสียงพื้นหลัง)
  เวกเตอร์ = ดัชนีเสียงของไฟล์ต้นฉบับในช่วงที่ได้ยิน (sound_index.py) หรือเวกเตอร์ของคลิปจากงาน Auto-checks
  โมเดล = logistic regression (ปรับสเกล) ทดสอบกับไฟล์ต้นฉบับที่ไม่ได้ใช้ฝึก แล้วตั้งเกณฑ์ให้แม่น ≥ 80%

ต่างจาก ML model ใน Review progress ซึ่งทายว่า ID ของ BirdNET น่าจะผิดไหม (ไม่ได้หาเสียงใหม่)
data/cache/recognizers/<ชื่อวิทยาศาสตร์>.npz   สเกล + coef + เกณฑ์ + ผลทดสอบ (สแกนด้วย numpy อย่างเดียว)
CLI (GUI เรียกเป็นงานในคิว): field_audio_to_ebird.py --recognizer train "Genus species" SUMMARY.xlsx ...
"""

import io
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

import paths
import sound_index

CACHE = paths.CACHE / "recognizers"
MIN_POSITIVE = 8               # คลิปที่ Approve อย่างน้อยเท่านี้ถึงจะฝึก
TARGET_PRECISION = 0.8         # เกณฑ์ตั้งให้เสียงที่ผ่านเป็นชนิดนั้นจริงอย่างน้อยเท่านี้ (ในการทดสอบ)
BACKGROUND_PER_POSITIVE = 4
OTHER_SPECIES_PER_POSITIVE = 3
WINDOWS_PER_CLIP = 6
C_VALUE = 0.1


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _two(name):
    return " ".join(str(name or "").split()[:2])


def reviewed_species(row):
    """(ชนิดที่ถูก, ชนิดที่ BirdNET ทายผิด) ของแถวนี้ตามผลตรวจ — None = บอกไม่ได้"""
    import species_names
    status = str(row.get("Review status") or "Pending")
    birdnet = _two(species_names.row_names(row)[1])
    reviewed = _two(row.get("Reviewed scientific"))
    if status == "Approved":
        right = reviewed or birdnet
        return right, (birdnet if right != birdnet and birdnet else None)
    if status == "Rejected":
        return None, birdnet or None
    return None, None


def examples(records, sound_type=""):
    """{ชื่อวิทยาศาสตร์: {"common", "positive": [แถว], "negative": [แถว]}} จากผลตรวจทุกผลชุด
    negative = คลิปที่ BirdNET ทายเป็นชนิดนั้นแต่ผิด"""
    import species_names
    from review_store import SOUND_TYPES
    if sound_type not in SOUND_TYPES:
        raise ValueError("Choose a supported sound type")
    found = {}
    seen = set()
    for row in records:
        right, wrong = reviewed_species(row)
        sid = row.get("Source ID") or row.get("Source path")
        if sid and row.get("Start offset (s)") is not None:
            key = (str(sid), row.get("Start offset (s)"), row.get("End offset (s)"),
                   row.get("Source channel") or "original", right, wrong, row.get("Sound type") or "")
            if key in seen:
                continue
            seen.add(key)
        kind = str(row.get("Sound type") or "")
        if right and (not sound_type or kind == sound_type):
            entry = found.setdefault(right, {"common": "", "positive": [], "negative": []})
            entry["positive"].append(row)
            entry["common"] = entry["common"] or str(row.get("Reviewed common") or species_names.row_names(row)[0])
        if wrong:
            entry = found.setdefault(wrong, {"common": "", "positive": [], "negative": []})
            entry["negative"].append(row)
            entry["common"] = entry["common"] or species_names.row_names(row)[0]
        if right and sound_type and kind and kind != sound_type:
            entry = found.setdefault(right, {"common": "", "positive": [], "negative": []})
            entry["negative"].append(row)
            entry["common"] = entry["common"] or str(row.get("Reviewed common") or species_names.row_names(row)[0])
    return found


def validation_group(row, sid=None):
    """Keep recordings from the same day together to avoid nearby-file leakage."""
    import manual_clips
    start = manual_clips.recording_start_of(row)
    return "day:" + start.date().isoformat() if start else sid or str(row.get("Generated sha256"))


def row_vectors(row):
    """เวกเตอร์ของช่วงที่ได้ยินในคลิปนี้ -> (เวกเตอร์ [n, d], sid หรือ None)"""
    sid = str(row.get("Source ID") or "").strip()
    if row.get("Source channel") in ("left", "right"):
        import sound_type
        vectors = sound_type.clip_vectors(str(row.get("Generated sha256") or ""))
        if vectors is not None and len(vectors) and vectors.shape[1] > 1:
            return np.asarray(vectors, np.float32), sid or None
        # A mixed-channel source index is not a valid substitute for a labelled single-channel clip.
        return np.zeros((0, 1), np.float32), sid or None
    first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
    if sid and first is not None and sound_index.has_index(sid):
        index = sound_index.load(sid)
        last = max(last if last is not None else first + sound_index.WINDOW, first + 0.5)
        overlap = np.minimum(index["starts"] + sound_index.WINDOW, last) - np.maximum(index["starts"], first)
        chosen = np.where(overlap >= min(sound_index.HOP, (last - first) / 2))[0]
        if len(chosen) > WINDOWS_PER_CLIP:          # ใช้ช่วงที่ BirdNET ว่าเป็นนกมากที่สุด (เสียงนกชัดสุด)
            chosen = chosen[np.argsort(-index["bird"][chosen])[:WINDOWS_PER_CLIP]]
        if len(chosen):
            return index["vectors"][np.sort(chosen)].astype(np.float32), sid
    import sound_type
    vectors = sound_type.clip_vectors(str(row.get("Generated sha256") or ""))
    if vectors is not None and len(vectors) and vectors.shape[1] > 1:
        return np.asarray(vectors, np.float32), sid or None
    return np.zeros((0, 1), np.float32), sid or None


def _background(records, scientific, rng, wanted):
    """ช่วงสุ่มจากไฟล์ที่มีดัชนี ที่ไม่ทับคลิปของชนิดนี้ -> (เวกเตอร์, กลุ่ม = sid)"""
    spans = {}
    sids = set()
    for row in records:
        sid = str(row.get("Source ID") or "").strip()
        if not sid or not sound_index.has_index(sid):
            continue
        sids.add(sid)
        right, wrong = reviewed_species(row)
        names = {right, wrong, _two(row.get("Species (scientific)"))}
        first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
        if scientific in names and first is not None:
            spans.setdefault(sid, []).append((first - 3, (last if last is not None else first) + 3))
    if not sids:
        return np.zeros((0, 1), np.float32), []
    per_source = max(1, wanted // len(sids))
    vectors, groups = [], []
    for sid in sorted(sids):
        index = sound_index.load(sid)
        free = [i for i, start in enumerate(index["starts"])
                if not any(a < start + sound_index.WINDOW and start < b for a, b in spans.get(sid, ()))]
        for i in rng.permutation(free)[:per_source]:
            vectors.append(index["vectors"][i].astype(np.float32))
            groups.append(sid)
    return np.asarray(vectors, np.float32), groups


def training_data(scientific, records, seed=0, sound_type=""):
    """(X, y, กลุ่ม = ไฟล์ต้นฉบับ, จำนวนคลิปบวก, จำนวนคลิปที่ BirdNET ทายผิดเป็นชนิดนี้)"""
    rng = np.random.default_rng(seed)
    pool = examples(records, sound_type)
    mine = pool.get(scientific, {"positive": [], "negative": []})
    X, y, groups = [], [], []
    positive_clips = 0
    for row in mine["positive"]:
        vectors, sid = row_vectors(row)
        if len(vectors) and vectors.shape[1] > 1:
            X += list(vectors)
            y += [1] * len(vectors)
            groups += [validation_group(row, sid)] * len(vectors)
            positive_clips += 1
    hard = 0
    for row in mine["negative"]:
        vectors, sid = row_vectors(row)
        if len(vectors) and vectors.shape[1] > 1:
            X += list(vectors)
            y += [0] * len(vectors)
            groups += [validation_group(row, sid)] * len(vectors)
            hard += 1
    others = [row for name, entry in examples(records).items() if name != scientific for row in entry["positive"]]
    for i in rng.permutation(len(others))[:OTHER_SPECIES_PER_POSITIVE * max(positive_clips, 1)]:
        vectors, sid = row_vectors(others[i])
        if len(vectors) and vectors.shape[1] > 1:
            X += list(vectors[:2])
            y += [0] * len(vectors[:2])
            groups += [validation_group(others[i], sid)] * len(vectors[:2])
    background, background_groups = _background(records, scientific, rng,
                                                BACKGROUND_PER_POSITIVE * max(positive_clips, 1) * 2)
    if len(background):
        X += list(background)
        y += [0] * len(background)
        by_sid = {str(row.get("Source ID")): validation_group(row, str(row.get("Source ID"))) for row in records}
        groups += [by_sid.get(sid, sid) for sid in background_groups]
    if X:
        common = max({len(v) for v in X}, key=lambda d: sum(len(v) == d for v in X))   # เวกเตอร์รุ่นเดียวกัน
        keep = [len(v) == common for v in X]
        X, y, groups = ([a for a, k in zip(items, keep) if k] for items in (X, y, groups))
    return np.asarray(X, np.float32), np.asarray(y, int), np.asarray(groups), positive_clips, hard


def _classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_VALUE, max_iter=4000, class_weight="balanced"))


def _cross_check(X, y, groups):
    """ทดสอบกับไฟล์ต้นฉบับที่ไม่ได้ใช้ฝึก -> (ความน่าจะเป็นของทุกช่วง, ทดสอบแยกไฟล์ได้ไหม)"""
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    names = sorted(set(groups[y == 1]))
    separate = len(names) >= 2
    if separate:
        splits = GroupKFold(n_splits=min(5, len(names))).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X, y)
    probability = np.zeros(len(X))
    for fit, test in splits:
        if len(set(y[fit])) < 2:
            probability[test] = y[fit].mean()
            continue
        probability[test] = _classifier().fit(X[fit], y[fit]).predict_proba(X[test])[:, 1]
    return probability, separate


def choose_threshold(probability, y):
    """เกณฑ์ต่ำสุดที่ยังแม่น (precision) ≥ TARGET_PRECISION -> (เกณฑ์, แม่น, เจอกี่ %) — ไม่ถึงเลย: เกณฑ์ที่ F1 ดีสุด"""
    probability, y = np.asarray(probability), np.asarray(y)
    order = np.argsort(-probability)
    hits = np.cumsum(y[order] == 1)
    # A threshold cannot split tied scores. Evaluate only complete tie groups.
    ends = np.r_[np.where(np.diff(probability[order]) != 0)[0], len(order) - 1]
    thresholds = np.clip(probability[order][ends], 0.05, 0.99)
    counts = np.searchsorted(-probability[order], -thresholds, side="right")
    true = np.where(counts > 0, hits[np.maximum(0, counts - 1)], 0)
    precision = true / np.maximum(counts, 1)
    recall = true / max(int((y == 1).sum()), 1)
    good = np.where(precision >= TARGET_PRECISION)[0]
    if len(good):
        at = good.max()
    else:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        at = int(np.argmax(f1))
    return float(thresholds[at]), float(precision[at]), float(recall[at])


def train(scientific, records, log=print, sound_type=""):
    """ฝึกตัวจำแนกของชนิดนี้แล้วเก็บไว้ -> dict ผลทดสอบ (ValueError ถ้าตัวอย่างไม่พอ)"""
    scientific = _two(scientific)
    X, y, groups, positives, hard = training_data(scientific, records, sound_type=sound_type)
    if positives < MIN_POSITIVE:
        raise ValueError(f"{scientific}: {positives} approved clip(s) with sound vectors — "
                         f"needs at least {MIN_POSITIVE}")
    if not (y == 0).any():
        raise ValueError(f"{scientific}: no other sounds to compare with yet")
    probability, separate = _cross_check(X, y, groups)
    threshold, precision, recall = choose_threshold(probability, y)
    model = _classifier().fit(X, y)
    scaler, regression = model[0], model[-1]
    meta = {"scientific": scientific, "common": examples(records, sound_type).get(scientific, {}).get("common", ""),
            "sound_type": sound_type,
            "trained": date.today().isoformat(), "positives": positives, "hard_negatives": hard,
            "windows": int(len(y)),
            "recordings": len({str(r.get("Source ID") or r.get("Generated sha256"))
                               for r in examples(records, sound_type).get(scientific, {}).get("positive", [])}),
            "validation_groups": int(len(set(groups[y == 1]))),
            "threshold": float(threshold), "precision": round(precision, 3), "recall": round(recall, 3),
            "tested_on_other_recordings": separate,
            "validation": "recording-day groups" if any(str(g).startswith("day:") for g in groups) else "recording groups",
            "false_positive_windows": int(((probability >= threshold) & (y == 0)).sum()),
            "false_negative_windows": int(((probability < threshold) & (y == 1)).sum()),
            "validation_note": "Cross-validation estimate; threshold selected on these predictions"}
    CACHE.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=regression.coef_[0].astype(np.float32), intercept=np.float32(regression.intercept_[0]),
             meta=json.dumps(meta))
    path = model_file(scientific, sound_type)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)
    log(f"recognizer {scientific}: {positives} clips (+{hard} wrong BirdNET IDs) — finds {recall:.0%} of its sounds "
        f"at {precision:.0%} precision" + ("" if separate else " (tested on the same recordings only)"))
    return meta


def model_file(scientific, sound_type=""):
    from review_store import SOUND_TYPES
    import re
    if sound_type not in SOUND_TYPES:
        raise ValueError("Unsupported sound type")
    name = re.sub(r"[^\w.-]+", "_", _two(scientific)).strip("._")
    if not name:
        raise ValueError("Enter a scientific name")
    suffix = "--" + sound_type.replace(" ", "_") if sound_type else ""
    return CACHE / f"{name}{suffix}.npz"


def load(scientific, sound_type=""):
    try:
        with np.load(model_file(scientific, sound_type)) as data:
            model = {key: data[key] for key in ("mean", "scale", "coef")}
            model["intercept"] = float(data["intercept"])
            model["meta"] = json.loads(str(data["meta"]))
            return model
    except (OSError, ValueError, KeyError):
        return None


def trained(sound_type=""):
    """{ชื่อวิทยาศาสตร์: meta} ของตัวจำแนกที่ฝึกแล้ว"""
    found = {}
    for path in CACHE.glob("*.npz"):
        name, _, kind = path.stem.partition("--")
        if kind.replace("_", " ") != sound_type:
            continue
        model = load(name.replace("_", " "), sound_type)
        if model:
            found[model["meta"]["scientific"]] = model["meta"]
    return found


def probabilities(model, index):
    """ความน่าจะเป็นว่าแต่ละช่วงในดัชนีเป็นชนิดนี้"""
    vectors = index["vectors"].astype(np.float32)
    if not len(vectors) or vectors.shape[1] != len(model["coef"]):
        return np.zeros(len(vectors), np.float32)
    logit = (vectors - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(logit, -60, 60)))


def scan(model, sids, threshold=None):
    """สแกนดัชนีของไฟล์เหล่านี้ -> [{"sid", "start", "end", "score" = ความน่าจะเป็น, "best"}] มากไปน้อย"""
    threshold = model["meta"]["threshold"] if threshold is None else threshold
    found = []
    for sid in sids:
        index = sound_index.load(sid)
        if index is None:
            continue
        for event in sound_index.merge_hits(index["starts"], probabilities(model, index), threshold):
            found.append(dict(event, sid=sid))
    found.sort(key=lambda event: -event["score"])
    return found


def describe(meta):
    if not meta:
        return "not trained"
    text = (f"finds {meta['recall']:.0%} at {meta['precision']:.0%} precision "
            f"({meta['positives']} clips, {meta['recordings']} recording(s); {meta['trained']})")
    if meta.get("sound_type"):
        text = meta["sound_type"] + ": " + text
    if meta.get("validation"):
        text += " · CV by " + meta["validation"]
    return text if meta.get("tested_on_other_recordings") else text + " — tested on the same recordings only"


def main(argv):
    """GUI อ่านบรรทัด TRAINED {...} / ERROR ..."""
    if len(argv) < 3 or argv[0] != "train":
        print('usage: --recognizer train "Genus species" SUMMARY.xlsx ...')
        return 2
    import review_progress
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=["train"])
        parser.add_argument("scientific")
        parser.add_argument("summaries", nargs="+")
        parser.add_argument("--sound-type", default="")
        args = parser.parse_args(argv)
        records = review_progress.collect([Path(p) for p in args.summaries])
        meta = train(args.scientific, records, log=lambda text: print(text, flush=True), sound_type=args.sound_type)
        print("TRAINED " + json.dumps(meta), flush=True)
    except KeyboardInterrupt:
        print("ERROR stopped", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
