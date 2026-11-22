"""เดาประเภทเสียงของคลิป (Song / Call) — BirdNET บอกได้แค่ชนิด ไม่บอกว่าเป็นเพลงหรือเสียงร้องเรียก

วิธี: เวกเตอร์เสียง (embedding) ของ BirdNET 3.0 จากช่วงที่ BirdNET ได้ยินนกในคลิป เข้าตัวแยก logistic
regression ที่ฝึกจากไฟล์ xeno-canto ที่คนอัดระบุประเภทไว้ (ทุกชนิดรวมกัน รวมไฟล์ของชนิดนั้นเอง)
+ คลิปที่ผู้ตรวจเลือก Sound type เอง (Approve หรือยัง Pending; ไม่นับ Reject/Not sure) น้ำหนักมากกว่า:
  เสียงจากเครื่องอัดของผู้ใช้เอง — เลือกได้ในแท็บ Review หรือหน้าต่าง Sound types (sound_type_window.py)

ทดลอง 2026-10-20 กับไฟล์ xeno-canto 679 ไฟล์ 110 ชนิด (ทดสอบกับไฟล์ที่ไม่ได้ใช้ฝึก):
  ถูก 79%; เกณฑ์มั่นใจคำนวณใหม่ทุกครั้งที่ฝึก ให้คำเดาที่มั่นใจถูก ~90% (ครอบคลุมราวครึ่งหนึ่ง)
  flight call / alarm call / drumming มีตัวอย่างไม่พอ — ผู้ตรวจเลือกเอง (ตอนฝึกนับ flight/alarm call เป็น Call)

ไฟล์ใน data/cache/sound-type/
  clips/<sha256 ของคลิป>.npy   เวกเตอร์ของคลิป (ว่าง = คลิปนี้ใช้ไม่ได้)
  species.json                 ชนิดที่หาเสียงอ้างอิงแยก song/call แล้ว {ชื่อวิทยาศาสตร์: วันที่}
  model.npz                    ตัวแยก (สเกล + coef, intercept, เกณฑ์มั่นใจ, ผลทดสอบ) — GUI คำนวณด้วย numpy อย่างเดียว
ประเภทเสียงของไฟล์ xeno-canto อยู่ใน xeno_canto.TYPES_FILE; เวกเตอร์ของไฟล์อ้างอิงใช้ร่วมกับ second opinion
"""

import io
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import paths

CACHE = paths.CACHE / "sound-type"
SURE = 0.55                    # เกณฑ์มั่นใจขั้นต่ำ (เกณฑ์จริงคำนวณตอนฝึก: ทดสอบแล้วถูก >= SURE_TARGET)
                               # ความน่าจะเป็นของโมเดลถูกบีบเข้าหา 50% (regularization) จึงไม่ใช้ค่าตายตัว
SURE_TARGET = 0.9              # คำเดาที่มั่นใจ (เติมช่อง Sound type ให้) ต้องถูกอย่างน้อยเท่านี้ในการทดสอบ
C_VALUE = 0.01
REFS_PER_TYPE = 4              # ไฟล์ xeno-canto ต่อชนิดต่อประเภท (song / call)
USER_WEIGHT = 3.0
CLIP_SEGMENTS = 3              # ช่วง 3 วิ ที่ BirdNET ให้คะแนนสูงสุดในคลิป
MANUAL_SEGMENTS = 6
MIN_TRAINING = 40
RECHECK_DAYS = 120             # ชนิดที่หาเสียงอ้างอิงไม่ครบ: ลองค้นใหม่หลังจากนี้
CALL_TYPES = ("Call", "Flight call", "Alarm call")


def kind(xc_type):
    """ประเภทที่คนอัดระบุใน xeno-canto -> 'Song' / 'Call' / None (มีทั้งสองอย่าง หรืออย่างอื่น)"""
    text = str(xc_type or "").lower()
    song, call = "song" in text, "call" in text
    if song and not call:
        return "Song"
    if call and not song:
        return "Call"
    return None


def reviewed_kind(sound_type):
    """ประเภทที่ผู้ตรวจเลือก -> 'Song' / 'Call' / None สำหรับฝึก"""
    sound_type = str(sound_type or "").strip()
    if sound_type == "Song":
        return "Song"
    return "Call" if sound_type in CALL_TYPES else None


def labelled(row):
    """ผู้ตรวจเลือก Sound type ไว้และใช้ฝึก/วัดได้: คลิปที่ Approve หรือยัง Pending (เลือกจากหน้าต่าง Sound types)
    ไม่ใช้คลิปที่ Reject (ชนิดผิด) หรือ Not sure (เสียงไม่ชัด)"""
    return (str(row.get("Review status") or "Pending") in ("Pending", "Approved")
            and reviewed_kind(row.get("Sound type")) is not None)


def _write_atomic(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")       # งานอื่นอาจเขียนพร้อมกัน
    temporary.write_bytes(data)
    temporary.replace(path)


def _save_npy(path, array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32))
    _write_atomic(path, buffer.getvalue())


# ---------- เวกเตอร์ของคลิป ----------
def _sha(row):
    sha = str((row or {}).get("Generated sha256") or "")
    return "" if sha == "nan" else sha


def clip_vectors(sha):
    try:
        return np.load(CACHE / "clips" / f"{sha}.npy") if sha else None
    except (OSError, ValueError):
        return None


def save_clip_vectors(sha, vectors):
    _save_npy(CACHE / "clips" / f"{sha}.npy", vectors)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def pick_segments(encoded, windows):
    """encoded = {วินาทีเริ่มของช่วงในคลิป: เวกเตอร์}, windows = [(t0, t1, คะแนน)] ช่วงที่ BirdNET เจอ
    -> เวกเตอร์ของช่วงคะแนนสูงสุด CLIP_SEGMENTS ช่วง (คลิปที่เพิ่มเอง: ช่วงกลางคลิป)"""
    if not encoded:
        return np.zeros((0, 1), np.float32)
    starts = sorted(encoded)
    chosen = []
    if windows:
        for t0, *_rest in sorted(windows, key=lambda w: -(w[2] if len(w) > 2 else 0)):
            nearest = min(starts, key=lambda s: abs(s - float(t0)))
            if abs(nearest - float(t0)) <= 0.75 and nearest not in chosen:
                chosen.append(nearest)
            if len(chosen) == CLIP_SEGMENTS:
                break
    if not chosen:
        middle = len(starts) // 2
        first = max(0, middle - MANUAL_SEGMENTS // 2)
        chosen = starts[first:first + MANUAL_SEGMENTS]
    return np.array([_unit(encoded[s]) for s in chosen], dtype=np.float32)


def needs_vectors(row):
    return bool(_sha(row)) and clip_vectors(_sha(row)) is None


# ---------- เสียงอ้างอิงแยกประเภท ----------
def species_of(row):
    """ชนิดของคลิป (ชื่อตาม eBird; ใช้ชื่อที่ผู้ตรวจแก้ถ้ามี)"""
    import species_names
    reviewed = " ".join(str(row.get("Reviewed scientific") or "").split()[:2])
    return reviewed if reviewed and reviewed != "nan" else species_names.row_names(row)[1]


def _checked():
    try:
        data = json.loads((CACHE / "species.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def needs_references(scientific):
    when = _checked().get(scientific)
    try:
        return when is None or date.fromisoformat(when) < date.today() - timedelta(days=RECHECK_DAYS)
    except ValueError:
        return True


def _mark_checked(scientific):
    checked = _checked()
    checked[scientific] = date.today().isoformat()
    _write_atomic(CACHE / "species.json", json.dumps(checked, indent=1, sort_keys=True).encode("utf-8"))


def needs_work(row, has_xc_key):
    """คลิปนี้ยังต้องให้งาน Auto-checks ทำส่วนประเภทเสียงไหม"""
    if not _sha(row):
        return False
    return needs_vectors(row) or (has_xc_key and needs_references(species_of(row)))


def _typed_references():
    """[(XC id, ชื่อวิทยาศาสตร์ในป้าย BirdNET, 'Song'/'Call', เวกเตอร์)] จาก cache ของ second opinion"""
    import second_opinion
    import xeno_canto
    types = xeno_canto.known_types()
    found = []
    for path in (second_opinion.CACHE / "references").glob("*.npy"):
        xc_id, _, scientific = path.stem.partition("_")
        label = kind(types.get(xc_id))
        if label:
            try:
                vectors = np.load(path)
            except (OSError, ValueError):
                continue
            if len(vectors):
                found.append((xc_id, scientific.replace("_", " "), label, vectors))
    return found


def ensure_references(model, scientific, label, key, lat=None, lon=None, log=print):
    """ให้ชนิดนี้มีไฟล์ xeno-canto ที่ระบุว่าเป็น song และ call อย่างละ REFS_PER_TYPE (โหลดเฉพาะที่ขาด)
    scientific = ชื่อในป้ายของ BirdNET, label = ป้ายเต็ม 'Scientific_Common'"""
    import second_opinion
    import xeno_canto
    have = {"Song": 0, "Call": 0}
    for _xc, species, kind_, _vectors in _typed_references():
        if species == scientific:
            have[kind_] += 1
    for wanted in ("Song", "Call"):
        if have[wanted] >= REFS_PER_TYPE:
            continue
        recordings, _total = xeno_canto.search(scientific, key, sound_type=wanted.lower(), lat=lat, lon=lon,
                                               limit=20, common=str(label).partition("_")[2])
        fresh = [r for r in recordings if kind(r["type"]) == wanted and r["file"]
                 and second_opinion._seconds(r["length"]) <= second_opinion.MAX_REF_SECONDS
                 and not second_opinion._ref_cache(r["id"], scientific).is_file()]
        fresh = fresh[:REFS_PER_TYPE - have[wanted]]
        if fresh:
            log(f"  {wanted.lower()} references for {scientific}: {len(fresh)} more")
            second_opinion.embed_references(model, scientific, label, fresh, log)


# ---------- ฝึก / เดา ----------
def training_data(summaries=()):
    """(X, y = เป็น song ไหม, น้ำหนัก, กลุ่ม = ไฟล์/คลิปที่มาของแต่ละช่วง, จำนวนไฟล์ xeno-canto, จำนวนคลิปของผู้ใช้)"""
    from review_store import load_reviews
    X, y, weight, group = [], [], [], []
    references = _typed_references()
    for xc_id, _species, label, vectors in references:
        X += list(vectors)
        y += [label == "Song"] * len(vectors)
        weight += [1.0] * len(vectors)
        group += [xc_id] * len(vectors)
    yours = 0
    for summary in summaries:
        try:
            rows = load_reviews(Path(summary))
        except Exception:  # noqa: BLE001  ไฟล์เสีย/กำลังเขียน: ข้าม
            continue
        for row in rows:
            label = reviewed_kind(row.get("Sound type"))
            vectors = clip_vectors(_sha(row))
            if not labelled(row) or vectors is None or not len(vectors):
                continue
            X += list(vectors)
            y += [label == "Song"] * len(vectors)
            weight += [USER_WEIGHT] * len(vectors)
            group += [_sha(row)] * len(vectors)
            yours += 1
    if X:
        dims = {len(v) for v in X}
        if len(dims) > 1:                         # เวกเตอร์คนละรุ่นโมเดล: ใช้ขนาดที่มีมากสุด
            common = max(dims, key=lambda d: sum(len(v) == d for v in X))
            keep = [len(v) == common for v in X]
            X, y, weight, group = ([a for a, k in zip(items, keep) if k] for items in (X, y, weight, group))
    return (np.array(X, dtype=np.float32), np.array(y, dtype=bool), np.array(weight, dtype=np.float32),
            np.array(group), len(references), yours)


def _classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # ทดลอง 2026-10-20 (679 ไฟล์ 110 ชนิด): C=0.01 ถูก 79% — C=0.5 ไม่ปรับสเกล ถูกแค่ 70%
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_VALUE, max_iter=4000, class_weight="balanced"))


def _cross_check(X, y, weight, group):
    """ทดสอบกับไฟล์ที่ไม่ได้ใช้ฝึก (5 ส่วน แบ่งตามไฟล์) -> (ถูกกี่ %, เกณฑ์มั่นใจที่ถูก >= SURE_TARGET, มั่นใจกี่ %)"""
    from sklearn.model_selection import GroupKFold
    names = sorted(set(group))
    if len(names) < 10:
        return None, SURE, None
    song = np.zeros(len(X))
    for fit, test in GroupKFold(n_splits=5).split(X, y, group):
        model = _classifier().fit(X[fit], y[fit], logisticregression__sample_weight=weight[fit])
        song[test] = model.predict_proba(X[test])[:, 1]
    p = np.array([song[group == name].mean() for name in names])
    truth = np.array([y[group == name][0] for name in names])
    confidence, right = np.maximum(p, 1 - p), (p >= 0.5) == truth
    order = np.argsort(-confidence)
    running = np.cumsum(right[order]) / np.arange(1, len(order) + 1)
    reached = np.where(running >= SURE_TARGET)[0]
    sure = float(np.clip(confidence[order][reached.max()], SURE, 0.95)) if len(reached) else 0.95
    return float(right.mean()), sure, float(np.mean(confidence >= sure))


def train(summaries=(), log=print):
    """ฝึกตัวแยกใหม่แล้วเก็บ model.npz -> dict ข้อมูลโมเดล หรือ None (ตัวอย่างไม่พอ)"""
    X, y, weight, group, references, yours = training_data(summaries)
    if len(X) < MIN_TRAINING or y.all() or not y.any():
        log(f"sound type: not enough labelled examples yet ({len(X)} segments)")
        return None
    tested, sure, covered = _cross_check(X, y, weight, group)
    model = _classifier().fit(X, y, logisticregression__sample_weight=weight)
    scaler, regression = model[0], model[-1]
    buffer = io.BytesIO()
    np.savez(buffer, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=regression.coef_[0].astype(np.float32), intercept=np.float32(regression.intercept_[0]),
             sure=np.float32(sure), tested=np.float32(np.nan if tested is None else tested),
             covered=np.float32(np.nan if covered is None else covered), references=references, yours=yours)
    _write_atomic(CACHE / "model.npz", buffer.getvalue())
    log(f"sound type: trained on {references} xeno-canto recordings + {yours} of your clips"
        + (f" — tested {tested:.0%} right; sure (>= {sure:.0%}) on {covered:.0%}" if tested is not None else ""))
    return {"references": references, "yours": yours, "tested": tested, "sure": sure, "covered": covered}


_loaded = {"stamp": None, "model": None}


def load_model():
    path = CACHE / "model.npz"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    if _loaded["stamp"] != stamp:
        try:
            with np.load(path) as data:
                model = {key: data[key] for key in ("mean", "scale", "coef")}
                model.update({key: float(data[key]) for key in ("intercept", "sure", "tested", "covered")})
                model.update({key: int(data[key]) for key in ("references", "yours")})
                _loaded["model"] = model
        except (OSError, ValueError, KeyError):   # model.npz รุ่นเก่า/เสีย: รอฝึกใหม่
            _loaded["model"] = None
        _loaded["stamp"] = stamp
    return _loaded["model"]


def guess(row):
    """คำเดาของคลิปนี้ -> {'type': 'Song'/'Call', 'p': ความน่าจะเป็นของประเภทนั้น, 'sure': bool} หรือ None"""
    model, vectors = load_model(), clip_vectors(_sha(row))
    if model is None or vectors is None or not len(vectors) or vectors.shape[1] != len(model["coef"]):
        return None
    logit = (vectors - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]
    song = float(np.mean(1 / (1 + np.exp(-np.clip(logit, -60, 60)))))
    label, p = ("Song", song) if song >= 0.5 else ("Call", 1 - song)
    return {"type": label, "p": round(p, 2), "sure": p >= model["sure"]}


def describe(found):
    """ข้อความสั้นข้างช่อง Sound type"""
    if not found:
        return ""
    if found["sure"]:
        return f"guess: {found['type'].lower()} (sure)"        # ทดสอบแล้วถูก ~9 ใน 10
    return f"guess: {found['type'].lower()}? (not sure)"


def guess_text(found):
    """ค่าที่เก็บในคอลัมน์ Sound type guess ตอนบันทึกผลตรวจ เช่น 'Song 0.86 sure' / 'Call 0.58'"""
    return f"{found['type']} {found['p']:.2f}" + (" sure" if found["sure"] else "") if found else ""


def parse_guess_text(text):
    """-> (ประเภท, ความน่าจะเป็น, มั่นใจไหม) หรือ None"""
    parts = str(text or "").split()
    try:
        if len(parts) in (2, 3) and parts[0] in ("Song", "Call"):
            return parts[0], float(parts[1]), parts[2:] == ["sure"]
    except ValueError:
        pass
    return None


def accuracy(records):
    """คำเดาตรงกับที่ผู้ตรวจเลือกกี่คลิป: {'sure': [ตรง, ทั้งหมด], 'unsure': [ตรง, ทั้งหมด]}"""
    result = {"sure": [0, 0], "unsure": [0, 0]}
    for row in records:
        guessed, chosen = parse_guess_text(row.get("Sound type guess")), reviewed_kind(row.get("Sound type"))
        if guessed and chosen and labelled(row):
            bucket = result["sure" if guessed[2] else "unsure"]
            bucket[0] += guessed[0] == chosen
            bucket[1] += 1
    return result


def unreviewed_guesses(records):
    """คำเดาของคลิปที่ยังไม่ได้ตรวจ (และยังไม่ได้เลือก Sound type)
    -> (Counter ของ 'call (sure)'/'song (sure)'/'not sure'/'no guess', [แถวที่ไม่มั่นใจ ก้ำกึ่งที่สุดก่อน])"""
    from collections import Counter
    counts, unsure = Counter(), []
    for row in records:
        if str(row.get("Review status") or "Pending") != "Pending" or str(row.get("Sound type") or "").strip():
            continue
        found = guess(row)
        if found is None:
            counts["no guess"] += 1
        elif found["sure"]:
            counts[f"{found['type'].lower()} (sure)"] += 1
        else:
            counts["not sure"] += 1
            unsure.append((found["p"], row))
    unsure.sort(key=lambda pair: pair[0])
    return counts, [row for _p, row in unsure]


def mismatches(records):
    """คำเดาที่ต่างจากที่ผู้ตรวจเลือก -> Counter {(เดา, เลือก): จำนวน}"""
    from collections import Counter
    found = Counter()
    for row in records:
        guessed, chosen = parse_guess_text(row.get("Sound type guess")), reviewed_kind(row.get("Sound type"))
        if guessed and chosen and labelled(row) and guessed[0] != chosen:
            found[(guessed[0], chosen)] += 1
    return found


def describe_accuracy(records):
    from collections import Counter
    model = load_model()
    if model is None:
        return "No sound type guesses yet — they come with the Auto-checks job (needs your xeno-canto key)."
    text = (f"Trained on {model['references']} xeno-canto recordings labelled by their recordists"
            f" + {model['yours']} of your clips with a sound type.")
    if model["tested"] == model["tested"]:          # ไม่ใช่ NaN
        text += (f" Tested on recordings it did not learn from: {model['tested']:.0%} right; "
                 f"guesses at {model['sure']:.0%}+ count as sure ({model['covered']:.0%} of recordings, ~90% right).")
    result = accuracy(records)
    for name, label in (("sure", "Sure guesses"), ("unsure", "Unsure guesses")):
        right, total = result[name]
        if total:
            text += f"  {label}: matched your choice on {right} of {total} ({right / total:.0%})."
    if not any(total for _right, total in result.values()):
        text += "  Not compared with your choices yet (choose a sound type when you approve clips)."
    differed = mismatches(records)
    if differed:
        text += "  Differed: " + ", ".join(f"guessed {g.lower()} but you chose {c.lower()} ×{n}"
                                          for (g, c), n in differed.most_common()) + "."
    chosen = Counter(str(row.get("Sound type")) for row in records
                     if labelled(row))
    if chosen:
        text += "  Your choices: " + ", ".join(f"{n} {kind_.lower()}" for kind_, n in chosen.most_common()) + "."
    return text
