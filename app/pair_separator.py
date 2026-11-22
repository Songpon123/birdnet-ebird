"""แยกชนิดที่เสียงคล้ายกันด้วยเสียงอ้างอิงที่โหลดไว้ในตัวเทียบเสียง (เช่น Chestnut-flanked กับ Swinhoe's White-eye)

ฝึก logistic regression จากเวกเตอร์ BirdNET 3.0 ของช่วง 3 วินาทีรอบ "เสียงหลัก" แต่ละเสียงในไฟล์อ้างอิง
(ไม่เกิน CALLS_PER_FILE ต่อไฟล์) ทดสอบแบบเว้นทีละไฟล์ (ทายไฟล์ที่ไม่ได้ใช้ฝึก) แล้วทายเสียงของคุณ
(ช่วง 3 วินาทีรอบกรอบที่เลือกในตัวเทียบ)

ทดลอง 2026-10-25 ด้วยไฟล์ที่โหลดไว้แล้ว:
  Pale-legged / Sakhalin Leaf Warbler 17 ไฟล์ 95 เสียง: ทายไฟล์ที่ไม่ได้ฝึกถูก 88% — แต่เสียง "ติ๊ก" ไกล ๆ ในคลิป 18 ต.ค.
    ได้ 49% (ตัดสินไม่ได้: เสียง 50 ms อยู่ในช่วง 3 วินาทีที่เป็นเสียงพื้นหลังเกือบทั้งหมด — ใช้การเทียบความถี่แทน)
  Chestnut-flanked / Swinhoe's White-eye 12 ไฟล์ 64 เสียง: ถูก 100% — คลิป 25 ต.ค. 09:46 เอียงไป Chestnut-flanked 69%
เสียงอ้างอิงส่วนใหญ่อัดใกล้ เสียงไกล/มีเสียงรบกวนจึงมักได้ค่ากลาง ๆ — เป็นตัวช่วยคัดก่อน ไม่ใช่การระบุชนิด

CLI (ตัวเทียบเรียกเป็นงานเบื้องหลัง): field_audio_to_ebird.py --separate SPEC.json -> บรรทัดสุดท้าย "SEPARATE {json}"
SPEC = {"species": [{"common", "scientific", "files": [{"name", "path", "centers": [วินาที]}]}],
        "yours": {"path": wav, "centers": [วินาที]}}
"""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

import paths

CACHE = paths.CACHE / "frequency-refs" / "vectors"
WINDOW = 3.0                   # ช่วงที่ BirdNET 3.0 ฟังต่อครั้ง
CALLS_PER_FILE = 6             # ไฟล์ที่ร้องมากไม่ให้น้ำหนักเกินไฟล์อื่น
YOUR_WINDOWS = 12
MIN_FILES = 5                  # ไฟล์อ้างอิงต่อชนิดอย่างน้อยเท่านี้ถึงจะฝึก (เท่ากับขั้นต่ำของตัวเทียบ)
C_VALUE = 0.1
LEAN = 0.65                    # ค่าเฉลี่ยของชนิดหนึ่งถึงเท่านี้ = "เอียงไป" ชนิดนั้น
GOOD_TEST = 0.8                # ทายไฟล์ที่ไม่ได้ฝึกถูกน้อยกว่านี้ = เสียงอ้างอิงแยกคู่นี้ไม่ได้ดี


def windows(path, centers, limit):
    """ช่วง 3 วินาทีรอบแต่ละจุด -> [(float32 mono, sr)] (ต้นไฟล์/ท้ายไฟล์เติมความเงียบ)"""
    from audio_io import read_mono
    found = []
    for center in list(centers)[:limit]:
        start = max(0.0, float(center) - WINDOW / 2)
        samples, sr = read_mono(path, start, start + WINDOW)
        if len(samples) >= sr:                                    # อย่างน้อย 1 วินาที
            found.append((np.pad(samples, (0, max(0, int(WINDOW * sr) - len(samples)))).astype(np.float32), sr))
    return found


def embed(model, arrays):
    """เวกเตอร์ BirdNET 3.0 (ความยาว 1) ของแต่ละช่วง -> (n, dim)"""
    if not arrays:
        return np.zeros((0, 1), np.float32)
    vectors = {}
    for r in model.encode_arrays(arrays, n_workers=1, show_stats=None).to_structured_array():
        vectors.setdefault(int(r["input"]), np.asarray(r["embedding"], np.float32))
    found = np.array([vectors[i] for i in sorted(vectors)], np.float32)
    return found / np.maximum(np.linalg.norm(found, axis=1, keepdims=True), 1e-9)


def _cache_file(path, centers):
    stat = Path(path).stat()
    key = f"{Path(path).name}|{stat.st_size}|{int(stat.st_mtime)}|{[round(float(c), 2) for c in centers]}"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.npy"


def reference_vectors(model, path, centers):
    """เวกเตอร์ของเสียงหลักในไฟล์อ้างอิง (เก็บ cache ต่อไฟล์ — ฝึกรอบหน้าไม่ต้องให้ BirdNET ฟังซ้ำ)"""
    centers = list(centers)[:CALLS_PER_FILE]
    target = _cache_file(path, centers)
    try:
        return np.load(target)
    except (OSError, ValueError):
        pass
    vectors = embed(model, windows(path, centers, CALLS_PER_FILE))
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.save(buffer, vectors)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(target)
    return vectors


def _classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_VALUE, max_iter=2000, class_weight="balanced"))


def test_each_file(X, y, groups):
    """ฝึกโดยเว้นทีละไฟล์แล้วทายไฟล์นั้น (ค่าเฉลี่ยของทุกเสียงในไฟล์) -> ผลรวมและผลต่อชนิด"""
    classes = sorted(set(y.tolist()))
    files = calls = files_right = calls_right = 0
    per_species = {label: [0, 0] for label in classes}
    for group in sorted(set(groups.tolist())):
        test = groups == group
        if len(set(y[~test].tolist())) < len(classes):            # ชนิดนั้นเหลือไฟล์เดียว: ทดสอบไม่ได้
            continue
        model = _classifier().fit(X[~test], y[~test])
        probability = model.predict_proba(X[test])
        truth = int(y[test][0])
        column = list(model.classes_).index(truth)
        right = int(np.argmax(probability.mean(axis=0)) == column)
        files, files_right = files + 1, files_right + right
        calls, calls_right = calls + int(test.sum()), calls_right + int((probability.argmax(axis=1) == column).sum())
        per_species[truth][0] += right
        per_species[truth][1] += 1
    return {"files_right": files_right, "files": files, "calls_right": calls_right, "calls": calls,
            "per_species": [per_species[label] for label in classes]}


def run(spec, log=print):
    species = spec["species"]
    short = [f"{s['common']} {len(s['files'])}" for s in species if len(s["files"]) < MIN_FILES]
    if len(species) < 2:
        return {"error": "add at least two species to train a separator"}
    if short:
        return {"error": f"needs at least {MIN_FILES} reference recordings per species ({', '.join(short)})"}
    import birdnet
    log("Loading BirdNET 3.0 …")
    model = birdnet.load("acoustic", "3.0", "onnx", precision="fp16")
    X, y, groups, counts = [], [], [], []
    total = sum(len(s["files"]) for s in species)
    done = 0
    for label, item in enumerate(species):
        used = 0
        for reference in item["files"]:
            done += 1
            log(f"Reference sounds {done}/{total}: {item['common']} {reference['name']}")
            try:
                vectors = reference_vectors(model, reference["path"], reference["centers"])
            except Exception as exc:  # noqa: BLE001  ไฟล์เสีย: ข้ามไฟล์นั้น
                log(f"  skipped {reference['name']}: {exc}")
                continue
            if len(vectors):
                used += 1
            X += list(vectors)
            y += [label] * len(vectors)
            groups += [f"{label}:{reference['name']}"] * len(vectors)
        counts.append(used)
    X, y, groups = np.array(X, np.float32), np.array(y), np.array(groups)
    if min(counts) < MIN_FILES:
        return {"error": f"BirdNET could read only {min(counts)} recording(s) of one species — needs {MIN_FILES}"}
    log(f"Testing on recordings left out of training ({len(X)} calls) …")
    result = {"species": [s["common"] for s in species], "files": counts,
              "calls": [int((y == label).sum()) for label in range(len(species))],
              "test": test_each_file(X, y, groups)}
    yours = spec.get("yours") or {}
    if yours.get("path") and yours.get("centers"):
        mine = embed(model, windows(yours["path"], yours["centers"], YOUR_WINDOWS))
        if len(mine):
            probability = _classifier().fit(X, y).predict_proba(mine)
            result["yours"] = {"windows": len(mine), "mean": probability.mean(axis=0).round(3).tolist(),
                               "votes": np.bincount(probability.argmax(axis=1), minlength=len(species)).tolist()}
    return result


def describe(result):
    """ผลการฝึก -> ข้อความในตัวเทียบ"""
    if "error" in result:
        return f"Separator not trained: {result['error']}"
    test, names = result["test"], result["species"]
    rate = test["files_right"] / test["files"] if test["files"] else 0.0
    text = (f"Trained on your references (BirdNET 3.0 sound vectors, {sum(result['calls'])} calls from "
            f"{sum(result['files'])} recordings) — on recordings it did not learn from: {test['files_right']} of "
            f"{test['files']} right ({rate:.0%}).")
    yours = result.get("yours")
    if yours:
        best = int(np.argmax(yours["mean"]))
        text += (f" Your sound ({yours['windows']} windows): "
                 + " · ".join(f"{name} {p:.0%}" for name, p in zip(names, yours["mean"]))
                 + (f" — leans {names[best]} ({yours['votes'][best]} of {yours['windows']} windows)."
                    if yours["mean"][best] >= LEAN else " — undecided."))
    if rate < GOOD_TEST:
        text += " ⚠ the references do not separate these species well — do not rely on this."
    return text + (" A hint only: references are mostly close recordings, so distant or noisy calls score nearer "
                   "the middle.")


def main(argv):
    if len(argv) != 1:
        print("usage: --separate SPEC.json", file=sys.stderr)
        return 2
    result = run(json.loads(Path(argv[0]).read_text(encoding="utf-8")), log=lambda text: print(text, flush=True))
    print("SEPARATE " + json.dumps(result), flush=True)
    return 0
