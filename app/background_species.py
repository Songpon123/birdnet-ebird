"""เสียงพื้นหลัง (Background species): ชนิดอื่นที่ได้ยินในไฟล์คลิปเดียวกัน สำหรับช่อง background species
ตอนอัปโหลดขึ้น eBird/Macaulay — แอปเติมให้จากชนิดอื่นที่ BirdNET เจอในไฟล์ต้นฉบับเดียวกัน ระหว่างช่วงเวลาของคลิปนี้
(และคลิปที่ผู้ตรวจเพิ่มเอง) ผู้ตรวจฟังแล้วแก้/ลบเอง"""

import json
import re

import species_names

SEPARATOR = "; "
MIN_OVERLAP_S = 1.0        # ทับกันน้อยกว่านี้ = แค่ขอบหน้าต่าง 3 วิของ BirdNET แตะคลิป ไม่น่าได้ยินในคลิปนี้
# ผู้ตรวจบอกแล้วว่าไม่ใช่ชนิดนี้ / ฟังไม่ออก: ไม่ใช้เป็นหลักฐาน
_NOT_EVIDENCE = ("Rejected", "Unsure")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _boxes(row):
    try:
        boxes = json.loads(row.get("Detection boxes") or "{}")
    except (TypeError, ValueError):
        return {}
    return boxes if isinstance(boxes, dict) else {}


def clip_span(row):
    """ช่วงของไฟล์คลิปในไฟล์ต้นฉบับ (วินาที) -> (เริ่ม, จบ) หรือ None
    คลิปของ BirdNET เริ่มก่อนเสียงแรก (lead) — หน้าต่างแรกใน Detection boxes บอกว่าก่อนกี่วินาที"""
    start = _number(row.get("Start offset (s)"))
    if start is None:
        return None
    boxes = _boxes(row)
    windows = boxes.get("windows") or []
    if boxes.get("manual"):                    # คลิปที่เพิ่มเอง: ไฟล์เริ่มตรง Start offset
        begin = start
    elif windows:
        begin = max(0.0, start - float(windows[0][0]))
    else:                                      # summary รุ่นเก่า: ไม่มีกรอบ ใช้ lead ที่ตั้งไว้ตอนวิเคราะห์
        try:
            lead = float(json.loads(row.get("Analysis settings") or "{}").get("lead") or 0)
        except (TypeError, ValueError, AttributeError):
            lead = 0.0
        begin = max(0.0, start - lead)
    duration = _number(row.get("Duration (s)"))
    end = begin + duration if duration is not None else _number(row.get("End offset (s)"))
    return begin, max(begin, end if end is not None else start)


def _heard(row):
    """ช่วงที่ได้ยินชนิดของแถวนี้ในเวลาของไฟล์ต้นฉบับ -> [(เริ่ม, จบ, คะแนน BirdNET หรือ None)]"""
    boxes, span = _boxes(row), clip_span(row)
    windows = boxes.get("windows") or []
    if span is not None and windows and not boxes.get("manual"):
        return [(span[0] + float(w[0]), span[0] + float(w[1]), _number(w[2]) if len(w) > 2 else None)
                for w in windows]
    start, end = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
    if start is None:
        return []
    score = None if boxes.get("manual") else _number(row.get("Max confidence"))
    return [(start, end if end is not None else start, score)]


def _source(row):
    return str(row.get("Source path") or row.get("Source file") or "").strip()


def _names(row):
    """ชื่อตาม eBird ของชนิดในแถวนี้ (ชื่อที่ผู้ตรวจแก้แล้วก่อน) -> dict หรือ None (ไม่ใช่นก, _Unknown)"""
    if str(row.get("Review status") or "") == "Approved" and row.get("Reviewed scientific"):
        return species_names.to_ebird(row["Reviewed scientific"], row.get("Reviewed common") or "")
    common, scientific = species_names.row_names(row)
    return species_names.to_ebird(scientific, common)


def suggest(row, rows, main_scientific=""):
    """ชนิดอื่นที่ได้ยินระหว่างคลิปนี้ -> [(common, scientific, หมายเหตุ)] ผู้ตรวจยืนยันแล้วก่อน แล้วคะแนน BirdNET สูงก่อน
    main_scientific = ชนิดหลักของคลิปตอนนี้ (อาจแก้ชื่อแล้ว) ไม่ใส่ซ้ำเป็นเสียงพื้นหลัง"""
    span, source = clip_span(row), _source(row)
    if span is None or not source:
        return []
    main = species_names.to_ebird(main_scientific) if main_scientific else None
    skip = {main["scientific"] if main else main_scientific, *(n["scientific"] for n in [_names(row)] if n)}
    found = {}
    for other in rows:
        if other is row or other.get("File") == row.get("File") or _source(other) != source:
            continue
        status = str(other.get("Review status") or "Pending")
        names = _names(other)
        if status in _NOT_EVIDENCE or names is None or names["scientific"] in skip:
            continue
        overlap, best = 0.0, None
        for start, end, score in _heard(other):
            shared = min(end, span[1]) - max(start, span[0])
            if shared >= min(MIN_OVERLAP_S, max(end - start, 0.0)) and shared > 0:
                overlap += shared
                if score is not None:
                    best = score if best is None else max(best, score)
        if overlap <= 0:
            continue
        confirmed = status == "Approved" or _boxes(other).get("manual")
        entry = found.setdefault(names["scientific"], {"names": names, "seconds": 0.0, "score": None,
                                                       "confirmed": False})
        entry["seconds"] += overlap
        entry["confirmed"] = entry["confirmed"] or bool(confirmed)
        if best is not None:
            entry["score"] = best if entry["score"] is None else max(entry["score"], best)
    ordered = sorted(found.values(), key=lambda e: (not e["confirmed"], -(e["score"] or 0.0), -e["seconds"]))
    result = []
    for entry in ordered:
        note = [f"{round(entry['seconds'])} s in this clip"]
        if entry["confirmed"]:
            note.insert(0, "confirmed by you")
        if entry["score"] is not None:
            note.insert(0, f"BirdNET {entry['score']:.2f}")
        result.append((entry["names"]["common"], entry["names"]["scientific"], ", ".join(note)))
    return result


def split(text):
    """ข้อความในช่อง -> [ชื่อ] (คั่นด้วย ; หรือ , ตัดซ้ำ)"""
    names, seen = [], set()
    for part in re.split(r"[;,]", str(text or "")):
        name = " ".join(part.split())
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    return names


def join(names):
    return SEPARATOR.join(names)
