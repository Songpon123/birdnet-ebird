"""Flag (MARK) ที่กดระหว่างอัด: Zoom H4essential เก็บในไฟล์ WAV เป็น chunk มาตรฐาน cue + LIST/adtl ต่อท้ายเสียง
(ตำแหน่งละเอียดระดับ sample, ป้ายเป็นแค่เลขลำดับ 01 02 … ไม่มีชื่อชนิด) อ่านได้ทั้ง RIFF และ RF64 โดยไม่ต้องอ่านเสียงทั้งไฟล์

flag เป็นจุดให้กลับมาดู: ไม่เปลี่ยนคะแนน BirdNET และไม่ยืนยันชนิด คนมักกดหลังได้ยินเสียง จึงถือว่าเสียงอยู่ก่อน flag ได้
BEFORE วินาที หลัง flag ได้ AFTER วินาที — ตรวจไฟล์จริง 2026-10-31: 8 จุดใน 3 ไฟล์ จาก 27 ไฟล์
ตำแหน่งเก็บแบบเวลาในไฟล์ จึงใช้ได้กับผลเก่าโดยไม่ต้องวิเคราะห์ใหม่ และจำไว้ใน data/cache/recorder-flags/<Source ID>.json
เผื่อย้ายไฟล์ต้นฉบับออกไป

คู่มือบอกว่า Pause แล้วอัดต่อเครื่องใส่ MARK ให้เอง จุดแบบนั้นแยกจากที่คนกดไม่ได้ (ป้ายเหมือนกัน)
"""

import json
import os
import re
import struct
from datetime import datetime, timedelta
from pathlib import Path

import paths

CACHE = paths.CACHE / "recorder-flags"
BEFORE = 30.0                  # เสียงอยู่ก่อน flag ได้เท่านี้วินาที (ได้ยินแล้วค่อยหยิบเครื่องกด)
AFTER = 10.0                   # เสียงอยู่หลัง flag ได้เท่านี้วินาที (กดรอเสียงที่กำลังจะร้อง)
PHOTO_WINDOW = 120.0           # รูปที่ถ่ายห่างจาก flag ไม่เกินเท่านี้วินาที (หลังปรับเวลากล้องแล้ว)
NAME_TIME = re.compile(r"(\d{6})_(\d{6})")     # 261030_122253_TrMic.WAV = เวลาเริ่มอัดในชื่อไฟล์ Zoom


def read_flags(path):
    """เวลาของ flag ในไฟล์ (วินาทีจากต้นไฟล์ เรียงจากน้อยไปมาก) — ไม่ใช่ WAV หรือไม่มี flag = []"""
    path = Path(path)
    if path.suffix.lower() != ".wav":
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            head = source.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return []
            rate, data_size, offsets = None, None, []
            while source.tell() + 8 <= size:
                name, length = struct.unpack("<4sI", source.read(8))
                body = source.tell()
                if name == b"ds64":
                    data_size = struct.unpack("<QQQ", source.read(24))[1]
                elif name == b"fmt ":
                    rate = struct.unpack("<HHI", source.read(8))[2]
                elif name == b"cue ":
                    for _ in range(struct.unpack("<I", source.read(4))[0]):
                        offsets.append(struct.unpack("<II4sIII", source.read(24))[5])
                if name == b"data" and length == 0xFFFFFFFF and data_size is not None:
                    length = data_size
                source.seek(body + length + (length & 1))
    except (OSError, struct.error):
        return []
    return sorted(round(offset / rate, 3) for offset in offsets) if rate else []


def _cache_file(sid):
    return CACHE / f"{sid}.json"


def flags_of(sid, path):
    """flag ของไฟล์ต้นฉบับ: อ่านจากไฟล์ถ้ายังอยู่ (จำไว้ตาม Source ID) ไม่งั้นใช้ค่าที่จำไว้ — ยังไม่เคยอ่าน = []"""
    target = _cache_file(sid) if sid else None
    if path and Path(path).is_file():
        flags = read_flags(path)
        if target is not None:
            CACHE.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"flags": flags, "source": str(path)}), encoding="utf-8")
            temporary.replace(target)
        return flags
    try:
        return json.loads(target.read_text(encoding="utf-8"))["flags"] if target is not None else []
    except (OSError, ValueError, KeyError):
        return []


def distance(flags, start, end):
    """คลิปช่วง start–end (วินาทีในไฟล์ต้นฉบับ) อยู่ใกล้ flag ไหม -> (flag, ระยะวินาที ลบ = คลิปอยู่ก่อน flag) ที่ใกล้สุด หรือ None"""
    found = None
    for flag in flags:
        if start - AFTER <= flag <= end + BEFORE:
            away = 0.0 if start <= flag <= end else (start - flag if flag < start else end - flag)
            if found is None or abs(away) < abs(found[1]):
                found = (flag, round(away, 1))
    return found


def by_source(rows):
    """{Source ID: [flag]} ของทุกไฟล์ต้นฉบับในแถวเหล่านี้"""
    found = {}
    for row in rows:
        sid = str(row.get("Source ID") or "")
        if sid and sid not in found:
            found[sid] = flags_of(sid, row.get("Source path"))
    return found


def recording_start(path):
    """เวลาเริ่มอัดจากชื่อไฟล์ของ Zoom (YYMMDD_HHMMSS…) -> datetime หรือ None"""
    match = NAME_TIME.search(Path(path).name)
    try:
        return datetime.strptime(match[1] + match[2], "%y%m%d%H%M%S") if match else None
    except ValueError:
        return None


def photos_near(when, photos, camera_offset=0.0, window=PHOTO_WINDOW):
    """รูปที่ถ่ายใกล้เวลา when -> [(ระยะวินาที บวก = ถ่ายหลัง flag, รูป)] เรียงจากใกล้สุด
    camera_offset = นาฬิกากล้องเร็วกว่าเครื่องอัดกี่วินาที (ช้ากว่า = ลบ)"""
    found = []
    for photo in photos:
        away = (datetime.fromisoformat(photo["time"]) - timedelta(seconds=camera_offset) - when).total_seconds()
        if abs(away) <= window:
            found.append((round(away, 1), photo))
    return sorted(found, key=lambda item: (abs(item[0]), not item[1]["common"]))

