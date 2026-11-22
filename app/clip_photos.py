"""รูปของคุณที่ถ่ายตอนได้ยินเสียงของคลิป (แท็บ Review): บอกเป็นชื่อชนิด ไม่แสดงภาพ

เวลาถ่ายจาก EXIF (DateTimeOriginal) ของรูปใน <วัน>/Pictures: JPG/DNG/TIFF อ่านเองด้วย PIL, CR3 และ raw อื่นใช้ exiftool
ถ้ามีในเครื่อง — จำเวลาของแต่ละไฟล์ไว้ตามขนาดและเวลาแก้ไข อ่านซ้ำเฉพาะไฟล์ใหม่
ชื่อชนิดและเวลาถ่ายมาจาก Log ของโปรแกรมจัดรูปด้วย (files.csv) ภาพเดียวกันหลายไฟล์ (.jpg .dng .CR3) นับใบเดียว
รูปที่ Log ไม่ได้ระบุชนิด ใช้ชื่อโฟลเดอร์ชนิดที่แยกรูปไว้แล้ว (Pictures/Done/Black Stork/…)
รูปที่ย้ายไป Google Drive แล้ว (เช่น My Drive/Pictures/Photos/2026/20261106_Trip): ใช้เวลาจาก Log แล้วหาไฟล์ตามชื่อ
ใน Google Drive บนเครื่อง — ดูแค่รายชื่อไฟล์ เปิดเฉพาะรูปที่แสดงภาพย่อ จึงไม่ทำให้ Drive โหลดทุกรูป
เวลาในรูปเป็นนาฬิกากล้อง ปรับด้วย Camera minus recorder clock ของหน้าต่าง Recorder flags
อยู่ใกล้กันในเวลาไม่ได้แปลว่าเป็นนกตัวเดียวกัน
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import field_log
import manual_clips
import paths
import platform_tools
import recorder_flags

CACHE = paths.CACHE / "photo-times"
WINDOW = recorder_flags.PHOTO_WINDOW             # รูปที่ถ่ายก่อน/หลังเสียงไม่เกินเท่านี้วินาที
VIEWABLE = (".jpg", ".jpeg", ".dng", ".tif", ".tiff")   # PIL อ่านเวลาและทำภาพย่อได้ (เรียงตามที่อยากแสดง)
RAW = (".cr3", ".cr2", ".nef", ".arw", ".orf", ".rw2", ".heic")
EXIF_TIME = "%Y:%m:%d %H:%M:%S"
CLOUD = Path.home() / "Library" / "CloudStorage"
NOT_SPECIES = {"done", "dxo", "raw", "pictures", "log", "sounds", "edited", "export", "jpg"}
ARCHIVES = ("GoogleDrive-*/My Drive/*/*/*/{day}*", "GoogleDrive-*/My Drive/*/*/{day}*", "GoogleDrive-*/My Drive/*/{day}*")


def _pil_time(path):
    """DateTimeOriginal (หรือ DateTime) ของรูป -> 'YYYY-MM-DDTHH:MM:SS' หรือ None"""
    from PIL import Image
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            text = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        return datetime.strptime(str(text).strip()[:19], EXIF_TIME).isoformat() if text else None
    except (OSError, ValueError, SyntaxError):
        return None


def _exiftool_times(files):
    """เวลาถ่ายของไฟล์ที่ PIL อ่านไม่ได้ ผ่าน exiftool (ไม่มี = {}) -> {path: iso}"""
    tool = platform_tools.bundled_tool("exiftool")
    if not tool or not files:
        return {}
    try:
        result = subprocess.run([tool, "-json", "-q", "-DateTimeOriginal", "-d", "%Y-%m-%dT%H:%M:%S",
                                 *map(str, files)], capture_output=True, text=True, timeout=120,
                                creationflags=platform_tools.NO_WINDOW)
        rows = json.loads(result.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    return {str(Path(row["SourceFile"])): row["DateTimeOriginal"] for row in rows
            if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", str(row.get("DateTimeOriginal") or ""))}


def picture_times(folder):
    """รูปทุกรูปใน folder (รวมโฟลเดอร์ย่อย) ที่มีเวลาถ่าย -> [{"time", "file", "path"}]"""
    folder = Path(folder)
    store = CACHE / f"{hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:20]}.json"
    try:
        known = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    found, unread = {}, []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if (path.suffix.lower() not in VIEWABLE + RAW or any(part.startswith(".") for part in relative.parts)
                or not path.is_file()):
            continue
        stat = path.stat()
        signature = [stat.st_size, int(stat.st_mtime)]
        key = relative.as_posix()
        if known.get(key, [None, None])[:2] == signature:
            found[key] = known[key]
            continue
        taken = _pil_time(path) if path.suffix.lower() in VIEWABLE else None
        found[key] = signature + [taken]
        if taken is None:
            unread.append(path)
    for path, taken in _exiftool_times(unread).items():
        found[Path(path).relative_to(folder).as_posix()][2] = taken
    if found != known:
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary = store.with_name(f"{store.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(found), encoding="utf-8")
        temporary.replace(store)
    return [{"time": value[2], "file": Path(key).name, "path": str(folder / key)}
            for key, value in found.items() if value[2]]


def _shot(photo):
    """ภาพเดียวกันต่างนามสกุล/ชื่อขึ้นต้นด้วยวันที่ (20261025-4X8A1266.jpg = 4X8A1266.CR3) -> key เดียวกัน"""
    return photo["time"][:19], re.sub(r"^\d{8}-", "", Path(photo["file"]).stem).upper()


def day_folders(summary, day):
    """โฟลเดอร์ <วัน> ที่อาจมี Pictures: เหนือ Sounds ของผลชุดนี้ และ <Birding>/<ปี>/<เดือน>/<YYYYMMDD> ที่รู้จัก"""
    candidates = [parent.parent for parent in Path(summary).parents if parent.name == "Sounds"][:1]
    candidates += [field_log.day_folder(root, day) for root in field_log.stored()[1]]
    found, seen = [], set()
    for folder in candidates:
        try:
            stat = (folder / "Pictures").stat()
        except OSError:
            continue
        if (stat.st_dev, stat.st_ino) not in seen:      # NOV กับ Nov คือโฟลเดอร์เดียวกันบน Mac
            seen.add((stat.st_dev, stat.st_ino))
            found.append(folder)
    return found


def folder_species(path):
    """ชื่อชนิดจากโฟลเดอร์ที่แยกรูปไว้ (…/Done/Black Stork/x.jpg -> 'Black Stork') หรือ ''
    ไม่นับโฟลเดอร์ทั่วไป (Done, DxO, Raw…), โฟลเดอร์ที่ขึ้นต้นด้วย _ (_Unidentified) หรือตัวเลข (วันที่/รอบ)"""
    name = Path(path).parent.name
    if name.casefold() in NOT_SPECIES or not re.fullmatch(r"[A-Z][A-Za-z'’ .()-]+", name):
        return ""
    return name


def archive_files(day):
    """ชื่อไฟล์ (ตัวเล็ก) -> [path] ของรูปวันนี้ที่ย้ายไปเก็บใน Google Drive บนเครื่อง (ดูแค่รายชื่อ ไม่เปิดไฟล์)"""
    found = {}
    for pattern in ARCHIVES:
        for folder in CLOUD.glob(pattern.format(day=f"{day:%Y%m%d}")):
            for root, _folders, names in os.walk(folder):
                for name in names:
                    found.setdefault(name.lower(), []).append(os.path.join(root, name))
    return found


def photos_at(row, summary, camera_offset=0.0, window=WINDOW):
    """รูปที่ถ่ายระหว่างหรือใกล้เสียงของคลิป -> [(ระยะวินาที: 0 = ระหว่างเสียง, ลบ = ก่อน, บวก = หลัง, รูป)]
    เรียงจากใกล้สุด — รูป: {"time", "file", "path", "common"}"""
    span = manual_clips.heard_span(row)
    if span is None:
        return []
    start, end = span
    order = {suffix: rank for rank, suffix in enumerate(VIEWABLE + RAW)}
    photos = {}
    for folder in day_folders(summary, start.date()):
        for photo in sorted(picture_times(folder / "Pictures"), key=lambda p: order[Path(p["file"]).suffix.lower()]):
            known = photos.setdefault(_shot(photo), dict(photo, common=""))
            known["common"] = known["common"] or folder_species(photo["path"])
    archive = None
    for logged in field_log.photos_on(field_log.for_day(start.date())):
        known = photos.get(_shot(logged))
        if known is not None:
            known["common"] = logged["common"] or known["common"]
            continue
        moved = []
        if not (logged["path"] and Path(logged["path"]).is_file()):   # ย้ายไปแล้ว: หาตามชื่อใน Google Drive
            archive = archive_files(start.date()) if archive is None else archive
            moved = archive.get(logged["file"].lower(), [])
        path = moved[0] if moved else logged["path"] if logged["path"] and Path(logged["path"]).is_file() else ""
        common = logged["common"] or next((folder_species(p) for p in moved if folder_species(p)), "")
        photos[_shot(logged)] = {"time": logged["time"], "file": logged["file"], "path": path, "common": common}
    found = []
    for photo in photos.values():
        taken = datetime.fromisoformat(photo["time"]) - timedelta(seconds=camera_offset)
        away = 0.0 if start <= taken <= end else (taken - (start if taken < start else end)).total_seconds()
        if abs(away) <= window:
            found.append((round(away, 1), photo))
    return sorted(found, key=lambda item: (abs(item[0]), item[1]["time"]))


def describe_away(away):
    if away == 0:
        return "during the sound"
    return f"{abs(away):.0f} s {'before' if away < 0 else 'after'}"


def describe(found):
    """รูปที่ตรงกับเสียง -> บรรทัดเดียว: 'Black Stork ×3 (38 s before) · Graylag Goose (102 s after) · 2 not sorted'"""
    if not found:
        return ""
    species = {}
    for away, photo in found:                         # found เรียงจากใกล้สุดแล้ว
        name = photo.get("common") or ""
        species.setdefault(name, []).append(away)
    parts = [f"{name}{f' ×{len(aways)}' if len(aways) > 1 else ''} ({describe_away(aways[0])})"
             for name, aways in species.items() if name]
    if species.get(""):
        parts.append(f"{len(species[''])} not sorted")
    return f"📷 Your photos then (within {WINDOW / 60:g} min): " + " · ".join(parts)
