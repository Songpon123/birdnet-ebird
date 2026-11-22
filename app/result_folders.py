"""ที่เก็บผลวิเคราะห์: <โฟลเดอร์ผลลัพธ์>/<วันที่อัด YYYY.MM.DD>/<เวลาเริ่มอัด HHMM><โมเดล>/

ไฟล์ที่อัดวันเดียวกัน (แต่ละไฟล์ = แต่ละเวลาเริ่ม) จึงอยู่รวมกันในโฟลเดอร์วันนั้น
คลิปที่ Approve แล้วอยู่ที่ <วัน>/Ready/<checklist>/<ชนิด>/ ข้างโฟลเดอร์เวลา (ชื่อไฟล์มีวัน_เวลาของผลชุด จึงไม่ชนกัน)
วันที่ไม่มี Log อยู่ที่ <วัน>/Ready/<ชนิด>/ — ชื่อโฟลเดอร์ checklist มาจาก field_log.ready_folder ย้ายคลิปเดิมด้วย ready_folders.py
รุ่นก่อนเก็บเป็น <โฟลเดอร์ผลลัพธ์>/<YYYY.MM.DD_HHMM><โมเดล>/ — ยังอ่านได้ และ organize() ย้ายเข้าโฟลเดอร์วันให้
"""

import hashlib
import json
import os
import re
from pathlib import Path

DAY = re.compile(r"\d{4}\.\d{2}\.\d{2}")
FLAT = re.compile(r"(\d{4}\.\d{2}\.\d{2})_(\d{4}.*)")   # แบบเก่า: วัน_เวลา<โมเดล>


def folder_for(root, when, suffix="", seconds=False):
    """โฟลเดอร์ของผลชุดที่เริ่มอัดเวลา when — มีผลแบบเก่าของเวลานี้อยู่แล้วใช้ที่เดิม (รันซ้ำไม่ต้องวิเคราะห์ใหม่)"""
    root = Path(root)
    clock = when.strftime("%H%M%S" if seconds else "%H%M")
    new = root / when.strftime("%Y.%m.%d") / f"{clock}{suffix}"
    old = root / f"{when:%Y.%m.%d}_{clock}{suffix}"
    return old if old.is_dir() and not new.exists() else new


def tag(folder):
    """ชื่อเต็มของผลชุดหนึ่ง YYYY.MM.DD_HHMM<โมเดล> ไม่ว่าเก็บแบบใหม่หรือแบบเก่า"""
    folder = Path(folder)
    if DAY.fullmatch(folder.parent.name) and not FLAT.fullmatch(folder.name):
        return f"{folder.parent.name}_{folder.name}"
    return folder.name


def root_of(summary):
    """โฟลเดอร์ผลลัพธ์ที่ summary นี้อยู่ (แบบใหม่ลึกกว่าแบบเก่า 1 ชั้น)"""
    folder = Path(summary).parent
    return folder.parent.parent if tag(folder) != folder.name else folder.parent


def ready_base(folder):
    """โฟลเดอร์ที่มี Ready/ ของผลชุดนี้ — คอลัมน์ Ready file เก็บ 'Ready/<checklist>/<ชนิด>/<ไฟล์>' เทียบกับโฟลเดอร์นี้
    แบบใหม่ = โฟลเดอร์วัน, แบบเก่า (ยังไม่ได้ย้าย) = โฟลเดอร์ผลชุดนั้นเอง"""
    folder = Path(folder)
    return folder.parent if tag(folder) != folder.name else folder


def ready_file(folder, relative):
    """ไฟล์ใน Ready ของแถวหนึ่ง — ยังอยู่ใน Ready เดิมของโฟลเดอร์เวลา (ย้ายไม่ได้เพราะชื่อชน) ใช้ที่เดิม"""
    path = ready_base(folder) / relative
    old = Path(folder) / relative
    return old if not path.exists() and old.exists() else path


def summaries(root):
    """summary.xlsx ทุกชุดในโฟลเดอร์ผลลัพธ์ (ในโฟลเดอร์วัน และแบบเก่าที่ยังไม่ได้ย้าย)"""
    root = Path(root).expanduser()
    found = set(root.glob("*/summary.xlsx"))
    found.update(path for path in root.glob("*/*/summary.xlsx") if DAY.fullmatch(path.parent.parent.name))
    return sorted(found)


def _digest(summary):
    return hashlib.sha256(str(summary).encode("utf-8")).hexdigest()[:12]


def _move_caches(old_summary, new_summary):
    """แคชที่ตั้งชื่อตาม path ของ summary (near misses, expected species, sound groups) ย้ายตามไปด้วย"""
    import paths
    old, new = _digest(old_summary), _digest(new_summary)
    for path in (paths.CACHE / "sound-index").glob(f"*-{old}*.json"):
        target = path.with_name(path.name.replace(old, new, 1))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("summary") == str(old_summary):
                data["summary"] = str(new_summary)
            target.write_text(json.dumps(data, indent=1), encoding="utf-8")
            path.unlink()
        except (OSError, ValueError):
            continue


def _move_last_summary(moves):
    import app_settings
    last = str(app_settings.load_settings().get("last_summary") or "")
    for old, new in moves:
        if last and Path(last).parent == old.resolve():
            app_settings.save_settings({"last_summary": str(new.resolve() / Path(last).name)})
            return


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gather_ready(session, log):
    """Ready/ ในโฟลเดอร์เวลา (รุ่นก่อน) -> <วัน>/Ready/ — ค่า 'Ready/<ชนิด>/<ไฟล์>' ใน summary ใช้ต่อได้เลย"""
    source = session / "Ready"
    if session.is_symlink() or source.is_symlink() or not source.is_dir():
        return 0
    moved = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            path.unlink()
            continue
        target = session.parent / "Ready" / path.relative_to(source)
        if target.exists():
            if _sha256(target) == _sha256(path):
                path.unlink()
            else:
                log(f"Not moved (a different file has the same name in the day's Ready): {path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(path, target)
        moved += 1
    for folder in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True) + [source]:
        try:
            folder.rmdir()
        except OSError:
            pass
    return moved


def organize(root, log=print):
    """ย้ายผลแบบเก่า <วัน_เวลา…>/ เข้า <วัน>/<เวลา…>/ และ Ready/ ของแต่ละเวลาไป <วัน>/Ready/ -> [(เดิม, ใหม่)]"""
    root = Path(root).expanduser()
    moves = []
    if not root.is_dir():
        return moves
    for old in sorted(root.iterdir()):
        match = FLAT.fullmatch(old.name)
        if not match or old.is_symlink() or not old.is_dir():
            continue
        new = root / match[1] / match[2]
        if new.exists():
            log(f"Not moved (already in the day folder): {old.name} -> {match[1]}/{match[2]}")
            continue
        new.parent.mkdir(exist_ok=True)
        old = old.resolve()
        os.rename(old, new)
        moves.append((old, new))
        _move_caches(old / "summary.xlsx", new.resolve() / "summary.xlsx")
    if moves:
        _move_last_summary(moves)
        log(f"Grouped results by recording day: moved {len(moves)} folder(s) into day folders in {root}")
    gathered = 0
    for day in sorted(root.iterdir()):
        if DAY.fullmatch(day.name) and day.is_dir() and not day.is_symlink():
            gathered += sum(_gather_ready(session, log) for session in sorted(day.iterdir())
                            if session.is_dir() and session.name != "Ready")
    if gathered:
        log(f"Approved clips: moved {gathered} file(s) into each day's Ready folder (one folder per species)")
    return moves
