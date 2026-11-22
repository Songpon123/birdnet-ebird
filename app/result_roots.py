"""ทุกโฟลเดอร์ผลลัพธ์ที่มีผลของคุณอยู่ — ผลเก็บแยกตามวันที่ <Birding>/<ปี>/<เดือน>/<YYYYMMDD>/Sounds/Done ได้
Review progress, เมนู Results, Recorder flags และตัวฝึกที่เรียนจากผลตรวจจึงอ่านจากทุกที่ ไม่ใช่แค่ Output folder ปัจจุบัน

ที่มา: Output folder ตอนนี้ + ที่จำไว้ใน settings (output_folders) + ค่าเริ่มต้น + โฟลเดอร์ของผลชุดล่าสุด
+ Sounds/Done ทุกวันใต้ <Birding> ที่เคยเพิ่ม Field log (หาเอง ไม่ต้องจำทีละวัน)
"""

import re
from pathlib import Path

import app_settings
import field_log
import result_folders
from review_store import list_results

DEFAULT = Path.home() / "BirdNET_eBird"
SETTING = "output_folders"
DAY_FOLDER = re.compile(r"\d{8}")


def done_folder(path):
    """ไฟล์/โฟลเดอร์ใน <YYYYMMDD>/Sounds/ -> <YYYYMMDD>/Sounds/Done (None ถ้าไม่ใช่โครงสร้างนี้)"""
    path = Path(path).expanduser()
    for folder in (path, *path.parents):
        if folder.name == "Sounds" and DAY_FOLDER.fullmatch(folder.parent.name):
            return folder / "Done"
    return None


def remembered(root):
    """รายการที่ต้องเก็บใน settings[SETTING] เมื่อจำโฟลเดอร์ผลลัพธ์นี้เพิ่ม (ไม่ซ้ำ)"""
    root = str(Path(root).expanduser().resolve())
    known = list(app_settings.load_settings().get(SETTING) or ())
    return known if root in known else known + [root]


def known_roots(current=None):
    """ทุกโฟลเดอร์ผลลัพธ์ที่มีอยู่จริง (current ก่อน) ไม่ซ้ำกัน"""
    settings = app_settings.load_settings()
    candidates = [current, DEFAULT, *(settings.get(SETTING) or ())]
    last = Path(settings.get("last_summary") or "")
    if last.is_file():
        candidates.append(result_folders.root_of(last))
    for birding in field_log.stored()[1]:
        candidates.extend(Path(birding).glob("*/*/*/Sounds/Done"))
    found = {}
    for candidate in candidates:
        path = Path(candidate).expanduser() if candidate else None
        if path is not None and path.is_dir():
            found.setdefault(path.resolve(), path)
    return list(found.values())


def all_results(current=None):
    """list_results ของทุกโฟลเดอร์ผลลัพธ์รวมกัน ใหม่สุดก่อน"""
    found = {}
    for root in known_roots(current):
        for info in list_results(root):
            found.setdefault(Path(info["path"]).resolve(), info)
    return sorted(found.values(), key=lambda info: (info["folder"][:15], info["modified"]), reverse=True)
