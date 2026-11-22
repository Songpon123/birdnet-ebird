"""รายการอัปโหลดของวันหนึ่ง: ทุกไฟล์ใน Ready พร้อมรายละเอียดที่ต้องกรอกใน Manage Media ของ eBird เปิดดูคู่กับหน้าเว็บ
เขียนที่ <วัน>/Ready/upload-sheet.csv ตอนกดปุ่ม Upload sheet… เท่านั้น จึงตรงกับผลตรวจล่าสุดเสมอ

เสียงพื้นหลังใส่ในช่อง Additional Species ซึ่งเลือกได้เฉพาะชนิดที่อยู่ใน checklist แล้ว ชนิดที่ยังไม่อยู่ต้องเพิ่มก่อน
(Edit Species List จำนวน X) — แอปเทียบกับ checklist ของไฟล์เสียงนั้นใน Field log ไม่มี Log = ไม่รู้ จึงเว้นว่าง
BirdNET also heard = ชนิดที่ BirdNET เจอระหว่างคลิปแต่ยังไม่ได้บันทึกเป็นเสียงพื้นหลัง: ฟังก่อน ไม่ใช่ผลที่ยืนยันแล้ว
"""

import csv
import os
from pathlib import Path

import background_species
import field_log
import manual_clips
import result_folders
from review_store import load_reviews

NAME = "upload-sheet.csv"
COLUMNS = ("Checklist", "Species", "Scientific name", "Time", "Quality", "Sound type", "Additional species",
           "Add to checklist first", "BirdNET also heard", "Notes", "File")


def summaries_of(summary):
    """ทุกผลชุดที่ใช้ Ready เดียวกับ summary นี้ (แบบเก่าที่ยังไม่ได้ย้าย = ชุดเดียว)"""
    folder = Path(summary).parent
    base = result_folders.ready_base(folder)
    return sorted(base.glob("*/summary.xlsx")) if base != folder else [Path(summary)]


def _species(name):
    """ชนิดย่อยใน checklist นับเป็นชนิดเดียวกัน: White Wagtail (Chinese) = White Wagtail"""
    return name.split(" (")[0].casefold()


def not_on_checklist(checklists, names):
    """ชื่อที่ยังไม่อยู่ใน checklist เหล่านี้"""
    if not checklists:
        return []
    listed = {_species(item["common"]) for checklist in checklists for item in checklist["species"]}
    return [name for name in names if _species(name) not in listed]


def clips(summaries):
    """ทุกคลิปที่อยู่ใน Ready -> [dict ตาม COLUMNS] เรียงตาม checklist ชนิด แล้วเวลา"""
    found, logged = [], {}
    for summary in summaries:
        rows = load_reviews(summary)
        for row in rows:
            if not row.get("Ready file"):
                continue
            common, scientific = str(row.get("Reviewed common") or ""), str(row.get("Reviewed scientific") or "")
            saved = background_species.split(row.get("Background species"))
            heard = [name for name, _scientific, _note in background_species.suggest(row, rows, scientific)
                     if name.casefold() not in {other.casefold() for other in saved}]
            start = manual_clips.recording_start_of(row)
            if start not in logged:
                logged[start] = field_log.for_recording(start)
            checklists = logged[start]
            found.append({"Checklist": checklists[0]["id"] if checklists else "", "Species": common, "Scientific name": scientific, "Time": row.get("Clock time") or "",
                          "Quality": row.get("Quality rating") or "", "Sound type": row.get("Sound type") or "",
                          "Additional species": background_species.join(saved),
                          "Add to checklist first": background_species.join(not_on_checklist(checklists, [common, *saved])),
                          "BirdNET also heard": background_species.join(heard), "Notes": row.get("Review notes") or "",
                          "File": Path(str(row["Ready file"])).name})
    return sorted(found, key=lambda clip: (clip["Checklist"], clip["Species"].casefold(), str(clip["Time"])))


def write(summary):
    """เขียนรายการของวันที่ summary นี้อยู่ -> (ไฟล์, [คลิป]) ไม่มีคลิปใน Ready = ไม่เขียน"""
    found = clips(summaries_of(summary))
    path = result_folders.ready_base(Path(summary).parent) / "Ready" / NAME
    if not found:
        return path, found
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8-sig", newline="") as sheet:   # utf-8-sig: Excel อ่านชื่อภาษาอื่นถูก
            writer = csv.DictWriter(sheet, COLUMNS)
            writer.writeheader()
            writer.writerows(found)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, found
