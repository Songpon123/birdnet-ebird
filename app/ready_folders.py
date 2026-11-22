"""คลิปที่ตรวจแล้วแยกตาม checklist: <วัน>/Ready/<checklist>/<ชนิด>/<ไฟล์> (ชื่อโฟลเดอร์จาก field_log.ready_folder)

Approve ใหม่เข้าที่ถูกเอง ไฟล์นี้ย้ายคลิปที่ตรวจไว้แล้วเมื่อ checklist ของไฟล์เสียงเปลี่ยนหรือเพิ่งมี Log พร้อมแก้คอลัมน์ Ready file
แอปสั่งผ่านคิวงานเท่านั้น (--sort-ready) ไม่ชนงานวิเคราะห์ที่เขียน summary.xlsx ชุดเดียวกัน
ไฟล์ที่หาย ถูกแก้นอกแอป หรือปลายทางมีไฟล์อยู่แล้ว = ไม่ย้าย และรายงานให้รู้

ไฟล์อื่นในโฟลเดอร์ชนิด (เช่นไฟล์ที่รวมจาก Merge clips) ไม่มีใน summary: ตามคลิปไปเมื่อคลิปทั้งโฟลเดอร์ไป checklist เดียวกัน
คลิปไปหลาย checklist = ไม่รู้ว่าไฟล์นั้นมาจากอันไหน จึงไม่เดา คงไว้ที่เดิมและรายงาน
"""

import os
from pathlib import Path

from openpyxl import load_workbook

import field_log
import result_folders
from manual_clips import recording_start_of
from result_roots import all_results
from review_store import _sha256, load_reviews


def day_summaries(day, output=None):
    """summary ทุกชุดของวันนี้จากทุกโฟลเดอร์ผลลัพธ์"""
    return [info["path"] for info in all_results(output) if info["folder"].startswith(f"{day:%Y.%m.%d}")]


def _followers(work, clips):
    """ไฟล์อื่นในโฟลเดอร์ชนิดที่คลิปกำลังย้ายออก -> ([(ไฟล์เดิม, ไฟล์ใหม่)], [ไฟล์ที่ย้ายไม่ได้])"""
    targets = {}
    for rows in work.values():
        for old, new, _relative in rows.values():
            targets.setdefault(old.parent, set()).add(new.parent)
    going = {old for rows in work.values() for old, _new, _relative in rows.values()}
    others, problems = [], []
    for folder, places in targets.items():
        rest = [path for path in folder.iterdir() if path.is_file() and path.name != ".DS_Store" and path not in going]
        one_place = len(places) == 1 and not any(path in clips for path in rest)
        for path in (path for path in rest if path not in clips):
            target = next(iter(places)) / path.name
            if not one_place:
                problems.append(f"Clips went to more than one checklist, so it stayed: {path}")
            elif target.exists():
                problems.append(f"Already exists: {target}")
            else:
                others.append((path, target))
    return others, problems


def plan(summaries):
    """-> ({summary: {แถว: (ไฟล์เดิม, ไฟล์ใหม่, ค่าใหม่ของ Ready file)}}, [(ไฟล์อื่นเดิม, ใหม่)], [ที่ย้ายไม่ได้])"""
    work, problems, folders, clips = {}, [], {}, set()
    for summary in map(Path, summaries):
        for row in load_reviews(summary):
            if not row.get("Ready file"):
                continue
            relative, start = Path(str(row["Ready file"])), recording_start_of(row)
            if start not in folders:
                folders[start] = field_log.ready_folder(start)
            wanted = str(Path("Ready", folders[start], relative.parent.name, relative.name))
            old = result_folders.ready_file(summary.parent, str(relative))
            clips.add(old)
            if wanted == str(relative):
                continue
            new = result_folders.ready_base(summary.parent) / wanted
            if not old.is_file():
                problems.append(f"Missing: {old}")
            elif new.exists():
                problems.append(f"Already exists: {new}")
            elif _sha256(old) != row.get("Ready sha256"):
                problems.append(f"Changed outside the app: {old}")
            else:
                work.setdefault(summary, {})[row["_row"]] = (old, new, wanted)
    others, stayed = _followers(work, clips)
    return work, others, problems + stayed


def _remove_empty(folder):
    """โฟลเดอร์ชนิด/checklist ที่ไม่เหลือไฟล์ (.DS_Store ของ Finder ไม่นับ) ไฟล์อื่นในนั้นไม่แตะ"""
    while folder.name != "Ready" and folder.is_dir() and {path.name for path in folder.iterdir()} <= {".DS_Store"}:
        (folder / ".DS_Store").unlink(missing_ok=True)
        folder.rmdir()
        folder = folder.parent


def apply(work, others=()):
    """เขียน summary ไว้ก่อน ย้ายไฟล์ แล้วค่อยแทน summary — ย้ายไม่สำเร็จ = ย้ายกลับหมด"""
    written, moved = [], []
    try:
        for summary, rows in work.items():
            book = load_workbook(summary)
            sheet = book.active
            column = next(cell.column for cell in sheet[1] if cell.value == "Ready file")
            for number, (_old, _new, relative) in rows.items():
                sheet.cell(number, column, relative)
            temporary = summary.with_name("summary.sort.tmp.xlsx")
            book.save(temporary)
            book.close()
            written.append((temporary, summary))
        for old, new in [(old, new) for rows in work.values() for old, new, _relative in rows.values()] + list(others):
            new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
            moved.append((old, new))
        for temporary, summary in written:
            os.replace(temporary, summary)
    except BaseException:
        for old, new in reversed(moved):
            os.rename(new, old)
        for temporary, _summary in written:
            temporary.unlink(missing_ok=True)
        raise
    for folder in {old.parent for old, _new in moved}:
        _remove_empty(folder)


def sort_day(day, output=None, log=print):
    work, others, problems = plan(day_summaries(day, output))
    apply(work, others)
    log(f"Ready: moved {sum(map(len, work.values()))} clip(s) and {len(others)} other file(s) of {day:%d %b %Y} "
        "into checklist folders")
    for problem in problems:
        log(f"Ready: not moved: {problem}")
