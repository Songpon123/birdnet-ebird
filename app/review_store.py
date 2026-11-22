"""Human review of BirdNET clips. Only approved clips are copied to Ready/."""

import hashlib
import os
import shutil
from pathlib import Path

from openpyxl import load_workbook

import background_species
import field_log
import result_folders


REVIEW_COLUMNS = ("Review status", "Reviewed common", "Reviewed scientific",
                  "Quality rating", "Review notes", "Ready file", "Ready sha256",
                  "Sound type", "Sound type guess", "Background species")
# Unsure = ฟังแล้วแต่เสียงไม่ชัดพอจะระบุชนิด: ไม่เข้า Ready และไม่นับเป็น ID ถูก/ผิด
STATUSES = ("Pending", "Approved", "Rejected", "Unsure")
# ประเภทเสียงตามแท็กของ eBird/Macaulay ("" = ยังไม่เลือก)
SOUND_TYPES = ("", "Song", "Call", "Drumming", "Flight call", "Alarm call", "Other")
WITH_NAMES = 3      # ชื่อชนิดที่ติดมาในชื่อไฟล์ Ready ไม่เกินเท่านี้ (ความยาวชื่อไฟล์มีจำกัด)


def _headers(sheet):
    return {str(cell.value): cell.column for cell in sheet[1] if cell.value is not None}


def _safe_part(value):
    result = "".join(c if c.isalnum() or c in " -_." else "_" for c in str(value)).strip(" .")
    return result or "Unknown"


def with_part(background):
    """ส่วนหน้าของชื่อไฟล์ใน Ready เมื่อคลิปมีชนิดอื่นติดมา เช่น 'with-Daurian-Redstart__'
    เห็นใน Finder ทันที และเรียงรวมกันท้ายโฟลเดอร์ชนิด (คลิปที่ไม่มีชนิดอื่นขึ้นต้นด้วยวันที่) -> เลือกอัปโหลดทีละกลุ่มได้"""
    names = sorted(background_species.split(background), key=str.casefold)
    if not names:
        return ""
    parts = [_safe_part(name).replace(" ", "-") for name in names[:WITH_NAMES]]
    more = f"_and-{len(names) - WITH_NAMES}-more" if len(names) > WITH_NAMES else ""
    return f"with-{'_'.join(parts)}{more}__"


def _within(parent: Path, relative: str) -> Path:
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("Clip path is outside the output folder")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviews(summary: Path):
    book = load_workbook(summary, read_only=True, data_only=True)
    try:
        sheet = book.active
        headers = _headers(sheet)
        if "File" not in headers:
            raise ValueError("summary.xlsx has no File column")
        result = []
        for row in sheet.iter_rows(min_row=2):
            values = {name: row[col - 1].value for name, col in headers.items()}
            if values.get("File"):
                values["_row"] = row[0].row
                result.append(values)
        return result
    finally:
        book.close()


def list_results(root):
    """ผลวิเคราะห์ทุกชุด (<root>/<วัน>/<เวลา>/summary.xlsx และแบบเก่า) ใหม่สุดก่อน -> list ของ dict"""
    found = []
    for summary in result_folders.summaries(root):
        try:
            rows = load_reviews(summary)
            modified = summary.stat().st_mtime
        except Exception:  # noqa: BLE001  ไฟล์เสียหรือกำลังถูกเขียน: ข้ามไปก่อน
            continue
        statuses = [str(row.get("Review status") or "Pending") for row in rows]
        found.append({
            "path": summary, "folder": result_folders.tag(summary.parent), "clips": len(rows),
            "approved": statuses.count("Approved"), "rejected": statuses.count("Rejected"),
            "unsure": statuses.count("Unsure"),
            "sources": sorted({str(row["Source file"]) for row in rows if row.get("Source file")}),
            "modified": modified,
        })
    # ชื่อโฟลเดอร์ขึ้นต้นด้วย YYYY.MM.DD_HHMM จึงเรียงตามเวลาบันทึกได้
    found.sort(key=lambda info: (info["folder"][:15], info["modified"]), reverse=True)
    return found


def save_review(summary: Path, row_number: int, *, status: str, common: str,
                scientific: str, rating=None, notes="", expected_file=None, sound_type=None, sound_type_guess=None,
                background=None):
    """sound_type / sound_type_guess / background = None: ไม่แตะค่าเดิม
    background = ชนิดเสียงพื้นหลัง คั่นด้วย ; (ไว้ใส่ช่อง background species ตอนอัปโหลด)"""
    if status not in STATUSES:
        raise ValueError("Invalid review status")
    if sound_type is not None and sound_type not in SOUND_TYPES:
        raise ValueError(f"Sound type must be one of: {', '.join(t for t in SOUND_TYPES if t)}")
    common, scientific = common.strip(), scientific.strip()
    if status == "Approved" and (not common or not scientific):
        raise ValueError("Enter both common and scientific names before approval")
    if status == "Approved" and rating not in (1, 2, 3, 4, 5):
        raise ValueError("Choose a human audio quality rating from 1 to 5")

    book = load_workbook(summary)
    sheet = book.active
    headers = _headers(sheet)
    if "File" not in headers or not 2 <= row_number <= sheet.max_row:
        book.close()
        raise ValueError("Selected clip is no longer in the summary")
    for name in REVIEW_COLUMNS:
        if name not in headers:
            col = sheet.max_column + 1
            sheet.cell(1, col, name)
            headers[name] = col

    def cell(name):
        return sheet.cell(row_number, headers[name])

    relative = str(cell("File").value or "")
    if not relative:
        book.close()
        raise ValueError("Selected row has no audio file")
    if expected_file is not None and relative != expected_file:
        book.close()
        raise ValueError("Summary changed since it was loaded; reload before reviewing")
    date_folder = summary.parent
    original = _within(date_folder, relative)
    if not original.is_file():
        book.close()
        raise FileNotFoundError(original)
    if status == "Approved" and original.stat().st_size > 500_000_000:
        book.close()
        raise ValueError("Clip exceeds the 500 MB eBird upload limit; shorten it before approval")
    previous = str(cell("Ready file").value or "")
    previous_hash = str(cell("Ready sha256").value or "")
    ready_base = result_folders.ready_base(date_folder)    # แบบใหม่: <วัน>/Ready/<ชนิด>/ ข้างโฟลเดอร์เวลา
    old_ready = None
    if previous:
        base = ready_base if result_folders.ready_file(date_folder, previous) == ready_base / previous else date_folder
        old_ready = _within(base, previous)
        if not old_ready.is_relative_to((base / "Ready").resolve()):
            book.close()
            raise ValueError("Saved Ready path is outside Ready folder")
        if old_ready.is_file() and (not previous_hash or _sha256(old_ready) != previous_hash):
            book.close()
            raise FileExistsError(f"Ready copy was changed outside the program: {old_ready}")
    ready_relative = ""
    ready_hash = ""
    new_ready = None

    kind = sound_type if sound_type is not None else str(cell("Sound type").value or "")
    others = background if background is not None else cell("Background species").value
    if status == "Approved":
        clip_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
        # ประเภทเสียงในชื่อไฟล์: ไว้ติดแท็กตอนอัปโหลดขึ้น eBird
        kind_part = f"{_safe_part(kind).replace(' ', '-')}_" if kind else ""
        out_name = (f"{with_part(others)}{result_folders.tag(date_folder)}_{clip_id}_"
                    f"{_safe_part(scientific).replace(' ', '.')}_{kind_part}R{rating}{original.suffix.lower()}")
        from manual_clips import recording_start_of       # manual_clips นำเข้าไฟล์นี้
        start = recording_start_of({name: cell(name).value for name in ("Analysis settings", "Start offset (s)", "Clock time")
                                    if name in headers})
        ready = ready_base / "Ready" / field_log.ready_folder(start) / _safe_part(common) / out_name
        if ready.is_file() and (old_ready is None or ready.resolve() != old_ready or
                                not previous_hash or _sha256(ready) != previous_hash):
            book.close()
            raise FileExistsError(f"Ready copy was changed outside the program: {ready}")
        ready.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready.with_name(ready.name + ".tmp")
        try:
            shutil.copy2(original, temporary)
            os.replace(temporary, ready)
        finally:
            temporary.unlink(missing_ok=True)
        ready_relative = str(ready.relative_to(ready_base))
        ready_hash = _sha256(ready)
        new_ready = ready.resolve()

    cell("Review status").value = status
    cell("Reviewed common").value = common
    cell("Reviewed scientific").value = scientific
    cell("Quality rating").value = rating if status == "Approved" else None
    cell("Review notes").value = notes.strip()
    cell("Ready file").value = ready_relative
    cell("Ready sha256").value = ready_hash
    if sound_type is not None:
        cell("Sound type").value = sound_type or None
    if sound_type_guess is not None:
        cell("Sound type guess").value = sound_type_guess or None
    if background is not None:
        cell("Background species").value = background.strip() or None
    temporary_summary = summary.with_name(summary.stem + ".review.tmp.xlsx")
    try:
        book.save(temporary_summary)
        os.replace(temporary_summary, summary)
    except Exception:
        if new_ready is not None and new_ready != old_ready and new_ready.is_file() \
                and _sha256(new_ready) == ready_hash:
            new_ready.unlink()
        raise
    finally:
        temporary_summary.unlink(missing_ok=True)
        book.close()

    if old_ready is not None and old_ready != new_ready and old_ready.is_file():
        old_ready.unlink()
    return ready_relative
