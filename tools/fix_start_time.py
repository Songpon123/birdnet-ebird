#!/usr/bin/env python3
"""ซ่อมผลวิเคราะห์ที่เวลาเริ่มอัดผิด: ไฟล์ Zoom ที่ ffprobe ให้แค่วันที่ เคยได้เวลา 00:00 (แก้สาเหตุแล้ว 2026-10-31)

แก้เวลาใน summary, ชื่อคลิป/ภาพ/ไฟล์ Ready และแยกไฟล์ต้นฉบับแต่ละไฟล์ไปโฟลเดอร์เวลาของตัวเอง
ผลตรวจอยู่ครบ เนื้อเสียงไม่เปลี่ยน (เปลี่ยนชื่ออย่างเดียว) ภาพ spectrogram แอปวาดเวลาใหม่ให้เองตอนเปิดผลชุดนั้น
ของเดิมสำรองไว้ที่ <โฟลเดอร์ผลลัพธ์>-backup-before-time-fix/ พร้อม moves.json (เดิม -> ใหม่ ทุกไฟล์) ก่อนแก้

    python/bin/python3 tools/fix_start_time.py <summary.xlsx>                  แสดงว่าจะแก้อะไร ไม่แตะไฟล์
    python/bin/python3 tools/fix_start_time.py <summary.xlsx> --apply          แก้จริง (ต้องปิดแอปก่อน)
    python/bin/python3 tools/fix_start_time.py <summary.xlsx> --apply --wait   รอจนปิดแอปแล้วแก้
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import app_settings  # noqa: E402
import manual_clips  # noqa: E402
import result_folders  # noqa: E402
from field_audio_to_ebird import read_audio_metadata  # noqa: E402
from openpyxl import load_workbook  # noqa: E402
from review_store import _sha256, load_reviews  # noqa: E402

CLIP = re.compile(r"\d{4}\.\d{2}\.\d{2}_\d{4}_\d{4}_(.+)")       # <วัน>_<เวลาเริ่มอัด>_<เวลาเสียง>_<ที่เหลือ>
CHANGED = ("Clock time", "Clock end", "File", "Ready file")      # คอลัมน์อื่นต้องเหมือนเดิมทุกช่อง
SUFFIX = re.compile(r"\d{4,6}(.*)")


def app_running():
    return subprocess.run(["pgrep", "-f", "birdnet_app.py|field_audio_to_ebird.py"], capture_output=True).returncode == 0


def clip_id(relative):
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]


def true_starts(rows):
    """{Source path: เวลาเริ่มอัดจริง} จาก metadata ของไฟล์ต้นฉบับ (ต้องตรงกับเวลาในชื่อไฟล์ Zoom ด้วย)"""
    starts = {}
    for path in sorted({str(row["Source path"]) for row in rows}):
        meta = read_audio_metadata(Path(path))
        if not meta["datetime"] or not meta["datetime_has_time"]:
            raise SystemExit(f"No start time in the file: {path}")
        starts[path] = meta["datetime"].replace(microsecond=0)
        named = datetime.strptime("_".join(Path(path).name.split("_")[:2]), "%y%m%d_%H%M%S")
        if named != starts[path]:
            raise SystemExit(f"File name and metadata disagree ({named} / {starts[path]}): {path}")
    return starts


def plan(summary):
    """ทุกอย่างที่จะเปลี่ยน -> {"sessions": {โฟลเดอร์ใหม่: {แถว: {คอลัมน์: ค่าใหม่}}}, "moves": [(เดิม, ใหม่)], "starts": …}"""
    rows = load_reviews(summary)
    folder = summary.parent
    root = result_folders.root_of(summary)
    suffix = SUFFIX.fullmatch(folder.name)[1]
    starts = true_starts(rows)
    sessions, moves = {}, []
    for row in rows:
        start = starts[str(row["Source path"])]
        shift = start - manual_clips.recording_start_of(row)
        target = result_folders.folder_for(root, start, suffix)

        def shifted(clock):
            return (datetime.strptime(str(clock), "%H:%M:%S") + shift).strftime("%H:%M:%S")

        clock = shifted(row["Clock time"])
        changes = {"Clock time": clock}
        if row.get("Clock end"):
            changes["Clock end"] = shifted(row["Clock end"])
        old_file = Path(str(row["File"]))
        name = f"{start:%Y.%m.%d_%H%M}_{clock[:2]}{clock[3:5]}_{CLIP.fullmatch(old_file.name)[1]}"
        changes["File"] = str(old_file.with_name(name))
        moves.append((folder / old_file, target / changes["File"]))
        if (folder / old_file).with_suffix(".png").exists():
            moves.append(((folder / old_file).with_suffix(".png"), (target / changes["File"]).with_suffix(".png")))
        if row.get("Ready file"):
            old_ready = Path(str(row["Ready file"]))
            before = f"{result_folders.tag(folder)}_{clip_id(str(row['File']))}_"
            if not old_ready.name.startswith(before):
                raise SystemExit(f"Unexpected Ready name: {old_ready}")
            ready = f"{result_folders.tag(target)}_{clip_id(changes['File'])}_{old_ready.name[len(before):]}"
            changes["Ready file"] = str(old_ready.with_name(ready))
            base = result_folders.ready_base(folder)
            moves.append((result_folders.ready_file(folder, str(old_ready)), base / changes["Ready file"]))
        sessions.setdefault(target, {})[row["_row"]] = changes
    if folder in sessions:
        raise SystemExit("Start times are already right: nothing to fix")
    problems = [f"Already exists: {folder}" for folder in sessions if folder.exists()]
    problems += [f"Missing: {old}" for old, _new in moves if not old.is_file()]
    problems += [f"Already exists: {new}" for _old, new in moves if new.exists()]
    problems += [f"Same new name: {new}" for new, count in Counter(new for _old, new in moves).items() if count > 1]
    if problems:
        raise SystemExit("\n".join(problems))
    return {"sessions": sessions, "moves": moves, "starts": starts, "rows": rows}


def write_session(summary, changes, target):
    """summary เดิมทั้งไฟล์ เหลือแค่แถวของผลชุดนี้ -> เซลล์อื่นคงเดิมทุกตัว"""
    book = load_workbook(summary)
    sheet = book.active
    headers = {cell.value: cell.column for cell in sheet[1]}
    for number in range(sheet.max_row, 1, -1):
        if number not in changes:
            sheet.delete_rows(number)
            continue
        for name, value in changes[number].items():
            sheet.cell(number, headers[name], value)
    book.save(target)
    book.close()


def back_up(summary, moves):
    folder = summary.parent
    root = result_folders.root_of(summary)
    target = root.with_name(f"{root.name}-backup-before-time-fix") / folder.parent.name
    if target.exists():
        raise SystemExit(f"Backup already exists: {target}")
    target.mkdir(parents=True)
    ready = result_folders.ready_base(folder) / "Ready"
    for source in (folder, ready):
        if source.is_dir():
            # APFS clone: ไม่กินที่ดิสก์เพิ่มจนกว่าไฟล์เดิมจะเปลี่ยน
            subprocess.run(["cp", "-cR", str(source), str(target / source.name)], check=True)
    for source in (folder, ready):
        count = sum(1 for path in source.rglob("*") if path.is_file())
        copied = sum(1 for path in (target / source.name).rglob("*") if path.is_file())
        if count != copied:
            raise SystemExit(f"Backup is incomplete: {source} {count} files, backup {copied}")
    (target / "moves.json").write_text(json.dumps(
        {"summary": str(summary), "moves": [[str(old), str(new)] for old, new in moves]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def remove_empty(folder):
    (folder / ".DS_Store").unlink(missing_ok=True)
    for path in sorted((p for p in folder.rglob("*") if p.is_dir()), reverse=True):
        (path / ".DS_Store").unlink(missing_ok=True)
        path.rmdir()
    folder.rmdir()


def apply(summary, work):
    backup = back_up(summary, work["moves"])
    moved, folders = [], list(work["sessions"])
    try:
        for target, changes in work["sessions"].items():
            target.mkdir(parents=True)
            write_session(summary, changes, target / "summary.tmp.xlsx")
        for old, new in work["moves"]:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
            moved.append((old, new))
        for target in folders:
            os.replace(target / "summary.tmp.xlsx", target / "summary.xlsx")
    except BaseException:
        for old, new in reversed(moved):
            os.rename(new, old)
        for target in folders:
            for path in (target / "summary.xlsx", target / "summary.tmp.xlsx"):
                path.unlink(missing_ok=True)
            remove_empty(target)
        raise
    summary.unlink()
    remove_empty(summary.parent)
    main_session = max(work["sessions"], key=lambda target: len(work["sessions"][target]))
    if app_settings.load_settings().get("last_summary") == str(summary):
        app_settings.save_settings({"last_summary": str(main_session / "summary.xlsx")})
    return backup


def verify(work):
    """อ่านผลที่แก้แล้วกลับมาเทียบกับแถวเดิมทุกช่อง และเทียบ hash ของไฟล์เสียงทุกไฟล์ (ชนิดต่างกันอาจมีคลิปเสียงเดียวกัน)"""
    before = {row["_row"]: row for row in work["rows"]}
    after = []
    for target, changes in work["sessions"].items():
        saved = {str(row["File"]): row for row in load_reviews(target / "summary.xlsx")}
        if len(saved) != len(changes):
            raise SystemExit(f"Rows lost: {target}")
        for number, change in changes.items():
            old, row = before.pop(number), saved[change["File"]]
            after.append(row)
            for name in (set(row) | set(old)) - {"_row"}:
                if row.get(name) != change.get(name, old.get(name)):
                    raise SystemExit(f"Changed {name}: {row['File']}")
            if manual_clips.recording_start_of(row) != work["starts"][str(row["Source path"])]:
                raise SystemExit(f"Wrong start time: {row['File']}")
            if _sha256(target / str(row["File"])) != row["Generated sha256"]:
                raise SystemExit(f"Clip differs: {row['File']}")
            if not (target / str(row["File"])).with_suffix(".png").is_file():
                raise SystemExit(f"No picture: {row['File']}")
            if row.get("Ready file") and \
                    _sha256(result_folders.ready_file(target, str(row["Ready file"]))) != row["Ready sha256"]:
                raise SystemExit(f"Ready file differs: {row['Ready file']}")
    if before:
        raise SystemExit(f"Rows lost: {len(before)}")
    return Counter(str(row.get("Review status") or "Pending") for row in after)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wait", action="store_true", help="wait until the app is closed")
    parser.add_argument("--ignore-app", action="store_true", help="for a copy of the results, not the real folder")
    args = parser.parse_args()
    summary = args.summary.expanduser().resolve()
    if args.apply and not args.ignore_app:
        while args.wait and app_running():
            time.sleep(5)
        if args.wait:
            time.sleep(15)        # งานเบื้องหลังที่ค้างหยุดเสร็จ / เผื่อเปิดแอปกลับมาทันที
        if app_running():
            raise SystemExit("The app is open: close it first")
    work = plan(summary)          # อ่านหลังปิดแอป: ได้ผลตรวจล่าสุด
    print(Counter(str(row.get("Review status") or "Pending") for row in work["rows"]))
    for target, changes in work["sessions"].items():
        print(f"{target}: {len(changes)} clips")
    print(f"{len(work['moves'])} files to rename")
    for old, new in work["moves"][:4]:
        print(f"  {old.name} -> {new.name}")
    if args.apply:
        backup = apply(summary, work)
        print(f"Backup: {backup}")
        print(f"Verified: {verify(work)}")


if __name__ == "__main__":
    main()
