#!/usr/bin/env python3
"""เปลี่ยนชื่อไฟล์ใน Ready ที่ Approve ไว้ก่อน 2026-10-31 ให้ขึ้นต้นด้วยชนิดที่ติดมา (with-<ชนิด>__…) แบบที่แอปตั้งตอนนี้

เปลี่ยนเฉพาะชื่อไฟล์ใน Ready และคอลัมน์ Ready file ของคลิปที่มี Background species เนื้อเสียงไม่เปลี่ยน
สำรอง summary ทุกชุดที่แก้ พร้อม moves.json (เดิม -> ใหม่) ไว้ที่ <โฟลเดอร์ผลลัพธ์>-backup-before-ready-rename/ ก่อนแก้

    python/bin/python3 tools/rename_ready_with_species.py                  แสดงว่าจะแก้อะไร ไม่แตะไฟล์
    python/bin/python3 tools/rename_ready_with_species.py --apply          แก้จริง (ต้องปิดแอปก่อน)
    python/bin/python3 tools/rename_ready_with_species.py --apply --wait   รอจนปิดแอปแล้วแก้
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import result_folders  # noqa: E402
import result_roots  # noqa: E402
from openpyxl import load_workbook  # noqa: E402
from review_store import _sha256, list_results, load_reviews, with_part  # noqa: E402


def app_running():
    return subprocess.run(["pgrep", "-f", "birdnet_app.py|field_audio_to_ebird.py"], capture_output=True).returncode == 0


def plan(results):
    """{summary: {แถว: (ไฟล์เดิม, ไฟล์ใหม่, ค่าใหม่ของ Ready file)}} เฉพาะคลิปที่ชื่อยังไม่บอกชนิดที่ติดมา"""
    work, problems = {}, []
    for info in results:
        summary = Path(info["path"])
        for row in load_reviews(summary):
            prefix = with_part(row.get("Background species"))
            relative = Path(str(row.get("Ready file") or ""))
            if not row.get("Ready file") or not prefix or relative.name.startswith(prefix):
                continue
            old = result_folders.ready_file(summary.parent, str(relative))
            new = old.with_name(prefix + relative.name)
            if not old.is_file():
                problems.append(f"Missing: {old}")
            elif new.exists():
                problems.append(f"Already exists: {new}")
            work.setdefault(summary, {})[row["_row"]] = (old, new, str(relative.with_name(new.name)))
    if problems:
        raise SystemExit("\n".join(problems))
    return work


def back_up(work):
    """สำรอง summary ทุกชุดก่อนแก้ แยกตามโฟลเดอร์ผลลัพธ์"""
    moves, targets = {}, []
    for summary, rows in work.items():
        root = result_folders.root_of(summary)
        target = root.with_name(f"{root.name}-backup-before-ready-rename")
        copy = target / summary.relative_to(root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, copy)
        moves.setdefault(target, []).extend([str(old), str(new)] for old, new, _relative in rows.values())
        targets.append(copy)
    for target, pairs in moves.items():
        (target / "moves.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False), encoding="utf-8")
    return targets


def apply(work):
    copies = back_up(work)
    written, moved = [], []
    try:
        for summary, rows in work.items():
            book = load_workbook(summary)
            sheet = book.active
            column = next(cell.column for cell in sheet[1] if cell.value == "Ready file")
            for number, (_old, _new, relative) in rows.items():
                sheet.cell(number, column, relative)
            temporary = summary.with_name("summary.rename.tmp.xlsx")
            book.save(temporary)
            book.close()
            written.append((temporary, summary))
        for rows in work.values():
            for old, new, _relative in rows.values():
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
    return copies


def verify(work, copies):
    """ทุกช่องเหมือนเดิมยกเว้น Ready file, ชื่อใหม่ตรงกับที่แอปตั้ง และเนื้อไฟล์ตรงกับ Ready sha256"""
    for (summary, rows), copy in zip(work.items(), copies):
        before = {row["_row"]: row for row in load_reviews(copy)}
        after = load_reviews(summary)
        if len(before) != len(after):
            raise SystemExit(f"Rows changed: {summary}")
        for row in after:
            old = before[row["_row"]]
            for name in (set(row) | set(old)) - {"Ready file"}:
                if row.get(name) != old.get(name):
                    raise SystemExit(f"Changed {name}: {row['File']}")
            wanted = rows[row["_row"]][2] if row["_row"] in rows else old.get("Ready file")
            if row.get("Ready file") != wanted:
                raise SystemExit(f"Wrong Ready file: {row['File']}")
            if row.get("Ready file") and \
                    _sha256(result_folders.ready_file(summary.parent, str(row["Ready file"]))) != row["Ready sha256"]:
                raise SystemExit(f"Ready file differs: {row['Ready file']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wait", action="store_true", help="wait until the app is closed")
    parser.add_argument("--root", type=Path, help="only this output folder, for a copy of the results")
    args = parser.parse_args()
    if args.apply and not args.root:
        while args.wait and app_running():
            time.sleep(5)
        if args.wait:
            time.sleep(15)        # งานเบื้องหลังที่ค้างหยุดเสร็จ / เผื่อเปิดแอปกลับมาทันที
        if app_running():
            raise SystemExit("The app is open: close it first")
    results = list_results(args.root) if args.root else result_roots.all_results()
    work = plan(results)          # อ่านหลังปิดแอป: ได้ผลตรวจล่าสุด
    for summary, rows in work.items():
        print(f"{result_folders.tag(summary.parent)}: {len(rows)} files")
    pairs = [pair for rows in work.values() for pair in rows.values()]
    print(f"{len(pairs)} files to rename")
    for _old, new, _relative in pairs[:5]:
        print(f"  {new.name}")
    if args.apply and work:
        copies = apply(work)
        verify(work, copies)
        print(f"Renamed and verified {len(pairs)} files. Backup: {copies[0].parents[2]}")


if __name__ == "__main__":
    main()
