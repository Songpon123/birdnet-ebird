#!/usr/bin/env python3
"""ย้ายคลิปที่ตรวจแล้วเข้าโฟลเดอร์ checklist: Ready/<ชนิด>/ -> Ready/<checklist>/<ชนิด>/

เปลี่ยนเฉพาะตำแหน่งไฟล์ใน Ready และคอลัมน์ Ready file เนื้อเสียง ผลตรวจ และคอลัมน์อื่นเหมือนเดิม วันที่ไม่มี Log ไม่เปลี่ยน
ไฟล์ที่รวมจาก Merge clips ตามคลิปไปเมื่อชนิดนั้นอยู่ checklist เดียว ถ้าอยู่หลาย checklist คงไว้ที่เดิมและรายงาน
สำรอง summary ทุกชุดที่แก้ พร้อม moves.json (เดิม -> ใหม่) ไว้ที่ <โฟลเดอร์ผลลัพธ์>-backup-before-ready-checklists

    python/bin/python3 tools/sort_ready_by_checklist.py                  แสดงว่าจะย้ายอะไร ไม่แตะไฟล์
    python/bin/python3 tools/sort_ready_by_checklist.py --apply          ย้ายจริง (ต้องปิดแอปก่อน)
    python/bin/python3 tools/sort_ready_by_checklist.py --apply --wait   รอจนปิดแอปแล้วย้าย
"""

import argparse
from collections import Counter
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import ready_folders  # noqa: E402
import result_folders  # noqa: E402
import result_roots  # noqa: E402
from review_store import _sha256, list_results, load_reviews  # noqa: E402


def app_running():
    return subprocess.run(["pgrep", "-f", "birdnet_app.py|field_audio_to_ebird.py"], capture_output=True).returncode == 0


def back_up(work, others):
    """สำรอง summary ทุกชุดก่อนแก้ แยกตามโฟลเดอร์ผลลัพธ์ -> {summary: สำเนา}"""
    moves, copies, roots = {}, {}, {}
    for summary, rows in work.items():
        root = result_folders.root_of(summary)
        target = root.with_name(f"{root.name}-backup-before-ready-checklists")
        copy = target / summary.relative_to(root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, copy)
        moves.setdefault(target, []).extend([str(old), str(new)] for old, new, _relative in rows.values())
        copies[summary], roots[root] = copy, target
    for old, new in others:
        moves[next(target for root, target in roots.items() if old.is_relative_to(root))].append([str(old), str(new)])
    for target, pairs in moves.items():
        (target / "moves.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False), encoding="utf-8")
    return copies


def verify(work, copies, others):
    """ทุกช่องเหมือนเดิมยกเว้น Ready file, ค่าใหม่ตรงตามแผน, ไฟล์ทุกไฟล์เนื้อเดิม"""
    for new, digest in others.items():
        if _sha256(new) != digest:
            raise SystemExit(f"File differs: {new}")
    for summary, rows in work.items():
        before = {row["_row"]: row for row in load_reviews(copies[summary])}
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
            time.sleep(15)        # งานเบื้องหลังที่ค้างหยุดเสร็จ / เขียน summary เสร็จ
        if app_running():
            raise SystemExit("The app is open: close it first")
    results = list_results(args.root) if args.root else result_roots.all_results()
    work, others, problems = ready_folders.plan(info["path"] for info in results)   # อ่านหลังปิดแอป: ได้ผลตรวจล่าสุด
    for summary, rows in work.items():
        print(f"{result_folders.tag(summary.parent)}: {len(rows)} files")
    pairs = [pair for rows in work.values() for pair in rows.values()]
    print(f"{len(pairs)} clips and {len(others)} other files to move")
    for folder, count in sorted(Counter(Path(relative).parts[1] for _old, _new, relative in pairs).items()):
        print(f"  {count:4}  {folder}")
    for _old, new in others:
        print(f"  with the clips: {new.parent.parent.name} / {new.parent.name} / {new.name}")
    for problem in problems:
        print(f"Not moved: {problem}")
    if args.apply and work:
        copies = back_up(work, others)
        digests = {new: _sha256(old) for old, new in others}
        ready_folders.apply(work, others)
        verify(work, copies, digests)
        print(f"Moved and verified {len(pairs) + len(others)} files. Backup: {sorted({str(copy.parents[2]) for copy in copies.values()})}")


if __name__ == "__main__":
    main()
