"""ผลวิเคราะห์อยู่ใน <ผลลัพธ์>/<วัน>/<เวลาเริ่ม><โมเดล>/ — อ่านแบบเก่าได้ และย้ายแบบเก่าเข้าโฟลเดอร์วันพร้อมแคช"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import app_settings  # noqa: E402
import paths  # noqa: E402
import result_folders  # noqa: E402
from review_store import list_results, load_reviews, save_review  # noqa: E402


def make_result(folder, statuses=("Pending",)):
    folder.mkdir(parents=True)
    pd.DataFrame([{"File": f"{n}.wav", "Source file": "REC-001.WAV", "Review status": s}
                  for n, s in enumerate(statuses)]).to_excel(folder / "summary.xlsx", index=False)
    return folder / "summary.xlsx"


def digest(summary):
    return hashlib.sha256(str(summary).encode("utf-8")).hexdigest()[:12]


class ResultFolderTests(unittest.TestCase):
    def test_new_results_go_into_the_recording_day(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            when = datetime(2026, 10, 17, 14, 51)
            self.assertEqual(result_folders.folder_for(root, when, "_v3preview"),
                             root / "2026.10.17" / "1451_v3preview")
            self.assertEqual(result_folders.folder_for(root, datetime(2026, 10, 25, 17, 24, 5), "_sound_events",
                                                       seconds=True), root / "2026.10.25" / "172405_sound_events")
            (root / "2026.10.17_1451_v3preview").mkdir()          # ผลเก่าของเวลานี้: รันซ้ำใช้ที่เดิม
            self.assertEqual(result_folders.folder_for(root, when, "_v3preview"), root / "2026.10.17_1451_v3preview")

    def test_names_and_listing_work_for_both_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            new = make_result(root / "2026.10.17" / "1551_v3preview", ["Approved", "Pending"])
            make_result(root / "2026.10.17" / "1451_v3preview")
            old = make_result(root / "2026.10.06_0928_v3preview")
            make_result(root / "2026.10.17" / "1451_v3preview" / "Ready" / "x")   # ไม่ใช่ผลชุด
            self.assertEqual(result_folders.tag(new.parent), "2026.10.17_1551_v3preview")
            self.assertEqual(result_folders.tag(old.parent), "2026.10.06_0928_v3preview")
            self.assertEqual(result_folders.root_of(new), root)
            self.assertEqual(result_folders.root_of(old), root)
            found = list_results(root)
        self.assertEqual([info["folder"] for info in found],
                         ["2026.10.17_1551_v3preview", "2026.10.17_1451_v3preview", "2026.10.06_0928_v3preview"])
        self.assertEqual(found[0]["approved"], 1)

    def test_organize_moves_old_results_with_their_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp).resolve()
            root, cache = temp / "out", temp / "cache"
            (cache / "sound-index").mkdir(parents=True)
            settings = temp / "settings.json"
            old_summary = make_result(root / "2026.10.17_1451_v3preview", ["Approved"])
            make_result(root / "2026.10.17_1551_v3preview")
            make_result(root / "2026.10.18_1439_v3preview")
            make_result(root / "2026.10.18" / "1439_v3preview")       # มีในโฟลเดอร์วันแล้ว: ไม่ทับ
            (root / "2026.10.17_1451_v3preview" / "Asian Tit").mkdir()
            near = cache / "sound-index" / f"near-{digest(old_summary)}.json"
            near.write_text(json.dumps({"summary": str(old_summary), "species": ["Parus cinereus"]}))
            other = cache / "sound-index" / "near-000000000000.json"
            other.write_text(json.dumps({"summary": "/elsewhere/summary.xlsx"}))
            app_settings.save_settings({"last_summary": str(old_summary), "keep": 1}, settings)
            notes = []
            with patch.object(paths, "CACHE", cache), patch.object(app_settings, "SETTINGS_FILE", settings):
                moves = result_folders.organize(root, log=notes.append)
            self.assertEqual([new.relative_to(root).as_posix() for _old, new in moves],
                             ["2026.10.17/1451_v3preview", "2026.10.17/1551_v3preview"])
            new_summary = root / "2026.10.17" / "1451_v3preview" / "summary.xlsx"
            self.assertTrue((new_summary.parent / "Asian Tit").is_dir())
            self.assertFalse((root / "2026.10.17_1451_v3preview").exists())
            self.assertTrue((root / "2026.10.18_1439_v3preview").is_dir())
            self.assertIn("Not moved", notes[0])
            moved = cache / "sound-index" / f"near-{digest(new_summary)}.json"
            self.assertFalse(near.exists())
            self.assertEqual(json.loads(moved.read_text())["summary"], str(new_summary))
            self.assertTrue(other.exists())
            self.assertEqual(app_settings.load_settings(settings), {"last_summary": str(new_summary), "keep": 1})
            self.assertEqual(result_folders.organize(root, log=notes.append), [])   # ครั้งที่สอง: ไม่มีอะไรต้องย้าย


    def test_approved_clips_go_to_the_day_ready_folder_by_species(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.25"
            summaries = []
            for session in ("0909_v3preview", "1032_v3preview"):
                folder = day / session
                (folder / "Asian Tit").mkdir(parents=True)
                (folder / "Asian Tit" / "clip.wav").write_bytes(session.encode() * 100)
                summary = folder / "summary.xlsx"
                pd.DataFrame([{"File": "Asian Tit/clip.wav", "Review status": "Pending"}]).to_excel(summary,
                                                                                                    index=False)
                summaries.append(summary)
            first = save_review(summaries[0], 2, status="Approved", common="Japanese Tit", scientific="Parus minor",
                                rating=4)
            second = save_review(summaries[1], 2, status="Approved", common="Japanese Tit",
                                 scientific="Parus minor", rating=3)
            self.assertTrue(first.startswith("Ready/Japanese Tit/2026.10.25_0909_v3preview_"), first)
            self.assertTrue(second.startswith("Ready/Japanese Tit/2026.10.25_1032_v3preview_"), second)
            self.assertEqual(sorted(p.relative_to(day).as_posix() for p in (day / "Ready").rglob("*.wav")),
                             sorted([first, second]))                       # ข้างโฟลเดอร์เวลา แยกตามชนิด
            self.assertFalse((day / "0909_v3preview" / "Ready").exists())
            self.assertEqual(result_folders.ready_file(summaries[0].parent, first), day / first)
            again = save_review(summaries[0], 2, status="Approved", common="Japanese Tit", scientific="Parus minor",
                                rating=5)
            self.assertFalse((day / first).exists())                        # คะแนนใหม่: เปลี่ยนชื่อ ไม่ซ้ำ
            self.assertTrue((day / again).is_file())
            self.assertEqual(load_reviews(summaries[0])[0]["Ready file"], again)
            save_review(summaries[0], 2, status="Rejected", common="Japanese Tit", scientific="Parus minor")
            self.assertFalse((day / again).exists())
            self.assertTrue((day / second).is_file())

    def test_organize_gathers_each_sessions_ready_into_the_day(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp).resolve()
            root = temp / "out"
            flat = root / "2026.10.17_1451_v3preview"                 # แบบเก่า: Ready อยู่ในโฟลเดอร์ผลชุด
            (flat / "Ready" / "Asian Tit").mkdir(parents=True)
            (flat / "Ready" / "Asian Tit" / "a.wav").write_bytes(b"a")
            (flat / "Ready" / ".DS_Store").write_bytes(b"x")
            make_result(root / "2026.10.17" / "1551_v3preview")
            session = root / "2026.10.17" / "1551_v3preview"
            (session / "Ready" / "Asian Tit").mkdir(parents=True)
            (session / "Ready" / "Asian Tit" / "b.wav").write_bytes(b"b")
            (session / "Ready" / "Asian Tit" / "same.wav").write_bytes(b"same")
            (session / "Ready" / "Asian Tit" / "clash.wav").write_bytes(b"mine")
            (root / "2026.10.17" / "Ready" / "Asian Tit").mkdir(parents=True)
            (root / "2026.10.17" / "Ready" / "Asian Tit" / "same.wav").write_bytes(b"same")
            (root / "2026.10.17" / "Ready" / "Asian Tit" / "clash.wav").write_bytes(b"other")
            notes = []
            with patch.object(paths, "CACHE", temp / "cache"), \
                    patch.object(app_settings, "SETTINGS_FILE", temp / "settings.json"):
                result_folders.organize(root, log=notes.append)
            ready = root / "2026.10.17" / "Ready" / "Asian Tit"
            self.assertEqual(sorted(p.name for p in ready.iterdir()), ["a.wav", "b.wav", "clash.wav", "same.wav"])
            self.assertEqual((ready / "clash.wav").read_bytes(), b"other")
            self.assertFalse((root / "2026.10.17" / "1451_v3preview" / "Ready").exists())
            kept = session / "Ready" / "Asian Tit" / "clash.wav"            # ชื่อชนแต่คนละไฟล์: ไม่ทับ
            self.assertEqual(sorted(p.name for p in (session / "Ready" / "Asian Tit").iterdir()), ["clash.wav"])
            self.assertTrue(any("clash.wav" in note for note in notes))
            self.assertTrue(any("2 file(s)" in note for note in notes))
            self.assertEqual(kept.read_bytes(), b"mine")
            self.assertEqual(result_folders.ready_file(session, "Ready/Asian Tit/b.wav"), ready / "b.wav")
            (session / "Ready" / "Asian Tit" / "only-here.wav").write_bytes(b"c")   # ยังไม่ได้ย้าย: ใช้ที่เดิม
            self.assertEqual(result_folders.ready_file(session, "Ready/Asian Tit/only-here.wav"),
                             session / "Ready" / "Asian Tit" / "only-here.wav")

if __name__ == "__main__":
    unittest.main()
