"""คลิปที่ Approve เข้า Ready/<checklist>/<ชนิด> — วันที่ไม่มี Log ใช้ Ready/<ชนิด> แบบเดิม"""

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import field_log  # noqa: E402
import ready_folders  # noqa: E402
from review_store import load_reviews, save_review  # noqa: E402

MORNING = {"id": "S2", "date": "2026-10-30T06:36", "location": "教字垭镇 (Jiaoziya Town), Hunan, CN", "species": []}
FOLDER = "0636 教字垭镇 (Jiaoziya Town) S2"


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.store = self.temp / "field-logs.json"
        patcher = patch.object(field_log, "STORE", self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def logs(self, *checklists):
        self.store.write_text(json.dumps({"checklists": {c["id"]: c for c in checklists}}), encoding="utf-8")

    def approve(self):
        session = self.temp / "2026.10.30" / "0634_v3pruned"
        (session / "Asian Tit").mkdir(parents=True)
        (session / "Asian Tit" / "a.wav").write_bytes(b"RIFF")
        pd.DataFrame([{"File": "Asian Tit/a.wav", "Start offset (s)": 100.0, "Clock time": "06:36:01",
                       "Analysis settings": json.dumps({"date": "2026-10-30"})}]).to_excel(session / "summary.xlsx", index=False)
        return save_review(session / "summary.xlsx", 2, status="Approved", common="Asian Tit",
                           scientific="Parus cinereus", rating=3), session

    def test_approved_clip_goes_under_its_checklist(self):
        self.logs(MORNING)
        relative, session = self.approve()
        self.assertEqual(Path(relative).parts[:3], ("Ready", FOLDER, "Asian Tit"))
        self.assertTrue((session.parent / relative).is_file())

    def test_day_without_a_log_keeps_the_old_layout(self):
        self.logs()
        relative, _session = self.approve()
        self.assertEqual(Path(relative).parts[:2], ("Ready", "Asian Tit"))
        self.assertEqual(len(Path(relative).parts), 3)

    def test_recording_set_to_no_checklist(self):
        self.logs(MORNING)
        field_log.choose({"start": datetime(2026, 10, 30, 6, 34, 21), "end": datetime(2026, 10, 30, 8, 7, 32)}, field_log.NONE)
        relative, _session = self.approve()
        self.assertEqual(Path(relative).parts[:3], ("Ready", "No checklist", "Asian Tit"))

    def test_folder_name_is_safe(self):
        name = field_log.folder_name(dict(MORNING, location="A/B: C, Hunan, CN"))
        self.assertEqual(name, "0636 A_B_ C S2")


class MovingTests(Case):
    def reviewed(self):
        """คลิปที่ตรวจก่อนมี Log อยู่ที่ Ready/<ชนิด>/"""
        self.logs()
        relative, session = self.approve()
        self.summary = session / "summary.xlsx"
        self.day = session.parent
        return relative

    def row(self):
        return load_reviews(self.summary)[0]

    def test_reviewed_clip_moves_with_its_review_and_sound(self):
        old = self.reviewed()
        content = (self.day / old).read_bytes()
        self.logs(MORNING)
        work, others, problems = ready_folders.plan([self.summary])
        self.assertEqual((others, problems), ([], []))
        ready_folders.apply(work)
        row = self.row()
        self.assertEqual(Path(row["Ready file"]).parts[:3], ("Ready", FOLDER, "Asian Tit"))
        self.assertEqual((self.day / row["Ready file"]).read_bytes(), content)
        self.assertEqual(row["Quality rating"], 3)
        self.assertFalse((self.day / "Ready" / "Asian Tit").exists())              # โฟลเดอร์ชนิดเดิมที่ว่างถูกลบ
        self.assertEqual(ready_folders.plan([self.summary]), ({}, [], []))         # รันซ้ำไม่มีอะไรต้องย้าย

    def test_changing_the_checklist_moves_it_again_and_leaves_no_empty_folder(self):
        self.reviewed()
        noon = dict(MORNING, id="S1", date="2026-10-30T11:28", location="崇山 (Chongshan Mountain), Hunan, CN")
        self.logs(MORNING, noon)
        ready_folders.apply(*ready_folders.plan([self.summary])[:2])
        field_log.choose({"start": datetime(2026, 10, 30, 6, 34, 21), "end": datetime(2026, 10, 30, 8, 7, 32)}, "S1")
        ready_folders.apply(*ready_folders.plan([self.summary])[:2])
        self.assertEqual(Path(self.row()["Ready file"]).parts[1], "1128 崇山 (Chongshan Mountain) S1")
        self.assertFalse((self.day / "Ready" / FOLDER).exists())

    def test_merged_file_follows_the_clips_to_their_one_checklist(self):
        old = self.reviewed()
        (self.day / old).with_name("Asian Tit merged.wav").write_bytes(b"mine")
        self.logs(MORNING)
        work, others, problems = ready_folders.plan([self.summary])
        self.assertEqual((len(others), problems), (1, []))
        ready_folders.apply(work, others)
        self.assertEqual((self.day / "Ready" / FOLDER / "Asian Tit" / "Asian Tit merged.wav").read_bytes(), b"mine")
        self.assertFalse((self.day / "Ready" / "Asian Tit").exists())

    def test_merged_file_stays_when_clips_split_between_checklists(self):
        old = self.reviewed()
        merged = (self.day / old).with_name("Asian Tit merged.wav")
        merged.write_bytes(b"mine")
        session = self.summary.parent
        relative = Path(old)
        second = relative.with_name("second.wav")
        (self.day / second).write_bytes((self.day / old).read_bytes())
        (session / "Asian Tit" / "b.wav").write_bytes(b"RIFF")
        book = load_workbook(self.summary)
        sheet = book.active
        row = [cell.value for cell in sheet[2]]
        heads = [cell.value for cell in sheet[1]]
        row[heads.index("File")], row[heads.index("Clock time")] = "Asian Tit/b.wav", "12:30:00"
        row[heads.index("Ready file")] = str(second)
        sheet.append(row)
        book.save(self.summary)
        noon = dict(MORNING, id="S1", date="2026-10-30T11:28", location="Chongshan, Hunan, CN")
        self.logs(MORNING, noon)
        work, others, problems = ready_folders.plan([self.summary])
        self.assertEqual(sum(map(len, work.values())), 2)
        self.assertEqual(others, [])
        self.assertIn("more than one checklist", problems[0])
        ready_folders.apply(work, others)
        self.assertEqual(merged.read_bytes(), b"mine")

    def test_merged_file_stays_beside_a_clip_that_could_not_move(self):
        old = self.reviewed()
        merged = (self.day / old).with_name("Asian Tit merged.wav")
        merged.write_bytes(b"mine")
        (self.day / old).write_bytes(b"edited")
        self.logs(MORNING)
        work, others, problems = ready_folders.plan([self.summary])
        self.assertEqual((work, others), ({}, []))

    def test_clip_edited_outside_the_app_is_left_and_reported(self):
        old = self.reviewed()
        (self.day / old).write_bytes(b"edited")
        self.logs(MORNING)
        work, _others, problems = ready_folders.plan([self.summary])
        self.assertEqual(work, {})
        self.assertIn("Changed outside the app", problems[0])

    def test_existing_file_at_the_target_is_never_replaced(self):
        old = self.reviewed()
        self.logs(MORNING)
        target = self.day / "Ready" / FOLDER / "Asian Tit" / Path(old).name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"other")
        work, _others, problems = ready_folders.plan([self.summary])
        self.assertEqual(work, {})
        self.assertIn("Already exists", problems[0])
        self.assertEqual(target.read_bytes(), b"other")

    def test_a_failed_move_puts_everything_back(self):
        old = self.reviewed()
        self.logs(MORNING)
        work = ready_folders.plan([self.summary])[0]
        before = self.summary.read_bytes()
        with patch.object(ready_folders.os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                ready_folders.apply(work)
        self.assertTrue((self.day / old).is_file())
        self.assertEqual(self.summary.read_bytes(), before)
        self.assertEqual(list(self.summary.parent.glob("summary.sort*")), [])


class AppTests(Case):
    def test_closing_match_recordings_queues_the_move_only_when_needed(self):
        import tkinter as tk
        import birdnet_gui as g
        import field_log_window
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        jobs = []
        with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                patch.object(g.App, "_start", lambda _self, *args, **kwargs: jobs.append((args, kwargs))), \
                patch.object(field_log_window, "recordings_on", return_value=[]), \
                patch.object(g.ready_folders, "day_summaries", return_value=["x"]):
            try:
                g.App(root)
                root.createcommand("::tk::mac::Quit", lambda: None)
                self.logs(MORNING)
                day = datetime(2026, 10, 30).date()
                with patch.object(g.ready_folders, "plan", return_value=({}, [], [])):
                    window = field_log_window.RecordingChecklists(root, None, day)
                    window.win.destroy()
                    root.update()
                self.assertEqual(jobs, [])
                with patch.object(g.ready_folders, "plan", return_value=({"x": {}}, [], [])):
                    window = field_log_window.RecordingChecklists(root, None, day)
                    window.win.destroy()
                    root.update()
                self.assertEqual(len(jobs), 1)
                self.assertIn("--sort-ready", jobs[0][0][0])
                self.assertIn("2026-10-30", jobs[0][0][0])
                self.assertEqual(root._matched_days, set())
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
