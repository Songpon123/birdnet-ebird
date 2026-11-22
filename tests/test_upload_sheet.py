"""รายการอัปโหลด: ไฟล์ใน Ready ของวัน + เสียงพื้นหลัง + ชนิดที่ต้องเพิ่มใน checklist ก่อน"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import upload_sheet  # noqa: E402

SETTINGS = json.dumps({"date": "2026-10-30"})
CHECKLIST = {"id": "S1", "species": [{"common": "Asian Tit"}, {"common": "White Wagtail (Chinese)"}]}


def row(common, scientific, start, ready="", status="Approved", background=None, clock="07:00:00"):
    return {"File": f"{common}/{start}.wav", "Species (common)": common, "Species (scientific)": scientific,
            "Start offset (s)": start, "End offset (s)": start + 3, "Duration (s)": 9, "Clock time": clock,
            "Detection boxes": json.dumps({"windows": [[3.0, 6.0, 0.8]]}), "Analysis settings": SETTINGS,
            "Source path": "/rec/a.wav", "Review status": status, "Reviewed common": common if ready else None,
            "Reviewed scientific": scientific if ready else None, "Quality rating": 3 if ready else None,
            "Ready file": ready or None, "Background species": background, "Sound type": "Call" if ready else None}


class UploadSheetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.day = Path(temporary.name) / "2026.10.30"
        (self.day / "Ready").mkdir(parents=True)
        patcher = patch.object(upload_sheet.field_log, "for_recording", return_value=[CHECKLIST])
        self.for_recording = patcher.start()
        self.addCleanup(patcher.stop)

    def result(self, session, rows):
        folder = self.day / session
        folder.mkdir()
        pd.DataFrame(rows).to_excel(folder / "summary.xlsx", index=False)
        return folder / "summary.xlsx"

    def test_lists_every_ready_clip_of_the_day(self):
        first = self.result("0623_v3pruned", [
            row("White Wagtail", "Motacilla alba", 100, "Ready/White Wagtail/w.wav", clock="07:02:00",
                background="Asian Tit; Oriental Magpie"),
            row("Asian Tit", "Parus cinereus", 500, "Ready/Asian Tit/b.wav", clock="08:00:00"),
            row("Daurian Redstart", "Phoenicurus auroreus", 101, status="Pending")])     # ยังไม่ Approve: ไม่อยู่ในรายการ
        self.result("0634_v3pruned", [row("Asian Tit", "Parus cinereus", 9, "Ready/Asian Tit/a.wav", clock="06:40:00")])
        path, clips = upload_sheet.write(first)
        self.assertEqual(path, self.day / "Ready" / "upload-sheet.csv")
        with open(path, encoding="utf-8-sig", newline="") as sheet:
            saved = list(csv.DictReader(sheet))
        self.assertEqual([c["File"] for c in saved], ["a.wav", "b.wav", "w.wav"])            # ชนิด แล้วเวลา
        wagtail = saved[2]
        self.assertEqual((wagtail["Quality"], wagtail["Sound type"], wagtail["Time"]), ("3", "Call", "07:02:00"))
        self.assertEqual(wagtail["Additional species"], "Asian Tit; Oriental Magpie")
        self.assertEqual(wagtail["Add to checklist first"], "Oriental Magpie")              # eBird เลือกไม่ได้จนกว่าจะเพิ่ม
        self.assertEqual(wagtail["BirdNET also heard"], "Daurian Redstart")                 # ทับช่วงคลิป แต่ไม่ได้บันทึกไว้

    def test_saved_species_are_not_listed_again_as_heard(self):
        summary = self.result("0623_v3pruned", [
            row("White Wagtail", "Motacilla alba", 100, "Ready/White Wagtail/w.wav", background="Daurian Redstart"),
            row("Daurian Redstart", "Phoenicurus auroreus", 101, status="Pending")])
        self.assertEqual(upload_sheet.clips([summary])[0]["BirdNET also heard"], "")

    def test_rejected_species_is_not_suggested(self):
        summary = self.result("0623_v3pruned", [
            row("White Wagtail", "Motacilla alba", 100, "Ready/White Wagtail/w.wav"),
            row("Daurian Redstart", "Phoenicurus auroreus", 101, status="Rejected")])
        self.assertEqual(upload_sheet.clips([summary])[0]["BirdNET also heard"], "")

    def test_no_field_log_means_unknown_not_missing(self):
        self.for_recording.return_value = []
        summary = self.result("0623_v3pruned", [row("White Wagtail", "Motacilla alba", 100, "Ready/White Wagtail/w.wav",
                                                    background="Oriental Magpie")])
        self.assertEqual(upload_sheet.clips([summary])[0]["Add to checklist first"], "")

    def test_each_recording_is_checked_against_its_own_checklist(self):
        hill = {"id": "S0", "species": [{"common": "Daurian Redstart"}]}
        first, second = {"id": "S1", "species": [{"common": "Asian Tit"}]}, hill
        self.for_recording.side_effect = lambda start: [first] if start.hour < 8 else [second]
        summary = self.result("0623_v3pruned", [
            row("Asian Tit", "Parus cinereus", 10, "Ready/Asian Tit/a.wav", clock="07:00:10",
                background="Daurian Redstart"),
            row("Asian Tit", "Parus cinereus", 20, "Ready/Asian Tit/b.wav", clock="09:00:20",
                background="Daurian Redstart")])
        clips = upload_sheet.clips([summary])
        self.assertEqual([(c["Checklist"], c["File"], c["Add to checklist first"]) for c in clips],
                         [("S0", "b.wav", "Asian Tit"),                        # เรียงตาม checklist: อัปโหลดทีละ checklist
                          ("S1", "a.wav", "Daurian Redstart")])

    def test_old_layout_keeps_ready_inside_the_session(self):
        folder = self.day.parent / "2026.10.30_0623_v3pruned"
        (folder / "Ready").mkdir(parents=True)
        pd.DataFrame([row("Asian Tit", "Parus cinereus", 9, "Ready/Asian Tit/a.wav")]).to_excel(
            folder / "summary.xlsx", index=False)
        path, clips = upload_sheet.write(folder / "summary.xlsx")
        self.assertEqual(path, folder / "Ready" / "upload-sheet.csv")
        self.assertEqual(len(clips), 1)

    def test_nothing_approved_writes_no_file(self):
        summary = self.result("0623_v3pruned", [row("Asian Tit", "Parus cinereus", 9, status="Pending")])
        path, clips = upload_sheet.write(summary)
        self.assertEqual(clips, [])
        self.assertFalse(path.exists())

    def test_button_opens_the_sheet(self):
        import birdnet_gui as g
        summary = self.result("0623_v3pruned", [row("Asian Tit", "Parus cinereus", 9, "Ready/Asian Tit/a.wav")])
        told = []
        app = SimpleNamespace(review_summary=SimpleNamespace(get=lambda: str(summary)),
                              status=SimpleNamespace(set=told.append))
        with patch.object(g, "open_path") as opened, patch.object(g.messagebox, "showinfo") as info:
            g.App.open_upload_sheet(app)
        opened.assert_called_once_with(self.day / "Ready" / "upload-sheet.csv")
        info.assert_not_called()
        self.assertTrue(told[0].startswith("Upload sheet: 1 clips"))


if __name__ == "__main__":
    unittest.main()
