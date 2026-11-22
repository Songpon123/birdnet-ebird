"""วางลิงก์ checklist แล้วดึงข้อมูลจาก eBird แทนการเพิ่มโฟลเดอร์ Log — API จำลอง ไม่ต่อเน็ต"""

from datetime import date, datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ebird  # noqa: E402
import field_log  # noqa: E402

VIEW = {"subId": "S300000001", "locId": "L1000001", "obsDt": "2026-10-30 06:36", "durationHrs": 2.833,
        "subnational1Code": "CN-43", "obs": [
            {"speciesCode": "spodov2", "howManyStr": "5", "mediaCounts": {"P": 2, "A": 1}},
            {"speciesCode": "owl1", "howManyStr": "X"}]}
TAXONOMY = [{"speciesCode": "spodov2", "comName": "Spotted Dove (Eastern)", "sciName": "Spilopelia chinensis chinensis/tigrina"},
            {"speciesCode": "owl1", "comName": "owl sp.", "sciName": "Strigiformes sp."}]
FEED = [{"locId": "L1", "loc": {"lat": 1.0, "lng": 2.0}},
        {"locId": "L1000001", "loc": {"name": "教字垭镇 (Jiaoziya Town)", "lat": 29.117096, "lng": 110.479191,
                                       "hierarchicalName": "教字垭镇 (Jiaoziya Town), Hunan, CN"}}]


def api(path, params=None, key=None, timeout=30):
    if path.startswith("product/checklist/view/"):
        if path.endswith("S999999"):
            raise ebird.EBirdError("eBird returned HTTP 400")
        return dict(VIEW)
    if path == "ref/taxonomy/ebird":
        return TAXONOMY
    if path.startswith("product/lists/"):
        return FEED
    raise ebird.EBirdError("Cannot reach eBird")


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.store = self.temp / "field-logs.json"
        for patcher in (patch.object(field_log, "STORE", self.store), patch.object(ebird, "_get", api)):
            patcher.start()
            self.addCleanup(patcher.stop)


class ReadingTests(Case):
    def test_finds_every_checklist_in_what_was_pasted(self):
        text = "https://ebird.org/checklist/S200000001 and\nebird.org/checklist/S300000001?x=1 S200000001 Sound123"
        self.assertEqual(field_log.checklist_ids(text), ["S200000001", "S300000001"])
        self.assertEqual(field_log.checklist_ids("no link here"), [])

    def test_link_gives_species_counts_photos_time_place_and_length(self):
        field_log.add_links("https://ebird.org/checklist/S300000001", "key")
        checklist = field_log.for_day(date(2026, 10, 30))[0]
        self.assertEqual((checklist["date"], checklist["location"], checklist["lat"], checklist["lon"], checklist["hours"]),
                         ("2026-10-30T06:36", "教字垭镇 (Jiaoziya Town), Hunan, CN", 29.117096, 110.479191, 2.833))
        dove, owl = checklist["species"]
        self.assertEqual((dove["common"], dove["count"], dove["photos"], dove["media"]), ("Spotted Dove (Eastern)", "5", 2, 3))
        self.assertEqual((owl["common"], owl["count"], owl["photos"]), ("owl sp.", "X", 0))
        self.assertEqual(field_log.describe_species([checklist], "Spilopelia chinensis", "Spotted Dove"),
                         "Spotted Dove: on your checklist S300000001 ×5, 2 photos")

    def test_personal_place_missing_from_the_feed_still_adds_the_checklist(self):
        with patch.object(ebird, "_get", lambda path, *a, **k: [] if path.startswith("product/lists/") else api(path, *a, **k)):
            added, problems = field_log.add_links("S300000001", "key")
        self.assertEqual(problems, [])
        self.assertEqual((added[0]["lat"], added[0]["location"]), (None, ""))

    def test_a_bad_link_is_reported_and_the_good_one_is_kept(self):
        added, problems = field_log.add_links("S999999 S300000001", "key")
        self.assertEqual([c["id"] for c in added], ["S300000001"])
        self.assertIn("S999999", problems[0])


class LogTests(Case):
    def log(self, species=("yebbun", "Yellow-browed Bunting", "Emberiza chrysophrys", "1", 0)):
        from test_field_log import write_log
        root = self.temp / "Birding"
        folder = write_log(root, date(2026, 10, 30), "S300000001", [species])
        (folder / "run1" / "files.csv").write_text(
            "filename,capture_time,common_name\na.jpg,2026-10-30T07:00:00+08:00,Eurasian Jay\n", encoding="utf-8")
        return folder

    def test_link_updates_species_and_keeps_the_photo_list_from_the_log(self):
        field_log.add(self.log())
        field_log.add_links("S300000001", "key")
        checklist = field_log.for_day(date(2026, 10, 30))[0]
        self.assertEqual([item["code"] for item in checklist["species"]], ["spodov2", "owl1"])   # ล่าสุดจาก eBird
        self.assertEqual(checklist["hours"], 2.833)
        self.assertEqual(len(checklist["photos"]), 1)                                            # เวลาถ่ายจาก Log

    def test_adding_the_log_again_does_not_undo_the_link(self):
        folder = self.log()
        field_log.add_links("S300000001", "key")
        field_log.add(folder)
        self.assertEqual([item["code"] for item in field_log.for_day(date(2026, 10, 30))[0]["species"]], ["spodov2", "owl1"])

    def test_without_a_link_the_log_still_decides(self):
        field_log.add(self.log())
        self.assertEqual(field_log.for_day(date(2026, 10, 30))[0]["species"][0]["code"], "yebbun")


class LengthTests(Case):
    def test_known_length_replaces_the_six_hour_guess(self):
        field_log.add_links("S300000001", "key")                                 # 06:36 + 2.833 ชม. = 09:26
        self.assertIsNotNone(field_log.spot_for(datetime(2026, 10, 30, 9, 30)))  # เผื่อ 15 นาที
        self.assertIsNone(field_log.spot_for(datetime(2026, 10, 30, 10, 0)))     # หลังจบ checklist แล้ว


class SettingsTests(Case):
    def test_pasting_a_link_adds_it_and_opens_match_recordings(self):
        import tkinter as tk
        import time
        import birdnet_gui as g
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        try:
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                    patch.object(g.App, "_start", lambda *_args, **_kw: None), \
                    patch("field_log_window.RecordingChecklists") as matching:
                app = g.App(root)
                root.createcommand("::tk::mac::Quit", lambda: None)
                with patch.object(g.messagebox, "showwarning") as warned:
                    app.checklist_link.set("https://ebird.org/checklist/S300000001")
                    app.add_checklist_links()                                    # ยังไม่มี key
                    warned.assert_called_once()
                app.ebird_key.set("key")
                app.add_checklist_links()
                for _ in range(200):
                    root.update()
                    time.sleep(.01)
                    if matching.called:
                        break
                self.assertEqual(matching.call_args.args[2], date(2026, 10, 30))
                self.assertEqual(app.checklist_link.get(), "")
                self.assertIn("S300000001", app.field_log_text.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
