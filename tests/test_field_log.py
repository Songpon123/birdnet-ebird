"""บันทึกภาคสนาม (eBird checklist จากโฟลเดอร์ Log ของโปรแกรมจัดรูป) — field_log.py"""

from datetime import date, datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import field_log  # noqa: E402


def write_log(root, day, checklist_id, species, runs=("run1",)):
    """<root>/<ปี>/<เดือน>/<YYYYMMDD>/Log/<รอบ>/checklist.json แบบที่โปรแกรมจัดรูปเขียน"""
    log = root / f"{day.year}" / field_log.MONTHS[day.month - 1] / day.strftime("%Y%m%d") / "Log"
    for run in runs:
        (log / run).mkdir(parents=True)
        (log / run / "checklist.json").write_text(json.dumps({
            "schema_version": 1, "checklist_id": checklist_id, "url": f"https://ebird.org/checklist/{checklist_id}",
            "date": f"{day:%d %b %Y} 8:08 AM", "location": "武汉大学 (Wuhan University), Hubei, CN",
            "species": [{"species_code": code, "common_name": common, "scientific_name": scientific, "count": count,
                         "media": [{"asset_id": str(n), "media_type": "photo"} for n in range(photos)]}
                        for code, common, scientific, count, photos in species]}), encoding="utf-8")
    return log


class FieldLogTests(unittest.TestCase):
    def test_add_log_then_find_other_days_under_the_same_birding_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root, store = Path(temp) / "Birding", Path(temp) / "data" / "field-logs.json"
            log = write_log(root, date(2026, 10, 25), "S200000001",
                            [("swiwhe1", "Swinhoe's White-eye", "Zosterops simplex", "13", 2),
                             ("spodov2", "Spotted Dove (Eastern)", "Spilopelia chinensis chinensis/tigrina", "5", 0)],
                            runs=("20261025T115911", "20261025T120222"))             # ส่งซ้ำ 2 รอบ = checklist เดียว
            write_log(root, date(2026, 10, 26), "S200000002",
                      [("chfwhe1", "Chestnut-flanked White-eye", "Zosterops erythropleurus", "2", 1)])
            self.assertEqual(field_log.birding_root(log), str(root.resolve()))
            self.assertIsNone(field_log.birding_root(Path(temp)))
            added = field_log.add(log, store)
            self.assertEqual([c["id"] for c in added], ["S200000001"])
            day = field_log.for_day(datetime(2026, 10, 25, 9, 46), store)
            self.assertEqual(field_log.describe_day(day), "S200000001 25 Oct 08:08 武汉大学 (Wuhan University), Hubei, CN")
            self.assertEqual(field_log.describe_species(day, "Zosterops simplex", "Swinhoe's White-eye"),
                             "Swinhoe's White-eye: on your checklist S200000001 ×13, 2 photos")
            self.assertEqual(field_log.describe_species(day, "Spilopelia chinensis"),
                             "Spotted Dove (Eastern): on your checklist S200000001 ×5")
            self.assertEqual(field_log.describe_species(day, "Zosterops erythropleurus", "Chestnut-flanked White-eye"),
                             "Chestnut-flanked White-eye: not on your eBird checklist S200000001")
            later = field_log.for_day(date(2026, 10, 26), store)                   # ไม่ได้กดเพิ่ม: หาเองใต้ Birding
            self.assertEqual([c["id"] for c in later], ["S200000002"])
            self.assertEqual(field_log.for_day(date(2026, 10, 21), store), [])
            self.assertEqual(field_log.describe_species([], "Zosterops simplex"), "")
            self.assertEqual(field_log.add(Path(temp) / "nothing-here", store), [])


    def test_log_under_pictures_is_found_and_the_newest_run_gives_the_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root, store = Path(temp) / "Birding", Path(temp) / "field-logs.json"
            store.write_text(json.dumps({"roots": [str(root)]}), encoding="utf-8")
            log = root / "2026/Oct/20261030/Pictures/Done/Done/Log"
            for run, place in (("20261031T043918", None),
                               ("20261101T033037", {"latitude": 29.117096, "longitude": 110.479191, "status": "available"})):
                (log / run).mkdir(parents=True)
                data = {"checklist_id": "S1", "date": "30 Oct 2026 6:36 AM", "species": []}
                (log / run / "checklist.json").write_text(json.dumps({**data, **({"location_details": place} if place else {})}),
                                                          encoding="utf-8")
            found = field_log.for_day(date(2026, 10, 30), store)
            self.assertEqual([(c["id"], c["lat"], c["lon"]) for c in found], [("S1", 29.117096, 110.479191)])

    def test_month_folder_may_be_in_capitals(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "Birding" / "2026" / "NOV" / "20261102" / "Log"
            folder.mkdir(parents=True)
            self.assertEqual(field_log.birding_root(folder), str((Path(temp) / "Birding").resolve()))

    def test_log_without_usable_coordinates_gives_none(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checklist.json"
            for place in (None, {"latitude": None, "longitude": None, "status": "unavailable"},
                          {"latitude": "x", "longitude": 500}):
                path.write_text(json.dumps({"checklist_id": "S1", "date": "30 Oct 2026 6:36 AM", "species": [],
                                            "location_details": place}), encoding="utf-8")
                checklist = field_log.read_checklist(path)
                self.assertEqual((checklist["lat"], checklist["lon"]), (None, None), place)

    def test_settings_tab_adds_a_field_log(self):
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp, patch.object(field_log, "STORE", Path(temp) / "field-logs.json"), \
                patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {"output_dir": temp}), \
                patch.object(g.App, "_start", lambda self, *a, **k: None):
            log = write_log(Path(temp) / "Birding", date(2026, 10, 25), "S200000001",
                            [("swiwhe1", "Swinhoe's White-eye", "Zosterops simplex", "13", 2)])
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"no display: {exc}")
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                app = g.App(root)
                root.createcommand("::tk::mac::Quit", lambda: None)               # App ตั้งของตัวเองแทน: ปิดแล้วอย่าให้เรียก
                self.assertTrue(app.field_log_text.get().startswith("Paste an eBird checklist link"))
                with patch.object(g.filedialog, "askdirectory", return_value=str(log)), \
                        patch("field_log_window.RecordingChecklists") as matching:
                    app.add_field_log()
                self.assertEqual(matching.call_args.args[2], date(2026, 10, 25))     # drop down เด้งขึ้นที่วันของ Log
                text = app.field_log_text.get()
                self.assertTrue(text.startswith("1 checklist(s), latest: S200000001 25 Oct 2026 武汉大学"), text)
                self.assertIn(f"other days found automatically under {(Path(temp) / 'Birding').resolve()}", text)
                self.assertIn("S200000001 (2026-10-25)", app.status.get())
            finally:
                root.destroy()

if __name__ == "__main__":
    unittest.main()
