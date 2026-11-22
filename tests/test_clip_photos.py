"""รูปของคุณที่ถ่ายตอนได้ยินเสียงของคลิป: เวลาใน EXIF, ภาพเดียวกันหลายไฟล์นับใบเดียว, เวลากล้อง, ชื่อชนิดจาก Log"""

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import clip_photos  # noqa: E402
import field_log  # noqa: E402

ROW = {"Analysis settings": json.dumps({"date": "2026-10-30"}), "Clock time": "07:01:50", "Start offset (s)": 100.0,
       "End offset (s)": 106.0}                                        # เสียง 07:01:50–07:01:56


def picture(path, taken):
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[306] = taken
    Image.new("RGB", (300, 200), (90, 120, 160)).save(path, exif=exif)


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.day = self.temp / "Birding" / "2026" / "Oct" / "20261030"
        self.summary = self.day / "Sounds" / "Done" / "2026.10.30" / "0634_v3pruned" / "summary.xlsx"
        for patcher in (patch.object(clip_photos, "CACHE", self.temp / "cache"),
                        patch.object(field_log, "STORE", self.temp / "field-logs.json"),
                        patch.object(clip_photos, "_exiftool_times", lambda _files: {})):
            patcher.start()
            self.addCleanup(patcher.stop)
        pictures = self.day / "Pictures"
        picture(pictures / "Done" / "20261030-4X8A0001.jpg", "2026:10:30 07:01:53")    # ระหว่างเสียง
        picture(pictures / "DxO" / "4X8A0001.tif", "2026:10:30 07:01:53")             # ภาพเดียวกัน
        picture(pictures / "Done" / "20261030-4X8A0002.jpg", "2026:10:30 07:03:00")    # 64 วินาทีหลังเสียง
        picture(pictures / "Done" / "20261030-4X8A0003.jpg", "2026:10:30 07:20:00")    # ไกลเกิน

    def test_photos_during_and_near_the_sound(self):
        found = clip_photos.photos_at(ROW, self.summary)
        self.assertEqual([(away, photo["file"]) for away, photo in found],
                         [(0.0, "20261030-4X8A0001.jpg"), (64.0, "20261030-4X8A0002.jpg")])   # jpg ก่อน tif
        self.assertEqual(clip_photos.describe_away(0.0), "during the sound")
        self.assertEqual(clip_photos.describe_away(-12.0), "12 s before")

    def test_camera_clock_offset_moves_the_photos(self):
        found = clip_photos.photos_at(ROW, self.summary, camera_offset=70.0)   # กล้องเร็วกว่า 70 วินาที
        self.assertEqual([(away, photo["file"]) for away, photo in found],
                         [(0.0, "20261030-4X8A0002.jpg"), (-67.0, "20261030-4X8A0001.jpg")])

    def test_species_name_comes_from_the_log(self):
        with patch.object(field_log, "photos_on", return_value=[
                {"time": "2026-10-30T07:01:53", "file": "20261030-4X8A0001.jpg", "path": "", "common": "Eurasian Jay"}]):
            found = clip_photos.photos_at(ROW, self.summary)
        self.assertEqual(found[0][1]["common"], "Eurasian Jay")
        self.assertEqual(found[1][1]["common"], "")

    def test_photos_moved_to_google_drive_come_from_the_log(self):
        drive = self.temp / "CloudStorage" / "GoogleDrive-me" / "My Drive" / "Pictures" / "By Years " / "2026"
        moved = drive / "20261030_Hunan" / "Pictures" / "Done" / "Black Stork" / "20261030-4X8A0050.jpg"
        picture(moved, "2026:10:30 07:02:10")
        logged = [{"time": "2026-10-30T07:02:10", "file": "20261030-4X8A0050.jpg", "common": "Black Stork",
                   "path": str(self.day / "Pictures" / "Done" / "20261030-4X8A0050.jpg")},          # ไม่อยู่แล้ว
                  {"time": "2026-10-30T07:02:20", "file": "4X8A0060.CR3", "common": "", "path": ""}]
        with patch.object(clip_photos, "CLOUD", self.temp / "CloudStorage"), \
                patch.object(field_log, "photos_on", return_value=logged):
            found = {photo["file"]: (away, photo) for away, photo in clip_photos.photos_at(ROW, self.summary)}
        self.assertEqual(found["20261030-4X8A0050.jpg"][0], 14.0)
        self.assertEqual(found["20261030-4X8A0050.jpg"][1]["path"], str(moved))
        self.assertEqual(found["20261030-4X8A0050.jpg"][1]["common"], "Black Stork")
        self.assertEqual(found["4X8A0060.CR3"][1]["path"], "")                     # หาไฟล์ไม่เจอ: ยังบอกเวลาได้

    def test_photo_times_are_read_once(self):
        clip_photos.photos_at(ROW, self.summary)
        with patch.object(clip_photos, "_pil_time", side_effect=AssertionError("read again")):
            self.assertEqual(len(clip_photos.photos_at(ROW, self.summary)), 2)

    def test_raw_files_use_exiftool_when_present(self):
        raw = self.day / "Pictures" / "Raw" / "4X8A0009.CR3"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"not readable by PIL")
        with patch.object(clip_photos, "_exiftool_times", lambda files: {str(f): "2026-10-30T07:01:55" for f in files
                                                                         if f.suffix == ".CR3"}):
            found = clip_photos.photos_at(ROW, self.summary)
        self.assertIn("4X8A0009.CR3", [photo["file"] for _away, photo in found])

    def test_no_time_or_no_pictures_shows_nothing(self):
        self.assertEqual(clip_photos.photos_at({}, self.summary), [])
        self.assertEqual(clip_photos.photos_at(ROW, self.temp / "elsewhere" / "summary.xlsx"), [])

    def test_species_folder_names_the_photo(self):
        sorted_photo = self.day / "Pictures" / "Done" / "Long-tailed Shrike" / "20261030-4X8A0070.jpg"
        picture(sorted_photo, "2026:10:30 07:01:58")
        found = {photo["file"]: photo for _away, photo in clip_photos.photos_at(ROW, self.summary)}
        self.assertEqual(found["20261030-4X8A0070.jpg"]["common"], "Long-tailed Shrike")
        for path, name in (("x/_Unidentified/a.jpg", ""), ("x/Done/a.jpg", ""), ("x/20261106_Wuhan/a.jpg", ""),
                           ("x/Pallas's Leaf Warbler/a.jpg", "Pallas's Leaf Warbler")):
            self.assertEqual(clip_photos.folder_species(path), name)

    def test_one_line_of_species(self):
        found = [(0.0, {"common": "Black Stork"}), (-12.0, {"common": "Black Stork"}), (40.0, {"common": ""}),
                 (102.0, {"common": "Graylag Goose"})]
        self.assertEqual(clip_photos.describe(found), "📷 Your photos then (within 2 min): Black Stork ×2 (during the sound)"
                         " · Graylag Goose (102 s after) · 1 not sorted")
        self.assertEqual(clip_photos.describe([]), "")


class ReviewTests(Case):
    def test_review_names_the_species_photographed_then(self):
        import tkinter as tk
        import time
        import birdnet_gui as g
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        self.addCleanup(root.destroy)
        with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                patch.object(g.App, "_start", lambda *_args, **_kw: None), \
                patch.object(field_log, "photos_on", return_value=[
                    {"time": "2026-10-30T07:01:53", "file": "20261030-4X8A0001.jpg", "path": "", "common": "Eurasian Jay"}]):
            app = g.App(root)
            root.createcommand("::tk::mac::Quit", lambda: None)
            app.review_summary.set(str(self.summary))
            app._show_review_photos(ROW)
            for _ in range(300):
                root.update()
                time.sleep(.01)
                if app.photo_text.get():
                    break
            self.assertEqual(app.photo_text.get(), "📷 Your photos then (within 2 min): Eurasian Jay (during the sound)"
                                                   " · 1 not sorted")
            app._clear_review_preview()
            self.assertEqual(app.photo_text.get(), "")


if __name__ == "__main__":
    unittest.main()
