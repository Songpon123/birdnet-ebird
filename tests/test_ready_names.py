"""ชื่อไฟล์ใน Ready บอกชนิดที่ติดมา: ดูใน Finder ได้ และเรียงรวมกันท้ายโฟลเดอร์ชนิด"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from review_store import load_reviews, save_review, with_part  # noqa: E402

NAMES = {"common": "Common Kingfisher", "scientific": "Alcedo atthis"}


class ReadyNameTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.day = Path(temporary.name) / "2026.10.30"
        self.folder = self.day / "0634_v3pruned"
        self.folder.mkdir(parents=True)
        (self.folder / "a.wav").write_bytes(b"a")
        (self.folder / "b.wav").write_bytes(b"b")
        self.summary = self.folder / "summary.xlsx"
        pd.DataFrame([{"File": "a.wav", "Species (common)": NAMES["common"]},
                      {"File": "b.wav", "Species (common)": NAMES["common"]}]).to_excel(self.summary, index=False)

    def approve(self, row=2, **kwargs):
        return save_review(self.summary, row, status="Approved", rating=3, **NAMES, **kwargs)

    def test_with_part(self):
        self.assertEqual(with_part(""), "")
        self.assertEqual(with_part(None), "")
        self.assertEqual(with_part("Daurian Redstart"), "with-Daurian-Redstart__")
        self.assertEqual(with_part("White Wagtail; Light-vented Bulbul"), "with-Light-vented-Bulbul_White-Wagtail__")
        self.assertEqual(with_part("Richard's Pipit"), "with-Richard_s-Pipit__")
        self.assertEqual(with_part("A b; C d; E f; G h; I j"), "with-A-b_C-d_E-f_and-2-more__")

    def test_clip_without_other_species_keeps_its_name(self):
        self.assertTrue(Path(self.approve()).name.startswith("2026.10.30_0634_v3pruned_"))

    def test_name_shows_other_species_and_groups_after_plain_clips(self):
        plain = self.approve()
        mixed = self.approve(row=3, background="Daurian Redstart")
        self.assertTrue(Path(mixed).name.startswith("with-Daurian-Redstart__2026.10.30_0634_v3pruned_"))
        self.assertEqual(Path(mixed).parent.name, "Common Kingfisher")
        self.assertTrue((self.day / mixed).is_file())
        self.assertEqual(sorted([Path(mixed).name, Path(plain).name]), [Path(plain).name, Path(mixed).name])

    def test_changing_other_species_renames_the_ready_file(self):
        first = self.approve(background="Daurian Redstart")
        second = self.approve(background="Gray Wagtail")
        self.assertFalse((self.day / first).exists())
        self.assertTrue(Path(second).name.startswith("with-Gray-Wagtail__"))
        third = self.approve(background="")
        self.assertFalse((self.day / second).exists())
        self.assertTrue(Path(third).name.startswith("2026.10.30_"))

    def test_saving_without_touching_other_species_keeps_them_in_the_name(self):
        self.approve(background="Daurian Redstart")
        again = save_review(self.summary, 2, status="Approved", rating=4, sound_type="Call", **NAMES)
        self.assertTrue(Path(again).name.startswith("with-Daurian-Redstart__"))
        self.assertEqual(len(list((self.day / "Ready").rglob("*.wav"))), 1)
        self.assertEqual(load_reviews(self.summary)[0]["Background species"], "Daurian Redstart")


if __name__ == "__main__":
    unittest.main()
