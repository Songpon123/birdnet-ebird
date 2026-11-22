"""Auto-checks ต้องไม่ทำงานซ้ำ: ฟังแต่ละคลิปรอบเดียว ค้น xeno-canto ครั้งเดียวต่อชนิด และโหลดไฟล์อ้างอิงพร้อมกัน"""

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import auto_checks  # noqa: E402
import second_opinion  # noqa: E402
import sound_type  # noqa: E402
import xeno_canto  # noqa: E402

RATE = 48000
LABELS = ["Phylloscopus inornatus_Yellow-browed Warbler", "Phylloscopus humei_Hume's Leaf Warbler"]


def wav(path, seconds=6):
    sf.write(path, np.zeros(RATE * seconds, np.float32), RATE)


class Result:
    def __init__(self, rows):
        self.rows = rows

    def to_structured_array(self):
        return self.rows


class Model:
    species_list = LABELS

    def __init__(self):
        self.heard, self.encoded = [], []

    def predict(self, paths, **kwargs):
        wanted = kwargs.get("custom_species_list")
        if not wanted:
            self.heard.append(sorted(Path(path).name for path in paths))
        return Result([{"input": path, "start_time": start, "end_time": start + 3, "species_name": label, "confidence": 0.5}
                       for path in paths for start in (0.0, 1.5) for label in wanted or LABELS])

    def encode(self, paths, **_kwargs):
        if not any("XC" in str(path) for path in paths):
            self.encoded.append(sorted(Path(path).name for path in paths))
        return Result([{"input": path, "start_time": start, "embedding": np.eye(4)[0]}
                       for path in paths for start in (0.0, 1.5)])


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.searches, self.downloads = [], []
        patches = (patch.object(second_opinion, "CACHE", self.temp / "second"),
                   patch.object(auto_checks, "CACHE", self.temp / "auto"),
                   patch.object(sound_type, "CACHE", self.temp / "types"),
                   patch.object(auto_checks, "_typed_references", lambda *_args: 0),
                   patch.object(auto_checks, "all_results", return_value=[]),
                   patch.object(sound_type, "train", lambda *_args, **_kwargs: None),
                   patch.object(xeno_canto, "search", self.fake_search),
                   patch.object(xeno_canto, "download", self.fake_download))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_search(self, scientific, _key, **_kwargs):
        self.searches.append(scientific)
        return [{"id": f"XC{scientific.split()[1]}{n}", "file": "https://x", "length": "0:30", "ext": ".mp3"}
                for n in range(2)], 2

    def fake_download(self, recording):
        self.downloads.append(recording["id"])
        path = self.temp / f"{recording['id']}.wav"
        wav(path)
        return path

    def result(self):
        day = self.temp / "2026.10.30_0634"
        day.mkdir()
        rows = []
        for n, score in enumerate((0.5, 0.55, 0.6, 0.9)):
            wav(day / f"{n}.wav")
            rows.append({"File": f"{n}.wav", "Generated sha256": f"s{n}", "Species (common)": "Yellow-browed Warbler",
                         "Species (scientific)": "Phylloscopus inornatus", "Max confidence": score})
        pd.DataFrame(rows).to_excel(day / "summary.xlsx", index=False)
        return day / "summary.xlsx"

    def test_each_clip_is_heard_once_and_each_species_searched_once(self):
        model = Model()
        summary = self.result()
        auto_checks.run(summary, "key", analyzer=types.SimpleNamespace(model=model), log=lambda _text: None)
        every = ["0.wav", "1.wav", "2.wav", "3.wav"]
        self.assertEqual(model.heard, [every])                     # รุ่นก่อน: ฟังซ้ำทีละคลิปอีก 3 รอบ
        self.assertEqual(model.encoded, [every])
        self.assertEqual(sorted(self.searches), ["Phylloscopus humei", "Phylloscopus inornatus"])
        self.assertEqual(len(self.downloads), 4)
        self.assertIsNotNone(second_opinion.cached_result("s2"))
        self.assertIsNone(second_opinion.cached_result("s3"))      # คะแนนสูง: ไม่ต้องใช้ second opinion

    def test_resuming_listens_to_the_remaining_clips_together(self):
        model = Model()
        summary = self.result()
        analyzer = types.SimpleNamespace(model=model)
        auto_checks.run(summary, "key", analyzer=analyzer, log=lambda _text: None)
        for sha in ("s0", "s1", "s2"):                             # หยุดก่อนถึง second opinion
            second_opinion.result_path(sha).unlink()
        model.heard, model.encoded, self.searches = [], [], []
        auto_checks.run(summary, "key", analyzer=analyzer, log=lambda _text: None)
        self.assertEqual(model.heard, [["0.wav", "1.wav", "2.wav"]])
        self.assertEqual(model.encoded, [["0.wav", "1.wav", "2.wav"]])
        self.assertEqual(self.searches, [])                        # จำผลค้นไว้แล้ว
        self.assertIsNotNone(second_opinion.cached_result("s0"))

    def test_clips_that_cannot_be_heard_together_are_tried_one_by_one(self):
        model = Model()
        summary = self.result()
        real = model.predict

        def fussy(paths, **kwargs):
            if len(paths) > 1 and kwargs.get("top_k") == 10:
                raise RuntimeError("one file is broken")
            return real(paths, **kwargs)

        model.predict = fussy
        analyzer = types.SimpleNamespace(model=model)
        with patch.object(auto_checks, "needs_check", lambda *_args: True):
            auto_checks.run(summary, "key", analyzer=analyzer, log=lambda _text: None)
            for sha in ("s0", "s1", "s2"):
                second_opinion.result_path(sha).unlink()
            auto_checks._save("s0", {"nonbird": None})
            auto_checks._save("s1", {"nonbird": None})
            auto_checks._save("s2", {"nonbird": None})
            auto_checks.run(summary, "key", analyzer=analyzer, log=lambda _text: None)
        self.assertIsNotNone(second_opinion.cached_result("s2"))

    def test_listening_to_more_species_does_not_add_candidates(self):
        """ฟังมา 25 อันดับ (ไว้หาเสียงที่ไม่ใช่นก) ต้องได้ตัวเลือกเหมือนฟัง 10 อันดับ แม้ตัวกรองพื้นที่ตัดชนิดอันดับต้นออก"""
        names = [f"Avis species{n}" for n in range(12)]
        labels = [f"{name}_Bird {n}" for n, name in enumerate(names)]
        model = Model()
        model.species_list = labels
        lucky = names[11]                                          # อันดับ 12 ของช่วงนี้ ผ่านตัวกรองพื้นที่
        analyzer = types.SimpleNamespace(model=model, _geo_probabilities=lambda *_args: (
            {names[0]: 1.0, lucky: 1.0}, {}))
        day = self.temp / "2026.10.30_0634"
        day.mkdir()
        wav(day / "a.wav")
        row = {"File": "a.wav", "Generated sha256": "s9", "Species (common)": "Bird 0", "Species (scientific)": names[0],
               "Latitude": 29.3, "Longitude": 110.3, "Analysis settings": json.dumps({"date": "2026-10-30"}),
               "_row": 2}
        pd.DataFrame([row]).drop(columns="_row").to_excel(day / "summary.xlsx", index=False)
        heard = [(0.0, 3.0, label, 0.9 - n * 0.05) for n, label in enumerate(labels)]
        result = second_opinion.run(day / "summary.xlsx", 2, "key", log=lambda _t: None, analyzer=analyzer, row=row,
                                    heard=heard, encoded={0.0: np.eye(4)[0]})
        self.assertEqual([item["scientific"] for item in result["ranking"]], [names[0]])

    def test_search_is_remembered_by_place_then_refreshed(self):
        first = second_opinion.chosen_references("Parus cinereus", "Parus cinereus_Asian Tit", "key", 29.3, 110.3)
        self.assertEqual(second_opinion.chosen_references("Parus cinereus", "x", "key", 29.4, 110.4), first)
        self.assertEqual(len(self.searches), 1)                    # จุดใกล้กัน
        second_opinion.chosen_references("Parus cinereus", "x", "key", 13.7, 100.5)
        self.assertEqual(len(self.searches), 2)                    # คนละที่
        path = next((self.temp / "second" / "searches").glob("*_29_110.json"))
        old = time.time() - (second_opinion.SEARCH_DAYS + 1) * 86400
        os.utime(path, (old, old))
        second_opinion.chosen_references("Parus cinereus", "x", "key", 29.3, 110.3)
        self.assertEqual(len(self.searches), 3)                    # เก่าเกิน: ค้นใหม่

    def test_failed_search_is_not_remembered(self):
        with patch.object(xeno_canto, "search", side_effect=xeno_canto.XenoCantoError("down")):
            with self.assertRaises(xeno_canto.XenoCantoError):
                second_opinion.chosen_references("Parus cinereus", "x", "key", 29.3, 110.3)
        self.assertEqual(second_opinion.chosen_references("Parus cinereus", "x", "key", 29.3, 110.3)[0]["id"],
                         "XCcinereus0")

    def test_references_download_at_the_same_time_and_keep_their_order(self):
        together = threading.Barrier(second_opinion.DOWNLOADS, timeout=5)

        def download(recording):
            if recording["id"] != "XC3":
                together.wait()                                    # ค้างถ้าโหลดทีละไฟล์
            path = self.temp / f"{recording['id']}.wav"
            wav(path)
            return path

        recordings = [{"id": f"XC{n}", "file": "https://x"} for n in range(4)]
        seen = []
        real = second_opinion._analyze

        def analyze(model, label, paths, log=print):
            seen.extend(Path(path).stem for path in paths)
            return real(model, label, paths, log)

        with patch.object(xeno_canto, "download", download), patch.object(second_opinion, "_analyze", analyze):
            found, used = second_opinion.embed_references(Model(), "Parus cinereus", LABELS[0], recordings, log=lambda _t: None)
        self.assertEqual(seen, ["XC0", "XC1", "XC2", "XC3"])
        self.assertEqual(used, 4)
        self.assertEqual(len(found), 8)

    def test_bad_reference_is_skipped_without_stopping_the_others(self):
        def download(recording):
            if recording["id"] == "XC1":
                raise xeno_canto.XenoCantoError("gone")
            path = self.temp / f"{recording['id']}.wav"
            wav(path)
            return path

        told = []
        with patch.object(xeno_canto, "download", download):
            _found, used = second_opinion.embed_references(Model(), "Parus cinereus", LABELS[0],
                                                           [{"id": f"XC{n}", "file": "x"} for n in range(3)], log=told.append)
        self.assertEqual(used, 2)
        self.assertTrue(any("XC1" in text for text in told))


if __name__ == "__main__":
    unittest.main()
