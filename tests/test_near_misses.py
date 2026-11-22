"""Near misses, second look, expected species and BirdNET labels — fake models, no BirdNET download."""

import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import field_audio_to_ebird as pipeline  # noqa: E402
import near_misses  # noqa: E402
import sound_index  # noqa: E402

SR = 16000
SETTINGS = json.dumps({"date": "2026-10-18", "min_conf": 0.5})


def tone_recording(path, seconds, bursts):
    """เงียบ (มีเสียงรบกวนเบา ๆ) + เสียง 3 kHz สั้น ๆ ที่เวลา bursts (วินาที)"""
    rng = np.random.default_rng(0)
    samples = rng.normal(scale=0.001, size=SR * seconds).astype(np.float32)
    t = np.arange(int(SR * 0.2)) / SR
    for at in bursts:
        first = int((at - 0.1) * SR)
        samples[first:first + len(t)] += 0.5 * np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    sf.write(path, samples, SR)


class CentredModel:
    """BirdNET จำลอง: คะแนน = สัดส่วนพลังงานในวินาทีกลางของหน้าต่าง 3 วิ (เสียงอยู่กลางหน้าต่าง = คะแนนสูง)"""

    def __init__(self, label, scale=1.0):
        self.label, self.scale, self.calls = label, scale, []

    def predict_arrays(self, arrays, **kwargs):
        self.calls.append((len(arrays), kwargs))
        rows = []
        for number, (samples, rate) in enumerate(arrays):
            energy = np.asarray(samples, np.float64) ** 2
            middle = energy[int(rate * 1.0):int(rate * 2.0)].sum() / max(energy.sum(), 1e-12)
            rows.append({"input": number, "species_name": self.label, "confidence": round(self.scale * middle, 3)})
        return types.SimpleNamespace(to_structured_array=lambda: rows)


class SecondLookTests(unittest.TestCase):
    def test_second_look_centres_the_window_on_the_sound(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "rec.wav"
            tone_recording(source, 12, [4.45])          # คร่อมรอยต่อของหน้าต่าง 3.0–6.0 / 4.5–7.5
            model = CentredModel("Motacilla cinerea_Grey Wagtail")
            spans = near_misses.second_look(model, source, [{"start": 3.0, "end": 7.5,
                                                             "label": "Motacilla cinerea_Grey Wagtail"}])
            self.assertGreater(spans[0]["look"], 0.9)
            self.assertAlmostEqual(spans[0]["look_at"], 4.45, delta=0.1)
            self.assertIn(spans[0]["look_how"], ("centred", "isolated"))
            # ถามเฉพาะชนิดเดิม ด้วยเกณฑ์ต่ำ (ได้คะแนนจริงแม้ต่ำ)
            self.assertEqual(model.calls[0][1]["custom_species_list"], ["Motacilla cinerea_Grey Wagtail"])
            self.assertEqual(model.calls[0][1]["default_confidence_threshold"], 0.01)

    def test_describe_look_explains_why_a_clip_was_cut(self):
        self.assertEqual(near_misses.describe_look(0.45, 0.58, "centred"),
                         "Second look: BirdNET 0.45 in its usual 3-s windows, 0.58 for a 3-s window centred on the sound")

    def fake_v3(self, model):
        class FakeV3:
            model_name = "BirdNET 3.0 preview ONNX fp16 + geo 3.0.4 (0.03)"
            direct_prediction = True

            def __init__(self):
                self.model = model

            def detections(self, _path, overlap, floor, *_args):
                self.floor = floor
                found = [
                    ("Pycnonotus sinensis", "Light-vented Bulbul", 0.0, 0.8, True),
                    ("Pycnonotus sinensis", "Light-vented Bulbul", 1.5, 0.4, True),     # ส่วนต่อของคลิปเดิม
                    ("Motacilla cinerea", "Grey Wagtail", 12.0, 0.45, True),           # เกือบเจอ -> ฟังซ้ำผ่าน
                    ("Strix uralensis", "Ural Owl", 24.0, 0.4, False),                  # ไม่ควรพบที่นี่: ไม่ฟังซ้ำ
                ]
                return [{"start_time": start, "end_time": start + 3, "common_name": common,
                         "scientific_name": scientific, "confidence": score,
                         "is_predicted_for_location_and_date": expected, "location_filter_available": True,
                         "birdnet_label": f"{scientific}_{common}"} for scientific, common, start, score, expected in found
                        if score >= floor]
        return FakeV3()

    def run_pipeline(self, model, second_look=True):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            tone_recording(source, 30, [1.0, 13.4, 25.0])
            analyzer = self.fake_v3(model)
            args = dict(lat_arg=30.5, lon_arg=114.4, cfg_lat=None, cfg_lon=None, date_override=datetime(2026, 10, 18),
                        use_meta=False, min_conf=0.5, gap=5, lead=3, tail=3, target_dbfs=-3, fmt="wav",
                        make_mono=True, incl_alt=False, place_arg=None, dt_regex=pipeline.FILENAME_DATETIME_REGEX)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rows, _day = pipeline.process_file(analyzer, source, root / "out", second_look=second_look, **args)
            return rows, analyzer, out.getvalue()

    def test_pipeline_clips_a_near_miss_that_passes_the_second_look(self):
        model = CentredModel("Motacilla cinerea_Grey Wagtail", scale=0.7)
        rows, analyzer, log = self.run_pipeline(model)
        self.assertEqual(analyzer.floor, near_misses.NEAR_FLOOR)
        names = sorted(row["Species (common)"] for row in rows)
        self.assertEqual(names, ["Gray Wagtail", "Light-vented Bulbul"])
        wagtail = next(row for row in rows if row["Species (common)"] == "Gray Wagtail")
        self.assertTrue(wagtail["Found by"].startswith("Second look: BirdNET 0.45"))
        self.assertGreaterEqual(wagtail["Max confidence"], 0.5)
        self.assertEqual(next(row for row in rows if row["Species (common)"] == "Light-vented Bulbul")["Found by"], "")
        self.assertIn('"second_look":0.2', wagtail["Analysis settings"])
        self.assertIn("second look at 1 near miss(es): 1 reached 0.5", log)
        self.assertEqual(len(model.calls), 1)                  # ทั้งไฟล์ถามครั้งเดียว

    def test_pipeline_skips_near_misses_that_stay_low_or_when_turned_off(self):
        rows, _analyzer, _log = self.run_pipeline(CentredModel("Motacilla cinerea_Grey Wagtail", scale=0.3))
        self.assertEqual([row["Species (common)"] for row in rows], ["Light-vented Bulbul"])
        model = CentredModel("Motacilla cinerea_Grey Wagtail", scale=0.7)
        rows, analyzer, _log = self.run_pipeline(model, second_look=False)
        self.assertEqual(analyzer.floor, 0.5)
        self.assertEqual([row["Species (common)"] for row in rows], ["Light-vented Bulbul"])
        self.assertNotIn("second_look", rows[0]["Analysis settings"])
        self.assertEqual(model.calls, [])


class NearMissIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = patch.object(sound_index, "CACHE", self.root / "index")
        self.cache.start()
        sound_index._loaded.clear()
        sound_index._label_lists.clear()

    def tearDown(self):
        self.cache.stop()
        sound_index._loaded.clear()
        sound_index._label_lists.clear()
        self.temp.cleanup()

    def make_result(self):
        source = self.root / "REC-001.WAV"
        tone_recording(source, 30, [7.0, 16.0])
        from field_audio_to_ebird import source_id
        sid = source_id(source)
        labels = ["Pycnonotus sinensis_Light-vented Bulbul", "Motacilla cinerea_Grey Wagtail",
                  "Gryllus campestris_Field Cricket"]
        sound_index.CACHE.mkdir(parents=True)
        (sound_index.CACHE / "labels-test.json").write_text(json.dumps(labels))
        n = 20
        top = np.full((n, 5), -1)
        scores = np.zeros((n, 5))
        top[2, 0], scores[2, 0] = 0, 0.45          # Bulbul ตรงคลิป Bulbul ที่มีอยู่แล้ว -> ไม่ใช่ near miss
        top[4, 0], scores[4, 0] = 1, 0.42          # Wagtail ไม่มีคลิป
        top[10, 0], scores[10, 0] = 1, 0.30
        top[6, 0], scores[6, 0] = 2, 0.9           # แมลง: ไม่นับ
        top[8, 0], scores[8, 0] = 0, 0.1           # ต่ำกว่า NEAR_FLOOR
        sound_index._save(sid, np.arange(n) * 1.5, np.ones((n, 4)) / 2, np.zeros(n), 30.0, top, scores,
                          np.zeros(n), "test")
        rows = [dict(row, **{"Source ID": sid, "Source path": str(source), "Source file": source.name,
                             "Analysis settings": SETTINGS, "Latitude": 30.5, "Longitude": 114.4})
                for row in ({"Species (common)": "Light-vented Bulbul", "Species (scientific)": "Pycnonotus sinensis",
                             "Start offset (s)": 3.0, "End offset (s)": 6.0, "Clock time": "14:39:47",
                             "Review status": "Approved", "File": "a.wav"},
                            {"Species (common)": "Common Pheasant", "Species (scientific)": "Phasianus colchicus",
                             "Start offset (s)": 20.0, "End offset (s)": 23.0, "Clock time": "14:40:04",
                             "Review status": "Rejected", "File": "b.wav"})]
        summary = self.root / "result" / "summary.xlsx"
        summary.parent.mkdir()
        pd.DataFrame(rows).to_excel(summary, index=False)
        return summary, source

    class FakeGeo:
        model = None

        def probability(self, scientific, common, lat, lon, when):
            return {"Motacilla cinerea": 0.82, "Pycnonotus sinensis": 0.9}.get(scientific)

        def probabilities(self, lat, lon, when=None):
            week = {"Motacilla cinerea": (0.82, "Grey Wagtail"), "Pycnonotus sinensis": (0.9, "Light-vented Bulbul"),
                    "Egretta garzetta": (0.98, "Little Egret"), "Gryllus campestris": (0.5, "Field Cricket")}
            if when is None:
                week["Anser fabalis"] = (0.4, "Taiga Bean-Goose")
            return week

    def test_find_near_misses_lists_unclipped_birds_with_second_look(self):
        summary, _source = self.make_result()
        model = CentredModel("Motacilla cinerea_Grey Wagtail", scale=0.8)
        with contextlib.redirect_stdout(io.StringIO()):
            found = near_misses.find_near_misses(summary, model=model, geo=self.FakeGeo(), log=lambda _t: None)
        self.assertEqual([item["common"] for item in found["species"]], ["Gray Wagtail"])
        wagtail = found["species"][0]
        self.assertEqual(wagtail["geo"], 0.82)
        self.assertEqual(len(wagtail["events"]), 2)
        self.assertEqual(wagtail["best"], 0.42)
        self.assertGreaterEqual(wagtail["passed"], 1)          # เสียงที่ 7 วิ อยู่กลางหน้าต่างเมื่อฟังซ้ำ
        self.assertEqual(found["min_conf"], 0.5)
        # คลิปที่ Reject ไม่นับว่าชนิดนั้นมีคลิป
        self.assertEqual(found["clips"], {"Pycnonotus sinensis": {"clips": 1, "approved": 1}})
        self.assertEqual(near_misses.load_near(summary)["species"][0]["common"], "Gray Wagtail")

    def test_expected_species_week_and_year_with_heard_scores(self):
        summary, _source = self.make_result()
        found = near_misses.find_expected(summary, geo=self.FakeGeo(), log=lambda _t: None)
        by_name = {item["common"]: item for item in found["species"]}
        self.assertEqual(list(by_name)[0], "Little Egret")              # เรียงตามความน่าจะเป็นสัปดาห์นี้
        self.assertNotIn("Field Cricket", by_name)                      # นกเท่านั้น
        self.assertEqual(by_name["Gray Wagtail"]["heard"], 0.42)
        self.assertEqual(by_name["Taiga Bean-Goose"]["week"], 0.0)       # มาเฉพาะบางฤดู
        self.assertEqual(by_name["Taiga Bean-Goose"]["year"], 0.4)
        self.assertEqual(found["start"], "2026-10-18T14:39:44")

    def test_index_labels_merge_neighbouring_windows(self):
        import manual_window
        summary, source = self.make_result()
        from field_audio_to_ebird import source_id
        sid = source_id(source)
        index = sound_index.load(sid)
        index["labels"][5, 0], index["scores"][5, 0] = 1, 0.6              # Wagtail 4 และ 5 ติดกัน
        spans = manual_window.index_labels(source, sid)
        self.assertEqual([(a, b, name, bird) for a, b, name, _score, _scientific, bird in spans],
                         [(3.0, 6.0, "Light-vented Bulbul", True), (6.0, 10.5, "Gray Wagtail", True),
                          (9.0, 12.0, "Field Cricket", False), (15.0, 18.0, "Gray Wagtail", True)])
        self.assertEqual(spans[1][3:5], (0.6, "Motacilla cinerea"))
        # ชนิดที่ควรพบของผลชุด (ถ้าเคยถามโมเดลพื้นที่แล้ว) ใช้ทำเครื่องหมายนกที่ไม่ควรพบ
        self.assertIsNone(manual_window._expected_of(summary))
        near_misses.find_expected(summary, geo=self.FakeGeo(), log=lambda _t: None)
        self.assertEqual(manual_window._expected_of(summary)["Motacilla cinerea"], 0.82)


class SecondLookReviewTests(unittest.TestCase):
    def test_review_progress_measures_second_look_clips(self):
        import review_progress
        row = {"Species (common)": "Gray Wagtail", "Species (scientific)": "Motacilla cinerea",
               "Max confidence": 0.58, "Found by": near_misses.describe_look(0.48, 0.58, "centred")}
        self.assertTrue(review_progress.flags(row)["second_look"])
        self.assertFalse(review_progress.flags(dict(row, **{"Found by": ""}))["second_look"])
        self.assertIn("second_look", dict(review_progress.FILTERS))

    def test_review_table_marks_second_look_scores(self):
        import birdnet_gui
        self.assertTrue(birdnet_gui.second_look_note({"Found by": "Second look: BirdNET 0.45 …"}))
        self.assertEqual(birdnet_gui.second_look_note({"Found by": float("nan")}), "")


class NearMissWindowTests(unittest.TestCase):
    def tearDown(self):
        import gc
        gc.collect()
        super().tearDown()

    def test_windows_show_cached_results_and_add_passed_sounds(self):
        import tkinter as tk
        import birdnet_gui
        import sound_search
        import workflow_presets
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(sound_index, "CACHE", Path(temp) / "index"), \
                patch.object(birdnet_gui, "load_settings", return_value={}), \
                patch.object(birdnet_gui, "save_settings"), \
                patch.object(birdnet_gui.App, "refresh_results"), \
                patch.object(birdnet_gui.App, "_preload_pipeline"), \
                patch.object(workflow_presets, "STORE", Path(temp) / "presets.json"):
            root = Path(temp)
            source = root / "REC-001.WAV"
            tone_recording(source, 30, [7.0])
            summary = root / "result" / "summary.xlsx"
            summary.parent.mkdir()
            pd.DataFrame([{"Species (common)": "Light-vented Bulbul", "Species (scientific)": "Pycnonotus sinensis",
                           "Start offset (s)": 3.0, "End offset (s)": 6.0, "Clock time": "14:39:47",
                           "Review status": "Pending", "File": "a.wav", "Source ID": "abc", "Source path": str(source),
                           "Source file": source.name, "Analysis settings": SETTINGS,
                           "Latitude": 30.5, "Longitude": 114.4}]).to_excel(summary, index=False)
            event = {"start": 6.0, "end": 9.0, "score": 0.42, "best": 6.0, "sid": "abc",
                     "label": "Motacilla cinerea_Grey Wagtail", "look": 0.61, "look_how": "centred", "look_at": 7.0}
            sound_index.CACHE.mkdir(parents=True)
            near_misses.near_file(summary).write_text(json.dumps({
                "summary": str(summary.resolve()), "made": "2026-10-24T10:00:00", "min_conf": 0.5, "floor": 0.2,
                "looked": 1, "clips": {}, "sources": {},
                "species": [{"common": "Gray Wagtail", "scientific": "Motacilla cinerea", "birdnet": event["label"],
                             "geo": 0.82, "checklist": "native in China", "events": [event], "best": 0.42,
                             "look": 0.61, "passed": 1},
                            {"common": "Common Ground Dove", "scientific": "Columbina passerina", "birdnet": "x",
                             "geo": None, "checklist": "", "events": [dict(event, score=0.7, look=0.0)],
                             "best": 0.7, "look": 0.0, "passed": 0}]}))
            near_misses.expected_file(summary).write_text(json.dumps({
                "summary": str(summary.resolve()), "made": "2026-10-24T10:00:00", "lat": 30.5, "lon": 114.4,
                "start": "2026-10-18T14:39:44", "indexed": 0, "recordings": 1, "clips": {},
                "species": [{"common": "Gray Wagtail", "scientific": "Motacilla cinerea", "geo_name": "Motacilla cinerea",
                             "week": 0.82, "year": 0.83, "heard": 0.42, "checklist": "native in China"},
                            {"common": "Light-vented Bulbul", "scientific": "Pycnonotus sinensis",
                             "geo_name": "Pycnonotus sinensis", "week": 0.91, "year": 0.93, "heard": 0.9,
                             "checklist": "native in China"}]}))
            tk_root = tk.Tk()
            tk_root.withdraw()
            tk_root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                app = birdnet_gui.App(tk_root)
                app.review_summary.set(str(summary))
                started = []
                app._start = lambda cmd, *args, **kwargs: started.append((cmd, kwargs))
                near = sound_search.NearMissWindow(app, summary)
                tk_root.update()
                self.assertEqual(list(near.species), ["Motacilla cinerea"])        # ไม่ควรพบที่นี่: ซ่อน
                self.assertIn("✓", near.tree.set("Motacilla cinerea", "look"))
                near.unexpected.set(True)
                near._refilter()
                self.assertEqual(len(near.species), 2)
                near.tree.selection_set("Motacilla cinerea")
                tk_root.update()
                self.assertEqual(near.list.scientific.get(), "Motacilla cinerea")
                self.assertEqual(len(near.list.matches), 1)
                with patch.object(sound_search.messagebox, "askokcancel", return_value=True):
                    near.add_passed()
                cmd, kwargs = started[-1]
                self.assertEqual(cmd[-3:-1], ["--manual-clip", "add-batch"])
                batch = json.loads(Path(cmd[-1]).read_text())
                self.assertEqual([(item["common"], item["start"], item["end"]) for item in batch],
                                 [("Gray Wagtail", 4.5, 10.5)])
                self.assertIn("second look 0.61", batch[0]["notes"])
                expected = sound_search.ExpectedSpeciesWindow(app, summary)
                tk_root.update()
                self.assertEqual(started[-1][0][-3:], ["--sound-index", "expected", str(summary.resolve())])
                self.assertEqual(expected.tree.get_children(), ("Motacilla cinerea", "Pycnonotus sinensis"))
                self.assertIn("heard", expected.tree.item("Motacilla cinerea", "tags"))
                expected.show.set(expected.SHOW[3])
                expected._fill()
                self.assertEqual(expected.tree.get_children(), ("Motacilla cinerea",))
                for window in (near, expected):
                    window.win.destroy()
                tk_root.update()
            finally:
                tk_root.destroy()


if __name__ == "__main__":
    unittest.main()
