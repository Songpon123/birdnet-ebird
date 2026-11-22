"""Prism viewer in Listen tools + the pruned BirdNET model option (fake models, no download)."""

import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import prism_view as kv  # noqa: E402


def tone(seconds=20, sr=24000, at=(5.0,), hz=3000.0, stereo=True):
    t = np.arange(int(sr * seconds)) / sr
    left = np.random.default_rng(0).normal(scale=1e-4, size=len(t))
    for start in at:
        mask = (t >= start) & (t < start + 0.5)
        left[mask] += 0.5 * np.sin(2 * np.pi * hz * t[mask])
    return (np.column_stack([left, left * 0.1]) if stereo else left).astype(np.float32), sr


class ViewerMathTests(unittest.TestCase):
    def test_palette_rises_from_the_background_to_pale_yellow(self):
        lut = kv.palette()
        self.assertEqual(lut.shape, (256, 3))
        self.assertEqual(tuple(lut[0]), (18, 21, 28))                    # เบาสุด = สีพื้น
        self.assertEqual(tuple(lut[255]), (252, 245, 190))               # ดังสุด = เหลืองอ่อน
        self.assertTrue((np.diff(lut.astype(float).sum(axis=1)) >= 0).all())   # ดังขึ้น = สว่างขึ้นตลอด
        self.assertGreater(int(lut[128][0]), int(lut[128][1]))           # กลางช่วงเป็นชมพู ไม่ใช่เขียว
        inverse = kv.palette(inverse=True)
        self.assertEqual(tuple(inverse[0]), (251, 250, 246))             # กลับสี: พื้นกระดาษ
        self.assertEqual(tuple(inverse[255]), (20, 40, 100))

    def test_full_scale_sine_is_zero_db(self):
        sr = 48000
        t = np.arange(sr) / sr
        db = kv.spectrum_db(np.sin(2 * np.pi * 3000 * t))
        self.assertEqual(db.shape[1], kv.FFT_SIZE // 2 + 1)
        self.assertEqual(len(db), 1 + (sr - kv.FFT_SIZE) // kv.FFT_HOP)
        self.assertAlmostEqual(float(db[100, 16]), 0.0, delta=0.2)        # 3000 Hz = bin 16 ที่ 48 kHz
        self.assertLess(float(db[100, 40]), -60)

    def test_rulers_and_labels_are_compact(self):
        self.assertEqual(kv.time_label(2), "2s")
        self.assertEqual(kv.time_label(0.5), "0.5s")
        self.assertEqual(kv.time_label(64), "1m4s")
        self.assertEqual(kv.freq_label(10000), "10kHz")
        self.assertEqual(kv.freq_label(500), "0.5kHz")
        self.assertEqual(kv.nice_step(12000, 300, 30), 2000)
        self.assertEqual(kv.nice_step(8.0, 1000, 90), 1.0)

    def test_analysis_stats_follow_the_checked_formulas(self):
        """ค่าที่ตรวจแล้ว (2026-10-24): sine -20 dBFS -> -20.02/-20.01/-20.02, sine ครึ่งเวลา -> dBmean -23.03,
        เสียงรบกวน RMS -40 dBFS -> -41.99/-38.05/-39.96, Fpmean 5.993 kHz, Fpmin/Fpmax 0"""
        sr = 48000
        t = np.arange(sr * 16) / sr
        rng = np.random.default_rng(1)
        floor = rng.normal(scale=1e-4, size=len(t))
        sine = kv.analysis_stats(kv.spectrum_db(0.1 * np.sin(2 * np.pi * 3000 * t) + floor), sr, 0, 12000)
        self.assertAlmostEqual(sine["dBmax"], -20.0, delta=0.1)
        self.assertAlmostEqual(sine["dBmean"], -20.0, delta=0.1)
        self.assertAlmostEqual(sine["Fppeak"], 3000, delta=5)
        self.assertAlmostEqual(sine["Fpmean"], 3000, delta=5)
        self.assertLess(sine["Fpmin"], 3000)
        self.assertGreater(sine["Fpmax"], 3000)
        half = kv.analysis_stats(kv.spectrum_db(np.where(t < 8, 0.1 * np.sin(2 * np.pi * 3000 * t), 0) + floor),
                                 sr, 0, 12000)
        self.assertAlmostEqual(half["dBmean"], -23.0, delta=0.1)
        self.assertLess(half["dBmin"], -75)
        noise = kv.analysis_stats(kv.spectrum_db(rng.normal(scale=0.01, size=len(t))), sr, 0, 12000)
        self.assertAlmostEqual(noise["dBmean"], -40.0, delta=0.2)
        self.assertAlmostEqual(noise["dBmin"], -42.0, delta=0.6)
        self.assertAlmostEqual(noise["dBmax"], -38.0, delta=0.6)
        self.assertAlmostEqual(noise["Fpmean"], 6000, delta=100)
        self.assertEqual((noise["Fpmin"], noise["Fpmax"]), (0.0, 0.0))     # สเปกตรัมเรียบ: หาขอบไม่ได้
        narrow = kv.analysis_stats(kv.spectrum_db(0.1 * np.sin(2 * np.pi * 3000 * t) + floor), sr, 2500, 3500)
        self.assertTrue(all(2500 <= f <= 3500 for f in narrow["freqs"]))

    def test_spectrogram_rgb_maps_bins_to_rows(self):
        db = np.full((10, 129), -96.0, np.float16)
        db[:, 32] = 0.0                                                   # 6 kHz ที่ 48 kHz
        rows = np.linspace(12000, 0, 120)
        rgb = kv.spectrogram_rgb(db, rows, 48000, kv.palette())
        self.assertEqual(rgb.shape, (120, 10, 3))
        brightest = int(np.argmax(rgb[:, 0, 0]))
        self.assertAlmostEqual(rows[brightest], 6000, delta=150)
        self.assertEqual(tuple(rgb[0, 0]), tuple(kv.palette()[0]))        # -96 dB = สีต่ำสุด
        pooled = kv.column_pool(np.arange(100.0).reshape(100, 1), 10)
        self.assertEqual(pooled[:, 0].tolist(), [9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 89.0, 99.0])


class ListenToolsPrismTests(unittest.TestCase):
    def tearDown(self):
        import gc
        gc.collect()
        super().tearDown()

    def test_listen_tools_open_with_the_prism_viewer(self):
        import tkinter as tk
        import listening_tools
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "REC-TEST.WAV"
            samples, sr = tone()
            sf.write(source, samples, sr)
            root = tk.Tk()
            root.withdraw()
            # ปิดหน้าต่างสุดท้ายแล้ว macOS ส่ง Quit มา (ค่าเริ่มต้นของ Tk = จบ process ทดสอบทั้งชุด)
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = listening_tools.ListeningToolsWindow(root, source, start=0.0, end=20.0,
                                                              clock_start=datetime(2026, 10, 18, 14, 39, 44))
                view = window.prism
                for _ in range(300):
                    root.update()
                    time.sleep(0.01)
                    if view.data:
                        break
                self.assertIs(window.view, view)
                self.assertEqual(window.channel.get(), "left")             # ทีละช่อง เริ่มซ้าย
                self.assertEqual(view.channel_text.get(), "Left")
                self.assertEqual(view.x_zoom, kv.DEFAULT_X_ZOOM)
                self.assertEqual((view.f_lo, view.f_hi), (0.0, kv.DEFAULT_FMAX))
                self.assertGreater(view.db_at(5.25, 3000), -10)            # เสียง 3 kHz ดังเกือบเต็มสเกล
                self.assertLess(view.db_at(2.0, 3000), -60)
                ruler = [view.spec.itemcget(i, "text") for i in view.spec.find_withtag("ruler")
                         if view.spec.type(i) == "text"]
                self.assertIn("2kHz", ruler)
                self.assertIn("0s", ruler)
                # ลากกรอบ -> ช่องเวลา/ความถี่ของ Listen tools
                view.selection = (4.9, 5.6, 2500.0, 3500.0)
                view.on_select(*view.selection)
                self.assertEqual((window.start.get(), window.finish.get()), ("0:04.90", "0:05.60"))
                self.assertEqual((window.low.get(), window.high.get()), ("2500", "3500"))
                played = []
                window.play = lambda: played.append((window.start.get(), window.finish.get(), window.low.get()))
                view.clear_selection()
                view.set_view(3.0)
                window.play_view()                                         # ไม่มีกรอบ: จากขอบซ้ายถึงท้ายช่วง ทุกความถี่
                self.assertEqual(played[-1], ("0:03.00", "0:20.00", "0"))
                view.zoom_time(-1)
                self.assertEqual(view.x_zoom, 4.0)
                view.fit_time()
                self.assertEqual(view.x_zoom, 20.0)
                view.toggle_osc()
                view.toggle_log_freq()
                view.time_mode = "clock"
                view.redraw()
                ruler = [view.spec.itemcget(i, "text") for i in view.spec.find_withtag("ruler")
                         if view.spec.type(i) == "text"]
                self.assertIn("14:39:50", ruler)
                view.time_mode = "elapsed"
                view.toggle_osc()
                view.toggle_log_freq()
                view.fit_time()
                # Prism Analysis: ภาพที่เห็น แล้วตามกรอบที่เลือก
                view.toggle_analysis()
                for _ in range(20):
                    root.update()
                    time.sleep(0.01)
                analysis = view.analysis
                self.assertTrue(analysis.exists())
                self.assertEqual(analysis.values["Tstart"].get(), "0.000000s")
                self.assertEqual(analysis.values["Fend"].get(), "12.000kHz")
                self.assertEqual(analysis.values["N"].get(), "—")              # zero-crossing: ไม่คำนวณ
                view.selection = (5.0, 5.5, 2000.0, 4000.0)
                view._draw_selection()
                for _ in range(20):
                    root.update()
                    time.sleep(0.01)
                self.assertEqual(analysis.values["Tstart"].get(), "5.000000s")
                self.assertAlmostEqual(float(analysis.values["Fppeak"].get()[:-3]), 3.0, delta=0.05)
                self.assertGreater(float(analysis.values["dBmax"].get()[:-2]), -10)
                self.assertIn("Fppeak: ", analysis.text())
                self.assertTrue(analysis.canvas.find_all())
                view.toggle_analysis()
                self.assertIsNone(view.analysis)
                view.update()
                self.assertEqual(view.spec.winfo_rootx(), view.osc.winfo_rootx())   # เวลาของสองภาพตรงกัน
                self.assertLess(view.spec.winfo_rooty(), view.osc.winfo_rooty())    # ภาพเสียงอยู่บนคลื่นเสียง
                self.assertLess(view.bar.winfo_rooty(), view.spec.winfo_rooty())    # แถบควบคุมอยู่บนสุด
                image = view.screen_image()
                self.assertGreater(image.width, 100)
                self.assertEqual(image.getpixel((5, 5)), (18, 21, 28))      # พื้นเทาเข้มของ Prism
                window.display.set("Macaulay")
                window.display_changed()
                self.assertIs(window.view, window.macaulay)
                window.close()
            finally:
                root.destroy()


class PrunedModelTests(unittest.TestCase):
    def test_pruned_model_loads_through_birdnet_load_custom_with_full_labels(self):
        import v3_model
        calls = {}

        def load_custom(*args, **kwargs):
            calls["custom"] = (args, kwargs)
            return types.SimpleNamespace(predict_arrays=lambda *_a, **_k: None)
        fake = types.SimpleNamespace(
            load=lambda *args, **kwargs: types.SimpleNamespace(species_list=["Parus cinereus_Asian Tit", "Pica serica_Oriental Magpie"]),
            load_custom=load_custom)
        with tempfile.TemporaryDirectory() as temp, patch.object(v3_model, "PRUNED_DIR", Path(temp)), \
                patch.dict(sys.modules, {"birdnet": fake}):
            (Path(temp) / v3_model.PRUNED_FILE).write_bytes(b"onnx")
            v3_model.load_pruned()
            args, kwargs = calls["custom"]
            self.assertEqual(args[:3], ("acoustic", "3.0", "onnx"))
            self.assertEqual(Path(args[3]).name, v3_model.PRUNED_FILE)
            self.assertEqual(Path(args[4]).read_text().split(), ["Parus", "cinereus_Asian", "Tit", "Pica",
                                                                 "serica_Oriental", "Magpie"])
            self.assertEqual(kwargs, {"precision": "fp16"})

    def test_damaged_download_is_not_kept(self):
        import v3_model

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False
        with tempfile.TemporaryDirectory() as temp, patch.object(v3_model, "PRUNED_DIR", Path(temp)), \
                patch.object(v3_model.urllib.request, "urlopen", lambda *_a, **_k: Response(b"not the model")):
            with self.assertRaises(RuntimeError):
                v3_model.pruned_model_path(log=lambda _text: None)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_pruned_results_go_to_their_own_folder_and_are_labelled(self):
        import birdnet_gui
        import field_audio_to_ebird as pipeline
        import workflow_presets
        from v3_model import PrunedV3Analyzer

        class FakePruned(PrunedV3Analyzer):
            def __init__(self):
                self.model = None

            def detections(self, *_args):
                return [{"start_time": 3.0, "end_time": 6.0, "common_name": "Light-vented Bulbul",
                         "scientific_name": "Pycnonotus sinensis", "confidence": 0.7,
                         "is_predicted_for_location_and_date": True, "location_filter_available": True}]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "REC-001.WAV"
            sf.write(source, np.zeros(8000 * 10, np.float32), 8000)
            with contextlib.redirect_stdout(io.StringIO()):
                rows, day = pipeline.process_file(
                    FakePruned(), source, Path(temp) / "out", lat_arg=30.5, lon_arg=114.4, cfg_lat=None,
                    cfg_lon=None, date_override=datetime(2026, 10, 18), use_meta=False, min_conf=0.5, gap=5,
                    lead=3, tail=3, target_dbfs=-3, fmt="wav", make_mono=True, incl_alt=False, place_arg=None,
                    dt_regex=pipeline.FILENAME_DATETIME_REGEX)
            self.assertTrue(day.name.endswith("_v3pruned"))
            self.assertIn("pruned", rows[0]["Analysis settings"])
        label = birdnet_gui.result_label({"folder": "2026.10.18_1439_v3pruned", "sources": ["REC-001.WAV"],
                                          "clips": 3, "approved": 0, "rejected": 0})
        self.assertTrue(label.endswith("BirdNET 3.0 pruned"))
        self.assertEqual(workflow_presets.validate({"model": "3.0-pruned"}), {"model": "3.0-pruned"})
        parser = pipeline.build_parser()
        self.assertEqual(parser.parse_args(["x.wav", "--model", "3.0-pruned"]).model, "3.0-pruned")


class FrequencyCompareTests(unittest.TestCase):
    """ตัวเทียบเสียงกับเสียงอ้างอิงหลายชนิด — ไฟล์สังเคราะห์ในคลังของคุณเอง ไม่ต่อ xeno-canto"""

    def tearDown(self):
        import gc
        gc.collect()
        super().tearDown()

    def calls(self, path, hz, sr=44100, extra=None):
        """เสียง "ติ๊ก" 5 ครั้งที่ hz (+ เสียงนกอื่นครั้งเดียวที่ extra Hz)"""
        t = np.arange(int(sr * 6)) / sr
        x = np.random.default_rng(2).normal(scale=1e-4, size=len(t))
        for start in (0.5, 1.5, 2.5, 3.5, 4.5):
            mask = (t >= start) & (t < start + 0.3)
            x[mask] += 0.3 * np.sin(2 * np.pi * hz * t[mask])
        if extra:
            mask = (t >= 5.2) & (t < 5.6)
            x[mask] += 0.5 * np.sin(2 * np.pi * extra * t[mask])
        sf.write(path, x.astype(np.float32), sr)
        return x, sr

    def test_reference_calls_are_measured_with_band_and_length(self):
        import frequency_compare as fc
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(fc, "LIBRARY", Path(temp) / "library"):
            source = Path(temp) / "ref.wav"
            self.calls(source, 4200, extra=2500)
            events = fc.measure_file(source)
            self.assertEqual(len(events), 6)
            self.assertEqual(fc.measure_file(source), events)                 # วัดครั้งเดียว (cache)
            main = fc.main_calls(events)                                      # เสียงนกอื่นครั้งเดียวถูกตัด
            self.assertEqual(len(main), 5)
            summary = fc.summarize(main)
            self.assertAlmostEqual(summary["peak"], 4200, delta=120)
            self.assertLess(summary["lo_edge"], 4200)
            self.assertGreater(summary["hi_edge"], 4200)
            self.assertAlmostEqual(summary["dur"], 0.3, delta=0.08)
            self.assertEqual(fc.matches({"Fppeak": 4190}, summary), "inside")
            self.assertEqual(fc.matches({"peak": 6000}, summary), "outside")
            self.assertIsNone(fc.matches({"Fppeak": 0}, summary))
            self.assertEqual(fc.matches({"Fppeak": 4000}, {"q1": 4400, "q3": 4600, "low": 3900, "high": 5000}), "near")
            added = fc.add_to_library("Dendrocopos hyperythrus", [source, source])
            self.assertEqual([p.name for p in added], ["ref.wav", "ref-2.wav"])
            self.assertEqual(fc.library_files("Dendrocopos hyperythrus"), sorted(added))
            order = fc.download_order([
                {"id": "XC1", "file": "x", "ext": ".wav", "quality": "A", "length": "0:05"},
                {"id": "XC2", "file": "x", "ext": ".mp3", "quality": "A", "length": "0:40"},
                {"id": "XC3", "file": "x", "ext": ".mp3", "quality": "C", "length": "0:03"},
                {"id": "XC4", "file": "x", "ext": ".mp3", "quality": "B", "length": "0:10"}])
            self.assertEqual([r["id"] for r in order], ["XC4", "XC2", "XC1", "XC3"])   # A/B, mp3, สั้นก่อน
            self.assertEqual(fc.length_seconds("1:02:03"), 3723)
            self.assertEqual(fc.parse_xc_name("XC1075519 - Sakhalin Leaf Warbler - Phylloscopus borealoides.mp3"),
                             ("XC1075519", "Sakhalin Leaf Warbler", "Phylloscopus borealoides"))
            self.assertIsNone(fc.parse_xc_name("REC-001.WAV"))

    def test_tone_purity_tells_a_clear_whistle_from_a_buzzy_call(self):
        import frequency_compare as fc
        sr = 44100
        t = np.arange(int(sr * 0.3)) / sr
        envelope = np.hanning(len(t))
        whistle = envelope * np.sin(2 * np.pi * (5200 - 2000 * t) * t)               # "tseeoo" ลากลงใส ๆ
        buzz = envelope * np.sin(2 * np.pi * 4600 * t + 3.3 * np.sin(2 * np.pi * 120 * t))   # สั่นเร็ว = แหบ
        clear, buzzy = (fc.tone_purity(x, sr, 0.0, 0.3, 4000, 5600) for x in (whistle, buzz))
        self.assertGreater(clear, 0.75)
        self.assertLess(buzzy, 0.45)
        clip = np.concatenate([np.zeros(sr // 2), 0.3 * whistle, np.zeros(sr // 2), 0.3 * buzz, np.zeros(sr // 2)])
        clip = clip + np.random.default_rng(3).normal(scale=1e-4, size=len(clip))
        events = fc.measure_samples(clip, sr)
        self.assertEqual(len(events), 2)
        self.assertGreater(events[0]["purity"], 0.7)
        self.assertLess(events[1]["purity"], 0.5)
        species = {"Chestnut-flanked White-eye": {"purity": 0.67, "purity_spread": 0.08},
                   "Swinhoe's White-eye": {"purity": 0.41, "purity_spread": 0.07}}
        text = fc.purity_verdict({"purity": 0.42}, species)
        self.assertTrue(text.startswith("Tone purity (high = clear whistle, low = buzzy): yours 0.42 — "
                                        "Swinhoe's White-eye 0.41 · Chestnut-flanked White-eye 0.67"))
        self.assertIn("→ closer to Swinhoe's White-eye", text)
        self.assertNotIn("closer to", fc.purity_verdict({"purity": 0.54}, species))   # อยู่กลาง ๆ
        self.assertEqual(fc.purity_verdict({"purity": None}, species), "")

    def test_verdict_picks_the_closer_species_or_says_inconclusive(self):
        import frequency_compare as fc

        def species(peak, spread=170):
            return {"peak": peak, "spread": spread, "q1": peak - 100, "q3": peak + 100, "low": peak - 300,
                    "high": peak + 300, "lo_edge": peak - 250, "hi_edge": peak + 300, "dur": 0.3}
        pooled = {"Pale-legged Leaf Warbler": species(5840), "Sakhalin Leaf Warbler": species(4710, 230)}
        yours = {"peak": 5790, "lo_edge": 5700, "hi_edge": 5950}
        text, best = fc.verdict(yours, pooled)
        self.assertEqual(best, "Pale-legged Leaf Warbler")
        self.assertTrue(text.startswith("Closer to Pale-legged Leaf Warbler"))
        self.assertEqual(fc.band_overlap(yours, pooled["Pale-legged Leaf Warbler"]), 1.0)
        self.assertEqual(fc.band_overlap(yours, pooled["Sakhalin Leaf Warbler"]), 0.0)
        text, best = fc.verdict({"peak": 5250}, pooled)                         # อยู่กลาง ๆ
        self.assertIsNone(best)
        self.assertTrue(text.startswith("Inconclusive"))
        text, best = fc.verdict({"peak": 9000}, pooled)
        self.assertIsNone(best)
        self.assertIn("far from every species", text)

    def test_window_compares_your_calls_with_two_species_offline(self):
        import types
        import tkinter as tk
        import frequency_compare as fc
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(fc, "LIBRARY", Path(temp) / "library"), \
                patch("app_settings.load_settings", return_value={}):
            pale = Path(temp) / "XC1 - Pale-legged Leaf Warbler - Phylloscopus tenellipes.wav"
            sakhalin = Path(temp) / "XC2 - Sakhalin Leaf Warbler - Phylloscopus borealoides.wav"
            self.calls(pale, 5850)
            self.calls(sakhalin, 4700)
            yours, sr = self.calls(Path(temp) / "mine.wav", 5790, extra=3000)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = fc.FrequencyCompareWindow(root, {}, species=("Pale-legged Leaf Warbler", "Phylloscopus tenellipes"),
                                                   audio=(yours, sr, (0.0, 6.0, 4000.0, 8000.0)))
                self.assertEqual(window.yours["calls"], 5)                        # ทุกเสียงในกรอบ 4–8 kHz
                self.assertAlmostEqual(window.yours["peak"], 5790, delta=120)
                self.assertEqual(window.add_paths([pale, sakhalin]), 2)           # ชื่อแบบ xeno-canto: รู้ชนิดเอง
                self.assertEqual([s["common"] for s in window.species], ["Pale-legged Leaf Warbler", "Sakhalin Leaf Warbler"])
                for _ in range(100):
                    root.update()
                    time.sleep(0.01)
                    if not window._busy:
                        break
                verdict = window.verdict.get()                                       # ไฟล์ละ 1: ยังไม่ถึงขั้นต่ำ
                self.assertTrue(verdict.startswith("Provisional — needs 5 reference recordings"), verdict)
                self.assertIn("Closer to Pale-legged Leaf Warbler", verdict)
                self.assertNotIn("best", window.species_tree.item("s0", "tags"))
                more = []
                for n in range(2, 6):                                                # ชนิดละ 5 ไฟล์
                    for name, hz in (("Pale-legged Leaf Warbler - Phylloscopus tenellipes", 5850 + 20 * n),
                                     ("Sakhalin Leaf Warbler - Phylloscopus borealoides", 4700 - 20 * n)):
                        path = Path(temp) / f"XC{n}{hz} - {name}.wav"
                        self.calls(path, hz)
                        more.append(path)
                self.assertEqual(window.add_paths(more), 8)
                self.assertEqual(window.reference_counts(), {0: (5, 5), 1: (5, 5)})
                self.assertTrue(window.verdict.get().startswith("Closer to Pale-legged Leaf Warbler"))
                self.assertIn("best", window.species_tree.item("s0", "tags"))
                self.assertRegex(window.species_tree.set("s0", "purity"), r"^[01]\.\d\d$")   # ความใส (เสียงสังเคราะห์ใส = 1.00)
                self.assertIn(", purity ", window.your_text.get())
                self.assertTrue(window.purity_text.get().startswith("Tone purity"))
                self.assertIn("recording date unknown", window.log_text.get())
                matches = {window.tree.item(iid, "values")[1]: window.tree.item(iid, "values")[10]
                           for iid in window.tree.get_children()}
                self.assertEqual(matches, {"Pale-legged Leaf Warbler": "inside ✓", "Sakhalin Leaf Warbler": "outside ✗"})
                self.assertEqual(sorted(p.name for p in (Path(temp) / "library").rglob("*.wav")),
                                 sorted([pale.name, sakhalin.name] + [p.name for p in more]))
                spec = window.separator_spec()                                       # Train on references
                self.assertEqual([len(item["files"]) for item in spec["species"]], [5, 5])
                self.assertEqual(len(spec["yours"]["centers"]), 1)                   # กรอบที่ลากไว้
                self.assertTrue(Path(spec["yours"]["path"]).is_file())
                result = {"species": ["Pale-legged Leaf Warbler", "Sakhalin Leaf Warbler"], "files": [5, 5],
                          "calls": [25, 25], "test": {"files_right": 10, "files": 10, "calls_right": 50, "calls": 50,
                                                       "per_species": [[5, 5], [5, 5]]},
                          "yours": {"windows": 1, "mean": [0.9, 0.1], "votes": [1, 0]}}
                job = types.SimpleNamespace(stdout=iter(["Loading BirdNET 3.0 …\n", "SEPARATE " + json.dumps(result)]),
                                            wait=lambda: 0, returncode=0, poll=lambda: 0)
                with patch("platform_tools.start_child", lambda cmd: job) as _start:
                    window.train_separator()
                    for _ in range(100):
                        root.update()
                        time.sleep(0.01)
                        if window._separator is None:
                            break
                self.assertIn("10 of 10 right", window.separation.get())
                self.assertIn("leans Pale-legged Leaf Warbler", window.separation.get())
                window.draw()
                self.assertTrue(window.canvas.find_all())
                window.open_spectrograms()                                           # ภาพเสียง 3 แหล่งเรียงกัน
                spectro = window.spectrograms
                root.update()
                self.assertEqual([r["kind"] for r in spectro.rows], ["yours", "species", "species"])
                self.assertTrue(all(r["audio"] is not None for r in spectro.rows))
                self.assertIn("Pale-legged Leaf Warbler: XC", spectro.rows[1]["text"].get())   # ไฟล์ใกล้ค่ากลางสุดก่อน
                self.assertIn("call 1 of 5", spectro.rows[2]["text"].get())
                self.assertTrue(all(any(r["canvas"].type(i) == "image" for i in r["canvas"].find_all())
                                    for r in spectro.rows))
                low, high = spectro._range()                                         # ครอบทั้ง 4.7 และ 5.85 kHz
                self.assertTrue(low < 4700 and high > 5850, (low, high))
                spectro.step(spectro.rows[1], 1)
                self.assertIn("call 2 of 5", spectro.rows[1]["text"].get())
                spectro.seconds.set("2")
                spectro.refresh()
                self.assertTrue(all(abs(len(r["audio"][0]) / r["audio"][1] - 2.0) < 0.05 for r in spectro.rows))
                with patch("platform_tools.play_audio", lambda path: types.SimpleNamespace(
                        path=Path(path), poll=lambda: None, terminate=lambda: None)):
                    spectro.play(spectro.rows[2])
                    self.assertEqual(window.player.path.name, "preview-row2.wav")
                    window.stop()
                window.species_tree.selection_set("s1")                              # ลบชนิด -> แถวหาย
                window.remove_species()
                for _ in range(60):
                    root.update()
                    time.sleep(0.01)
                self.assertEqual([r["kind"] for r in spectro.rows], ["yours", "species"])
                window.close()
            finally:
                root.destroy()


    def test_pick_your_sounds_from_birdnet_boxes_auto_click_or_drag(self):
        import types
        import tkinter as tk
        import frequency_compare as fc
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(fc, "LIBRARY", Path(temp) / "library"), \
                patch("app_settings.load_settings", return_value={}):
            samples, sr = self.calls(Path(temp) / "clip.wav", 5790, extra=3000)
            boxes = {"windows": [[0.0, 6.0, 0.62]],                                   # แบบ Detection boxes ของคลิป
                     "sounds": [[s, s + 0.3, 5600, 6000] for s in (0.5, 1.5, 2.5, 3.5, 4.5)] + [[5.2, 5.6, 2900, 3100]]}
            shifted = fc.shift_boxes(boxes, 1.0)                     # Listen tools เริ่มที่วินาที 1: กรอบก่อนหน้านั้นตัดออก
            self.assertEqual([[round(v, 2) for v in box] for box in shifted["sounds"][:1]], [[0.5, 0.8, 5600, 6000]])
            self.assertEqual(len(shifted["sounds"]), 5)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = fc.FrequencyCompareWindow(root, None, describe="clip.wav", audio=(samples, sr, None),
                                                   boxes=boxes)

                def wait():
                    for _ in range(300):
                        root.update()
                        if window.clip["db"] is not None and window.candidates:
                            return
                        time.sleep(0.01)
                wait()
                self.assertEqual(window.source.get(), "BirdNET boxes")
                self.assertEqual(len(window.candidates), 6)
                self.assertEqual([c["on"] for c in window.candidates], [True] * 5 + [False])   # Auto: เสียงที่ร้องซ้ำ
                self.assertEqual(window.yours["calls"], 5)
                self.assertAlmostEqual(window.yours["peak"], 5790, delta=120)
                self.assertIn("5 box(es) chosen", window.your_text.get())
                self.assertTrue(window.clip_canvas.find_all())
                self.assertEqual(fc.auto_groups(window.candidates), [[0, 1, 2, 3, 4], [5]])
                window.auto_pick(step=True)                                          # กด Auto อีกครั้ง = กลุ่มถัดไป
                self.assertEqual([c["on"] for c in window.candidates], [False] * 5 + [True])
                self.assertIn("sound group 2 of 2", window.state.get())
                window.auto_pick(step=True)
                self.assertEqual(window.yours["calls"], 5)

                class Player:                                                        # แทน afplay
                    def __init__(self, path):
                        self.path, self.stopped = Path(path), False

                    def poll(self):
                        return 0 if self.stopped else None

                    def terminate(self):
                        self.stopped = True
                with patch("platform_tools.play_audio", Player):
                    window.play_mine()                                               # ช่วงของ 5 กรอบที่เลือก
                    player = window.player
                    self.assertEqual(player.path.name, "preview-mine.wav")
                    self.assertAlmostEqual(sf.info(str(player.path)).duration, 4.8 + 0.3 - 0.2, delta=0.05)
                    root.update()
                    self.assertEqual(len(window.clip_canvas.find_withtag("playhead")), 1)   # เส้นตำแหน่งบนภาพ
                    window.stop()
                    self.assertTrue(player.stopped)
                    self.assertIsNone(window.player)
                    self.assertFalse(window.clip_canvas.find_withtag("playhead"))
                    window.clear_pick()
                    window.play_mine()                                               # ไม่ได้เลือก = ทั้งคลิป
                    self.assertAlmostEqual(sf.info(str(window.player.path)).duration, 6.0, delta=0.05)
                    first = window.player
                    window.play_mine()                                               # กดซ้ำ: หยุดเสียงเดิมก่อน
                    self.assertTrue(first.stopped)
                    window.stop()
                window.auto_pick()
                x0, plot_w, plot_h, fmax = window._clip_geometry()

                def at(t, hz):
                    return types.SimpleNamespace(x=x0 + t / window.clip["duration"] * plot_w, y=plot_h * (1 - hz / fmax))
                for _ in range(2):                                                   # คลิก = ใช้ / ไม่ใช้
                    window._clip_press(at(5.4, 3000))
                    window._clip_release(at(5.4, 3000))
                    self.assertEqual(window.yours["calls"], 6 if window.candidates[5]["on"] else 5)
                self.assertFalse(window.candidates[5]["on"])
                window.clear_pick()
                self.assertIsNone(window.yours)
                self.assertIn("nothing chosen yet", window.your_text.get())
                window._clip_press(at(0.2, 7000))                                    # ลากกรอบเอง
                window._clip_drag(at(3.0, 4500))
                window._clip_release(at(3.0, 4500))
                self.assertEqual(window.candidates[-1]["kind"], "drawn")
                self.assertEqual(window.yours["calls"], 3)                            # 3 เสียงแรกในกรอบ
                window.pick_all()
                self.assertEqual(window.yours["calls"], 6)
                window.source.set("Sound detector")                                  # ไม่ใช้กรอบของ BirdNET
                window._load_candidates(auto=True)
                window.clip["db"], window.candidates = None, [c for c in window.candidates if c["kind"] == "drawn"]
                for _ in range(300):
                    root.update()
                    if any(c["kind"] == "detector" for c in window.candidates):
                        break
                    time.sleep(0.01)
                self.assertEqual(sum(c["kind"] == "detector" for c in window.candidates), 6)
                self.assertEqual(window.yours["calls"], 5)                            # Auto ตัดเสียงนกอื่น, ปิดกรอบที่ลาก
                window.close()
            finally:
                root.destroy()

    def test_field_log_line_follows_the_log_added_in_settings(self):
        import tkinter as tk
        import field_log
        import frequency_compare as fc
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_field_log import write_log
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(fc, "LIBRARY", Path(temp) / "library"), \
                patch.object(field_log, "STORE", Path(temp) / "field-logs.json"), \
                patch("app_settings.load_settings", return_value={}):
            log = write_log(Path(temp) / "Birding", datetime(2026, 10, 25).date(), "S200000001",
                            [("swiwhe1", "Swinhoe's White-eye", "Zosterops simplex", "13", 2)])
            samples, sr = self.calls(Path(temp) / "clip.wav", 4300)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = fc.FrequencyCompareWindow(root, None, species=("Chestnut-flanked White-eye",
                                                                        "Zosterops erythropleurus"),
                                                   audio=(samples, sr, None), when=datetime(2026, 10, 25, 9, 46))
                self.assertIn("no eBird checklist of yours for 25 Oct 2026", window.log_text.get())
                self.assertIn("Settings › Field logs", window.log_text.get())
                window.close()
                field_log.add(log)
                window = fc.FrequencyCompareWindow(root, None, species=("Chestnut-flanked White-eye",
                                                                        "Zosterops erythropleurus"),
                                                   audio=(samples, sr, None), when=datetime(2026, 10, 25, 9, 46))
                window._add("Swinhoe's White-eye", "Zosterops simplex")
                text = window.log_text.get()
                self.assertTrue(text.startswith("Your checklist S200000001 25 Oct 08:08"), text)
                self.assertIn("Chestnut-flanked White-eye: not on your eBird checklist S200000001", text)
                self.assertIn("Swinhoe's White-eye: on your checklist S200000001 ×13, 2 photos", text)
                window.close()
            finally:
                root.destroy()

    def test_controls_wrap_on_a_narrow_window_and_bottom_buttons_stay_visible(self):
        import tkinter as tk
        import frequency_compare as fc
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(fc, "LIBRARY", Path(temp) / "library"), \
                patch("app_settings.load_settings", return_value={}):
            samples, sr = self.calls(Path(temp) / "clip.wav", 5790)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = fc.FrequencyCompareWindow(root, None, species=("Pale-legged Leaf Warbler", "Phylloscopus tenellipes"),
                                                   audio=(samples, sr, None))
                window.win.geometry("660x560")                                       # หน้าต่างแคบและเตี้ย
                for _ in range(30):
                    root.update()
                    time.sleep(0.01)
                for bar in (window.picker_bar, window.species_bar, window.reference_bar, window.button_bar):
                    width = bar.winfo_width()
                    for widget, _space in bar.items:                                 # ทุกปุ่มอยู่ในแถว ไม่ล้นขวา
                        self.assertTrue(widget.winfo_ismapped())
                        self.assertLessEqual(widget.winfo_x() + widget.winfo_reqwidth(), width + 1)
                self.assertGreater(window.reference_bar.winfo_height(), 40)          # ขึ้นบรรทัดใหม่แล้ว
                bottom = window.button_bar.winfo_rooty() + window.button_bar.winfo_height()
                self.assertLessEqual(bottom, window.win.winfo_rooty() + window.win.winfo_height())  # ปุ่มล่างไม่ตกขอบ
                self.assertTrue(window.scroller.bar.winfo_ismapped())                # เนื้อหาเลื่อนได้
                self.assertGreater(window.tree.winfo_height(), 120)                  # ตารางไฟล์อ้างอิงไม่ถูกบีบหาย
                window.win.geometry("1000x1000")                                     # หน้าต่างสูงพอ: ตารางยืด แถบเลื่อนซ่อน
                for _ in range(30):
                    root.update()
                    time.sleep(0.01)
                if window.win.winfo_height() >= 990:
                    self.assertFalse(window.scroller.bar.winfo_ismapped())
                    self.assertGreater(window.tree.winfo_height(), 160)
                window.close()
            finally:
                root.destroy()

class IconTests(unittest.TestCase):
    def test_a_png_in_the_icon_folder_replaces_the_text(self):
        import tkinter as tk
        from PIL import Image
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as temp, patch.object(kv, "ICON_FOLDER", Path(temp)):
                Image.new("RGBA", (64, 64), (200, 0, 0, 255)).save(Path(temp) / "play.png")
                (Path(temp) / "stop.png").write_bytes(b"not a picture")
                view = kv.PrismView(root)
                self.assertEqual(str(view.play_btn.cget("text")), "")
                self.assertTrue(str(view.play_btn.cget("image")))
                self.assertEqual(view._face("stop", "■ Stop"), {"image": "", "text": "■ Stop"})     # ไฟล์เสีย = ตัวหนังสือ
                self.assertEqual(view._face("invert", "Invert"), {"image": "", "text": "Invert"})   # ไม่มีไฟล์ = ตัวหนังสือ
                view.destroy()
        finally:
            root.destroy()

    def test_every_button_name_is_listed_for_the_person_adding_icons(self):
        source = Path(kv.__file__).read_text(encoding="utf-8")
        names = set(re.findall(r'_face\("([a-z-]+)"', source))
        names |= set(re.findall(r'\("([a-z-]+)", "[A-Z][\w ]*", self\.toggle', source))
        guide = (kv.ICON_FOLDER / "README.txt").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(names), 14)
        for name in names:
            self.assertIn(f"{name}.png", guide)


if __name__ == "__main__":
    unittest.main()
