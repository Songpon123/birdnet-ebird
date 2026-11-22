import contextlib
from datetime import datetime
import hashlib
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
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import audio_io
import event_groups
import detections_import
import recognizers
import signal_events
import sound_index
import workflow_presets


class SoundToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sr = 16000

    def tearDown(self):
        self.temp.cleanup()

    def audio(self, name="source.wav", seconds=12, stereo=False):
        rng = np.random.default_rng(11)
        data = rng.normal(0, .0005, int(self.sr * seconds)).astype(np.float32)
        for a, b, freq in ((2, 2.3, 1000), (6, 6.25, 3000), (6.5, 6.75, 3000)):
            t = np.arange(int((b - a) * self.sr)) / self.sr
            at = int(a * self.sr)
            if at + len(t) <= len(data):
                data[at:at + len(t)] += .2 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
        if stereo:
            data = np.column_stack((data, np.zeros_like(data)))
        source = self.root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        sf.write(source, data, self.sr, subtype="FLOAT")
        return source

    def test_channels_filter_tempo_and_source_unchanged(self):
        source = self.audio(stereo=True)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        left, sr = audio_io.read_channels(source, 2, 2.3, "left")
        right, _ = audio_io.read_channels(source, 2, 2.3, "right")
        self.assertGreater(np.max(np.abs(left)), .15)
        self.assertEqual(np.max(np.abs(right)), 0)
        target = self.root / "stereo.wav"
        audio_io.listening_preview(source, 1, 4, target, channel="stereo", low=700, high=1300, speed=1.5)
        preview, rate = sf.read(target, always_2d=True)
        self.assertEqual(preview.shape[1], 2)
        self.assertAlmostEqual(len(preview) / rate, 2, delta=.12)
        fft = abs(np.fft.rfft(preview[:, 0]))
        peak = np.fft.rfftfreq(len(preview), 1 / rate)[np.argmax(fft)]
        self.assertAlmostEqual(peak, 1000, delta=12)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        t = np.arange(sr) / sr
        high = np.sin(2 * np.pi * 3500 * t)
        filtered = audio_io.bandpass(high, sr, 700, 1300)
        self.assertLess(np.std(filtered[sr // 10:-sr // 10]), .001)

    def test_event_detection_frequency_and_syllable_gap(self):
        source = self.audio()
        data, sr = sf.read(source)
        params = signal_events.DetectionParams(low=500, high=5000, maximum=3, gap=.3)
        events = signal_events.detect_samples(data, sr, params)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]["start"], 2, delta=.05)
        self.assertGreater(events[1]["end"], 6.7)
        narrow = signal_events.detect_samples(data, sr, signal_events.DetectionParams(low=700, high=1400))
        self.assertEqual(len(narrow), 1)
        for kwargs in (dict(low=float("nan")), dict(minimum=0), dict(gap=-1)):
            with self.assertRaises(ValueError):
                signal_events.DetectionParams(**kwargs).validate()

    def test_streaming_boundary_cache_and_channel(self):
        source = self.audio(stereo=True)
        params = signal_events.DetectionParams(low=500, high=5000, maximum=3)
        with patch.object(signal_events, "CACHE", self.root / "events"):
            result = signal_events.scan_source(source, params=params, chunk_seconds=6.2, log=lambda _: None)
            self.assertEqual(len(result["events"]), 2)
            # A cached run neither rereads audio nor recomputes the detector.
            with patch.object(signal_events, "read_channels", side_effect=AssertionError("cache missed")):
                self.assertEqual(signal_events.scan_source(source, params=params)["events"], result["events"])
            right = signal_events.scan_source(source, params=signal_events.DetectionParams(channel="right"),
                                               log=lambda _: None)
            self.assertEqual(right["events"], [])
            sf.write(source, np.zeros(self.sr * 12), self.sr)
            with self.assertRaisesRegex(ValueError, "changed"):
                signal_events.scan_source(source, sid=result["sid"], params=params)

    def test_event_embedding_isolates_sound_and_caches(self):
        source = self.audio()
        captured = []
        class Encoder:
            def encode(_self, files, **_kwargs):
                data, sr = sf.read(files[0])
                captured.append((data, sr))
                return types.SimpleNamespace(to_structured_array=lambda: [dict(input=files[0], embedding=np.array([1., 0., 1.]))])
        result = dict(sid="sample", events=[dict(start=2, end=2.3, channel="mix")])
        with patch.object(signal_events, "CACHE", self.root / "events"):
            vectors = event_groups.event_vectors(source, result, Encoder(), log=lambda _: None)
            self.assertEqual(vectors.shape, (1, 3))
            self.assertEqual(len(captured[0][0]), self.sr * 3)
            self.assertLess(np.max(abs(captured[0][0][:self.sr])), 1e-8)
            again = event_groups.event_vectors(source, result, Encoder())
            np.testing.assert_allclose(again, vectors, atol=.001)
            self.assertEqual(len(captured), 1)

    def test_large_grouping_keeps_every_event(self):
        vectors = np.random.default_rng(7).normal(size=(40, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        with patch.object(sound_index, "MAX_GROUP_WINDOWS", 20):
            labels, method = event_groups.cluster_vectors(vectors)
        self.assertEqual(len(labels), len(vectors))
        self.assertIn("all events", method)

    def test_sound_type_training_separates_labels_and_models(self):
        def row(kind, n):
            return {"Review status": "Approved", "Species (scientific)": "Avis testus", "Species (common)": "Test bird",
                    "Sound type": kind, "Source ID": f"source-{n}", "Start offset (s)": n * 3,
                    "End offset (s)": n * 3 + 3, "Generated sha256": f"hash-{kind}-{n}"}
        calls = [row("Call", n) for n in range(8)]
        songs = [row("Song", n + 10) for n in range(8)]
        unspecified = row("", 99)
        pool = recognizers.examples(calls + songs + [unspecified] + calls, "Call")["Avis testus"]
        self.assertEqual(len(pool["positive"]), 8)
        self.assertEqual(len(pool["negative"]), 8)
        def vectors(row):
            sign = 1 if row["Sound type"] == "Call" else -1
            return np.array([[sign * 3., 1., 0.], [sign * 2.9, .9, .1]], np.float32), row["Source ID"]
        with patch.object(recognizers, "CACHE", self.root / "models"), \
                patch.object(recognizers, "row_vectors", side_effect=vectors), \
                patch.object(recognizers, "_background", return_value=(np.zeros((0, 3)), [])):
            meta = recognizers.train("Avis testus", calls + songs, sound_type="Call", log=lambda _: None)
            self.assertEqual(meta["sound_type"], "Call")
            self.assertTrue(meta["tested_on_other_recordings"])
            self.assertIsNotNone(recognizers.load("Avis testus", "Call"))
            self.assertIsNone(recognizers.load("Avis testus"))
            self.assertIn("Avis testus", recognizers.trained("Call"))
        common = {"Analysis settings": json.dumps({"date": "2026-10-23"}), "Clock time": "07:00:00", "Start offset (s)": 0}
        self.assertEqual(recognizers.validation_group(common, "A"), recognizers.validation_group(common, "B"))

    def test_import_windows_paths_channels_duplicates_and_bad_rows(self):
        source = self.audio("audio/site/REC.WAV", stereo=True)
        csv = self.root / "cluster.csv"
        csv.write_text('\ufeffINDIR;FOLDER;IN FILE;CHANNEL;OFFSET;DURATION;MANUAL ID;TOP1DIST\n'
                       'C:\\recordings;site;REC.WAV;0;2;0.3;Test bird;0.2\n'
                       'C:\\recordings;site;REC.WAV;1;6;0.5;;0.4\n'
                       'C:\\recordings;site;REC.WAV;0;2;0.3;Another label;0.2\n'
                       'C:\\recordings;site;REC.WAV;0;99;3;;\n', encoding="utf-8")
        result = detections_import.read_candidates(csv, self.root / "audio")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["source"], source.resolve())
        self.assertEqual(result["candidates"][1]["channel"], "right")
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["candidates"][0]["padding"], 0)
        self.assertIn("unverified", result["candidates"][0]["note"])
        self.assertIn("Another label", result["candidates"][0]["label"])
        with self.assertRaises(ValueError):
            detections_import.resolve_source({"INFILE": "../cluster.csv"}, csv, self.root / "audio")

    def test_presets_whitelist_and_atomic_validation(self):
        path = self.root / "presets.json"
        workflow_presets.save_preset("Forest", dict(min_conf=.4, model="3.0-preview", mono=False), path)
        self.assertEqual(workflow_presets.read_store(path)["Forest"]["min_conf"], .4)
        before = path.read_bytes()
        for settings in (dict(xeno_canto_key="secret"), dict(force=True), dict(min_conf=float("nan"))):
            with self.assertRaises(ValueError):
                workflow_presets.save_preset("Bad", settings, path)
        self.assertEqual(path.read_bytes(), before)

    def test_compressed_format_preview_uses_bundled_decoder(self):
        import subprocess
        import platform_tools
        source = self.audio(stereo=True)
        compressed = self.root / "source.m4a"
        subprocess.run([platform_tools.bundled_tool("ffmpeg"), "-v", "error", "-y", "-i", str(source),
                        "-c:a", "aac", str(compressed)], check=True, capture_output=True)
        # libsndfile cannot read AAC on every supported platform; exercise that fallback explicitly.
        real_info = sf.info
        def info(file, *args, **kwargs):
            if str(file) == str(compressed):
                raise RuntimeError("unsupported by soundfile")
            return real_info(file, *args, **kwargs)
        with patch.object(sf, "info", side_effect=info):
            properties = audio_io.audio_info(compressed)
            samples, sr = audio_io.read_channels(compressed, 1.9, 2.4, "left")
        self.assertEqual(properties["channels"], 2)
        self.assertEqual(sr, self.sr)
        self.assertGreater(np.max(abs(samples)), .1)

    def test_channel_clip_import_keeps_pending_and_extracts_correct_track(self):
        import manual_clips
        from review_store import load_reviews
        source = self.audio(stereo=True)
        summary = self.root / "result/summary.xlsx"
        with patch.object(manual_clips, "location_expected", return_value=True):
            row = manual_clips.add_clip(summary, source, 1.9, 2.4, "Test bird", "Avis testus",
                    recorded=datetime(2026, 10, 23, 7), lat=13.8, lon=100.5, channel="left",
                    notes="Imported label (unverified)", spectrogram=False, log=lambda _: None)
        self.assertEqual(row["Review status"], "Pending")
        self.assertEqual(row["Source channel"], "left")
        clip = summary.parent / row["File"]
        self.assertEqual(sf.info(clip).channels, 1)
        self.assertFalse((summary.parent / "Ready").exists())
        self.assertEqual(load_reviews(summary)[0]["Origin"], "Manual")
        self.assertTrue(manual_clips._already_clipped(summary, source, 1.9, 2.4, "Avis testus", "left"))
        self.assertFalse(manual_clips._already_clipped(summary, source, 1.9, 2.4, "Avis testus", "right"))

    def test_threshold_never_splits_ties_or_reports_unreachable_precision(self):
        probability = np.array([.9, .9, .9, .2])
        truth = np.array([1, 0, 0, 1])
        threshold, precision, recall = recognizers.choose_threshold(probability, truth)
        mask = probability >= threshold
        self.assertAlmostEqual(precision, truth[mask].sum() / mask.sum())
        self.assertAlmostEqual(recall, truth[mask].sum() / truth.sum())

    def test_channel_training_never_substitutes_a_mixed_source_embedding(self):
        import sound_type
        row = {"Source ID": "stereo-recording", "Source channel": "right", "Generated sha256": "clip-hash"}
        expected = np.array([[0., 1., 0.]], np.float32)
        with patch.object(sound_type, "clip_vectors", return_value=expected), \
                patch.object(sound_index, "has_index", side_effect=AssertionError("must use right-channel clip")):
            actual, sid = recognizers.row_vectors(row)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(sid, "stereo-recording")


class CandidateGuiTests(unittest.TestCase):
    def tearDown(self):
        # Tk objects must be collected on the main thread, after test locals are released.
        import gc
        gc.collect()
        super().tearDown()

    def test_app_and_all_new_tool_windows_construct_together(self):
        import tkinter as tk
        import birdnet_gui
        import sound_search
        from import_window import DetectionsImportWindow
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(birdnet_gui, "load_settings", return_value={}), \
                patch.object(birdnet_gui, "save_settings"), \
                patch.object(birdnet_gui.App, "refresh_results"), \
                patch.object(birdnet_gui.App, "_preload_pipeline"), \
                patch.object(workflow_presets, "STORE", Path(temp) / "presets.json"):
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                app = birdnet_gui.App(root)
                app.out.set(temp)
                app.review_summary.set("")
                presets = workflow_presets.PresetsWindow(app)
                snapshot = workflow_presets.snapshot(app)
                self.assertNotIn("force", snapshot)
                self.assertNotIn("ebird_key", snapshot)
                workflow_presets.apply(app, dict(min_conf=.65, model="2.4", mono=True))
                self.assertAlmostEqual(app.min_conf.get(), .65)
                importer = DetectionsImportWindow(app)
                recognizer = sound_search.RecognizersWindow(app)
                groups = sound_search.SoundGroupsWindow(app, Path(temp) / "summary.xlsx")
                for _ in range(30):
                    root.update()
                    time.sleep(.01)
                self.assertEqual(groups.group_mode.get(), "Sound events")
                self.assertEqual(recognizer.target_type.get(), "Any sound")
                self.assertEqual(importer.win.title(), "Import detections")
                for tool in (groups, recognizer, importer, presets):
                    self.assertTrue(tool.win.winfo_exists())
                    tool.win.destroy()
                # Let the preload worker finish before collecting the Tk-owning App object.
                root.update()
            finally:
                root.destroy()

    def test_listening_loop_stop_and_speed_playhead(self):
        import tkinter as tk
        import listening_tools
        import spectrogram_view
        class Player:
            returncode = None
            def poll(self): return self.returncode
            def terminate(self): self.returncode = -15
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "test.wav"
            sf.write(source, np.zeros((16000 * 5, 2)), 16000)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                win = listening_tools.ListeningToolsWindow(root, source, end=5)
                win.display.set("Macaulay")              # this test covers the Macaulay display's playhead
                win.display_changed()
                until = time.monotonic() + 3
                while win.view._data is None and time.monotonic() < until:
                    root.update()
                    time.sleep(.01)
                self.assertIsNotNone(win.view._data)
                with patch.object(listening_tools, "play_audio", side_effect=lambda _: Player()) as play:
                    with patch.object(spectrogram_view.time, "monotonic", return_value=10):
                        win._start_player(source, 0, 3, 1.5)
                    with patch.object(spectrogram_view.time, "monotonic", return_value=11):
                        win.view._tick()
                    self.assertAlmostEqual(win.view._position, 1.5)
                    first = win.player
                    win.loop.set(True)
                    first.returncode = 0
                    win._poll()
                    self.assertIsNot(win.player, first)
                    self.assertEqual(play.call_count, 2)
                    current = win.player
                    win.stop()
                    self.assertEqual(current.returncode, -15)
                    win._poll()
                    self.assertEqual(play.call_count, 2)
                win.close()
                self.assertFalse(Path(win._temp.name).exists())
            finally:
                root.destroy()
                import gc
                gc.collect()

    def test_scanner_ui_and_metadata_without_birdnet(self):
        import tkinter as tk
        from event_window import SignalEventsWindow, source_entry
        class App:
            def _install_clipboard_fix(self): pass
            def review_stop(self): pass
            def _launch(self): return []
            def _start(self, cmd, button, title, is_analyze, on_done):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = signal_events.main(cmd[1:])
                on_done(code, output.getvalue().splitlines())
        with tempfile.TemporaryDirectory() as temp, patch.object(signal_events, "CACHE", Path(temp) / "cache"):
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            app = App()
            app.root, app.job_running = root, False
            app.review_summary, app.out = tk.StringVar(root, value=""), tk.StringVar(root, value=temp)
            try:
                source = Path(temp) / "recording.wav"
                sr = 16000
                data = np.random.default_rng(3).normal(0, .0005, sr * 5)
                t = np.arange(sr // 4) / sr
                data[sr * 2:sr * 2 + len(t)] += .2 * np.sin(2 * np.pi * 2000 * t)
                sf.write(source, data, sr)
                win = SignalEventsWindow(app)
                sid, entry = source_entry(source, win.summary)
                win.sources[sid] = entry
                win.refresh_sources()
                win.date.set("20261023")
                win.time.set("073005")
                win.coords.set("13.8,100.5")
                win.apply_context()
                self.assertEqual(entry["context"]["start"], datetime(2026, 10, 23, 7, 30, 5))
                with patch("birdnet.load", side_effect=AssertionError("detector must not load a species model")):
                    win.scan()
                self.assertGreater(len(win.list.matches), 0)
                self.assertFalse(win.summary.exists())  # scanning alone creates no clips or review records
                win.win.destroy()
                # Finish spectrogram worker while Tcl is still alive.
                for _ in range(20):
                    root.update()
                    time.sleep(.01)
            finally:
                root.destroy()
                import gc
                gc.collect()


if __name__ == "__main__":
    unittest.main()
