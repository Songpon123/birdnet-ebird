import contextlib
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# app/ first, like the real app: its tflite_runtime shim routes birdnetlib to LiteRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import birdnetlib  # noqa: E402
import field_audio_to_ebird as pipeline
import field_log
from review_store import load_reviews, save_review


def make_wav(path: Path, seconds=1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b"\0\0" * (8000 * seconds))


def fake_recording(starts, interrupt_on=None):
    """BirdNET stand-in: one 'Test bird' detection per start second."""
    class FakeRecording:
        def __init__(self, _analyzer, path, **_kwargs):
            self.path = path
            self.detections = [
                {"start_time": start, "end_time": start + 1, "common_name": "Test bird",
                 "scientific_name": "Avis testus", "confidence": 0.8}
                for start in starts
            ]

        def analyze(self):
            if Path(self.path).name == interrupt_on:
                raise KeyboardInterrupt   # what the CLI sees when the GUI window is closed

    return FakeRecording


PROCESS_ARGS = dict(lat_arg=13.8, lon_arg=100.5, cfg_lat=None, cfg_lon=None,
                    date_override=datetime(2026, 7, 15), use_meta=False, min_conf=0.5, gap=5,
                    lead=3, tail=3, target_dbfs=-3, fmt="wav", make_mono=True, incl_alt=False,
                    place_arg=None, dt_regex=pipeline.FILENAME_DATETIME_REGEX)


class TkTestCase(unittest.TestCase):
    """เก็บกวาดหน้าต่าง Tk ที่ปิดแล้วใน thread หลักทันทีหลังแต่ละ test — ถ้าปล่อยให้ garbage collector เก็บทีหลัง
    มันอาจทำใน thread อื่น (งานเบื้องหลัง, sklearn) แล้ว Tk ล้มทั้ง process (Tcl_AsyncDelete … wrong thread)"""

    def tearDown(self):
        import gc
        gc.collect()
        super().tearDown()


class WorkflowTests(unittest.TestCase):
    def test_filename_date_and_time_need_no_manual_date(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "2026-10-01 12_22.wav"
            make_wav(source)
            context = pipeline.recording_context_preview(source)
            self.assertEqual(context["datetime"], datetime(2026, 10, 1, 12, 22))
            self.assertEqual(context["datetime_source"], "filename")
            self.assertTrue(context["has_time"])
            pipeline.require_recording_context([source], None, True, 30.593, 114.306)

    def test_file_without_geotag_uses_the_coordinates_of_its_checklist(self):
        class Fake:
            model_name = "fake v3"
            direct_prediction = True

            def detections(self, _path, _overlap, _floor, latitude, longitude, _recorded_at):
                return [{"start_time": 3.0, "end_time": 6.0, "common_name": "Light-vented Bulbul",
                         "scientific_name": "Pycnonotus sinensis", "confidence": 0.8,
                         "is_predicted_for_location_and_date": True, "location_filter_available": True,
                         "latitude": latitude, "longitude": longitude}]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = root / "field-logs.json"
            store.write_text(json.dumps({"checklists": {"S1": {
                "id": "S1", "date": "2026-10-30T06:36", "species": [], "lat": 29.117096, "lon": 110.479191}}}), encoding="utf-8")
            source, evening = root / "2026-10-30 06_34.wav", root / "2026-10-30 20_00.wav"
            make_wav(source, seconds=30)
            make_wav(evening, seconds=30)
            args = dict(PROCESS_ARGS, lat_arg=None, lon_arg=None, date_override=None, use_meta=True)
            with patch.object(field_log, "STORE", store):
                context = pipeline.recording_context_preview(source)
                self.assertEqual((context["lat"], context["lon"], context["coords_source"]),
                                 (29.117096, 110.479191, "checklist S1, by time"))
                pipeline.require_recording_context([source], None, True, None, None)
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    rows, _ = pipeline.process_file(Fake(), source, root / "out", **args)
                self.assertEqual((rows[0]["Latitude"], rows[0]["Longitude"]), (29.117096, 110.479191))
                self.assertIn("coords 29.117096,110.479191 (checklist S1, by time)", printed.getvalue())
                rows, _ = pipeline.process_file(Fake(), source, root / "typed", **dict(args, lat_arg=1.5, lon_arg=2.5))
                self.assertEqual(rows[0]["Latitude"], 1.5)                         # พิมพ์เองชนะ checklist
                tagged = {"lat": 7.5, "lon": 8.5, "datetime": None, "datetime_has_time": False}
                with patch.object(pipeline, "read_audio_metadata", return_value=tagged):
                    context = pipeline.recording_context_preview(source)
                self.assertEqual((context["lat"], context["coords_source"]), (7.5, "embedded metadata"))   # geotag ชนะ
                with self.assertRaisesRegex(ValueError, "Recording location missing"):
                    pipeline.require_recording_context([evening], None, True, None, None)  # ไกลจาก checklist: ถามเหมือนเดิม

    def test_fill_missing_metadata_keeps_tagged_file_details(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recordings = root / "recordings"
            recordings.mkdir()
            tagged = recordings / "TAGGED.WAV"
            missing = recordings / "MISSING.WAV"
            make_wav(tagged, seconds=8)
            make_wav(missing, seconds=9)
            tagged_meta = {"datetime": datetime(2026, 10, 1, 7, 30),
                           "datetime_has_time": True,
                           "lat": 30.593, "lon": 114.306, "place": None}
            empty_meta = {"datetime": None, "datetime_has_time": False,
                          "lat": None, "lon": None, "place": None}

            def metadata(path):
                return tagged_meta if Path(path).name == tagged.name else empty_meta

            argv = ["field_audio_to_ebird.py", str(recordings), "-o", str(root / "out"),
                    "--model", "2.4", "--no-spectrogram", "--fill-missing-metadata",
                    "--date", "2026-10-02", "--coords", "31.0,115.0",
                    "--start-time", "09:00"]
            analyzer = types.ModuleType("birdnetlib.analyzer")
            analyzer.Analyzer = lambda: None
            with patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, {"birdnetlib.analyzer": analyzer}), \
                    patch.object(birdnetlib, "Recording", fake_recording((1,))), \
                    patch.object(pipeline, "read_audio_metadata", side_effect=metadata), \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline.main()
            tagged_rows = pd.read_excel(root / "out" / "2026.10.01" / "0730" / "summary.xlsx")
            missing_rows = pd.read_excel(root / "out" / "2026.10.02" / "0900" / "summary.xlsx")
            self.assertEqual((tagged_rows.iloc[0]["Latitude"], tagged_rows.iloc[0]["Longitude"]),
                             (30.593, 114.306))
            self.assertEqual((missing_rows.iloc[0]["Latitude"], missing_rows.iloc[0]["Longitude"]),
                             (31.0, 115.0))
            self.assertEqual(tagged_rows.iloc[0]["Clock time"], "07:30:01")
            self.assertEqual(missing_rows.iloc[0]["Clock time"], "09:00:01")

    def test_v3_preview_is_separate_and_uses_recording_location(self):
        class FakeV3:
            model_name = "BirdNET 3.0 preview ONNX fp16 + geo 3.0.4 (0.03)"
            direct_prediction = True

            def detections(self, _path, overlap, confidence_floor,
                           latitude, longitude, recorded_at):
                self.options = (overlap, confidence_floor,
                                latitude, longitude, recorded_at)
                return [{
                    "start_time": 3.0, "end_time": 6.0,
                    "common_name": "Light-vented Bulbul",
                    "scientific_name": "Pycnonotus sinensis",
                    "confidence": 0.608,
                    "is_predicted_for_location_and_date": True,
                    "location_filter_available": True,
                }]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=10)
            analyzer = FakeV3()
            rows, day = pipeline.process_file(
                analyzer, source, root / "out", **dict(PROCESS_ARGS, min_conf=0.25)
            )
            self.assertEqual(analyzer.options[:4], (1.5, 0.25, 13.8, 100.5))
            self.assertEqual(analyzer.options[4].date(), datetime(2026, 7, 15).date())
            self.assertTrue(day.name.endswith("_v3preview"))
            self.assertEqual(rows[0]["Expected by location/date"], "Yes")

    def test_v3_geo_rejects_out_of_area_false_positive_but_keeps_asian_tit(self):
        from v3_model import V3Analyzer, birdnet_week

        class FakeGeo:
            def predict(self, lat, lon, **kwargs):
                self.options = (lat, lon, kwargs)
                return types.SimpleNamespace(
                    species_list=["Vireolanius pulchellus_Green Shrike-Vireo",
                                  "Parus cinereus_Asian Tit"],
                    species_probs=[0.0, 0.916],
                )

        class FakeAcoustic:
            def predict(self, *_args, **_kwargs):
                return types.SimpleNamespace(to_structured_array=lambda: [
                    {"start_time": 1, "end_time": 4,
                     "species_name": "Vireolanius pulchellus_Green Shrike-Vireo",
                     "confidence": 0.589},
                    {"start_time": 4, "end_time": 7,
                     "species_name": "Parus cinereus_Asian Tit",
                     "confidence": 0.804},
                ])

        analyzer = V3Analyzer.__new__(V3Analyzer)
        analyzer.model = FakeAcoustic()
        analyzer.geo_model = FakeGeo()
        analyzer._geo_cache = {}
        date = datetime(2026, 10, 1, 10, 6)
        detections = analyzer.detections("sample.wav", 1.5, 0.5,
                                       30.5928, 114.3055, date)
        self.assertEqual(birdnet_week(date), 37)
        self.assertEqual(analyzer.geo_model.options[2]["week"], 37)
        kept = [d for d in detections if d["is_predicted_for_location_and_date"] or
                d["confidence"] >= 0.7]
        self.assertEqual([d["common_name"] for d in kept], ["Asian Tit"])

    def test_v3_uses_ebird_name_for_yellow_billed_grosbeak(self):
        from v3_model import V3Analyzer

        analyzer = V3Analyzer.__new__(V3Analyzer)
        analyzer.geo_model = types.SimpleNamespace(predict=lambda *_args, **_kwargs:
            types.SimpleNamespace(species_list=["Eophona migratoria_Chinese Grosbeak"],
                                  species_probs=[0.66]))
        analyzer.model = types.SimpleNamespace(predict=lambda *_args, **_kwargs:
            types.SimpleNamespace(to_structured_array=lambda: [{
                "start_time": 1, "end_time": 4,
                "species_name": "Eophona migratoria_Chinese Grosbeak",
                "confidence": 0.7,
            }]))
        analyzer._geo_cache = {}
        rows = analyzer.detections("sample.wav", 1.5, 0.5, 30.55, 114.36,
                                   datetime(2026, 10, 1))
        self.assertEqual(rows[0]["common_name"], "Yellow-billed Grosbeak")
        self.assertEqual(rows[0]["scientific_name"], "Eophona migratoria")

    def test_location_filter_drops_out_of_area_and_unchecked_species_by_default(self):
        class FakeV3:
            model_name = "fake v3"
            direct_prediction = True

            def detections(self, *_args):
                def det(start, common, sci, conf, predicted, available=True):
                    return {"start_time": start, "end_time": start + 3, "common_name": common,
                            "scientific_name": sci, "confidence": conf,
                            "is_predicted_for_location_and_date": predicted,
                            "location_filter_available": available}
                return [det(1, "Light-vented Bulbul", "Pycnonotus sinensis", 0.74, True),
                        det(10, "Chimpanzee", "Pan troglodytes", 0.81, False),
                        det(20, "Meimuna opalifera", "Meimuna opalifera", 0.92, True, False)]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=30)
            rows, _ = pipeline.process_file(FakeV3(), source, root / "out", **PROCESS_ARGS)
            self.assertEqual([r["Species (common)"] for r in rows], ["Light-vented Bulbul"])

    def test_detection_boxes_frame_each_note_inside_birdnet_windows(self):
        import numpy as np
        from pydub import AudioSegment
        sr = 32000
        t = np.arange(sr * 10) / sr
        y = 0.02 * np.random.default_rng(0).standard_normal(t.size)

        def note(start, stop, f0, f1):
            inside = (t >= start) & (t < stop)
            tt = t[inside] - start
            out = np.zeros_like(t)
            out[inside] = 0.3 * np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * (stop - start))))
            out[inside] *= np.hanning(inside.sum())
            return out
        y += note(2.0, 4.0, 3000, 3000)                         # whistle
        for k in range(4):
            y += note(7.0 + k * 0.25, 7.15 + k * 0.25, 5000, 6500)   # four rising notes
        clip = AudioSegment((y * 32767).astype("<i2").tobytes(), frame_rate=sr,
                            sample_width=2, channels=1)
        windows = [(101.5, 104.5, 0.71), (103.0, 106.0, 0.66), (106.0, 109.0, 0.80)]
        boxes = pipeline.detection_boxes(windows, {"start": 101.5, "end": 109.0}, 100.0, clip)
        self.assertEqual(boxes["windows"], [[1.5, 9.0, 0.8]])
        self.assertEqual(len(boxes["sounds"]), 5)
        t0, t1, low, high = boxes["sounds"][0]
        self.assertTrue(2.0 <= t0 < t1 <= 4.0 and 2800 < low < 3000 < high < 3200)
        self.assertTrue(all(5000 <= b[2] and b[3] <= 6600 for b in boxes["sounds"][1:]))

    def test_spectrogram_redrawn_when_boxes_change(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp)
            make_wav(day / "clip.wav", seconds=4)
            (day / "clip.png").write_bytes(b"old image without boxes")
            boxes = {"windows": [[0.5, 3.5, 0.9]], "sounds": []}
            pd.DataFrame([{"File": "clip.wav", "Species (common)": "A",
                           "Detection boxes": json.dumps(boxes)}]).to_excel(day / "summary.xlsx", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.gen_spectrograms([day])
                self.assertEqual(pipeline._png_description(day / "clip.png"), pipeline._boxes_tag(boxes))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.gen_spectrograms([day])
            self.assertIn("created 0, skipped existing 1", output.getvalue())

    def test_xeno_canto_search_orders_by_quality_then_distance(self):
        import urllib.error
        import xeno_canto

        def rec(number, quality, lat, lon, key="lon"):
            return {"id": str(number), "q": quality, "lat": str(lat), key: str(lon), "type": "call",
                    "cnt": "China", "loc": "Wuhan", "rec": "A. Recordist", "length": "0:12",
                    "lic": "//creativecommons.org/licenses/by-nc-sa/4.0/",
                    "file": f"https://xeno-canto.org/{number}/download", "url": f"//xeno-canto.org/{number}",
                    "file-name": f"XC{number}-call.mp3"}
        nearby = {"numRecordings": "2", "recordings": [rec(1, "B", 30.6, 114.3), rec(2, "A", 31.0, 115.0)]}
        world = {"numRecordings": "900", "recordings": [rec(2, "A", 31.0, 115.0), rec(3, "A", 60.0, 30.0, "lng"),
                                                        rec(4, "C", 30.5, 114.4)]}
        queries = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def fake_urlopen(request, timeout):
            query = request.full_url.split("query=")[1].split("&")[0]
            queries.append(query)
            return Response(json.dumps(nearby if "box" in query else world).encode())

        with patch.object(xeno_canto.urllib.request, "urlopen", fake_urlopen):
            found, total = xeno_canto.search("Phylloscopus inornatus", "user-key", "call", 30.5, 114.4,
                                             limit=10)
        self.assertEqual([r["id"] for r in found], ["XC2", "XC3", "XC1", "XC4"])   # A near, A far, B, C
        self.assertEqual(total, 900)
        self.assertIn("gen%3APhylloscopus+sp%3Ainornatus+type%3Acall+box%3A", queries[0])
        first = found[0]
        self.assertEqual((first["license"], first["page"], first["ext"]),
                         ("CC BY-NC-SA 4.0", "https://xeno-canto.org/2", ".mp3"))
        self.assertIsNotNone(found[1]["distance_km"])                    # 'lng' field also understood
        self.assertEqual(xeno_canto.build_query("Anas crecca", "flight call"),
                         'gen:Anas sp:crecca type:"flight call"')

        def rejected(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {},
                                         io.BytesIO(b'{"message": "Missing or invalid key"}'))
        with patch.object(xeno_canto.urllib.request, "urlopen", rejected):
            with self.assertRaisesRegex(xeno_canto.XenoCantoError, "Missing or invalid key"):
                xeno_canto.search("Phylloscopus inornatus", "wrong")
        with self.assertRaisesRegex(xeno_canto.XenoCantoError, "API key"):
            xeno_canto.search("Phylloscopus inornatus", "")

    def test_second_opinion_flags_clip_closer_to_another_species(self):
        import numpy as np
        import second_opinion
        import xeno_canto
        from birdnet_gui import format_second_opinion

        clip_vec, ybw_vec, hume_vec = np.eye(4)[0], np.eye(4)[1], np.eye(4)[0] * 0.9 + np.eye(4)[2] * 0.1
        labels = ["Phylloscopus inornatus_Yellow-browed Warbler", "Phylloscopus humei_Hume's Leaf Warbler"]

        class FakeModel:
            species_list = labels

            def __init__(self):
                self.ref_paths = {}

            def predict(self, paths, **kwargs):
                rows = []
                for path in paths:
                    for start in (0.0, 1.5, 3.0):
                        if kwargs.get("custom_species_list"):        # reference recording
                            rows.append({"input": path, "start_time": start, "end_time": start + 3,
                                         "species_name": kwargs["custom_species_list"][0], "confidence": 0.8})
                        else:                                         # the clip
                            rows += [{"input": path, "start_time": start, "end_time": start + 3,
                                      "species_name": labels[0], "confidence": 0.9},
                                     {"input": path, "start_time": start, "end_time": start + 3,
                                      "species_name": labels[1], "confidence": 0.3}]
                return types.SimpleNamespace(to_structured_array=lambda: rows)

            def encode(self, paths, **kwargs):
                def vector(path):
                    if "XC" not in str(path):
                        return clip_vec
                    return hume_vec if "humei" in str(path) else ybw_vec
                rows = [{"input": path, "start_time": start, "end_time": start + 3, "embedding": vector(path)}
                        for path in paths for start in (0.0, 1.5, 3.0)]
                return types.SimpleNamespace(to_structured_array=lambda: rows)

        analyzer = types.SimpleNamespace(model=FakeModel(),
                                         _geo_probabilities=lambda *_a: ({"Phylloscopus humei": 0.4}, {}))
        downloads = []

        def fake_search(scientific, key, lat=None, lon=None, limit=30, common=""):
            species = scientific.split()[1]
            return [{"id": f"XC{species}{n}", "file": "https://x", "length": "0:30", "ext": ".mp3"}
                    for n in range(2)], 2

        def fake_download(recording):
            downloads.append(recording["id"])
            path = Path(temp) / f"{recording['id']}.wav"
            make_wav(path, seconds=6)
            return path

        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp)
            make_wav(day / "clip.wav", seconds=6)
            pd.DataFrame([{"File": "clip.wav", "Generated sha256": "abc123", "Species (common)": "Yellow-browed Warbler",
                           "Species (scientific)": "Phylloscopus inornatus", "Latitude": 30.5, "Longitude": 114.4,
                           "Analysis settings": json.dumps({"date": "2026-10-06"}),
                           "Detection boxes": json.dumps({"windows": [[0.0, 6.0, 0.9]], "sounds": []})}]
                         ).to_excel(day / "summary.xlsx", index=False)
            with patch.object(second_opinion, "CACHE", day / "cache"), \
                    patch.object(xeno_canto, "search", fake_search), \
                    patch.object(xeno_canto, "download", fake_download):
                result = second_opinion.run(day / "summary.xlsx", 2, "user-key", log=lambda _t: None,
                                            analyzer=analyzer)
                self.assertEqual([r["scientific"] for r in result["ranking"]],
                                 ["Phylloscopus humei", "Phylloscopus inornatus"])
                # BirdNET's "Hume's Leaf Warbler" is shown with its eBird name
                self.assertIn("⚠ Closer to Hume's Warbler than", format_second_opinion(result))
                self.assertEqual(second_opinion.cached_result("abc123")["ranking"], result["ranking"])
                second_opinion.run(day / "summary.xlsx", 2, "user-key", log=lambda _t: None, analyzer=analyzer)
            self.assertEqual(len(downloads), 4)            # references reused from cache on the 2nd run

    def test_reference_files_are_made_mono_and_bad_ones_skipped(self):
        import numpy as np
        import soundfile as sf
        import second_opinion

        class Model:
            def predict(self, paths, **_kwargs):
                if any("bad" in path for path in paths):
                    raise RuntimeError("Analysis was cancelled")      # what BirdNET does on a bad file
                return types.SimpleNamespace(to_structured_array=lambda: [
                    {"input": p, "start_time": 0.0, "end_time": 3.0, "species_name": "x", "confidence": 0.5}
                    for p in paths])

            def encode(self, paths, **_kwargs):
                return types.SimpleNamespace(to_structured_array=lambda: [
                    {"input": p, "start_time": 0.0, "end_time": 3.0, "embedding": [1.0, 0.0]} for p in paths])

        scores, _encoded = second_opinion._analyze(Model(), "x", ["good1.wav", "bad.wav", "good2.wav"],
                                                   log=lambda _text: None)
        self.assertEqual(sorted(scores), ["good1.wav", "good2.wav"])
        with tempfile.TemporaryDirectory() as temp:
            stereo = Path(temp) / "XC1.wav"
            sf.write(str(stereo), np.zeros((48000 * 2, 2), dtype="float32"), 48000)
            with patch.object(second_opinion, "CACHE", Path(temp) / "cache"):
                prepared = second_opinion._prepare(stereo)
            self.assertEqual(sf.info(str(prepared)).channels, 1)

    def test_xeno_canto_falls_back_to_english_name_when_genus_differs(self):
        import xeno_canto

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def fake_urlopen(request, timeout):
            if "en%3A" not in request.full_url:                 # gen:Periparus sp:venustulus -> nothing
                return Response(b'{"numRecordings": "0", "recordings": []}')
            return Response(json.dumps({"numRecordings": "155", "recordings": [
                {"id": "7", "gen": "Pardaliparus", "sp": "venustulus", "q": "A", "file": "https://x/7"}]}).encode())

        with patch.object(xeno_canto.urllib.request, "urlopen", fake_urlopen):
            found, total = xeno_canto.search("Periparus venustulus", "user-key", common="Yellow-bellied Tit")
            page = xeno_canto.species_page("Periparus venustulus", "user-key", "Yellow-bellied Tit")
        self.assertEqual(([r["id"] for r in found], total), (["XC7"], 155))
        self.assertEqual(page, "https://xeno-canto.org/species/Pardaliparus-venustulus")

    def test_ebird_check_region_recent_and_hotspot(self):
        from datetime import date, timedelta
        import ebird
        from birdnet_gui import format_ebird

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        replies = {
            "ref/taxonomy/ebird": [{"sciName": "Phylloscopus inornatus", "comName": "Yellow-browed Warbler",
                                    "speciesCode": "yebwar3"},
                                   {"sciName": "Mareca strepera", "comName": "Gadwall", "speciesCode": "gadwal"}],
            "ref/hotspot/geo": [{"locId": "L2", "locName": "Far", "lat": 31.0, "lng": 115.0,
                                 "subnational1Code": "CN-42", "countryCode": "CN"},
                                {"locId": "L1", "locName": "Jiufeng NFP", "lat": 30.52, "lng": 114.46,
                                 "subnational1Code": "CN-42", "countryCode": "CN"}],
            "product/spplist/CN-42": ["yebwar3"],
            "ref/region/info/CN-42": {"result": "Hubei, China"},
            "data/obs/geo/recent": [{"speciesCode": "yebwar3", "obsDt": "2026-10-05 08:30",
                                     "locName": "Jiufeng NFP"}],
        }
        keys_sent = {}

        def fake_urlopen(request, timeout):
            path = request.full_url.split("/v2/")[1].split("?")[0]
            keys_sent[path] = request.get_header("X-ebirdapitoken")
            return Response(json.dumps(replies[path]).encode())

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(ebird, "CACHE", Path(temp)), \
                patch.object(ebird.urllib.request, "urlopen", fake_urlopen):
            yesterday = date.today() - timedelta(days=1)
            warbler = ebird.check("Phylloscopus inornatus", 30.5928, 114.3055, yesterday, "user-key")
            duck = ebird.check("Mareca strepera", 30.5928, 114.3055, yesterday, "user-key")
            old = ebird.check("Phylloscopus inornatus", 30.5928, 114.3055, yesterday - timedelta(days=60), "user-key")
            unknown = ebird.check("Avis inventus", 30.5, 114.4, None, "user-key")
        self.assertEqual((warbler["in_region"], warbler["region_name"], warbler["hotspot"]["name"]),
                         (True, "Hubei", "Jiufeng NFP"))
        self.assertEqual(warbler["recent"], {"date": "2026-10-05 08:30", "where": "Jiufeng NFP"})
        self.assertIs(duck["in_region"], False)
        self.assertIn("⚠ never reported in Hubei", format_ebird(duck))
        self.assertIn("not checked", format_ebird(old))
        self.assertIn("not in the eBird taxonomy", format_ebird(unknown))
        self.assertIsNone(keys_sent["ref/taxonomy/ebird"])               # taxonomy needs no key
        self.assertEqual(keys_sent["data/obs/geo/recent"], "user-key")

    def test_settings_file_is_private(self):
        from app_settings import load_settings, save_settings
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sub" / "settings.json"
            self.assertEqual(load_settings(path), {})
            save_settings({"xeno_canto_key": "abc"}, path)
            save_settings({"other": 1}, path)
            self.assertEqual(load_settings(path), {"xeno_canto_key": "abc", "other": 1})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_windows_stop_request_interrupts_child_cleanly(self):
        import subprocess
        import platform_tools
        app = Path(platform_tools.__file__).parent
        child = (f"import sys, time; sys.path.insert(0, {str(app)!r})\n"
                 "import platform_tools as p; p.IS_WINDOWS = True; p.watch_for_stop()\n"
                 "print('ready', flush=True)\n"
                 "try:\n    for _ in range(600):\n        time.sleep(0.05)\nexcept KeyboardInterrupt:\n"
                 "    print('rolled back', flush=True); sys.exit(130)\n")
        with patch.object(platform_tools, "IS_WINDOWS", True):
            proc = platform_tools.start_child([sys.executable, "-c", child])
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            platform_tools.request_stop(proc)            # ส่ง STOP ไม่ใช่ฆ่าทิ้ง
            rest = proc.stdout.read()
            self.assertEqual(proc.wait(timeout=10), 130)
        self.assertIn("rolled back", rest)

    def test_windows_helpers_pick_bundled_tools_and_console_python(self):
        import platform_tools
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bin").mkdir()
            (root / "bin" / "ffplay.exe").write_text("")
            (root / "python").mkdir()
            for name in ("python.exe", "pythonw.exe"):
                (root / "python" / name).write_text("")
            with patch.object(platform_tools, "PROGRAM", root):
                self.assertEqual(platform_tools.bundled_tool("ffplay"), str(root / "bin" / "ffplay.exe"))
            with patch.object(platform_tools.sys, "executable", str(root / "python" / "pythonw.exe")):
                self.assertEqual(platform_tools.child_python(), str(root / "python" / "python.exe"))
        env = platform_tools.child_env()
        self.assertEqual((env["PYTHONUTF8"], env[platform_tools.GUI_CHILD_ENV]), ("1", "1"))

    def test_app_data_lives_in_program_folder(self):
        import app_settings
        import ebird
        import paths
        import second_opinion
        import xeno_canto
        data = Path(__file__).resolve().parents[1] / "data"
        self.assertEqual(app_settings.SETTINGS_FILE, data / "settings.json")
        for cache in (ebird.CACHE, xeno_canto.CACHE, second_opinion.CACHE):
            self.assertEqual(cache.parent, data / "cache")
        self.assertEqual(Path(os.environ["BIRDNET_APP_DATA"]), paths.BIRDNET_MODELS)

    def test_old_library_data_is_moved_without_overwriting(self):
        import paths
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            old = {"settings": temp / "lib/AS/BirdNET-eBird/settings.json",
                   "cache": temp / "lib/Caches/BirdNET-eBird",
                   "birdnet": temp / "lib/AS/birdnet"}
            data = temp / "program/data"
            old["settings"].parent.mkdir(parents=True)
            old["settings"].write_text('{"ebird_key": "k", "last_summary": "old"}')
            (old["birdnet"] / "acoustic-models").mkdir(parents=True)
            (old["birdnet"] / "acoustic-models/model.onnx").write_text("model")
            (old["cache"] / "xeno-canto").mkdir(parents=True)
            (old["cache"] / "xeno-canto/XC1.mp3").write_text("old copy")
            (old["cache"] / "xeno-canto/XC2.mp3").write_text("two")
            (data / "cache/xeno-canto").mkdir(parents=True)
            (data / "cache/xeno-canto/XC1.mp3").write_text("kept")
            (data / "settings.json").write_text('{"last_summary": "new"}')
            os.utime(old["settings"], (1, 1))                  # ไฟล์เดิมเก่ากว่า -> ค่าใหม่ชนะ
            with patch.object(paths, "_OLD", old), patch.object(paths, "DATA", data), \
                    patch.object(paths, "SETTINGS_FILE", data / "settings.json"), \
                    patch.object(paths, "CACHE", data / "cache"), \
                    patch.object(paths, "BIRDNET_MODELS", data / "birdnet"):
                paths.migrate_old_locations()
                paths.migrate_old_locations()                   # รันซ้ำได้
            self.assertEqual(json.loads((data / "settings.json").read_text()),
                             {"ebird_key": "k", "last_summary": "new"})
            self.assertEqual((data / "settings.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((data / "birdnet/acoustic-models/model.onnx").read_text(), "model")
            self.assertEqual((data / "cache/xeno-canto/XC1.mp3").read_text(), "kept")
            self.assertEqual((data / "cache/xeno-canto/XC2.mp3").read_text(), "two")
            self.assertFalse(old["settings"].exists() or old["birdnet"].exists())
            self.assertFalse(old["settings"].parent.exists())

    def test_previous_results_are_listed_newest_first(self):
        from review_store import list_results
        from birdnet_gui import result_label
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder, statuses in (("2026.10.01_1105_v3preview", ["Pending"]),
                                     ("2026.10.06_0928_v3preview", ["Approved", "Rejected", "Pending"]),
                                     ("2026.09.30_0700", ["Pending", "Pending"])):
                (root / folder).mkdir()
                pd.DataFrame([{"File": f"{n}.wav", "Source file": "REC-002.WAV", "Review status": s}
                              for n, s in enumerate(statuses)]).to_excel(root / folder / "summary.xlsx", index=False)
            (root / "broken").mkdir()
            (root / "broken" / "summary.xlsx").write_text("not a workbook")
            found = list_results(root)
        self.assertEqual([info["folder"] for info in found],
                         ["2026.10.06_0928_v3preview", "2026.10.01_1105_v3preview", "2026.09.30_0700"])
        self.assertEqual(result_label(found[0]), "2026-10-06 09:28  ·  REC-002.WAV  ·  3 clips, 2 reviewed")
        self.assertTrue(result_label(found[2]).endswith("BirdNET 2.4"))

    def test_ebird_and_macaulay_links(self):
        from birdnet_gui import species_links
        links = species_links("yebwar3", "CN-42")
        self.assertEqual(links["species"], "https://ebird.org/species/yebwar3")
        self.assertIn("taxonCode=yebwar3", links["photo"])
        self.assertIn("mediaType=photo", links["photo"])
        self.assertIn("mediaType=audio&regionCode=CN-42", links["audio"])
        self.assertNotIn("regionCode", species_links("yebwar3")["audio"])

    def test_xeno_canto_species_link(self):
        from birdnet_gui import xeno_canto_url
        self.assertEqual(xeno_canto_url("Phylloscopus inornatus"),
                         "https://xeno-canto.org/species/Phylloscopus-inornatus")
        self.assertEqual(xeno_canto_url(" parus  cinereus minor "),
                         "https://xeno-canto.org/species/Parus-cinereus")   # subspecies dropped
        for name in ("", None, "_Unknown", "Acrididae", "Meimuna opalifera2"):
            self.assertIsNone(xeno_canto_url(name))

    @unittest.skipUnless(hasattr(os.stat_result, "st_birthtime"), "needs file creation times")
    def test_consistent_file_times_give_recording_start(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = Path(temp) / "REC-002.WAV"
            make_wav(recorder, seconds=40)
            started = datetime(2026, 10, 6, 9, 28).timestamp()
            os.utime(recorder, (started, started))            # birth time follows an earlier mtime
            os.utime(recorder, (started + 40, started + 40))  # closed when the recording ended
            context = pipeline.recording_context_preview(recorder, use_meta=False)
            self.assertEqual((context["datetime"], context["has_time"]), (datetime(2026, 10, 6, 9, 28), True))
            copied = Path(temp) / "REC-003.WAV"                # new file: created == modified
            make_wav(copied, seconds=40)
            self.assertIsNone(pipeline.recording_context_preview(copied, use_meta=False)["datetime"])

    def test_compact_date_and_time_entries(self):
        from entry_formats import parse_date_entry, parse_time_entry
        self.assertEqual(parse_date_entry("20261006"), datetime(2026, 10, 6))
        self.assertEqual(parse_date_entry(" 2026-10-06 "), datetime(2026, 10, 6))
        for text, expected in (("0929", "09:29:00"), ("929", "09:29:00"),
                               ("092930", "09:29:30"), ("09:29", "09:29:00")):
            self.assertEqual(parse_time_entry(text).strftime("%H:%M:%S"), expected)
        for bad in ("2026106", "20261332", "2026/10/06"):
            with self.assertRaises(ValueError):
                parse_date_entry(bad)
        for bad in ("2575", "09-29", "12345"):
            with self.assertRaises(ValueError):
                parse_time_entry(bad)

    def test_habitat_rules(self):
        from habitat import fits_site, parse_site_habitats, species_habitat
        self.assertEqual(species_habitat("Mareca strepera"), "Wetland")        # Gadwall
        self.assertEqual(species_habitat("Unknown species", "Parus cinereus"), "Woodland")
        self.assertEqual(species_habitat("Not a bird"), "")
        self.assertIs(fits_site("Wetland", ("forest",)), False)
        self.assertIs(fits_site("Wetland", ("forest", "marsh")), True)
        self.assertIs(fits_site("Marine", ("marsh", "river")), False)
        self.assertIs(fits_site("Forest", ("sea",)), True)                     # land birds fit anywhere
        self.assertIsNone(fits_site("Wetland", ()))
        self.assertEqual(parse_site_habitats(" Marsh,forest "), ("forest", "marsh"))
        with self.assertRaises(ValueError):
            parse_site_habitats("forest,jungle")

    def test_country_from_coordinates(self):
        from country_checklist import country_at
        places = {
            (30.5928, 114.3055): "CN",     # Wuhan
            (13.75, 100.5): "TH",          # Bangkok
            (22.49, 114.04): "HK",         # Mai Po
            (25.03, 121.56): "TW",         # Taipei
            (13.15, 99.98): "TH",          # Gulf of Thailand off Pak Thale (sea, near our coast)
            (17.97, 102.61): None,         # Vientiane, across the Mekong from Thailand
            (20.45, 99.88): None,          # Tachileik, across the river from Mae Sai
            (9.983, 98.55): None,          # Kawthaung, across the estuary from Ranong
            (16.8, 96.15): None,           # Yangon
            (20.0, 115.0): None,           # South China Sea, far from land
            (None, None): None,
        }
        self.assertEqual({place: country_at(*place) for place in places}, places)

    def test_country_checklist_status_and_splits(self):
        from birdnet_gui import format_checklist
        from country_checklist import check, describe, needs_check
        wuhan, bangkok = (30.5928, 114.3055), (13.75, 100.5)
        self.assertEqual(describe(check("Pycnonotus sinensis", *wuhan)), "native in China")
        bulbul = check("Pycnonotus sinensis", *bangkok)
        self.assertEqual(describe(bulbul), "rarity in Thailand — verify by ear")
        self.assertTrue(needs_check(bulbul))
        self.assertEqual(format_checklist(bulbul), "Checklist: ⚠ rarity in Thailand — verify by ear")
        # BirdNET 3.0 still calls the Asian bird by the pre-split name
        self.assertEqual(describe(check("Butorides striata", *wuhan)),
                         "not on the China checklist, but split species Little Heron "
                         "(Butorides atricapilla) is — check the name")
        # BirdNET name differs from eBird (Thinornis dubius): fall back to the English name
        self.assertEqual(describe(check("Charadrius dubius", *wuhan, "Little Ringed Plover")), "native in China")
        self.assertEqual(format_checklist(check("Parus cinereus", *bangkok)),
                         "Checklist: native in Thailand ✓")
        self.assertIsNone(check("Homo sapiens", *wuhan))                 # not a bird
        self.assertIsNone(check("Parus cinereus", 16.8, 96.15))          # Myanmar: no checklist
        self.assertEqual(format_checklist(None), "")

    def test_country_checklist_column_flags_rarity(self):
        class FakeV3:
            model_name = "fake v3"
            direct_prediction = True

            def detections(self, *_args):
                return [{"start_time": start, "end_time": start + 3, "common_name": common,
                         "scientific_name": sci, "confidence": 0.8,
                         "is_predicted_for_location_and_date": True,
                         "location_filter_available": True}
                        for start, common, sci in ((1, "Asian Tit", "Parus cinereus"),
                                                   (10, "Light-vented Bulbul", "Pycnonotus sinensis"),
                                                   (20, "Test bird", "Avis testus"))]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=30)
            rows, _ = pipeline.process_file(FakeV3(), source, root / "out", **PROCESS_ARGS)   # Bangkok
        self.assertEqual({r["Species (common)"]: r["Country checklist"] for r in rows},
                         {"Asian Tit": "native in Thailand",
                          "Light-vented Bulbul": "rarity in Thailand — verify by ear",
                          "Test bird": ""})

    def test_site_habitat_drops_weak_water_bird_but_keeps_strong_flyover(self):
        class FakeV3:
            model_name = "fake v3"
            direct_prediction = True

            def detections(self, *_args):
                return [{"start_time": start, "end_time": start + 3, "common_name": common,
                         "scientific_name": sci, "confidence": conf,
                         "is_predicted_for_location_and_date": True,
                         "location_filter_available": True}
                        for start, common, sci, conf in (
                            (1, "Asian Tit", "Parus cinereus", 0.82),
                            (10, "Gadwall", "Mareca strepera", 0.52),
                            (20, "Mallard", "Anas platyrhynchos", 0.85))]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=30)
            rows, _ = pipeline.process_file(FakeV3(), source, root / "forest",
                                            **dict(PROCESS_ARGS, site_habitats=("forest",)))
            fits = {r["Species (common)"]: r["Fits site habitat"] for r in rows}
            self.assertEqual(fits, {"Asian Tit": "Yes", "Mallard": "No — verify by ear"})
            rows, _ = pipeline.process_file(FakeV3(), source, root / "forest+marsh",
                                            **dict(PROCESS_ARGS, site_habitats=("forest", "marsh")))
            self.assertEqual({r["Species (common)"] for r in rows}, {"Asian Tit", "Gadwall", "Mallard"})
            rows, _ = pipeline.process_file(FakeV3(), source, root / "unspecified", **PROCESS_ARGS)
            self.assertEqual({r["Fits site habitat"] for r in rows}, {""})

    def test_v3_matches_geo_by_common_name_when_taxonomy_differs(self):
        from v3_model import V3Analyzer

        analyzer = V3Analyzer.__new__(V3Analyzer)
        analyzer.geo_model = types.SimpleNamespace(predict=lambda *_args, **_kwargs:
            types.SimpleNamespace(species_list=["Thinornis dubius_Little Ringed Plover",
                                                "Hesperiphona vespertina_Evening Grosbeak"],
                                  species_probs=[0.4, 0.0]))
        analyzer.model = types.SimpleNamespace(predict=lambda *_args, **_kwargs:
            types.SimpleNamespace(to_structured_array=lambda: [
                {"start_time": 1, "end_time": 4, "confidence": 0.8,
                 "species_name": "Charadrius dubius_Little Ringed Plover"},
                {"start_time": 5, "end_time": 8, "confidence": 0.74,
                 "species_name": "Coccothraustes vespertinus_Evening Grosbeak"},
            ]))
        analyzer._geo_cache = {}
        plover, grosbeak = analyzer.detections("sample.wav", 1.5, 0.5, 30.5, 114.4,
                                               datetime(2026, 10, 6))
        self.assertTrue(plover["location_filter_available"])
        self.assertTrue(plover["is_predicted_for_location_and_date"])
        self.assertTrue(grosbeak["location_filter_available"])
        self.assertFalse(grosbeak["is_predicted_for_location_and_date"])

    def test_strong_out_of_range_candidate_kept_for_review_when_opted_in(self):
        used_options = {}

        class FakeRecording:
            def __init__(self, _analyzer, _path, **kwargs):
                used_options.update(kwargs)
                self.detections = [
                    {"start_time": 1, "end_time": 2, "common_name": "Cinereous Tit",
                     "scientific_name": "Parus cinereus", "confidence": 0.81,
                     "is_predicted_for_location_and_date": False},
                    {"start_time": 5, "end_time": 6, "common_name": "Unexpected weak bird",
                     "scientific_name": "Avis weakus", "confidence": 0.4,
                     "is_predicted_for_location_and_date": False},
                    {"start_time": 10, "end_time": 11, "common_name": "Light-vented Bulbul",
                     "scientific_name": "Pycnonotus sinensis", "confidence": 0.31,
                     "is_predicted_for_location_and_date": True},
                ]

            def analyze(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=16)
            with patch.object(birdnetlib, "Recording", FakeRecording):
                rows, _ = pipeline.process_file(
                    None, source, root / "out",
                    **dict(PROCESS_ARGS, min_conf=0.25, out_of_range_min_conf=0.7)
                )
            self.assertEqual(used_options["overlap"], 1.5)
            self.assertTrue(used_options["return_all_detections"])
            # names follow eBird (Parus cinereus is "Asian Tit" there); BirdNET's name is kept alongside
            self.assertEqual({r["Species (common)"] for r in rows},
                             {"Asian Tit", "Light-vented Bulbul"})
            self.assertEqual({r["Species (common)"]: r["BirdNET name"] for r in rows},
                             {"Asian Tit": "Cinereous Tit", "Light-vented Bulbul": ""})
            flags = {r["Species (common)"]: r["Expected by location/date"] for r in rows}
            self.assertEqual(flags["Asian Tit"], "No — verify by ear")
            self.assertEqual(flags["Light-vented Bulbul"], "Yes")

    def test_changed_detection_settings_reanalyze_existing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=8)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                pipeline.write_summary(rows, day / "summary.xlsx", {str(source.resolve())})
                skipped, _ = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                refreshed, _ = pipeline.process_file(
                    None, source, root / "out", **dict(PROCESS_ARGS, overlap=0)
                )
            self.assertIsNone(skipped)
            self.assertIsNotNone(refreshed)
            self.assertNotEqual(rows[0]["Analysis settings"],
                                refreshed[0]["Analysis settings"])

    def test_changed_settings_do_not_replace_approved_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=8)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                summary = day / "summary.xlsx"
                pipeline.write_summary(rows, summary, {str(source.resolve())})
                ready = save_review(summary, 2, status="Approved", common="Test bird",
                                    scientific="Avis testus", rating=3,
                                    expected_file=rows[0]["File"])
                refreshed, _ = pipeline.process_file(
                    None, source, root / "out", **dict(PROCESS_ARGS, overlap=0)
                )
            self.assertIsNone(refreshed)
            self.assertTrue((day.parent / ready).is_file())   # <วัน>/Ready/ ข้างโฟลเดอร์เวลา
            self.assertEqual(load_reviews(summary)[0]["Review status"], "Approved")

    def test_missing_metadata_requires_manual_date_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "REC-001.WAV"
            make_wav(path)
            with self.assertRaisesRegex(ValueError, "Recording date missing"):
                pipeline.require_recording_context([path], None, True, None, None)
            with self.assertRaisesRegex(ValueError, "Recording location missing"):
                pipeline.require_recording_context([path], datetime(2026, 7, 15), True, None, None)
            pipeline.require_recording_context([path], datetime(2026, 7, 15), True, 13.8, 100.5)

    def test_approval_creates_ready_copy_and_rejection_removes_it(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.07.15_1200"
            original = day / "Unknown" / "clip.wav"
            original.parent.mkdir(parents=True)
            make_wav(original)
            summary = day / "summary.xlsx"
            pd.DataFrame([{"File": "Unknown/clip.wav", "Species (common)": "Unknown",
                           "Species (scientific)": "", "Max confidence": 0.71,
                           "Review status": "Pending"}]).to_excel(summary, index=False)
            rows = load_reviews(summary)
            self.assertEqual(rows[0]["_row"], 2)
            ready = save_review(summary, 2, status="Approved", common="Asian Koel",
                                scientific="Eudynamys scolopaceus", rating=3,
                                notes="Checked by ear", expected_file="Unknown/clip.wav")
            self.assertTrue((day / ready).is_file())
            self.assertEqual(load_reviews(summary)[0]["Review status"], "Approved")
            save_review(summary, 2, status="Rejected", common="Asian Koel",
                        scientific="Eudynamys scolopaceus", notes="Different bird",
                        expected_file="Unknown/clip.wav")
            self.assertFalse((day / ready).exists())
            self.assertEqual(load_reviews(summary)[0]["Review status"], "Rejected")

    def test_rerun_replaces_summary_and_superseded_generated_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp)
            first = day / "old.wav"
            first.write_bytes(b"old clip")
            source = str(day / "REC-001.WAV")
            first_row = {"File": first.name, "Generated sha256": pipeline.file_sha256(first),
                         "Source path": source, "Source ID": "abc123",
                         "Species (common)": "A", "Occurrence #": 1}
            summary = day / "summary.xlsx"
            pipeline.write_summary([first_row], summary, {source})
            second = day / "new.wav"
            second.write_bytes(b"new clip")
            second_row = dict(first_row, File=second.name,
                              **{"Generated sha256": pipeline.file_sha256(second)})
            pipeline.write_summary([second_row], summary, {source})
            rows = pd.read_excel(summary)
            self.assertEqual(list(rows["File"]), ["new.wav"])
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())

    def test_continuous_mode_exports_one_original_span(self):
        class FakeRecording:
            def __init__(self, *_args, **_kwargs):
                self.detections = [
                    {"start_time": start, "end_time": start + 1,
                     "common_name": "Test bird", "scientific_name": "Avis testus",
                     "confidence": 0.8}
                    for start in (1, 10)
                ]

            def analyze(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=16)
            with patch.object(birdnetlib, "Recording", FakeRecording):
                rows, day = pipeline.process_file(
                    None, source, root / "output", lat_arg=13.8, lon_arg=100.5,
                    cfg_lat=None, cfg_lon=None, date_override=datetime(2026, 7, 15),
                    use_meta=False, min_conf=0.5, gap=5, lead=3, tail=3,
                    target_dbfs=-3, fmt="wav", make_mono=True, incl_alt=False,
                    place_arg=None, dt_regex=pipeline.FILENAME_DATETIME_REGEX,
                    keep_continuous=True,
                )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Start offset (s)"], 1)
            self.assertEqual(rows[0]["End offset (s)"], 11)
            clip = day / rows[0]["File"]
            self.assertTrue(clip.is_file())
            with wave.open(str(clip), "rb") as audio:
                self.assertAlmostEqual(audio.getnframes() / audio.getframerate(), 14, delta=0.1)

    def test_failed_second_clip_does_not_publish_first_clip(self):
        class FakeRecording:
            def __init__(self, *_args, **_kwargs):
                self.detections = [
                    {"start_time": start, "end_time": start + 1,
                     "common_name": "Test bird", "scientific_name": "Avis testus",
                     "confidence": 0.8}
                    for start in (1, 10)
                ]

            def analyze(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=16)
            original_export = pipeline._export_clip
            calls = [0]

            def fail_second(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("simulated export failure")
                return original_export(*args, **kwargs)

            with patch.object(birdnetlib, "Recording", FakeRecording), \
                    patch.object(pipeline, "_export_clip", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "simulated export failure"):
                    pipeline.process_file(
                        None, source, root / "output", lat_arg=13.8, lon_arg=100.5,
                        cfg_lat=None, cfg_lon=None, date_override=datetime(2026, 7, 15),
                        use_meta=False, min_conf=0.5, gap=5, lead=3, tail=3,
                        target_dbfs=-3, fmt="wav", make_mono=True, incl_alt=False,
                        place_arg=None, dt_regex=pipeline.FILENAME_DATETIME_REGEX,
                    )
            published = [p for p in (root / "output").rglob("*.wav")
                         if ".staging" not in p.parts]
            self.assertEqual(published, [])

    def test_rerun_without_force_keeps_reviews_and_ready_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=40)
            with patch.object(birdnetlib, "Recording", fake_recording((1, 30))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                summary = day / "summary.xlsx"
                pipeline.write_summary(rows, summary, {str(source.resolve())})
                first, second = load_reviews(summary)
                ready = save_review(summary, first["_row"], status="Approved", common="Test bird",
                                    scientific="Avis testus", rating=4, expected_file=first["File"])
                save_review(summary, second["_row"], status="Rejected", common="Test bird",
                            scientific="Avis testus", expected_file=second["File"])
                (day / first["File"]).unlink()      # clip lost -> the source is analyzed again
                rows, _ = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                pipeline.write_summary(rows, summary, {str(source.resolve())})
                reviews = load_reviews(summary)
                self.assertEqual([r["Review status"] for r in reviews], ["Approved", "Rejected"])
                self.assertEqual(reviews[0]["Quality rating"], 4)
                self.assertTrue((day.parent / ready).is_file())

                # --force keeps its documented meaning: regenerate and reset reviews
                rows, _ = pipeline.process_file(None, source, root / "out", force=True, **PROCESS_ARGS)
                pipeline.write_summary(rows, summary, {str(source.resolve())}, keep_reviews=False)
                self.assertEqual({r["Review status"] for r in load_reviews(summary)}, {"Pending"})
                self.assertFalse((day.parent / ready).exists())

    def test_deleting_rejected_clip_does_not_trigger_reanalysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=8)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                summary = day / "summary.xlsx"
                pipeline.write_summary(rows, summary, {str(source.resolve())})
                row = load_reviews(summary)[0]
                save_review(summary, row["_row"], status="Rejected", common="Test bird",
                            scientific="Avis testus", expected_file=row["File"])
                (day / row["File"]).unlink()
                rows, _ = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
            self.assertIsNone(rows)   # skipped as already processed

    def test_interrupted_batch_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recordings = root / "recordings"
            recordings.mkdir()
            make_wav(recordings / "REC-001.WAV", seconds=8)
            make_wav(recordings / "REC-002.WAV", seconds=9)
            argv = ["field_audio_to_ebird.py", str(recordings), "-o", str(root / "out"),
                    "--model", "2.4",
                    "--date", "20260715", "--same-date-for-all", "--coords", "13.8,100.5",
                    "--start-time", "0600", "--no-use-metadata", "--no-spectrogram"]
            summary = root / "out" / "2026.07.15" / "0600" / "summary.xlsx"
            analyzer = types.ModuleType("birdnetlib.analyzer")
            analyzer.Analyzer = lambda: None

            def run(recording_class):
                output = io.StringIO()
                with patch.object(sys, "argv", argv), \
                        patch.dict(sys.modules, {"birdnetlib.analyzer": analyzer}), \
                        patch.object(birdnetlib, "Recording", recording_class), \
                        contextlib.redirect_stdout(output):
                    pipeline.main()
                return output.getvalue()

            with self.assertRaises(SystemExit) as stopped:
                run(fake_recording((1,), interrupt_on="REC-002.WAV"))
            self.assertEqual(stopped.exception.code, 130)
            self.assertEqual(set(pd.read_excel(summary)["Source file"]), {"REC-001.WAV"})

            output = run(fake_recording((1,)))
            self.assertIn("skip: already processed", output)
            self.assertEqual(set(pd.read_excel(summary)["Source file"]),
                             {"REC-001.WAV", "REC-002.WAV"})
            self.assertIn(f"summary already up to date -> {summary}", run(fake_recording((1,))))

    def test_leftover_clips_from_interrupted_run_are_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=8)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                # stopped before summary.xlsx was written; the identical clip is ours to replace
                rows, _ = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
            self.assertEqual(len(rows), 1)

    def test_force_redo_drops_spectrogram_of_changed_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=20)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                def redo(**changes):
                    rows, day = pipeline.process_file(None, source, root / "out", force=True,
                                                      **dict(PROCESS_ARGS, **changes))
                    pipeline.write_summary(rows, day / "summary.xlsx", {str(source.resolve())},
                                           keep_reviews=False)
                    return rows, day

                rows, day = redo()
                png = (day / rows[0]["File"]).with_suffix(".png")
                png.write_bytes(b"spectrogram")
                redo()
                self.assertTrue(png.exists())       # same clip -> spectrogram still valid
                rows, _ = redo(tail=10)
            self.assertEqual((day / rows[0]["File"]).with_suffix(".png"), png)
            self.assertFalse(png.exists())          # regenerated by gen_spectrograms instead

    def test_file_timestamp_fallback_uses_recording_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "REC-001.WAV"
            make_wav(source, seconds=120)
            closed = datetime(2026, 7, 15, 6, 32).timestamp()   # recorder closes the file at the end
            os.utime(source, (closed, closed))
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                _rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
            self.assertEqual(day.relative_to(root / "out"), Path("2026.07.15", "0630"))   # วัน/เวลาเริ่ม

    def test_recordings_of_one_day_share_a_day_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = []
            for name, hour in (("REC-001.WAV", 6), ("REC-002.WAV", 7)):
                source = root / name
                make_wav(source, seconds=120)
                closed = datetime(2026, 7, 15, hour, 32).timestamp()
                os.utime(source, (closed, closed))
                with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                    _rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                folders.append(day.relative_to(root / "out"))
            self.assertEqual(folders, [Path("2026.07.15", "0630"), Path("2026.07.15", "0730")])

    def test_timestamps_with_timezone_become_local_time(self):
        self.addCleanup(time.tzset)   # runs after patch.dict restores TZ
        with patch.dict(os.environ, {"TZ": "Asia/Bangkok"}):
            time.tzset()
            self.assertEqual(pipeline._parse_meta_datetime("2026-07-14T23:30:00.000000Z"),
                             datetime(2026, 7, 15, 6, 30))
            self.assertEqual(pipeline._parse_meta_datetime("2026-07-15 06:30:00"),
                             datetime(2026, 7, 15, 6, 30))

    def test_occurrences_sort_as_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp) / "summary.xlsx"
            rows = [{"File": f"{n}.wav", "Species (common)": "A", "Occurrence #": n,
                     "Source ID": "abc", "Source file": "REC.WAV"} for n in (10, 2, 1)]
            pipeline.write_summary(rows, summary)
            self.assertEqual(list(pd.read_excel(summary)["Occurrence #"]), [1, 2, 10])

    def test_gps_and_map_link_parsing(self):
        self.assertAlmostEqual(pipeline._gps_exif("12,48,30N"), 12 + 48 / 60 + 30 / 3600)
        self.assertAlmostEqual(pipeline._gps_exif("99,37.2W"), -99.62)
        link = ("https://www.google.com/maps/place/X/@13.80,100.50,15z/data=!3m1!4b1"
                "!4m6!3m5!1s0x0:0x0!8m2!3d13.7563309!4d100.5017651")
        self.assertEqual(pipeline.parse_coords(link), (13.7563309, 100.5017651))


def make_tone(path: Path, seconds=40, rate=48000, freq=4000, subtype="PCM_24", channels=2):
    """Quiet 24-bit stereo tone, like a field recorder file."""
    import numpy as np
    import soundfile as sf
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.02 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), np.column_stack([tone] * channels) if channels > 1 else tone, rate, subtype=subtype)


class ManualClipTests(unittest.TestCase):
    def test_time_entries_for_position_and_clock(self):
        from manual_clips import format_offset, parse_clock, parse_offset
        self.assertEqual(parse_offset("15:04"), 904)
        self.assertEqual(parse_offset("15:04.5"), 904.5)
        self.assertEqual(parse_offset("904"), 904)
        self.assertEqual(parse_offset("1:00:05"), 3605)
        for bad in ("15:75", "abc", ""):
            with self.assertRaises(ValueError):
                parse_offset(bad)
        start = datetime(2026, 10, 18, 14, 39, 44)
        self.assertEqual(parse_clock("14:54:49", start), 905)
        self.assertEqual(parse_clock("145449", start), 905)
        self.assertEqual(parse_clock("1455", start), 916)
        night = datetime(2026, 10, 18, 23, 50, 0)
        self.assertEqual(parse_clock("00:10", night), 1200)          # after midnight
        self.assertEqual(format_offset(904.5), "15:04.5")
        self.assertEqual(format_offset(3725, 0), "1:02:05")

    def test_recording_start_comes_from_existing_rows(self):
        from manual_clips import source_context, sources_in
        settings = json.dumps({"date": "2026-10-18", "habitat": ["forest"]})
        rows = [{"Source path": "/r/REC-001.WAV", "Start offset (s)": 189.0, "Clock time": "14:42:53",
                 "Analysis settings": settings, "Latitude": 30.5, "Longitude": 114.3, "Source ID": "abc"},
                {"Source path": "/r/REC-002.WAV", "Start offset (s)": 1800.0, "Clock time": "00:10:00",
                 "Analysis settings": json.dumps({"date": "2026-10-18"}), "Latitude": 30.5, "Longitude": 114.3}]
        self.assertEqual([p.name for p in sources_in(rows)], ["REC-001.WAV", "REC-002.WAV"])
        first = source_context(rows, "/r/REC-001.WAV")
        self.assertEqual(first["start"], datetime(2026, 10, 18, 14, 39, 44))
        self.assertEqual((first["lat"], first["habitat"], first["source_id"]), (30.5, ["forest"], "abc"))
        # 23:40 start, clip 30 min later after midnight: the start stays on the 18th
        self.assertEqual(source_context(rows, "/r/REC-002.WAV")["start"], datetime(2026, 10, 18, 23, 40))
        self.assertIsNone(source_context(rows, "/r/other.wav"))

    def test_clock_times_are_exact_for_half_second_offsets(self):
        from manual_clips import heard_span, recording_start_of
        # started 14:39:44; BirdNET windows start on half seconds, Clock time drops the fraction
        row = {"Start offset (s)": 904.5, "End offset (s)": 911.5, "Clock time": "14:54:48",
               "Analysis settings": json.dumps({"date": "2026-10-18"})}
        self.assertEqual(recording_start_of(row), datetime(2026, 10, 18, 14, 39, 44))
        self.assertEqual(heard_span(row), (datetime(2026, 10, 18, 14, 54, 48, 500000),
                                           datetime(2026, 10, 18, 14, 54, 55, 500000)))
        self.assertIsNone(heard_span({"Start offset (s)": 3.0}))       # old summary without settings

    def test_clip_spectrogram_has_clock_axis_and_old_images_are_redrawn(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        pipeline._clock_axis(ax, datetime(2026, 10, 18, 14, 54, 44, 500000), 9.5)
        self.assertEqual([round(t, 2) for t in ax.get_xticks()][:2], [0.5, 1.5])
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()][:2], ["14:54:45", "14:54:46"])
        plt.close(fig)
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp)
            make_wav(day / "clip.wav", seconds=12)
            boxes = {"windows": [[3.0, 9.0, 0.9]], "sounds": []}
            pd.DataFrame([{"File": "clip.wav", "Species (common)": "A", "Detection boxes": json.dumps(boxes),
                           "Start offset (s)": 904.5, "Clock time": "14:54:48",
                           "Analysis settings": json.dumps({"date": "2026-10-18"})}]).to_excel(
                day / "summary.xlsx", index=False)
            (day / "clip.png").write_bytes(b"image drawn before clock time")
            self.assertEqual(pipeline.spectrograms_to_update(day), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.gen_spectrograms([day])
            # clip starts 3 s before BirdNET's first window: 14:39:44 + 904.5 - 3 = 14:54:45.5
            self.assertTrue(pipeline._png_description(day / "clip.png").endswith("clock 14:54:45.500000"))
            self.assertEqual(pipeline.spectrograms_to_update(day), 0)

    def test_added_clip_is_cut_from_original_and_survives_reanalysis(self):
        import soundfile as sf
        from manual_clips import add_clip, source_context
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            make_tone(source)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                self.assertEqual(rows[0]["Clock end"],
                                 (datetime.strptime(rows[0]["Clock time"], "%H:%M:%S")
                                  + pipeline.timedelta(seconds=1)).strftime("%H:%M:%S"))
                summary = day / "summary.xlsx"
                pipeline.write_summary(rows, summary, {str(source)}, {pipeline.source_id(source)})
                context = source_context(load_reviews(summary), source)
                row = add_clip(summary, source, 20.0, 27.5, "Rufous-bellied Woodpecker",
                               "Dendrocopos hyperythrus", recorded=context["start"], lat=context["lat"],
                               lon=context["lon"], sound_type="Call", spectrogram=False,
                               check_location=False, log=lambda _text: None)
                clip = day / row["File"]
                info = sf.info(str(clip))
                self.assertEqual((info.subtype, info.channels), ("PCM_24", 1))
                self.assertAlmostEqual(info.duration, 7.5, places=2)
                self.assertIn("_M001_", clip.name)
                saved = {r["File"]: r for r in load_reviews(summary)}[row["File"]]
                self.assertEqual((saved["Origin"], saved["Sound type"], saved["Review status"]),
                                 ("Manual", "Call", "Pending"))
                self.assertIsNone(saved["Max confidence"])
                self.assertEqual(saved["Clock time"],
                                 (context["start"] + pipeline.timedelta(seconds=20)).strftime("%H:%M:%S"))
                self.assertEqual(saved["Clock end"],
                                 (context["start"] + pipeline.timedelta(seconds=27.5)).strftime("%H:%M:%S"))

                # a normal rerun still sees the BirdNET result as done
                again, _ = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
                self.assertIsNone(again)
                # force redo replaces BirdNET clips but keeps the reviewer's clip, its file and review
                ready = save_review(summary, saved["_row"], status="Approved", common=saved["Species (common)"],
                                    scientific=saved["Species (scientific)"], rating=4,
                                    expected_file=saved["File"])
                rows, _ = pipeline.process_file(None, source, root / "out", force=True, **PROCESS_ARGS)
                pipeline.write_summary(rows, summary, {str(source)}, {pipeline.source_id(source)},
                                       keep_reviews=False)
                after = {r["File"]: r for r in load_reviews(summary)}
                self.assertEqual(after[row["File"]]["Review status"], "Approved")
                self.assertTrue(clip.is_file() and (day.parent / ready).is_file())
                self.assertEqual(len(after), 2)
                # a second clip of the same species gets the next number
                second = add_clip(summary, source, 30.0, 33.0, "Rufous-bellied Woodpecker",
                                  "Dendrocopos hyperythrus", recorded=context["start"], lat=context["lat"],
                                  lon=context["lon"], spectrogram=False, check_location=False,
                                  log=lambda _text: None)
                self.assertIn("_M002_", second["File"])

    def test_added_clip_refuses_bad_input_and_changed_recording(self):
        from manual_clips import add_clip
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            make_tone(source, seconds=12)
            summary = root / "day" / "summary.xlsx"
            args = dict(recorded=datetime(2026, 10, 18, 14, 0), lat=30.5, lon=114.3, spectrogram=False,
                        check_location=False, log=lambda _text: None)
            with self.assertRaises(ValueError):
                add_clip(summary, source, 2, 2.2, "Bird", "Avis testus", **args)        # too short
            with self.assertRaises(ValueError):
                add_clip(summary, source, 2, 5, "Bird", "Avis", **args)                 # no species epithet
            add_clip(summary, source, 2, 5, "Bird", "Avis testus", **args)              # no result yet: new summary
            make_tone(source, seconds=13)                                               # file replaced
            with self.assertRaises(ValueError):
                add_clip(summary, source, 2, 5, "Bird", "Avis testus", **args)

    def test_birdnet_suggestions_for_a_short_selection(self):
        from manual_clips import suggest
        calls = []

        class FakeAnalyzer:
            def detections(self, wav, overlap, floor, lat, lon, recorded):
                import soundfile as sf
                calls.append((round(sf.info(str(wav)).duration, 2), floor, recorded))
                return [
                    {"start_time": 0, "end_time": 3, "common_name": "Rufous-bellied Woodpecker",
                     "scientific_name": "Dendrocopos hyperythrus", "confidence": 0.38,
                     "is_predicted_for_location_and_date": True, "location_filter_available": True},
                    {"start_time": 0, "end_time": 3, "common_name": "Buff-necked Woodpecker",
                     "scientific_name": "Meiglyptes tukki", "confidence": 0.12,
                     "is_predicted_for_location_and_date": False, "location_filter_available": True},
                    {"start_time": 1.5, "end_time": 4.5, "common_name": "Rufous-bellied Woodpecker",
                     "scientific_name": "Dendrocopos hyperythrus", "confidence": 0.2,
                     "is_predicted_for_location_and_date": True, "location_filter_available": True},
                ]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "REC-001.WAV"
            make_tone(source, seconds=12)
            start = datetime(2026, 10, 18, 14, 39, 44)
            ranking = suggest(source, 5.0, 6.0, 30.54, 114.36, start, analyzer=FakeAnalyzer())
        self.assertEqual(calls[0][0], 3.0)                   # widened to one BirdNET window
        self.assertEqual(calls[0][2], datetime(2026, 10, 18, 14, 39, 48))   # 4.0-7.0 s around 5.5 s
        self.assertEqual([(r["common"], r["confidence"], r["expected"]) for r in ranking],
                         [("Rufous-bellied Woodpecker", 0.38, True), ("Buff-necked Woodpecker", 0.12, False)])
        self.assertEqual(ranking[0]["checklist"], "native in China")

    def test_location_check_uses_geo_model_only(self):
        from v3_model import location_check

        class Prediction:
            species_list = ["Dendrocopos hyperythrus_Rufous-bellied Woodpecker", "Meiglyptes tukki_Buff-necked Woodpecker"]
            species_probs = [0.2, 0.001]

        class Geo:
            def predict(self, *_args, **_kwargs):
                return Prediction()
        when = datetime(2026, 10, 18)
        self.assertTrue(location_check("Dendrocopos hyperythrus", "", 30.5, 114.3, when, geo_model=Geo()))
        self.assertFalse(location_check("Meiglyptes tukki", "", 30.5, 114.3, when, geo_model=Geo()))
        self.assertTrue(location_check("Other name", "Rufous-bellied Woodpecker", 30.5, 114.3, when, geo_model=Geo()))
        self.assertIsNone(location_check("Avis testus", "Test bird", 30.5, 114.3, when, geo_model=Geo()))

    def test_species_search_lists_birds_of_the_recording_country_first(self):
        from manual_window import species_matches
        found = species_matches("rufous woodpecker", "CN")
        self.assertIn(("Rufous-bellied Woodpecker", "Dendrocopos hyperythrus"), found)
        import country_checklist
        statuses = [country_checklist._species()[0][sci]["CN"] for _common, sci in found]
        natives = [status in ("native", "endemic") for status in statuses]
        self.assertEqual(natives, sorted(natives, reverse=True))
        self.assertEqual(species_matches("dendrocopos hyper", None)[0][1], "Dendrocopos hyperythrus")
        self.assertEqual(species_matches("   ", "CN"), [])

    def test_view_audio_and_spectrogram_from_long_file(self):
        import numpy as np
        import soundfile as sf
        from manual_window import read_mono, spectrogram_image, write_preview
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "REC-001.WAV"
            make_tone(source, seconds=20, freq=6000)
            samples, sr = read_mono(source, 5.0, 15.0)
            self.assertEqual((len(samples), sr), (480000, 48000))
            with patch("soundfile.info", side_effect=RuntimeError("unsupported")):   # m4a path: ffmpeg
                decoded, decoded_sr = read_mono(source, 5.0, 15.0)
            self.assertEqual(decoded_sr, 48000)
            self.assertAlmostEqual(len(decoded) / decoded_sr, 10.0, places=1)
            image = np.asarray(spectrogram_image(samples, sr, 900, 250))
            self.assertEqual(image.shape, (250, 900, 3))
            brightest = int(np.argmax(image.sum(axis=(1, 2))))          # 6 kHz of 12 kHz = middle row
            self.assertLess(abs(brightest - 125), 6)
            preview = Path(temp) / "p.wav"
            write_preview(samples, sr, preview)
            data, _ = sf.read(str(preview))
            self.assertAlmostEqual(float(np.abs(data).max()), 0.7, places=2)   # quiet tone made audible


class SpeciesNameTests(unittest.TestCase):
    def test_birdnet_names_become_ebird_names(self):
        from species_names import to_ebird
        self.assertEqual(to_ebird("Aegithalos concinnus", "Black-throated Bushtit"),
                         {"common": "Black-throated Tit", "scientific": "Aegithalos concinnus", "code": "blttit2"})
        self.assertEqual(to_ebird("Ardea cinerea", "Grey Heron")["common"], "Gray Heron")
        # different scientific name: matched by the English name
        self.assertEqual(to_ebird("Charadrius dubius", "Little Ringed Plover")["scientific"], "Thinornis dubius")
        # BirdNET's split species that eBird lumps: matched ignoring Bushtit/Tit and hyphens
        self.assertEqual(to_ebird("Aegithalos bonvaloti", "Black-browed Bushtit")["common"], "Black-browed Tit")
        self.assertEqual(to_ebird("Ptilinopus jambu", "Jambu Fruit Dove")["common"], "Jambu Fruit-Dove")
        self.assertIsNone(to_ebird("Engine", "Engine"))
        self.assertIsNone(to_ebird("Avis testus", "Test bird"))

    def test_reviews_follow_a_clip_renamed_to_its_ebird_name(self):
        old = pd.DataFrame([{"File": "Black-throated Bushtit/a_001_R0.wav", "Generated sha256": "s1",
                             "Start offset (s)": 12.0, "Review status": "Approved", "Quality rating": 4,
                             "Ready file": "Ready/Black-throated Tit/a_R4.wav"},
                            {"File": "Black-throated Bushtit/a_002_R0.wav", "Generated sha256": "s1",
                             "Start offset (s)": 40.5, "Review status": "Rejected"}])
        new = pd.DataFrame([{"File": "Black-throated Tit/a_001_R0.wav", "Generated sha256": "s1",
                             "Start offset (s)": 12.0, "Review status": "Pending"},
                            {"File": "Black-throated Tit/a_003_R0.wav", "Generated sha256": "s1",
                             "Start offset (s)": 90.0, "Review status": "Pending"}])
        carried = pipeline._carry_over_reviews(new, old)
        self.assertEqual(list(carried["Review status"]), ["Approved", "Pending"])   # same audio, same place only
        self.assertEqual(carried.loc[0, "Ready file"], "Ready/Black-throated Tit/a_R4.wav")

    def test_old_results_show_ebird_names_and_keep_birdnet_name(self):
        from species_names import birdnet_name, row_names
        old = {"Species (common)": "Black-throated Bushtit", "Species (scientific)": "Aegithalos concinnus"}
        self.assertEqual(row_names(old), ("Black-throated Tit", "Aegithalos concinnus"))
        self.assertEqual(birdnet_name(old), "Black-throated Bushtit")
        new = {"Species (common)": "Black-throated Tit", "Species (scientific)": "Aegithalos concinnus",
               "BirdNET name": "Black-throated Bushtit"}
        self.assertEqual(birdnet_name(new), "Black-throated Bushtit")
        same = {"Species (common)": "Light-vented Bulbul", "Species (scientific)": "Pycnonotus sinensis",
                "BirdNET name": ""}
        self.assertEqual(birdnet_name(same), "")
        self.assertEqual(row_names({"Species (common)": "Test bird", "Species (scientific)": "Avis testus"}),
                         ("Test bird", "Avis testus"))


class ReviewProgressTests(unittest.TestCase):
    @staticmethod
    def _records(rows):
        import review_progress
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp) / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            return review_progress.collect([summary])

    def test_outcome_counts_rejects_and_renamed_approvals_as_wrong_ids(self):
        base = {"Species (common)": "Black-throated Bushtit", "Species (scientific)": "Aegithalos concinnus"}
        records = self._records([
            dict(base, File="a.wav", **{"Review status": "Approved", "Reviewed scientific": "Aegithalos concinnus"}),
            # poor sound of the right species is approved with quality 1: still a correct ID
            dict(base, File="b.wav", **{"Review status": "Approved", "Reviewed scientific": "Aegithalos concinnus",
                                        "Quality rating": 1}),
            dict(base, File="c.wav", **{"Review status": "Approved", "Reviewed scientific": "Aegithalos glaucogularis"}),
            dict(base, File="d.wav", **{"Review status": "Rejected"}),                 # reject = wrong species
            dict(base, File="e.wav", **{"Review status": "Pending"}),
        ])
        self.assertEqual([r["_outcome"] for r in records], ["correct", "correct", "wrong", "wrong", "pending"])
        self.assertEqual({r["_species"] for r in records}, {"Black-throated Tit"})       # eBird name

    def test_filter_report_and_next_clip(self):
        import review_progress
        rows = []
        for n in range(12):     # 12 warned by the location model: 9 wrong
            rows.append({"File": f"w{n}.wav", "Species (common)": "Red-whiskered Bulbul",
                         "Species (scientific)": "Pycnonotus jocosus", "Max confidence": 0.9,
                         "Expected by location/date": "No — verify by ear",
                         "Review status": "Rejected" if n < 9 else "Approved"})
        for n in range(12):     # 12 not warned: 1 wrong
            rows.append({"File": f"c{n}.wav", "Species (common)": "Light-vented Bulbul",
                         "Species (scientific)": "Pycnonotus sinensis", "Max confidence": 0.9,
                         "Expected by location/date": "Yes",
                         "Review status": "Rejected" if n == 0 else "Approved"})
        rows += [{"File": "p1.wav", "Species (common)": "Light-vented Bulbul", "Species (scientific)": "Pycnonotus sinensis",
                  "Max confidence": 0.95, "Expected by location/date": "Yes", "Review status": "Pending"},
                 {"File": "p2.wav", "Species (common)": "Asian Tit", "Species (scientific)": "Parus cinereus",
                  "Max confidence": 0.7, "Expected by location/date": "Yes", "Review status": "Pending"},
                 {"File": "p3.wav", "Species (common)": "Asian Tit", "Species (scientific)": "Parus cinereus",
                  "Max confidence": 0.5, "Expected by location/date": "Yes", "Review status": "Pending"},
                 {"File": "m1.wav", "Species (common)": "Rufous-bellied Woodpecker", "Origin": "Manual",
                  "Species (scientific)": "Dendrocopos hyperythrus", "Review status": "Pending"}]
        records = self._records(rows)
        report = review_progress.filter_report(records)
        self.assertEqual(report["location"], {"flagged": [12, 9], "clear": [12, 1]})
        self.assertEqual(review_progress.verdict(report["location"]),
                         "catches 90% of wrong IDs; 25% of its warnings were right IDs")
        self.assertEqual(review_progress.verdict(report["habitat"]), "need more reviews")
        # Asian Tit has no reviews yet; its low-score (warned) clip comes first; the manual clip is never picked
        self.assertEqual(review_progress.next_clip(records)["File"], "p3.wav")
        table = review_progress.species_table(records)
        self.assertEqual((table["Red-whiskered Bulbul"]["wrong"], table["Light-vented Bulbul"]["pending"],
                          table["Rufous-bellied Woodpecker"]["clips"]), (9, 1, 1))


class AutoCheckTests(unittest.TestCase):
    def test_non_bird_sound_as_strong_as_the_bird(self):
        from auto_checks import nonbird_competitor
        dets = [(0.0, 3.0, "Aegithalos concinnus_Black-throated Bushtit", 0.61),   # BirdNET name, eBird = Black-throated Tit
                (0.0, 3.0, "Gryllus bimaculatus_Field cricket", 0.83),
                (9.0, 12.0, "Hyla chinensis_Chinese tree toad", 0.95)]                # outside the bird's span
        found = nonbird_competitor(dets, "Aegithalos concinnus", spans=[(0.0, 4.5)])
        self.assertEqual((found["common"], found["score"], found["bird"]), ("Field cricket", 0.83, 0.61))
        self.assertIsNone(nonbird_competitor(dets[:1] + [(0.0, 3.0, "Gryllus bimaculatus_Field cricket", 0.4)],
                                             "Aegithalos concinnus", spans=[(0.0, 4.5)]))
        self.assertIsNone(nonbird_competitor([(0.0, 3.0, "Gryllus bimaculatus_Field cricket", 0.2)],
                                             "Aegithalos concinnus"))           # too weak to matter

    def test_second_opinion_disagreement(self):
        from second_opinion import disagrees
        result = {"current": "Pycnonotus jocosus", "ranking": [
            {"scientific": "Pycnonotus sinensis", "similarity": 0.8},
            {"scientific": "Pycnonotus jocosus", "similarity": 0.7}]}
        self.assertTrue(disagrees(result))
        result["ranking"].reverse()
        self.assertFalse(disagrees(result))
        self.assertIsNone(disagrees({"current": "A b", "ranking": [{"scientific": "A b", "similarity": 0.9}]}))
        self.assertIsNone(disagrees(None))

    def test_unchecked_clips_are_left_out_of_the_filter_table(self):
        import auto_checks
        import review_progress
        import second_opinion
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            rows = [{"File": f"{n}.wav", "Generated sha256": f"sha{n}", "Species (common)": "Light-vented Bulbul",
                     "Species (scientific)": "Pycnonotus sinensis", "Max confidence": 0.55,
                     "Review status": "Rejected" if n == 0 else "Approved"} for n in range(3)]
            with patch.object(auto_checks, "CACHE", cache / "auto"), \
                    patch.object(second_opinion, "CACHE", cache / "second"), \
                    patch.object(auto_checks.sound_type, "needs_work", lambda _row, _key: False):
                auto_checks._save("sha0", {"nonbird": {"common": "Field cricket", "score": 0.8, "bird": 0.55}})
                auto_checks._save("sha1", {"nonbird": None})
                self.assertTrue(auto_checks.needs_check(rows[2], has_xc_key=False))     # never checked
                self.assertFalse(auto_checks.needs_check(rows[1], has_xc_key=False))
                self.assertTrue(auto_checks.needs_check(rows[1], has_xc_key=True))      # score < 0.65: wants a 2nd opinion
                with tempfile.TemporaryDirectory() as day:
                    summary = Path(day) / "summary.xlsx"
                    pd.DataFrame(rows).to_excel(summary, index=False)
                    records = review_progress.collect([summary])
                    report = review_progress.filter_report(records)
                self.assertEqual(report["nonbird"], {"flagged": [1, 1], "clear": [1, 0]})   # sha2 not counted
                self.assertEqual(report["second"], {"flagged": [0, 0], "clear": [0, 0]})
                self.assertEqual(auto_checks.describe_nonbird(rows[0]), "Field cricket 0.80 vs bird 0.55")


class XenoCantoDownloadTests(unittest.TestCase):
    def test_reference_download_with_curl_and_without(self):
        import shutil
        import xeno_canto
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "source.wav"
            make_wav(source, seconds=2)
            recording = {"id": "XC1", "file": source.as_uri(), "ext": ".wav"}
            got = xeno_canto.download(recording, cache=temp / "with-curl")
            self.assertEqual(got.read_bytes(), source.read_bytes())
            with patch.object(shutil, "which", lambda _name: None):            # no curl: Python download
                got = xeno_canto.download(recording, cache=temp / "python")
            self.assertEqual(got.read_bytes(), source.read_bytes())
            with self.assertRaises(xeno_canto.XenoCantoError):
                xeno_canto.download({"id": "XC2", "file": (temp / "missing.wav").as_uri(), "ext": ".wav"},
                                    cache=temp / "with-curl")
            self.assertEqual(sorted(p.name for p in (temp / "with-curl").iterdir()), ["XC1.wav"])  # no .part left


class SlowXenoCantoTests(unittest.TestCase):
    def tearDown(self):
        import xeno_canto
        xeno_canto.MIN_SPEED, xeno_canto.downloads_paused = None, False

    def test_background_downloads_pause_when_slow_but_yours_do_not(self):
        import subprocess
        import xeno_canto
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            return types.SimpleNamespace(returncode=28 if "--speed-limit" in command else 0, stderr="", stdout="")

        with tempfile.TemporaryDirectory() as temp, patch.object(subprocess, "run", fake_run):
            cache = Path(temp)
            (cache / "XC9.mp3").write_bytes(b"cached")
            recording = {"id": "XC1", "file": "https://xeno-canto.org/1/download", "ext": ".mp3"}
            with self.assertRaises(xeno_canto.XenoCantoError):          # your download: no speed limit
                xeno_canto.download(recording, cache=cache)
            self.assertNotIn("--speed-limit", calls[-1])
            xeno_canto.MIN_SPEED = 50 * 1024                             # what Auto-checks sets
            with self.assertRaises(xeno_canto.XenoCantoSlow):
                xeno_canto.download(recording, cache=cache)
            self.assertIn("51200", calls[-1])
            self.assertTrue(xeno_canto.downloads_paused)
            count = len(calls)
            with self.assertRaises(xeno_canto.XenoCantoSlow):
                xeno_canto.download(dict(recording, id="XC2"), cache=cache)   # paused: not even tried
            self.assertEqual(len(calls), count)
            self.assertEqual(xeno_canto.download({"id": "XC9", "file": "x", "ext": ".mp3"}, cache=cache).name,
                             "XC9.mp3")                                         # cached files still work
            with self.assertRaises(xeno_canto.XenoCantoSlow):
                xeno_canto.search("Avis testus", "key")
            self.assertEqual(sorted(p.name for p in cache.iterdir()), ["XC9.mp3"])   # no .part left

    def test_second_opinion_does_not_use_an_incomplete_reference_set(self):
        import second_opinion
        import xeno_canto

        def slow(_recording):
            raise xeno_canto.XenoCantoSlow("slow")

        with tempfile.TemporaryDirectory() as temp, patch.object(second_opinion, "CACHE", Path(temp)), \
                patch.object(xeno_canto, "download", slow):
            with self.assertRaises(xeno_canto.XenoCantoSlow):
                second_opinion.embed_references(None, "Avis testus", "Avis testus_Test bird",
                                                [{"id": "XC1"}, {"id": "XC2"}], log=lambda _text: None)

    def test_auto_checks_finish_the_rest_and_say_slow(self):
        import numpy as np
        import auto_checks
        import second_opinion
        import sound_type
        import xeno_canto

        def slow_second(*_args, **_kwargs):
            xeno_canto.downloads_paused = True
            raise xeno_canto.XenoCantoSlow("XC1: slower than 50 KB/s — downloads paused")

        def slow_references(*_args, **_kwargs):
            raise xeno_canto.XenoCantoSlow("paused")

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            day = temp / "day"
            day.mkdir()
            rows = []
            for n in range(3):
                make_wav(day / f"{n}.wav", seconds=3)
                rows.append({"File": f"{n}.wav", "Generated sha256": f"s{n}", "Species (common)": "Test bird",
                             "Species (scientific)": "Avis testus", "Max confidence": 0.55})
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            logs = []
            analyzer = types.SimpleNamespace(model=types.SimpleNamespace(species_list=["Avis testus_Test bird"]))
            with patch.object(auto_checks, "CACHE", temp / "auto"), patch.object(sound_type, "CACHE", temp / "st"), \
                    patch.object(second_opinion, "CACHE", temp / "so"), \
                    patch.object(second_opinion, "hear", lambda _m, clips, **_k: {str(c): [] for c in clips}), \
                    patch.object(second_opinion, "vectors",
                                 lambda _m, clips, **_k: {str(c): {0.0: np.eye(8)[0]} for c in clips}), \
                    patch.object(second_opinion, "run", slow_second), \
                    patch.object(sound_type, "ensure_references", slow_references), \
                    patch.object(sound_type, "train", lambda *_a, **_k: None), \
                    patch.object(auto_checks, "all_results", lambda _root: []):
                auto_checks.run(summary, xc_key="key", analyzer=analyzer, log=logs.append)
                slow_lines = [line for line in logs if line.startswith("SLOW ")]
                self.assertEqual(len(slow_lines), 1)
                self.assertIn("4 second opinion(s)/reference set(s)", slow_lines[0])   # 3 clips + 1 species
                self.assertTrue(all("nonbird" in auto_checks.cached(f"s{n}") for n in range(3)))   # offline part done
                self.assertTrue(all("second_error" not in auto_checks.cached(f"s{n}") for n in range(3)))
                self.assertTrue(sound_type.needs_references("Avis testus"))     # retried next time
                self.assertTrue(auto_checks.needs_check(rows[0], True))


class ModelCheckTests(unittest.TestCase):
    @staticmethod
    def _row(n, score, outcome, season_warned=False, source="s0"):
        return {"_manual": False, "_outcome": outcome, "_summary": Path("/r/summary.xlsx"), "Source ID": source,
                "Species (common)": "Test bird", "Species (scientific)": "Avis testus", "Max confidence": score,
                "Alt1 conf": 0.1, "Generated sha256": f"none{n}", "_season_key": ("2026-10-18", f"Avis {n}"),
                "_season_warned": season_warned}

    def test_model_waits_for_enough_reviews(self):
        import review_progress
        rows = [self._row(n, 0.9, "correct") for n in range(25)] + [self._row(n, 0.55, "wrong") for n in range(25, 32)]
        result = review_progress.model_check(rows)
        self.assertFalse(result["ready"])
        self.assertEqual(result["need"], {"reviews": 118, "wrong": 23, "correct": 5})
        self.assertTrue(review_progress.describe_model(result).startswith(
            "Needs 118 more reviewed clips and 23 more wrong IDs and 5 more right IDs (now 32 reviewed, 7 wrong)"))

    def test_model_beats_the_score_when_another_signal_matters(self):
        import numpy as np
        import review_progress
        random = np.random.RandomState(0)
        rows = []
        for n in range(240):
            score = float(random.uniform(0.5, 1.0))
            warned = bool(random.rand() < 0.3)
            wrong = (warned and random.rand() < 0.8) or (score < 0.6 and random.rand() < 0.5)
            rows.append(self._row(n, round(score, 3), "wrong" if wrong else "correct", warned, f"s{n % 24}"))
        warned_keys = {row["_season_key"] for row in rows if row["_season_warned"]}
        result = review_progress.model_check(rows, warned_keys)
        self.assertTrue(result["ready"] and result["by_recording"])
        self.assertTrue(result["better"], result["auc"])
        self.assertIn("eBird past years warns", result["top_signals"])
        self.assertGreater(result["same_warnings"]["model"], result["same_warnings"]["score_rule"])
        self.assertIn("→ The model is better", review_progress.describe_model(result))


class EBirdSeasonTests(unittest.TestCase):
    def test_same_dates_in_past_years_skip_recording_day_and_future(self):
        import ebird
        from datetime import date
        days = ebird.season_dates(datetime(2024, 2, 29, 7, 30))
        self.assertEqual(len(days), 7 * 3 + 6)                     # 2021-2023 full, 2024 without the day itself
        self.assertNotIn(date(2024, 2, 29), days)
        self.assertIn(date(2021, 2, 25), days)                     # 29 Feb -> 28 Feb in other years
        self.assertIn(date(2021, 3, 3), days)
        self.assertTrue(all(day < date.today() for day in ebird.season_dates(datetime.now())))

    def test_season_reports_count_days_and_judge_sparse_regions(self):
        import ebird
        from birdnet_gui import format_season
        from datetime import date
        reported = {date(2023, 10, 17): ["rubwoo1", "livbul1"], date(2025, 10, 19): ["livbul1"]}
        busy = [f"sp{n}" for n in range(60)]
        with patch.object(ebird, "historic_day", lambda _region, day, _key: reported.get(day, busy[:1])):
            season = ebird.season_reports("CN-42", date(2026, 10, 18), "key")
        self.assertEqual(season["reported"]["rubwoo1"], ["2023-10-17"])
        self.assertEqual(len(season["reported"]["livbul1"]), 2)
        self.assertEqual(ebird.season_status(season, "rubwoo1"), "reported")
        self.assertEqual(ebird.season_status(season, "rewbul"), "few reports")       # only 3 species in total
        self.assertIn("too few eBird reports", format_season(season, "rewbul", "Hubei"))
        text = format_season(season, "rubwoo1", "Hubei")
        self.assertTrue(text.startswith("Hubei, ±3 days of Oct 18, 2023–2026: reported on 1 of"), text)
        with patch.object(ebird, "historic_day", lambda _region, _day, _key: busy):
            rich = ebird.season_reports("CN-42", date(2026, 10, 18), "key")
        self.assertEqual(ebird.season_status(rich, "rewbul"), "not reported")
        self.assertTrue(format_season(rich, "rewbul", "Hubei").startswith("⚠ Hubei"))

    def test_review_flags_species_not_reported_at_this_time_of_year(self):
        import ebird
        from birdnet_gui import season_flags
        settings = json.dumps({"date": "2026-10-18"})
        rows = [{"Species (scientific)": "Pycnonotus jocosus", "Species (common)": "Red-whiskered Bulbul",
                 "Latitude": 30.5928, "Longitude": 114.3055, "Analysis settings": settings},
                {"Species (scientific)": "Pycnonotus sinensis", "Species (common)": "Light-vented Bulbul",
                 "Latitude": 30.5928, "Longitude": 114.3055, "Analysis settings": settings},
                {"Species (scientific)": "Avis testus", "Species (common)": "Test bird",       # not an eBird name
                 "Latitude": 30.5928, "Longitude": 114.3055, "Analysis settings": settings}]
        season = {"reported": {"livbul1": ["2025-10-17"], **{f"sp{n}": ["2025-10-17"] for n in range(50)}},
                  "species_total": 51}
        codes = {"Pycnonotus jocosus": "rewbul", "Pycnonotus sinensis": "livbul1"}
        with patch.object(ebird, "species", lambda sci, _common="": {"code": codes[sci]} if sci in codes else None):
            flagged = season_flags(rows, {("2026-10-18", 30.59, 114.31): season})
        self.assertEqual(flagged, {("2026-10-18", "Pycnonotus jocosus")})


class ClipFormatTests(unittest.TestCase):
    def test_clips_keep_stereo_by_default(self):
        import soundfile as sf
        self.assertFalse(pipeline.MAKE_MONO)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            make_tone(source)
            with patch.object(birdnetlib, "Recording", fake_recording((10,))):
                rows, day = pipeline.process_file(None, source, root / "out",
                                                  **dict(PROCESS_ARGS, make_mono=pipeline.MAKE_MONO))
            info = sf.info(str(day / rows[0]["File"]))
            self.assertEqual((info.samplerate, info.subtype, info.channels), (48000, "PCM_24", 2))

    def test_old_mono_results_are_not_recut_when_stereo_becomes_the_default(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            make_tone(source)
            with patch.object(birdnetlib, "Recording", fake_recording((10,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)   # old: mono
                summary = day / "summary.xlsx"
                pipeline.write_summary(rows, summary, {str(source)}, {pipeline.source_id(source)})
                save_review(summary, 2, status="Rejected", common="Test bird", scientific="Avis testus")
                again, _day = pipeline.process_file(None, source, root / "out",
                                                    **dict(PROCESS_ARGS, make_mono=False))
            self.assertIsNone(again)                                        # skipped, not re-cut
            self.assertEqual(sf.info(str(day / rows[0]["File"])).channels, 1)
            self.assertEqual(load_reviews(summary)[0]["Review status"], "Rejected")

    def test_32_bit_float_louder_than_0_db_is_turned_down_not_clipped(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            t = np.arange(40 * 48000) / 48000
            tone = (0.02 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
            tone[10 * 48000:12 * 48000] *= 100                  # +6 dB over full scale, kept by a 32-bit recorder
            sf.write(str(source), tone, 48000, subtype="FLOAT")
            with patch.object(birdnetlib, "Recording", fake_recording((10,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
            self.assertEqual(rows[0]["Clipping"], "")
            data, _rate = sf.read(str(day / rows[0]["File"]))
            self.assertAlmostEqual(np.abs(data).max(), 10 ** (-3 / 20), places=2)   # normalized to -3 dB
            loud, quiet = np.abs(data[3 * 48000:5 * 48000]).max(), np.abs(data[:int(2.9 * 48000)]).max()
            self.assertAlmostEqual(loud / quiet, 100, delta=3)  # clipping at 0 dB would give 50


class SoundTypeTests(unittest.TestCase):
    def setUp(self):
        import second_opinion
        import sound_type
        import xeno_canto
        self.temp = tempfile.TemporaryDirectory()
        cache = Path(self.temp.name)
        self.patches = [patch.object(sound_type, "CACHE", cache / "sound-type"),
                        patch.object(second_opinion, "CACHE", cache / "second-opinion"),
                        patch.object(xeno_canto, "TYPES_FILE", cache / "xc-types.json")]
        for item in self.patches:
            item.start()
        sound_type._loaded.update(stamp=None, model=None)

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    @staticmethod
    def _vectors(center, count, seed):
        import numpy as np
        rng = np.random.default_rng(seed)
        vectors = np.zeros((count, 8), np.float32)
        vectors[:, center] = 1.0
        vectors += rng.normal(0, 0.15, vectors.shape).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _references(self):
        """xeno-canto: 15 song + 15 call recordings, 3 segments each (songs point one way, calls another)"""
        import numpy as np
        import second_opinion
        import xeno_canto
        folder = second_opinion.CACHE / "references"
        folder.mkdir(parents=True)
        types = {}
        for n in range(30):
            song = n % 2 == 0
            np.save(folder / f"XC{n}_Pycnonotus_sinensis.npy", self._vectors(0 if song else 1, 3, n))
            types[f"XC{n}"] = "song" if song else "call, flight call"
        types["XC99"] = "call, song"                                  # both: not used
        xeno_canto.remember_types([{"id": key, "type": value} for key, value in types.items()])

    def test_xeno_canto_and_reviewer_types(self):
        from sound_type import kind, reviewed_kind
        self.assertEqual([kind(t) for t in ("song", "dawn song", "call, flight call", "alarm call", "call, song",
                                            "drumming", "", None)],
                         ["Song", "Song", "Call", "Call", None, None, None, None])
        self.assertEqual([reviewed_kind(t) for t in ("Song", "Flight call", "Alarm call", "Drumming", "", None)],
                         ["Song", "Call", "Call", None, None, None])

    def test_clip_segments_follow_birdnet_windows(self):
        import numpy as np
        from sound_type import pick_segments
        encoded = {t: np.eye(8, dtype=np.float32)[i] * 2 for i, t in enumerate((0.0, 1.5, 3.0, 4.5, 6.0, 7.5))}
        picked = pick_segments(encoded, [[3.0, 6.0, 0.4], [4.5, 7.5, 0.9], [6.0, 9.0, 0.7], [7.5, 10.5, 0.2]])
        self.assertEqual([int(np.argmax(v)) for v in picked], [3, 4, 2])           # best 3 windows, unit length
        self.assertAlmostEqual(float(np.linalg.norm(picked[0])), 1.0, places=5)
        self.assertEqual(len(pick_segments(encoded, None)), 6)                      # added clips: middle of clip
        self.assertEqual(pick_segments({}, None).shape[0], 0)

    def test_guess_song_or_call_after_training(self):
        import sound_type
        self._references()
        self.assertIsNone(sound_type.guess({"Generated sha256": "a"}))              # no model yet
        info = sound_type.train()
        self.assertEqual((info["references"], info["yours"]), (30, 0))
        self.assertGreater(info["tested"], 0.9)                                     # clean clusters: easy
        self.assertTrue(sound_type.SURE <= info["sure"] <= 0.95)
        sound_type.save_clip_vectors("song", self._vectors(0, 3, 100))
        sound_type.save_clip_vectors("call", self._vectors(1, 3, 101))
        sound_type.save_clip_vectors("odd", self._vectors(0, 3, 102)[:, :4])       # other model: no guess
        song = sound_type.guess({"Generated sha256": "song"})
        self.assertEqual((song["type"], song["sure"]), ("Song", True))
        self.assertTrue(sound_type.describe(song).startswith("guess: song ("))
        self.assertEqual(sound_type.guess({"Generated sha256": "call"})["type"], "Call")
        self.assertIsNone(sound_type.guess({"Generated sha256": "odd"}))
        unsure = dict(song, p=0.6, sure=False)
        self.assertEqual(sound_type.describe(unsure), "guess: song? (not sure)")
        self.assertEqual(sound_type.parse_guess_text(sound_type.guess_text(unsure)), ("Song", 0.6, False))
        self.assertEqual(sound_type.parse_guess_text(sound_type.guess_text(song))[2], True)
        counts, unsure = sound_type.unreviewed_guesses([
            {"Generated sha256": "song", "Review status": "Pending"}, {"Generated sha256": "call"},
            {"Generated sha256": "odd"}, {"Generated sha256": "song", "Sound type": "Song"},      # already chosen
            {"Generated sha256": "call", "Review status": "Approved"}])                         # reviewed
        self.assertEqual(dict(counts), {"song (sure)": 1, "call (sure)": 1, "no guess": 1})
        self.assertEqual(unsure, [])

    def test_reviewer_choices_train_and_measure_the_guess(self):
        import sound_type
        self._references()
        sound_type.save_clip_vectors("mine", self._vectors(1, 3, 7))
        rows = [{"File": "a.wav", "Generated sha256": "mine", "Review status": "Approved", "Sound type": "Flight call",
                 "Sound type guess": "Call 0.91 sure"},
                {"File": "b.wav", "Generated sha256": "x", "Review status": "Approved", "Sound type": "Song",
                 "Sound type guess": "Call 0.55"},
                {"File": "c.wav", "Generated sha256": "y", "Review status": "Rejected", "Sound type": "Song",
                 "Sound type guess": "Song 0.95 sure"}]                        # wrong species: not counted
        with tempfile.TemporaryDirectory() as day:
            summary = Path(day) / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            self.assertEqual(sound_type.train([summary])["yours"], 1)
        self.assertEqual(sound_type.accuracy(rows), {"sure": [1, 1], "unsure": [0, 1]})
        text = sound_type.describe_accuracy(rows)
        self.assertIn("30 xeno-canto recordings", text)
        self.assertIn("Sure guesses: matched your choice on 1 of 1", text)
        self.assertEqual(dict(sound_type.mismatches(rows)), {("Call", "Song"): 1})
        self.assertIn("guessed call but you chose song ×1", text)
        self.assertIn("Your choices: 1 flight call, 1 song", text)

    def test_sound_type_is_saved_and_named_in_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.18_1439"
            (day / "Test bird").mkdir(parents=True)
            make_wav(day / "Test bird" / "clip.wav")
            summary = day / "summary.xlsx"
            pd.DataFrame([{"File": "Test bird/clip.wav", "Review status": "Pending"}]).to_excel(summary, index=False)
            with self.assertRaises(ValueError):
                save_review(summary, 2, status="Pending", common="", scientific="", sound_type="Chirp")
            ready = save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus",
                                rating=3, sound_type="Flight call", sound_type_guess="Call 0.91")
            self.assertTrue(ready.endswith("_Avis.testus_Flight-call_R3.wav"), ready)
            row = load_reviews(summary)[0]
            self.assertEqual((row["Sound type"], row["Sound type guess"]), ("Flight call", "Call 0.91"))
            ready = save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus", rating=3)
            self.assertTrue(ready.endswith("_Flight-call_R3.wav"), ready)           # None keeps the saved type
            ready = save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus",
                                rating=3, sound_type="")
            self.assertTrue(ready.endswith("_Avis.testus_R3.wav"), ready)
            self.assertEqual(len(list((day / "Ready").rglob("*.wav"))), 1)          # renamed, not copied twice

    def test_auto_checks_store_clip_vectors_for_every_clip(self):
        import numpy as np
        import auto_checks
        import sound_type

        class Result:
            def __init__(self, rows):
                self.rows = rows

            def to_structured_array(self):
                return self.rows

        class Model:
            species_list = ["Avis testus_Test bird"]

            def predict(self, paths, **_kwargs):
                return Result([])

            def encode(self, paths, **_kwargs):
                return Result([{"input": p, "start_time": t, "embedding": np.eye(8)[int(t / 1.5)]}
                               for p in paths for t in (0.0, 1.5, 3.0, 4.5)])

        analyzer = types.SimpleNamespace(model=Model())
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.18_1439"
            day.mkdir()
            for name in ("a.wav", "b.wav"):
                make_wav(day / name, seconds=6)
            rows = [{"File": "a.wav", "Generated sha256": "sa", "Species (common)": "Test bird",
                     "Species (scientific)": "Avis testus", "Max confidence": 0.9,
                     "Detection boxes": json.dumps({"windows": [[3.0, 6.0, 0.9]]})},
                    {"File": "b.wav", "Generated sha256": "sb", "Species (common)": "Test bird",
                     "Species (scientific)": "Avis testus", "Origin": "Manual", "Sound type": "Call"},
                    {"File": "gone.wav", "Generated sha256": "sc", "Species (common)": "Test bird",
                     "Species (scientific)": "Avis testus", "Max confidence": 0.9}]
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            with patch.object(auto_checks, "CACHE", Path(temp) / "auto"), \
                    patch.object(auto_checks, "all_results", return_value=[]):
                self.assertTrue(auto_checks.needs_check(rows[1], has_xc_key=False))   # added clip: vectors only
                auto_checks.run(summary, xc_key="", analyzer=analyzer, log=lambda _text: None)
                self.assertFalse(auto_checks.needs_check(rows[1], has_xc_key=False))
            self.assertEqual(int(np.argmax(sound_type.clip_vectors("sa")[0])), 2)    # the 3.0 s window
            self.assertEqual(len(sound_type.clip_vectors("sb")), 4)
            self.assertEqual(len(sound_type.clip_vectors("sc")), 0)                  # missing clip: not retried


class WarningModelTests(unittest.TestCase):
    @staticmethod
    def _records(wrong_when):
        import random
        rng = random.Random(3)
        records = []
        for n in range(170):
            score, gap = rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.5)
            records.append({"File": f"{n}.wav", "Generated sha256": f"x{n}", "Species (common)": "Test bird",
                            "Species (scientific)": "Avis testus", "Max confidence": score,
                            "Alt1 conf": score - gap, "Source ID": f"rec{n % 12}", "_summary": Path("/r/summary.xlsx"),
                            "_species": "Test bird", "_manual": False,
                            "_outcome": "wrong" if wrong_when(score, gap) else "correct"})
        return records

    def test_model_warns_only_when_it_beats_the_score(self):
        import review_progress as rp
        records = self._records(lambda score, gap: gap < 0.12)       # wrong IDs: 2nd species almost as likely
        warner, result = rp.warning_model(records)
        self.assertTrue(result["better"])
        self.assertIsNotNone(warner)
        self.assertTrue(0.0 < warner["threshold"] < 1.0)
        close = dict(records[0], **{"Max confidence": 0.8, "Alt1 conf": 0.75})
        clear = dict(records[0], **{"Max confidence": 0.8, "Alt1 conf": 0.4})
        self.assertGreater(rp.warning_risk(warner, close), warner["threshold"])
        self.assertLess(rp.warning_risk(warner, clear), warner["threshold"])
        self.assertIn("marks risky clips", rp.describe_model(result))
        warner, result = rp.warning_model(self._records(lambda score, gap: score < 0.65))  # score says it all
        self.assertIsNone(warner)
        self.assertIn("no model warnings", rp.describe_model(result))


class ReviewTableSortTests(TkTestCase):
    def test_headers_sort_up_then_down_and_keep_blanks_last(self):
        import tkinter as tk
        from tkinter import ttk
        import birdnet_gui
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        try:
            settings = json.dumps({"date": "2026-10-10"})
            rows = {"2": {"Clock time": "23:59:58", "Start offset (s)": 10.0, "Analysis settings": settings,
                          "Max confidence": 0.9},
                    "3": {"Clock time": "00:00:05", "Start offset (s)": 17.0, "Analysis settings": settings,
                          "Max confidence": 0.5},                                  # after midnight: later
                    "4": {"Clock time": "23:59:59", "Start offset (s)": 11.0, "Analysis settings": settings,
                          "Origin": "Manual"}}                                     # added: no score
            app = types.SimpleNamespace(review_rows=rows, _review_sort=None,
                                        _review_headings={"time": "Time", "confidence": "Confidence",
                                                          "status": "Status"})
            app.review_tree = ttk.Treeview(root, columns=("time", "confidence", "status"), show="headings")
            for iid, status in (("2", "Approved"), ("3", "Pending"), ("4", "")):
                app.review_tree.insert("", "end", iid=iid, values=("", "", status))
            for name in ("_review_sort_key", "_apply_review_sort"):
                setattr(app, name, getattr(birdnet_gui.App, name).__get__(app))
            sort = birdnet_gui.App._sort_review.__get__(app)
            order = lambda: list(app.review_tree.get_children())
            sort("confidence")
            self.assertEqual(order(), ["3", "2", "4"])
            self.assertEqual(app.review_tree.heading("confidence", "text"), "Confidence ▲")
            sort("confidence")
            self.assertEqual(order(), ["2", "3", "4"])                         # descending, blank still last
            self.assertEqual(app.review_tree.heading("confidence", "text"), "Confidence ▼")
            sort("time")
            self.assertEqual(order(), ["2", "4", "3"])                         # real time, not "00:00" first
            self.assertEqual(app.review_tree.heading("confidence", "text"), "Confidence")
            sort("status")
            self.assertEqual(order(), ["2", "3", "4"])
        finally:
            root.destroy()


class SoundTypeWindowTests(TkTestCase):
    def test_groups_put_unsure_guesses_first_and_skip_rejected(self):
        import sound_type
        import sound_type_window as stw
        guesses = {"a": {"type": "Song", "p": 0.9, "sure": True}, "b": {"type": "Call", "p": 0.6, "sure": False},
                   "c": {"type": "Call", "p": 0.52, "sure": False}, "d": {"type": "Call", "p": 0.95, "sure": True},
                   "e": {"type": "Call", "p": 0.9, "sure": True}, "f": None}
        rows = [{"Generated sha256": "a"}, {"Generated sha256": "b"}, {"Generated sha256": "c"},
                {"Generated sha256": "d", "Sound type": "Song", "Review status": "Approved"},
                {"Generated sha256": "e", "Review status": "Rejected"}, {"Generated sha256": "f"}]
        with patch.object(sound_type, "guess", lambda row: guesses[row["Generated sha256"]]):
            groups = stw.build_groups(rows)
        sha = {name: [row["Generated sha256"] for row, _found in pairs] for name, pairs in groups.items()}
        self.assertEqual(sha["Not sure — most uncertain first"], ["c", "b"])
        self.assertEqual(sha["Guessed song (sure)"], ["a"])
        self.assertEqual(sha["Guessed call (sure)"], [])                      # e was rejected
        self.assertEqual(sha["Different from your choice"], ["d"])
        self.assertEqual(sha["Your choices"], ["d"])
        self.assertEqual(stw.guess_label(guesses["b"]), "call 60%?")

    def test_label_saves_sound_type_without_changing_the_review(self):
        import tkinter as tk
        import birdnet_gui as g
        import sound_type
        import sound_type_window as stw
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.10_1111_v3preview"
            day.mkdir()
            rows = []
            for n in range(2):
                make_wav(day / f"{n}.wav", seconds=3)
                rows.append({"File": f"{n}.wav", "Species (common)": "Asian Tit", "Species (scientific)": "Parus cinereus",
                             "Generated sha256": f"s{n}", "Max confidence": 0.7, "Review status": "Pending"})
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            retrained = []
            unsure = {"type": "Call", "p": 0.6, "sure": False}
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                    patch.object(stw, "all_results", lambda _root: [{"path": summary}]), \
                    patch.object(g, "DEF_OUT", temp), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None), \
                    patch.object(g.App, "_retrain_sound_type", lambda self: retrained.append(1)), \
                    patch.object(sound_type, "guess", lambda row: unsure):
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    app.out.set(temp)
                    app.review_summary.set(str(summary))
                    window = stw.SoundTypeWindow(app)
                    window.win.withdraw()
                    deadline = time.time() + 10
                    while not window.items and time.time() < deadline:
                        root.update()
                        time.sleep(0.05)
                    self.assertEqual(window.count.get(), "2 clips")
                    window.label("Song")
                    saved = load_reviews(summary)[0]
                    self.assertEqual((saved["Sound type"], saved["Sound type guess"], saved["Review status"]),
                                     ("Song", "Call 0.60", "Pending"))
                    self.assertEqual(window.tree.selection(), ("1",))                 # moved to the next clip
                    self.assertEqual(retrained, [1])
                    self.assertTrue(sound_type.labelled(saved))                       # pending labels train too
                    window.close()
                finally:
                    root.destroy()


class JobQueueTests(unittest.TestCase):
    def test_background_jobs_never_jump_ahead_of_your_jobs(self):
        import birdnet_gui as g
        launched, stops, timers = [], [], []

        def launch(job):
            launched.append(job["title"])
            app.job_running, app.current_job, app.proc = True, job, types.SimpleNamespace(poll=lambda: None)

        app = types.SimpleNamespace(job_running=False, job_queue=[], current_job=None, proc=None, _closing=False,
                                    status=types.SimpleNamespace(set=lambda _text: None),
                                    root=types.SimpleNamespace(after=lambda _ms, fn: timers.append(fn)),
                                    _update_job_controls=lambda: None, _launch_job=launch)
        for name in ("_start", "_yield_to_user_jobs", "_start_next_job"):
            setattr(app, name, getattr(g.App, name).__get__(app))
        with patch.object(g, "request_stop", lambda proc: stops.append(proc)):
            # an analysis just finished; the next one waits 1.5 s, and meanwhile the new result's check is queued
            app.job_queue.append({"title": "Analyze REC-002", "background": False})
            app._start(["x"], None, "Auto-checks", is_analyze=False, background=True)
            self.assertEqual(launched, [])                                     # did not jump ahead
            app._start_next_job()
            self.assertEqual(launched, ["Analyze REC-002"])
            self.assertEqual([job["title"] for job in app.job_queue], ["Auto-checks"])
            # a background job is running and you start a job: it is asked to stop and runs again later
            app.job_running, app.current_job, app.job_queue = True, {"title": "Auto-checks", "background": True}, []
            app._start(["y"], None, "Analyze REC-003")
            self.assertEqual(len(stops), 1)
            self.assertTrue(app.current_job["preempted"])
            self.assertEqual([job["title"] for job in app.job_queue], ["Analyze REC-003"])


class SpeciesPickerTests(TkTestCase):
    def test_typing_part_of_a_name_offers_ebird_species(self):
        import tkinter as tk
        from tkinter import ttk
        from species_picker import SpeciesPicker
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        try:
            text, picked, exact = tk.StringVar(master=root), [], []
            entry = ttk.Entry(root, textvariable=text)
            entry.pack()
            picker = SpeciesPicker(entry, lambda: "CN", lambda common, sci: picked.append((common, sci)),
                                   on_exact=exact.append)
            text.set("black-throated t")
            picker._typed()
            self.assertIn(("Black-throated Tit", "Aegithalos concinnus"), picker.names)
            self.assertIsNotNone(picker.top)                                  # list shown under the field
            index = picker.names.index(("Black-throated Tit", "Aegithalos concinnus"))
            picker.pick(index)
            self.assertEqual(picked, [("Black-throated Tit", "Aegithalos concinnus")])
            self.assertIsNone(picker.top)                                     # closed after choosing
            text.set("Light-vented Bulbul")                                   # typed in full: no list needed
            picker._typed()
            self.assertEqual(exact, ["Pycnonotus sinensis"])
            self.assertIsNone(picker.top)
            text.set("x")
            picker._typed()
            self.assertEqual(picker.names, [])                                # 1 letter: too short
            # first click: this clip's candidates (second opinion first); a hint row cannot be picked
            picker.candidates_of = lambda: [("Black-throated Tit", "Aegithalos concinnus", "second opinion 0.84"),
                                            ("Light-vented Bulbul", "Pycnonotus sinensis", "BirdNET's other guess 0.65"),
                                            (None, None, "No second opinion yet")]
            picker.show_candidates()
            self.assertEqual(picker.box.get(0), "Black-throated Tit  —  Aegithalos concinnus   ·  second opinion 0.84")
            picker.pick(2)
            self.assertIsNotNone(picker.top)                                  # hint row: nothing happens
            picker.pick(1)
            self.assertEqual(picked[-1], ("Light-vented Bulbul", "Pycnonotus sinensis"))
            text.set("")
            picker._typed()                                                   # cleared: candidates again
            self.assertEqual(len(picker.names), 3)
        finally:
            root.destroy()

    def test_several_names_in_one_field(self):
        import tkinter as tk
        from tkinter import ttk
        from species_picker import SpeciesPicker
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        try:
            text = tk.StringVar(master=root)
            entry = ttk.Entry(root, textvariable=text)
            entry.pack()
            picker = SpeciesPicker(entry, lambda: "CN", lambda _common, _sci: None, multiple=True,
                                   candidates_of=lambda: [("Spotted Dove", "Spilopelia chinensis", "BirdNET 0.9")])
            text.set("Asian Tit; black-throated t")                         # searches only the name being typed
            picker._typed()
            picker.pick(picker.names.index(("Black-throated Tit", "Aegithalos concinnus")))
            self.assertEqual(text.get(), "Asian Tit; Black-throated Tit; ")
            picker._typed()                                                   # nothing typed yet: this clip's list
            self.assertEqual(picker.names, [("Spotted Dove", "Spilopelia chinensis")])
            picker.pick(0)
            self.assertEqual(text.get(), "Asian Tit; Black-throated Tit; Spotted Dove; ")
            text.set("Asian Tit")                                             # chosen from the list: added, not replaced
            picker.show_candidates()
            picker.pick(0)
            self.assertEqual(text.get(), "Asian Tit; Spotted Dove; ")
        finally:
            root.destroy()


class ShutDownWhenDoneTests(TkTestCase):
    def test_shuts_down_only_after_jobs_stay_finished_and_countdown_ends(self):
        import tkinter as tk
        import birdnet_gui as g
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        root.withdraw()
        # หน้าต่างนับถอยหลังเป็นหน้าต่างเดียวของ process ทดสอบ: ปิดแล้ว macOS ส่ง Quit มา (ค่าเริ่มต้นของ Tk = exit)
        root.createcommand("::tk::mac::Quit", lambda: None)
        calls, countdowns = [], []
        try:
            app = types.SimpleNamespace(root=root, job_running=True, job_queue=[], _closing=False,
                                        _shutdown_idle=0, _shutdown_window=None,
                                        SHUTDOWN_IDLE_S=g.App.SHUTDOWN_IDLE_S, SHUTDOWN_COUNTDOWN_S=2,
                                        shutdown_after=tk.BooleanVar(master=root, value=True),
                                        status=types.SimpleNamespace(set=lambda _text: None))
            app._shutdown_watch = g.App._shutdown_watch.__get__(app)
            app._shutdown_countdown = lambda: countdowns.append(1)
            # ส่วนแรกเรียก _shutdown_watch เอง: ไม่ตั้งเวลาเรียกซ้ำจริง (ไม่งั้นค้างไปกวนส่วนนับถอยหลัง)
            app.root = types.SimpleNamespace(after=lambda _ms, _fn: None)
            app._shutdown_watch()                       # a job is running: keep waiting
            self.assertEqual((app._shutdown_idle, countdowns), (0, []))
            app.job_running = False
            for _ in range(3):
                app._shutdown_watch()
            self.assertEqual(countdowns, [1])           # 15 s with nothing running
            app.root = root
            app._shutdown_countdown = g.App._shutdown_countdown.__get__(app)
            with patch.object(g.platform_tools, "shut_down_computer", lambda: calls.append(1) or (True, "")):
                app._shutdown_countdown()
                deadline = time.time() + 20                 # นับ 2 วิ — เผื่อเครื่องยุ่ง
                while not calls and time.time() < deadline:
                    root.update()
                    time.sleep(0.05)
            self.assertEqual(calls, [1])
            # cancel: no shut down, option switched off
            app.shutdown_after.set(True)
            with patch.object(g.platform_tools, "shut_down_computer", lambda: calls.append(2) or (True, "")):
                app._shutdown_countdown()
                window = app._shutdown_window
                cancel = [w for w in window.winfo_children()[0].winfo_children()[1].winfo_children()
                          if w.cget("text") == "Cancel"][0]
                cancel.invoke()
                for _ in range(60):
                    root.update()
                    time.sleep(0.05)
            self.assertEqual(calls, [1])
            self.assertFalse(app.shutdown_after.get())
        finally:
            root.destroy()


class ReviewSpeciesFilterTests(TkTestCase):
    def test_species_filter_and_jump_from_progress(self):
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.10_1111_v3preview"
            day.mkdir()
            rows = []
            for n, (common, scientific) in enumerate((("Asian Tit", "Parus cinereus"), ("Light-vented Bulbul",
                                                      "Pycnonotus sinensis"), ("Asian Tit", "Parus cinereus"))):
                make_wav(day / f"{n}.wav")
                rows.append({"File": f"{n}.wav", "Species (common)": common, "Species (scientific)": scientific,
                             "Max confidence": 0.7, "Review status": "Pending"})
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {"output_dir": temp}), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None):
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    app.review_summary.set(str(summary))
                    app.review_load()
                    self.assertEqual(list(app.review_species_box.cget("values")),
                                     ["All", "Asian Tit", "Light-vented Bulbul"])
                    app.review_species_filter.set("Asian Tit")
                    app._fill_review_tree()
                    self.assertEqual(len(app.review_tree.get_children()), 2)
                    self.assertEqual(app.review_species_count.get(), "2 of 3 clips")
                    app.show_clip(summary, "1.wav")                  # another species: filter falls back to All
                    self.assertEqual(app.review_species_filter.get(), "All")
                    self.assertEqual(app.review_tree.set(app.review_tree.selection()[0], "species"),
                                     "Light-vented Bulbul")
                    app.review_species_filter.set("Asian Tit")      # what a double-click in Review progress does
                    app.show_clip(summary, "2.wav")
                    self.assertEqual(len(app.review_tree.get_children()), 2)
                    self.assertEqual(app.review_tree.selection(), ("4",))
                finally:
                    root.destroy()


    def test_arrow_keys_move_between_clips_from_table_or_picture(self):
        import types
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.25" / "0909_v3preview"
            day.mkdir(parents=True)
            for n in range(3):
                make_wav(day / f"{n}.wav")
            summary = day / "summary.xlsx"
            pd.DataFrame([{"File": f"{n}.wav", "Species (common)": "Test bird", "Species (scientific)": "Avis testus",
                           "Max confidence": 0.7, "Review status": "Pending"} for n in range(3)]).to_excel(summary,
                                                                                                         index=False)
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {"output_dir": temp}), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None):
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    app.review_summary.set(str(summary))
                    app.review_load()
                    tree = app.review_tree

                    def key(name, widget, state=0):
                        return app._review_key(types.SimpleNamespace(keysym=name, keycode=0, state=state,
                                                                     widget=widget))
                    self.assertEqual(key("Down", tree), "break")              # ยังไม่เลือก: ลงไปคลิปแรก
                    self.assertEqual(tree.selection(), ("2",))
                    tree.selection_set("3")                                   # เลือกจากโค้ด (เปิดแอป/คลิปที่แนะนำ)
                    root.update()
                    self.assertEqual(tree.focus(), "3")                       # ลูกศรของตารางเริ่มจากแถวนี้
                    key("Down", app.review_view.main)                         # บนภาพเสียงก็ได้
                    self.assertEqual((tree.selection(), tree.focus()), (("4",), "4"))
                    key("Down", tree)
                    self.assertEqual(tree.selection(), ("4",))                # แถวสุดท้าย: อยู่ที่เดิม
                    key("Up", tree)
                    self.assertEqual(tree.selection(), ("3",))
                    self.assertIsNone(key("Down", tree, state=0x08 if sys.platform == "darwin" else 0x04))  # Cmd/Ctrl
                    root.update()
                    self.assertEqual(app.review_common.get(), "Test bird")    # โหลดคลิปที่เลือกแล้ว
                finally:
                    root.destroy()

class NotSureTests(unittest.TestCase):
    def test_not_sure_leaves_ready_and_is_not_counted_as_right_or_wrong(self):
        import review_progress
        from birdnet_gui import result_label
        from review_store import list_results
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.18_1439_v3preview"
            day.mkdir()
            rows = []
            for n, status in enumerate(("Approved", "Rejected", "Pending", "Pending")):
                make_wav(day / f"{n}.wav")
                rows.append({"File": f"{n}.wav", "Review status": status, "Species (common)": "Test bird",
                             "Species (scientific)": "Avis testus", "Max confidence": 0.5,
                             "Reviewed scientific": "Avis testus" if status == "Approved" else None})
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            ready = save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus", rating=4)
            self.assertTrue((day / ready).is_file())
            self.assertEqual(save_review(summary, 2, status="Unsure", common="Test bird", scientific="Avis testus",
                                         rating=4, notes="too faint"), "")
            self.assertFalse((day / ready).exists())                                # no longer in Ready/
            saved = load_reviews(summary)[0]
            self.assertEqual((saved["Review status"], saved["Quality rating"]), ("Unsure", None))
            info = list_results(temp)[0]
            self.assertEqual((info["approved"], info["rejected"], info["unsure"]), (0, 1, 1))
            self.assertIn("4 clips, 2 reviewed", result_label(info))
            records = review_progress.collect([summary])
        self.assertEqual([r["_outcome"] for r in records], ["unsure", "wrong", "pending", "pending"])
        self.assertEqual(review_progress.filter_report(records)["low_score"], {"flagged": [1, 1], "clear": [0, 0]})
        table = review_progress.species_table(records)["Test bird"]
        self.assertEqual((table["reviewed"], table["wrong"], table["unsure"], table["pending"]), (1, 1, 1, 2))
        self.assertEqual(review_progress.model_check(records)["reviewed"], 1)
        self.assertEqual(review_progress.next_clip(records)["File"], "2.wav")      # not the not-sure one


class SpectrogramViewTests(TkTestCase):
    @staticmethod
    def _tone(seconds=8.0, sr=48000, hz=4000, start=2.0, end=3.0):
        import numpy as np
        rng = np.random.default_rng(1)
        samples = rng.normal(0, 0.01, int(seconds * sr)).astype(np.float32)
        t = np.arange(int(start * sr), int(end * sr)) / sr
        samples[int(start * sr):int(end * sr)] += 0.3 * np.sin(2 * np.pi * hz * t)
        return samples, sr

    def test_clip_spectrogram_is_0_to_12_khz_with_high_frequencies_on_top(self):
        import numpy as np
        import spectrogram_view as sv
        samples, sr = self._tone()
        db, seconds_per_column, fmax = sv.clip_db(samples, sr)
        self.assertEqual(fmax, 12000)
        self.assertEqual(db.shape[0], int(12000 / (sr / 1024)) + 1)      # 46.875 Hz bins up to 12 kHz
        self.assertAlmostEqual(seconds_per_column, 256 / sr)
        row = int(np.argmax(db[:, int(2.5 / seconds_per_column)]))
        self.assertAlmostEqual((db.shape[0] - 1 - row) / (db.shape[0] - 1) * fmax, 4000, delta=100)
        self.assertEqual(sv.clip_db(samples[:22050 * 4], 22050)[2], 11025)  # low sample rate: up to Nyquist

    def test_picture_is_white_background_with_dark_bird_sound(self):
        import numpy as np
        import spectrogram_view as sv
        samples, sr = self._tone()
        db, seconds_per_column, fmax = sv.clip_db(samples, sr)
        relative, top = sv.whiten(db)
        self.assertGreaterEqual(top, sv.MIN_TOP_DB)
        image = np.asarray(sv.gray_image(relative, top, 400, 120))
        self.assertEqual(image.shape, (120, 400))
        y = round(120 * (1 - 4000 / fmax))
        tone, quiet = image[y - 1:y + 2, 110:140], image[:, 300:]       # 2.5 s vs 6-8 s of 8 s
        self.assertLess(tone.min(), 60)
        self.assertGreater(np.mean(quiet), 245)                          # noise speckle is smoothed away
        overview = np.asarray(sv.gray_image(relative, top, 100, 20, pool=True))
        self.assertEqual(overview.shape, (20, 100))
        self.assertLess(overview[:, 25:38].min(), 60)                    # 1 s tone kept when squeezed

    def test_review_clip_spans_and_clock_of_clip_start(self):
        from birdnet_gui import _clip_spans_and_clock
        # recording started 14:39:44; BirdNET clip starts 2 s before its first window at 904.5 s
        row = {"Start offset (s)": 904.5, "Clock time": "14:54:48",
               "Analysis settings": json.dumps({"date": "2026-10-18"}),
               "Detection boxes": json.dumps({"windows": [[2.0, 5.0, 0.8], [5.0, 8.0, 0.7]]})}
        spans, clock = _clip_spans_and_clock(row)
        self.assertEqual(spans, [(2.0, 5.0), (5.0, 8.0)])
        self.assertEqual(clock, datetime(2026, 10, 18, 14, 54, 46, 500000))
        self.assertEqual(_clip_spans_and_clock({"Detection boxes": "not json"}), ([], None))

    def test_viewer_shows_six_seconds_and_pages_through_the_clip(self):
        import tkinter as tk
        import spectrogram_view as sv
        samples, sr = self._tone(seconds=15.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sr)
                out.writeframes((samples * 32767).astype("<i2").tobytes())
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"no display: {exc}")
            root.withdraw()
            try:
                seeks = []
                view = sv.SpectrogramView(root, width=300, height=80, on_seek=seeks.append)
                view.pack()
                view.load(path, clock_start=datetime(2026, 10, 18, 7, 0, 0), spans=[(2.0, 3.0)])
                deadline = time.time() + 10
                while view._data is None and time.time() < deadline:
                    root.update()
                    time.sleep(0.02)
                self.assertAlmostEqual(view.duration, 15.0, places=2)
                self.assertEqual(view.window, sv.WINDOW_S)
                self.assertTrue(view.info.get().startswith("0:00–0:06 of 0:15  ·  clock 07:00:00"), view.info.get())
                view.page(1)
                self.assertAlmostEqual(view.view_start, 4.8)
                view.set_view(100)
                self.assertAlmostEqual(view.view_start, 9.0)                # clamped to the last 6 s
                view._clicked(types.SimpleNamespace(x=sv.LEFT + view.width / 2))
                self.assertAlmostEqual(seeks[-1], 12.0)
                view.message("Select a clip")
                self.assertIsNone(view._data)
            finally:
                root.destroy()

    def test_only_where_birdnet_heard_it(self):
        import tkinter as tk
        import spectrogram_view as sv
        self.assertEqual(sv.merge_spans([(10, 13), (2, 5), (3.5, 6.5)]), [(2.0, 6.5), (10.0, 13.0)])
        samples, sr = self._tone(seconds=15.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sr)
                out.writeframes((samples * 32767).astype("<i2").tobytes())
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"no display: {exc}")
            root.withdraw()
            try:
                view = sv.SpectrogramView(root, width=300, height=80)
                view.pack()

                def load(spans):
                    view.load(path, spans=spans)
                    deadline = time.time() + 10
                    while view._data is None and time.time() < deadline:
                        root.update()
                        time.sleep(0.02)

                load([(2.0, 5.0), (3.5, 6.5), (10.0, 13.0)])
                self.assertEqual((view.compressed, round(view.duration)), (False, 15))
                view.only_heard.set(True)
                self.assertTrue(view.compressed)
                self.assertAlmostEqual(view.duration, 7.5, delta=0.02)               # 4.5 s + 3 s heard
                self.assertAlmostEqual(view.to_real(0.0), 2.0, delta=0.01)
                self.assertAlmostEqual(view.to_real(5.0), 10.5, delta=0.02)          # second part
                self.assertIsNone(view.to_shown(8.0))                                # cut out
                plan = view.audio_plan(4.0)
                self.assertEqual(len(plan), 2)
                self.assertAlmostEqual(plan[0][0], 6.0, delta=0.02)
                self.assertAlmostEqual(plan[1][0], 10.0, delta=0.01)
                self.assertIn("of 0:15 (heard 0:08)", view.info.get())                # 7.5 s rounds up
                self.assertTrue(view.time_var.get().endswith("/ 0:15"))              # real clip length
                view.only_heard.set(False)
                self.assertEqual((view.compressed, round(view.duration)), (False, 15))
                self.assertEqual(view.audio_plan(3.0), [(3.0, view.duration)])
                load([])                                                             # added clip: no BirdNET parts
                self.assertTrue(view._only_box.instate(["disabled"]))
                view.only_heard.set(True)
                self.assertFalse(view.compressed)
            finally:
                root.destroy()

    def test_playback_joins_only_the_parts_shown(self):
        import soundfile as sf
        import birdnet_gui
        samples, sr = self._tone(seconds=8.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            sf.write(str(path), samples, sr)
            started = []
            view = types.SimpleNamespace(audio_plan=lambda _s: [(1.0, 2.0), (5.0, 6.5)],
                                         start_playhead=lambda seconds, player: started.append(seconds))
            app = types.SimpleNamespace(_review_preview_number=0, review_stop=lambda: None, audio_player=None)
            with patch.object(birdnet_gui.paths, "CACHE", Path(tmp) / "cache"), \
                    patch.object(birdnet_gui, "play_audio", lambda target: target):
                birdnet_gui.App.play_from(app, path, 0.0, view)
            self.assertAlmostEqual(sf.info(str(app.audio_player)).duration, 2.5, places=2)
            self.assertEqual(started, [0.0])

    def test_zoom_and_playhead_stay_in_the_middle_like_macaulay(self):
        import tkinter as tk
        import spectrogram_view as sv
        self.assertEqual([sv.time_steps(w) for w in (1.5, 3, 6, 15, 30, 60)],
                         [(0.5, 0.125), (1, 0.25), (1, 0.25), (1, 0.25), (2, 0.5), (5, 1)])
        samples, sr = self._tone(seconds=30.0)

        class Player:
            done = False

            def poll(self):
                return 0 if self.done else None

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sr)
                out.writeframes((samples * 32767).astype("<i2").tobytes())
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"no display: {exc}")
            root.withdraw()
            try:
                seeks = []
                view = sv.SpectrogramView(root, width=300, height=80, on_seek=seeks.append)
                view.pack()
                view.load(path)
                deadline = time.time() + 10
                while view._data is None and time.time() < deadline:
                    root.update()
                    time.sleep(0.02)
                view.set_view(10.0)                                  # 10-16 s, middle 13 s
                view.zoom(1)
                self.assertEqual((view.window, view.view_start), (3.0, 11.5))     # zoom in around the middle
                view.zoom(-2)
                self.assertEqual((view.window, view.view_start), (10.0, 8.0))
                view.set_zoom(0)
                self.assertEqual((view.window, view.view_start), (30.0, 0.0))     # whole clip; can't go wider
                self.assertEqual(view.zoom_var.get(), 0)
                view.set_zoom(sv.ZOOM_LEVELS.index(6.0))
                player = Player()
                view.start_playhead(14.0, player)
                self.assertAlmostEqual(view.view_start, 11.0, delta=0.05)          # playhead in the middle
                self.assertEqual(view.time_var.get(), "0:14 / 0:30")
                view.zoom(1)                                                       # zoom around the playhead
                self.assertAlmostEqual(view.view_start + view.window / 2, 14.0, delta=0.05)
                view.start_playhead(29.0, player)
                self.assertEqual(view.view_start, 30.0 - view.window)              # end: picture stops moving
                player.done = True
                view._tick()
                self.assertIsNone(view._play)
                view._progress_clicked(types.SimpleNamespace(x=sv.LEFT + view.width / 2))
                self.assertAlmostEqual(seeks[-1], 15.0)                            # progress bar: play from there
            finally:
                view.stop_playhead()
                root.destroy()


def background_rows():
    """One recording: a Light-vented Bulbul clip (source 97-113 s) and what else was heard around it."""
    def row(n, common, scientific, start, end, windows=None, status="Pending", source="/rec/a.wav", **extra):
        boxes = {"windows": windows or [], "sounds": []}
        if windows is None:
            boxes["manual"] = True
        lead = windows[0][0] if windows else 0.0
        return {"File": f"{n}.wav", "Species (common)": common, "Species (scientific)": scientific,
                "Start offset (s)": start, "End offset (s)": end, "Duration (s)": end - start + lead + 3,
                "Detection boxes": json.dumps(boxes), "Source path": source, "Source file": Path(source).name,
                "Max confidence": max((w[2] for w in windows), default=None) if windows else None,
                "Review status": status, **extra}
    return [
        row(0, "Light-vented Bulbul", "Pycnonotus sinensis", 100, 110, [[3.0, 13.0, 0.8]]),
        row(1, "Asian Tit", "Parus cinereus", 105, 111, [[3.0, 9.0, 0.7]]),
        row(2, "Spotted Dove", "Spilopelia chinensis", 112.5, 115.5, [[3.0, 6.0, 0.9]]),      # 0.5 s inside: no
        row(3, "Chinese Blackbird", "Turdus mandarinus", 101, 104, [[3.0, 6.0, 0.95]], status="Rejected"),
        row(4, "Long-tailed Tit", "Aegithalos caudatus", 99, 102, [[3.0, 6.0, 0.6]], status="Approved",
            **{"Reviewed common": "Silver-throated Tit", "Reviewed scientific": "Aegithalos glaucogularis"}),
        row(5, "Asian Tit", "Parus cinereus", 100, 110, [[3.0, 13.0, 0.9]], source="/rec/b.wav"),  # other file
        row(6, "_Unknown", "", 100, 103, [[3.0, 6.0, 0.3]]),
        row(7, "Yellow-bellied Tit", "Periparus venustulus", 98, 104),                    # added by the reviewer
    ]


class BackgroundSpeciesTests(TkTestCase):
    def test_suggests_other_species_heard_during_the_clip(self):
        import background_species as bg
        rows = background_rows()
        self.assertEqual(bg.clip_span(rows[0]), (97.0, 113.0))
        found = bg.suggest(rows[0], rows, "Pycnonotus sinensis")
        self.assertEqual([(common, scientific) for common, scientific, _note in found],
                         [("Silver-throated Tit", "Aegithalos glaucogularis"),       # confirmed by you first
                          ("Yellow-bellied Tit", "Periparus venustulus"),
                          ("Asian Tit", "Parus cinereus")])
        self.assertEqual(found[0][2], "BirdNET 0.60, confirmed by you, 3 s in this clip")
        self.assertEqual(found[2][2], "BirdNET 0.70, 6 s in this clip")
        # renamed to Asian Tit: it is the clip's own species now, and BirdNET's ID is not background either
        self.assertNotIn("Asian Tit", [item[0] for item in bg.suggest(rows[0], rows, "Parus cinereus")])
        self.assertNotIn("Light-vented Bulbul", [item[0] for item in bg.suggest(rows[0], rows, "Parus cinereus")])
        old = dict(rows[0], **{"Detection boxes": None, "Analysis settings": json.dumps({"lead": 3})})
        self.assertEqual(bg.clip_span(old), (97.0, 113.0))                            # summary without boxes
        self.assertEqual(bg.split("Asian Tit;  Spotted Dove, asian tit;"), ["Asian Tit", "Spotted Dove"])
        self.assertEqual(bg.join(["Asian Tit", "Spotted Dove"]), "Asian Tit; Spotted Dove")

    def test_saved_with_the_review_and_kept_when_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.22_0900_v3preview"
            day.mkdir()
            make_wav(day / "0.wav")
            summary = day / "summary.xlsx"
            pd.DataFrame([{"File": "0.wav", "Species (common)": "Test bird", "Species (scientific)": "Avis testus",
                           "Review status": "Pending"}]).to_excel(summary, index=False)
            save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus", rating=3,
                        background="Asian Tit; Spotted Dove")
            self.assertEqual(load_reviews(summary)[0]["Background species"], "Asian Tit; Spotted Dove")
            save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus", rating=4)
            self.assertEqual(load_reviews(summary)[0]["Background species"], "Asian Tit; Spotted Dove")
            save_review(summary, 2, status="Approved", common="Test bird", scientific="Avis testus", rating=4,
                        background="")
            self.assertIsNone(load_reviews(summary)[0]["Background species"])
        self.assertIn("Background species", pipeline.REVIEW_COLUMNS)                 # kept when re-analysed

    def test_review_tab_fills_picks_and_saves(self):
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.22_0900_v3preview"
            day.mkdir()
            rows = background_rows()
            for row in rows:
                make_wav(day / row["File"])
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)

            def fail(title, message, **_kw):
                raise AssertionError(f"{title}: {message}")
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {"output_dir": temp}), \
                    patch.object(g.messagebox, "showerror", fail), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None):
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    app.review_summary.set(str(summary))
                    app.review_load()

                    def select(row_number):
                        app.review_tree.selection_set(str(row_number))
                        root.update()                             # delivers the selection event, like a click
                    select(2)                                         # pending: filled from what BirdNET heard
                    self.assertEqual(app.review_background.get(), "Silver-throated Tit; Yellow-bellied Tit; Asian Tit")
                    app.review_background.set("Silver-throated Tit; ")   # removed two; click offers them again
                    app.background_picker.show_candidates()
                    self.assertEqual(app.background_picker.names, [("Yellow-bellied Tit", "Periparus venustulus"),
                                                                   ("Asian Tit", "Parus cinereus")])
                    app.background_picker.pick(1)
                    self.assertEqual(app.review_background.get(), "Silver-throated Tit; Asian Tit; ")
                    select(2)
                    app._review_species_picked("Asian Tit", "Parus cinereus")   # BirdNET was wrong: it's the tit
                    root.update()
                    self.assertEqual(app.review_background.get(), "Silver-throated Tit; Yellow-bellied Tit")
                    app.review_rating.set("4")
                    app.review_save("Approved")
                    self.assertEqual(load_reviews(summary)[0]["Background species"],
                                     "Silver-throated Tit; Yellow-bellied Tit")
                    select(5)                                         # reviewed earlier, nothing saved: not filled
                    self.assertEqual(app.review_background.get(), "")
                finally:
                    root.destroy()



def unit_vectors(rows):
    import numpy as np
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class SoundIndexTests(unittest.TestCase):
    """ดัชนีเสียง (Find similar / Sound groups / recognizers) — เวกเตอร์จำลอง ไม่เรียก BirdNET"""

    def setUp(self):
        import sound_index
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(sound_index, "CACHE", Path(self.temp.name))
        self.cache.start()
        sound_index._loaded.clear()

    def tearDown(self):
        import sound_index
        self.cache.stop()
        sound_index._loaded.clear()
        self.temp.cleanup()

    def fake_index(self, sid, vectors, bird=None, activity=None, labels=None, scores=None):
        import numpy as np
        import sound_index
        n = len(vectors)
        sound_index._save(sid, np.arange(n) * 1.5, unit_vectors(vectors),
                          np.zeros(n) if activity is None else activity, n * 1.5 + 1.5,
                          np.full((n, 5), -1) if labels is None else labels,
                          np.zeros((n, 5)) if scores is None else scores,
                          np.zeros(n) if bird is None else bird, "test")

    def test_merge_hits_joins_neighbours_and_skips_the_example(self):
        import numpy as np
        import sound_index
        starts = np.arange(10) * 1.5
        scores = np.array([0.2, 0.9, 0.95, 0.85, 0.1, 0.8, 0.1, 0.1, 0.99, 0.9])
        events = sound_index.merge_hits(starts, scores, 0.75)
        self.assertEqual([(e["start"], e["end"], e["score"], e["best"]) for e in events],
                         [(1.5, 7.5, 0.95, 3.0), (7.5, 10.5, 0.8, 7.5), (12.0, 16.5, 0.99, 12.0)])
        # windows overlapping the example by half a window or more are not results
        events = sound_index.merge_hits(starts, scores, 0.75, exclude=[(12.0, 15.0)])
        self.assertEqual([e["start"] for e in events], [1.5, 7.5])

    def test_find_similar_ranks_matching_sounds_across_recordings(self):
        import numpy as np
        import sound_index
        rng = np.random.default_rng(1)
        call = rng.normal(size=16)
        noise = rng.normal(size=(30, 16))
        first = noise.copy()
        first[10:13] = call + rng.normal(scale=0.05, size=(3, 16))          # the example
        first[20] = call + rng.normal(scale=0.1, size=16)                   # same sound later
        second = rng.normal(size=(20, 16))
        second[5] = call + rng.normal(scale=0.1, size=16)                   # and in another recording
        self.fake_index("aaa", first, activity=np.full(30, 15.0))
        self.fake_index("bbb", second)
        queries = sound_index.query_vectors("aaa", 15.0, 21.0)          # heard span = windows 10-12
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(sound_index.query_vectors("aaa", 15.0, 19.5)), 2)     # edge window only half inside
        found = sound_index.find_similar(queries, ["aaa", "bbb"], 0.8, exclude={"aaa": [(15.0, 21.0)]})
        self.assertEqual([(m["sid"], m["start"]) for m in found][:2], [("aaa", 30.0), ("bbb", 7.5)] if
                         found[0]["sid"] == "aaa" else [("bbb", 7.5), ("aaa", 30.0)])
        self.assertTrue(all(m["score"] >= 0.8 for m in found))
        self.assertFalse(any(m["sid"] == "aaa" and 13.5 <= m["start"] <= 18.0 for m in found))

    def test_loudness_measure_finds_a_call_in_noise(self):
        import numpy as np
        import sound_index
        rate = 24000
        rng = np.random.default_rng(0)
        audio = rng.normal(scale=0.01, size=rate * 9).astype(np.float32)
        t = np.arange(int(rate * 0.5)) / rate
        audio[rate * 4:rate * 4 + len(t)] += 0.2 * np.sin(2 * np.pi * 3000 * t)      # a 0.5 s whistle at 4 s
        times, scores = sound_index.frame_activity(audio, rate)
        loud = times[scores > 20]
        self.assertTrue(len(loud) and loud.min() >= 3.9 and loud.max() <= 4.6)

    def test_groups_keep_similar_sounds_together(self):
        import numpy as np
        import sound_index
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=32), rng.normal(size=32)
        vectors = np.vstack([a + rng.normal(scale=0.1, size=(6, 32)), b + rng.normal(scale=0.1, size=(4, 32)),
                             rng.normal(size=(1, 32))])
        labels = sound_index.group_labels(unit_vectors(vectors), 0.6)
        self.assertEqual(len(set(labels[:6])), 1)
        self.assertEqual(len(set(labels[6:10])), 1)
        self.assertNotEqual(labels[0], labels[6])
        self.assertNotIn(labels[10], set(labels[:10]))                      # an odd sound gets its own group

    def test_sound_groups_of_a_result_skip_sounds_already_clipped(self):
        import numpy as np
        import sound_index
        rng = np.random.default_rng(3)
        background = rng.normal(size=32)
        call = rng.normal(size=32) * 2
        vectors = np.tile(background, (40, 1)) + rng.normal(scale=0.3, size=(40, 32))
        vectors[[5, 6, 20, 21, 30]] += call                                  # the same call five times
        bird = np.zeros(40)
        bird[[5, 6, 20, 21, 30]] = 0.3
        bird[[2, 12]] = 0.1                                                  # two odd bird-like sounds
        self.fake_index("ccc", vectors, bird=bird)
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp) / "2026.10.18_1439_v3preview" / "summary.xlsx"
            summary.parent.mkdir()
            pd.DataFrame([{"File": "x.wav", "Source ID": "ccc", "Source path": str(Path(temp) / "REC.WAV"),
                           "Source file": "REC.WAV", "Start offset (s)": 45.0, "End offset (s)": 48.0,
                           "Clock time": "14:40:29", "Analysis settings": json.dumps({"date": "2026-10-18"}),
                           "Latitude": 30.5, "Longitude": 114.3, "Species (common)": "A",
                           "Species (scientific)": "Avis testus"}]).to_excel(summary, index=False)
            with patch.object(sound_index._Geo, "expected", lambda *_a: None), \
                    contextlib.redirect_stdout(io.StringIO()):
                found = sound_index.find_groups(summary, log=lambda _text: None)
            self.assertEqual(found["windows"], 6)                           # window 30 (45-48 s) has a clip
            biggest = found["groups"][0]
            self.assertEqual(sorted(e["start"] for e in biggest["events"]), [7.5, 30.0])   # 5-6 and 20-21 merged
            self.assertEqual(sound_index.load_groups(summary)["windows"], 6)

    def test_index_build_stores_vectors_guesses_and_bird_score(self):
        import numpy as np
        import sound_index

        class Rows(list):
            def to_structured_array(self):
                return self

        class FakeModel:
            species_list = ["Avis testus_Test bird", "Gryllus campestris_Field Cricket"]

            def encode(self, paths, **_kw):
                return Rows({"start_time": t, "embedding": np.ones(8) * (1 + t)} for t in (0.0, 1.5, 3.0))

            def predict(self, paths, **_kw):
                return Rows([{"start_time": 1.5, "species_name": "Avis testus_Test bird", "confidence": 0.4},
                             {"start_time": 1.5, "species_name": "Gryllus campestris_Field Cricket", "confidence": 0.7},
                             {"start_time": 3.0, "species_name": "Gryllus campestris_Field Cricket", "confidence": 0.9}])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "REC.WAV"
            make_tone(source, seconds=6, channels=1)
            with patch("species_names.is_bird_label", lambda scientific, common="": scientific == "Avis testus"):
                sound_index.build(source, "ddd", FakeModel(), log=lambda _text: None)
        index = sound_index.load("ddd")
        self.assertEqual(list(index["starts"]), [0.0, 1.5, 3.0])
        self.assertAlmostEqual(float(np.linalg.norm(index["vectors"][0].astype(np.float32))), 1.0, places=2)
        self.assertEqual([round(float(b), 2) for b in index["bird"]], [0.0, 0.4, 0.0])   # crickets are not birds
        self.assertEqual(sound_index.window_guesses(index, 1)[0][:2], ("Gryllus campestris", "Field Cricket"))

    def test_recognizer_learns_from_reviews_and_scans(self):
        import numpy as np
        import recognizers
        import sound_index
        rng = np.random.default_rng(4)
        call = rng.normal(size=24) * 2
        records = []
        for n in range(4):                                  # four recordings, three approved calls each
            vectors = rng.normal(size=(40, 24))
            for start in (3, 15, 27):
                vectors[start:start + 2] += call
                records.append({"Source ID": f"r{n}", "Start offset (s)": start * 1.5, "End offset (s)": start * 1.5 + 4.5,
                                "Review status": "Approved", "Species (common)": "Test bird",
                                "Species (scientific)": "Avis testus", "Generated sha256": f"{n}-{start}"})
            self.fake_index(f"r{n}", vectors, bird=np.full(40, 0.1))
        records.append({"Source ID": "r0", "Start offset (s)": 50.0, "End offset (s)": 54.0, "Review status": "Rejected",
                        "Species (common)": "Test bird", "Species (scientific)": "Avis testus", "Generated sha256": "w"})
        new = rng.normal(size=(30, 24))
        new[10:12] += call
        self.fake_index("fresh", new)
        with patch.object(recognizers, "CACHE", Path(self.temp.name) / "recognizers"), \
                patch("species_names.to_ebird", lambda scientific, common="": {"common": "Test bird",
                                                                                 "scientific": "Avis testus", "code": "x"}):
            meta = recognizers.train("Avis testus", records, log=lambda _text: None)
            self.assertEqual((meta["positives"], meta["hard_negatives"], meta["recordings"]), (12, 1, 4))
            self.assertTrue(meta["tested_on_other_recordings"])
            self.assertGreaterEqual(meta["precision"], 0.8)
            model = recognizers.load("Avis testus")
            found = recognizers.scan(model, ["fresh"])
            self.assertTrue(found and 13.5 <= found[0]["best"] <= 16.5)
            self.assertIn("Avis testus", recognizers.trained())
            with self.assertRaises(ValueError):                             # too few approved clips
                recognizers.train("Avis testus", records[:5], log=lambda _text: None)

    def test_threshold_keeps_precision(self):
        import numpy as np
        import recognizers
        probability = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4])
        truth = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        threshold, precision, recall = recognizers.choose_threshold(probability, truth)
        self.assertEqual((threshold, round(precision, 2), recall), (0.7, 0.8, 1.0))


class ReviewKeysTests(unittest.TestCase):
    def test_shortcut_keys_work_with_thai_layout(self):
        import birdnet_gui as g
        event = types.SimpleNamespace
        self.assertEqual(g.physical_key(event(keysym="a", keycode=0)), "a")
        self.assertEqual(g.physical_key(event(keysym="Return", keycode=0)), "return")
        # Thai layout: the keysym is a Thai letter, the physical key is still R (mac code 15, Windows 82)
        thai = event(keysym="Thai_phophan", keycode=(15 << 24) if sys.platform == "darwin" else 82)
        if sys.platform == "darwin" or os.name == "nt":
            self.assertEqual(g.physical_key(thai), "r")
        self.assertIsNone(g.physical_key(event(keysym="x", keycode=0)))

    def test_status_filter(self):
        import birdnet_gui as g
        self.assertTrue(g.review_status_shown("All", "Approved", ""))
        self.assertTrue(g.review_status_shown("Pending", "Pending", ""))
        self.assertFalse(g.review_status_shown("Pending", "Approved", "Check"))
        self.assertTrue(g.review_status_shown("Check", "Approved", "Check"))
        self.assertTrue(g.review_status_shown("Approved", "Ready file missing", ""))


class AddBatchTests(unittest.TestCase):
    def test_batch_adds_clips_and_skips_species_already_clipped(self):
        from manual_clips import add_batch, source_context
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "REC-001.WAV"
            make_tone(source)
            with patch.object(birdnetlib, "Recording", fake_recording((1,))):
                rows, day = pipeline.process_file(None, source, root / "out", **PROCESS_ARGS)
            summary = day / "summary.xlsx"
            pipeline.write_summary(rows, summary, {str(source)}, {pipeline.source_id(source)})
            context = source_context(load_reviews(summary), source)
            base = {"summary": str(summary), "source": str(source), "recorded": context["start"].isoformat(),
                    "lat": context["lat"], "lon": context["lon"]}
            items = [dict(base, start=20.0, end=24.0, common="Rufous-bellied Woodpecker",
                          scientific="Dendrocopos hyperythrus", notes="Found by Find similar (0.91)"),
                     dict(base, start=0.0, end=4.0, common="Test bird", scientific="Avis testus"),    # BirdNET clip there
                     dict(base, start=30.0, end=34.0, common="Test bird", scientific="Avis testus")]
            lines = []
            with patch("manual_clips.location_expected", lambda *_a: None), \
                    patch.object(pipeline, "gen_spectrograms", lambda dirs: 0):
                added = add_batch(items, log=lines.append)
            self.assertEqual([row["Species (common)"] for row in added], ["Rufous-bellied Woodpecker", "Test bird"])
            self.assertTrue(any(line.startswith("SKIPPED Test bird 0.0-4.0") for line in lines))
            self.assertEqual(lines[-1], "BATCH added 2 of 3")
            saved = {row["File"]: row for row in load_reviews(summary)}
            self.assertEqual(saved[added[0]["File"]]["Review notes"], "Found by Find similar (0.91)")
            self.assertEqual(sum(row.get("Origin") == "Manual" for row in saved.values()), 2)



class ReviewFlowTests(TkTestCase):
    def test_keys_approve_then_next_pending_and_status_filter(self):
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp:
            day = Path(temp) / "2026.10.10_1111_v3preview"
            day.mkdir()
            rows = []
            for n, status in enumerate(("Pending", "Approved", "Pending", "Pending")):
                make_wav(day / f"{n}.wav")
                rows.append({"File": f"{n}.wav", "Species (common)": "Asian Tit", "Species (scientific)": "Parus cinereus",
                             "Max confidence": 0.7, "Review status": status, "Generated sha256": f"s{n}"})
            summary = day / "summary.xlsx"
            pd.DataFrame(rows).to_excel(summary, index=False)
            with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                    patch.object(g, "load_settings", lambda *_args, **_kw: {"output_dir": temp}), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_check_index", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None), \
                    patch.object(g.App, "_retrain_sound_type", lambda self: None):
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    app.auto_next.set(True)
                    app.review_summary.set(str(summary))
                    app.review_load()
                    app.review_tree.selection_set("2")
                    app._review_select()
                    key = lambda sym: types.SimpleNamespace(keysym=sym, keycode=0, state=0)
                    self.assertEqual(app._review_key(key("4")), "break")
                    self.assertEqual(app.review_rating.get(), "4")
                    app._review_key(key("Return"))                          # approve -> next pending (skips approved)
                    self.assertEqual(load_reviews(summary)[0]["Review status"], "Approved")
                    self.assertEqual(app.review_tree.selection(), ("4",))
                    app.review_status_filter.set("Pending")
                    app._fill_review_tree()
                    self.assertEqual(set(app.review_tree.get_children()), {"4", "5"})
                    self.assertEqual(app.review_species_count.get(), "2 of 4 clips")
                    app.review_tree.selection_set("4")
                    app._review_select()
                    app._review_key(key("u"))                               # not sure; hidden by the filter
                    self.assertEqual(app.review_tree.selection(), ("5",))
                    self.assertEqual(set(app.review_tree.get_children()), {"5"})
                finally:
                    root.destroy()


if __name__ == "__main__":
    unittest.main()
