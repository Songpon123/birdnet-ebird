"""Flag (MARK) ที่กดตอนอัดบน Zoom H4essential: อ่านจาก cue chunk, แท็บ Review, หน้าต่าง Recorder flags, รูปถ่ายใกล้ flag"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import struct
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
import audio_io  # noqa: E402
import field_log  # noqa: E402
import recorder_flags  # noqa: E402

RATE = 8000


def write_wav(path, seconds, flags, rf64=False):
    """WAV แบบที่ Zoom เขียน: fmt + data แล้ว cue + LIST/adtl ต่อท้ายเสียง"""
    samples = np.zeros(int(seconds * RATE), "<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
    cue = struct.pack("<I", len(flags)) + b"".join(
        struct.pack("<II4sIII", n, int(t * RATE), b"data", 0, 0, int(t * RATE)) for n, t in enumerate(flags, 1))
    labels = b"".join(b"labl" + struct.pack("<II", 8, n) + b"%02d\x00\x00" % n for n in range(1, len(flags) + 1))
    tail = b"cue " + struct.pack("<I", len(cue)) + cue if flags else b""
    tail += b"LIST" + struct.pack("<I", 4 + len(labels)) + b"adtl" + labels if flags else b""
    head = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if rf64:
        body = b"ds64" + struct.pack("<IQQQI", 28, 0, len(samples), len(samples) // 2, 0) + head
        data = b"data" + struct.pack("<I", 0xFFFFFFFF) + samples
        Path(path).write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + body + data + tail)
    else:
        data = b"data" + struct.pack("<I", len(samples)) + samples
        Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(head) + len(data) + len(tail)) + b"WAVE"
                               + head + data + tail)
    return Path(path)


def write_log(folder, day, species, photos):
    """โฟลเดอร์ Log หนึ่งรอบ: checklist.json + files.csv"""
    run = folder / "Log" / "run1"
    run.mkdir(parents=True)
    (run / "checklist.json").write_text(json.dumps({
        "checklist_id": "S1", "date": f"{day:%d %b %Y} 8:08 AM", "location": "Wuhan University",
        "species": [{"common_name": common, "scientific_name": scientific, "count": "1", "media": []}
                    for common, scientific in species]}), encoding="utf-8")
    pd.DataFrame(photos, columns=["filename", "original_path", "capture_time", "common_name", "scientific_name"]).to_csv(
        run / "files.csv", index=False)
    return folder / "Log"


class ReadFlagsTests(unittest.TestCase):
    def test_reads_cue_points_from_riff_and_rf64(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_wav(Path(temp) / "261030_122253_TrMic.WAV", 4, [3.5, 1.25])
            self.assertEqual(recorder_flags.read_flags(path), [1.25, 3.5])
            self.assertEqual(recorder_flags.read_flags(write_wav(Path(temp) / "big.wav", 4, [2.0], rf64=True)), [2.0])
            self.assertEqual(recorder_flags.read_flags(write_wav(Path(temp) / "none.wav", 2, [])), [])
            (Path(temp) / "note.mp3").write_bytes(b"not a wav")
            (Path(temp) / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            self.assertEqual(recorder_flags.read_flags(Path(temp) / "note.mp3"), [])
            self.assertEqual(recorder_flags.read_flags(Path(temp) / "broken.wav"), [])
            self.assertEqual(recorder_flags.read_flags(Path(temp) / "missing.wav"), [])
            self.assertEqual(recorder_flags.recording_start(path), datetime(2026, 10, 30, 12, 22, 53))
            self.assertIsNone(recorder_flags.recording_start("REC-001.WAV"))

    def test_remembers_flags_after_the_recording_moves(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(recorder_flags, "CACHE", Path(temp) / "cache"):
            path = write_wav(Path(temp) / "a.wav", 4, [1.0])
            self.assertEqual(recorder_flags.flags_of("abc", path), [1.0])
            path.unlink()
            self.assertEqual(recorder_flags.flags_of("abc", path), [1.0])
            self.assertEqual(recorder_flags.flags_of("other", path), [])

    def test_a_clip_is_near_a_flag_from_30_s_before_to_10_s_after(self):
        near = recorder_flags.distance
        self.assertEqual(near([2238.4], 2230, 2245), (2238.4, 0.0))              # flag อยู่ในคลิป
        self.assertEqual(near([2238.4], 2242, 2252), (2238.4, 3.6))              # คลิปอยู่หลัง flag
        self.assertEqual(near([2238.4], 2217, 2220), (2238.4, -18.4))            # คลิปอยู่ก่อน flag
        self.assertIsNone(near([2238.4], 2264, 2280))                            # หลัง flag เกิน 10 วินาที
        self.assertIsNone(near([2238.4], 2190, 2200))                            # ก่อน flag เกิน 30 วินาที
        self.assertEqual(near([100.0, 200.0], 205, 210), (200.0, 5.0))           # flag ที่ใกล้สุด

    def test_photos_near_a_flag_use_the_camera_clock_offset(self):
        photos = [{"time": "2026-10-25T09:46:31", "common": "Pallas's Leaf Warbler", "file": "a.jpg"},
                  {"time": "2026-10-25T09:50:00", "common": "", "file": "b.jpg"}]
        flag = datetime(2026, 10, 25, 9, 46, 30)
        self.assertEqual([(away, photo["file"]) for away, photo in recorder_flags.photos_near(flag, photos)],
                         [(1.0, "a.jpg")])
        late = recorder_flags.photos_near(flag, photos, camera_offset=120)        # กล้องเร็วกว่าเครื่องอัด 2 นาที
        self.assertEqual([(away, photo["file"]) for away, photo in late], [(90.0, "b.jpg"), (-119.0, "a.jpg")])

    def test_field_log_keeps_one_photo_per_shot_with_its_capture_time(self):
        with tempfile.TemporaryDirectory() as temp:
            log = write_log(Path(temp) / "20261025", datetime(2026, 10, 25), [("Pallas's Leaf Warbler", "Phylloscopus proregulus")], [
                ("20261025-4X8A1891.dng", "/x/a.dng", "2026-10-25T09:46:31.05+08:00", "", ""),
                ("4X8A1891.CR3", "/x/a.cr3", "2026-10-25T09:46:31.05+08:00", "", ""),
                ("20261025-4X8A1891.jpg", "/x/a.jpg", "2026-10-25T09:46:31.05+08:00", "Pallas's Leaf Warbler", "Phylloscopus proregulus"),
                ("20261025-4X8A1892.jpg", "/x/b.jpg", "2026-10-25T09:46:40+08:00", "", ""),
                ("no-time.jpg", "/x/c.jpg", "", "", "")])
            checklists = field_log.find_checklists(log)
            self.assertEqual(len(checklists[0]["photos"]), 4)
            photos = field_log.photos_on(checklists)
            self.assertEqual([(p["time"], p["file"], p["common"]) for p in photos],
                             [("2026-10-25T09:46:31", "20261025-4X8A1891.jpg", "Pallas's Leaf Warbler"),
                              ("2026-10-25T09:46:40", "20261025-4X8A1892.jpg", "")])


class RecordingStartTests(unittest.TestCase):
    def test_bext_time_is_used_when_ffprobe_gives_only_a_date(self):
        import field_audio_to_ebird as pipeline
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "261030_063421_TrMic.WAV"
            bext = b"zTAKE=002\r\n".ljust(256, b"\0") + b"ZOOM H4essential".ljust(32, b"\0") + bytes(32)
            bext += b"2026-10-3006:34:21" + bytes(64)
            data = np.zeros(RATE, "<i2").tobytes()
            fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
            body = (b"WAVE" + b"bext" + struct.pack("<I", len(bext)) + bext + b"fmt " + struct.pack("<I", len(fmt)) + fmt
                    + b"data" + struct.pack("<I", len(data)) + data)
            path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
            only_date = {"datetime": datetime(2026, 10, 30), "datetime_has_time": False}
            pipeline._read_wav_meta(path, only_date)
            self.assertEqual(only_date, {"datetime": datetime(2026, 10, 30, 6, 34, 21), "datetime_has_time": True})
            with_time = {"datetime": datetime(2026, 10, 30, 7, 0), "datetime_has_time": True}   # มีเวลาแล้ว: ไม่ทับ
            pipeline._read_wav_meta(path, with_time)
            self.assertEqual(with_time["datetime"], datetime(2026, 10, 30, 7, 0))
            context = pipeline.recording_context_preview(path)
            self.assertEqual((context["datetime"], context["has_time"]), (datetime(2026, 10, 30, 6, 34, 21), True))


class CollectAudioTests(unittest.TestCase):
    def test_finds_recordings_in_subfolders_but_not_results_or_hidden_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("Raw/261030_062349/261030_062349_TrMic.WAV", "Raw/261030_063421/261030_063421_TrMic.WAV",
                         "top.mp3", ".Trash/old.wav", "Raw/._261030.WAV", "Raw/ZOOM.SYS",
                         "2026.10.25/0909_v3preview/Asian Tit/clip.wav", "2026.10.25/Ready/Asian Tit/ready.wav"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            (root / "2026.10.25" / "0909_v3preview" / "summary.xlsx").write_bytes(b"x")
            found = [p.relative_to(root).as_posix() for p in audio_io.collect_audio(root)]
            self.assertEqual(found, ["Raw/261030_062349/261030_062349_TrMic.WAV",
                                     "Raw/261030_063421/261030_063421_TrMic.WAV", "top.mp3"])


class FlagScreensTests(unittest.TestCase):
    """แท็บ Review และหน้าต่าง Recorder flags กับไฟล์จำลอง"""

    def result(self, temp):
        """ไฟล์ 09:09 (flag ที่ 60 วินาที) มีคลิปใกล้ flag 1 คลิป ไกล 1 คลิป + ไฟล์ 12:22 ยังไม่ได้วิเคราะห์ (flag 2 จุด)"""
        from field_audio_to_ebird import source_id
        sounds = Path(temp) / "Birding" / "2026" / "Oct" / "20261025" / "Sounds" / "Raw"
        (sounds / "261025_090912").mkdir(parents=True)
        (sounds / "261025_122253").mkdir()
        source = write_wav(sounds / "261025_090912" / "261025_090912_TrMic.WAV", 200, [60.0])
        write_wav(sounds / "261025_122253" / "261025_122253_TrMic.WAV", 30, [5.0, 12.0])
        log = write_log(Path(temp) / "Birding" / "2026" / "Oct" / "20261025", datetime(2026, 10, 25),
                        [("Swinhoe's White-eye", "Zosterops simplex")],
                        [("a.jpg", str(Path(temp) / "a.jpg"), "2026-10-25T09:10:20+08:00", "Swinhoe's White-eye", "Zosterops simplex")])
        field_log.add(log)
        day = Path(temp) / "out" / "2026.10.25" / "0909_v3preview"
        (day / "Test bird").mkdir(parents=True)
        rows, sid = [], source_id(source)
        for n, (start, end) in enumerate(((55.0, 58.0), (150.0, 153.0))):
            clip = day / "Test bird" / f"{n}.wav"
            sf.write(clip, np.zeros(RATE * 9, np.float32), RATE)
            rows.append({"File": f"Test bird/{n}.wav", "Species (common)": "Test bird", "Species (scientific)": "Avis testus",
                         "Max confidence": 0.7, "Review status": "Pending", "Source file": source.name,
                         "Source path": str(source), "Source ID": sid, "Start offset (s)": start, "End offset (s)": end,
                         "Clock time": f"{datetime(2026, 10, 25, 9, 9, 12) + timedelta(seconds=start):%H:%M:%S}",
                         "Detection boxes": json.dumps({"windows": [[3.0, 6.0, 0.7]], "sounds": []}),
                         "Analysis settings": json.dumps({"date": "2026-10-25"})})
        summary = day / "summary.xlsx"
        pd.DataFrame(rows).to_excel(summary, index=False)
        return summary, source

    def stores(self, temp):
        return (patch.object(field_log, "STORE", Path(temp) / "field-logs.json"),
                patch.object(recorder_flags, "CACHE", Path(temp) / "flags"))

    def test_review_shows_flagged_clips_and_can_filter_them(self):
        import tkinter as tk
        import birdnet_gui as g
        with tempfile.TemporaryDirectory() as temp:
            one, two = self.stores(temp)
            with one, two, patch.object(g, "save_settings", lambda *_a, **_k: None), \
                    patch.object(g, "load_settings", lambda *_a, **_k: {"output_dir": temp}), \
                    patch.object(g.App, "_start", lambda self, *a, **k: None), \
                    patch.object(g.App, "_check_auto", lambda self, s: None), \
                    patch.object(g.App, "_check_season", lambda self, s: None), \
                    patch.object(g.App, "_check_spectrograms", lambda self, s: None), \
                    patch.object(g.App, "_check_model", lambda self, s: None), \
                    patch.object(g.App, "_start_ebird_check", lambda self, row: None):
                summary, _source = self.result(temp)
                try:
                    root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"no display: {exc}")
                root.withdraw()
                try:
                    app = g.App(root)
                    root.createcommand("::tk::mac::Quit", lambda: None)
                    app.review_summary.set(str(summary))
                    app.review_load()
                    tree = app.review_tree
                    self.assertEqual(tree.set("2", "flag"), "⚑ -2 s")                # คลิปอยู่ก่อน flag 2 วินาที
                    self.assertEqual(tree.set("3", "flag"), "")
                    tree.selection_set("2")
                    root.update()
                    self.assertIn("⚑ 2 s before your recorder flag at 1:00", app.review_context.get())
                    self.assertEqual(app.review_view.flags, [8.0])                   # เวลาในคลิป: คลิปเริ่มที่ 52 วินาที
                    app.review_status_filter.set("Near a flag")
                    app._fill_review_tree()
                    self.assertEqual(tree.get_children(), ("2",))
                    self.assertEqual(app.review_species_count.get(), "1 of 2 clips")
                    app.review_status_filter.set("All")
                    app._fill_review_tree()
                    app._sort_review("flag")
                    self.assertEqual(tree.get_children(), ("2", "3"))                # มี flag ก่อน
                finally:
                    root.destroy()

    def test_flags_window_lists_flags_near_clips_and_photos(self):
        import tkinter as tk
        import flags_window as fw
        with tempfile.TemporaryDirectory() as temp:
            one, two = self.stores(temp)
            with one, two, patch("app_settings.load_settings", return_value={}), \
                    patch("result_roots.DEFAULT", Path(temp) / "no-default"), \
                    patch("flags_window.load_settings", return_value={}), \
                    patch("flags_window.save_settings") as saved, \
                    patch("flags_window.platform_tools.open_path") as opened:
                summary, source = self.result(temp)
                (Path(temp) / "a.jpg").write_bytes(b"x")
                root = tk.Tk()
                root.withdraw()
                root.createcommand("::tk::mac::Quit", lambda: None)
                try:
                    calls = []
                    app = types.SimpleNamespace(root=root, out=types.SimpleNamespace(get=lambda: str(Path(temp) / "out")),
                                                analyze_path=calls.append, show_clip=lambda *a: calls.append(a),
                                                open_manual_at=lambda *a: calls.append(a))
                    window = fw.RecorderFlagsWindow(app)
                    for _ in range(500):
                        root.update()
                        time.sleep(0.01)
                        if not window._busy:
                            break
                    self.assertEqual(len(window.items), 3)
                    self.assertTrue(window.state.get().startswith("3 flag(s) in 2 recording(s)"))
                    first = window.tree.item("0", "values")
                    self.assertEqual(first[:3], ("2026-10-25 09:10:12", source.name, "1:00"))
                    self.assertEqual(first[3], "Test bird -2 s")
                    self.assertEqual((first[4], first[5]), ("Swinhoe's White-eye", "analyzed"))   # รูปห่าง 8 วินาที
                    self.assertEqual(window.tree.item("1", "values")[3:], ("not analyzed yet", "", "not analyzed"))
                    window.tree.selection_set("0")
                    root.update()
                    near = [window.near_tree.item(iid, "values") for iid in window.near_tree.get_children()]
                    self.assertEqual([row[0] for row in near], ["Clip", "Photo"])
                    self.assertEqual(near[1][2:4], ("+8 s", "Swinhoe's White-eye"))
                    window.near_tree.selection_set("c0")
                    window.open_near()
                    self.assertEqual(calls.pop(), (summary, "Test bird/0.wav"))
                    window.near_tree.selection_set("p0")
                    window.open_near()
                    opened.assert_called_once()
                    window.add_clip()
                    self.assertEqual(calls.pop(), (summary, source, 55.0))
                    with patch("listening_tools.ListeningToolsWindow") as listen:
                        window.listen()
                    keywords = listen.call_args.kwargs
                    self.assertEqual((keywords["start"], keywords["end"], keywords["flags"]), (30.0, 70.0, [60.0]))
                    self.assertEqual(keywords["clock_start"], datetime(2026, 10, 25, 9, 9, 42))
                    window.offset.set("600")                                         # กล้องเร็ว 10 นาที: รูปไม่ใกล้แล้ว
                    window._offset_changed()
                    saved.assert_called_with({fw.OFFSET_SETTING: 600.0})
                    self.assertEqual(window.tree.item("0", "values")[4], "")
                    window.tree.selection_set("1")
                    root.update()
                    window.add_clip()
                    self.assertIn("has not been analyzed yet", window.state.get())
                    window.analyze()
                    self.assertEqual(calls.pop().name, "261025_122253_TrMic.WAV")
                    window.close()
                finally:
                    root.destroy()

    def test_listen_tools_draws_flags_and_keeps_the_clock_right_mid_file(self):
        import tkinter as tk
        from listening_tools import ListeningToolsWindow
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "take.wav"
            sf.write(path, np.random.default_rng(1).normal(scale=0.01, size=RATE * 12).astype(np.float32), RATE)
            root = tk.Tk()
            root.withdraw()
            root.createcommand("::tk::mac::Quit", lambda: None)
            try:
                window = ListeningToolsWindow(root, path, start=2.0, end=10.0, clock_start=datetime(2026, 10, 25, 9, 0, 2),
                                              flags=[4.0, 11.5])
                for _ in range(300):
                    root.update()
                    time.sleep(0.01)
                    if window.prism.data:
                        break
                view = window.prism
                view.redraw()
                self.assertEqual(view.clock_start, datetime(2026, 10, 25, 9, 0, 0))   # เวลาจริงที่ต้นไฟล์
                self.assertEqual(window.macaulay.flags, [2.0, 9.5])                   # เวลานับจากต้นช่วงที่เปิด
                self.assertEqual(len(view.spec.find_withtag("flags")), 2)             # 1 จุดในจอ: เส้น + ป้าย
                window.close()
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
