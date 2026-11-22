"""ระดับเสียง: ค่าเดียวทุกหน้า จำไว้ ปรับระหว่างเล่นแล้วเล่นต่อจากจุดเดิม ขยายเกิน 100% แล้วไม่แตก — ไม่แตะไฟล์คลิป"""

from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import playback  # noqa: E402
import platform_tools  # noqa: E402

RATE = 8000


class Process:
    """แทน Popen ของ afplay"""

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.started = []

        def start(path, volume=1.0):
            self.started.append((Path(path), volume, Process()))
            return self.started[-1][2]

        self.saved = {}
        for patcher in (patch.object(playback, "FOLDER", self.temp / "playback"),
                        patch.object(playback, "_percent", 100), patch.object(playback, "_playing", []),
                        patch.object(platform_tools, "start_player", start),
                        patch.object(playback, "load_settings", lambda: dict(self.saved)),
                        patch.object(playback, "save_settings", self.saved.update)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.clip = self.temp / "clip.wav"
        tone = 0.5 * np.sin(np.arange(RATE * 4) * 0.2)
        tone[:RATE] *= 0.1                                                         # เสียงเบาช่วงแรก
        sf.write(self.clip, tone, RATE)

    def written(self, index=-1):
        return sf.read(str(self.started[index][0]), dtype="float32")[0]


class VolumeTests(Case):
    def test_at_or_below_full_level_the_clip_file_itself_is_played(self):
        playback.set_volume(40)
        playback.play(self.clip)
        self.assertEqual(self.started[0][:2], (self.clip, 0.4))
        self.assertFalse((self.temp / "playback").exists())

    def test_above_full_level_makes_quiet_sound_louder_without_clipping(self):
        playback.set_volume(300)
        before = sf.read(str(self.clip), dtype="float32")[0]
        playback.play(self.clip)
        path, level, _process = self.started[0]
        self.assertNotEqual(path, self.clip)
        self.assertEqual(level, 1.0)
        louder = self.written()
        quiet = before[:RATE]
        np.testing.assert_allclose(louder[:RATE], quiet * 3, atol=1e-5)            # เสียงเบา: ดังขึ้น 3 เท่าตรง ๆ
        self.assertLess(float(np.max(np.abs(louder))), 1.0)                        # เสียงดัง: อัดลง ไม่เกิน full scale
        self.assertGreater(float(np.max(np.abs(louder))), 0.8)
        np.testing.assert_array_equal(sf.read(str(self.clip), dtype="float32")[0], before)   # ไฟล์คลิปไม่เปลี่ยน

    def test_unreadable_file_still_plays(self):
        playback.set_volume(200)
        broken = self.temp / "reference.mp3"
        broken.write_bytes(b"not audio")
        playback.play(broken)
        self.assertEqual(self.started[0][0], broken)

    def test_volume_is_kept_in_range_and_read_from_settings(self):
        for saved, expected in ((150, 150), (999, 300), (-5, 0), ("x", 100), (None, 100)):
            self.saved[playback.SETTING] = saved
            with patch.object(playback, "_percent", None):
                self.assertEqual(playback.volume(), expected, saved)

    def test_players_are_called_with_the_volume(self):
        with patch.object(platform_tools.subprocess, "Popen") as popen, patch.object(platform_tools, "start_player",
                                                                                     self.real_start):
            with patch.object(platform_tools, "IS_MAC", True):
                platform_tools.start_player("a.wav", 0.25)
            self.assertEqual(popen.call_args.args[0], ["afplay", "-v", "0.25", "a.wav"])
            with patch.object(platform_tools, "IS_MAC", False), patch.object(platform_tools, "bundled_tool",
                                                                             return_value="ffplay"):
                platform_tools.start_player("a.wav", 0.25)
            self.assertEqual(popen.call_args.args[0][-3:], ["-volume", "25", "a.wav"])

    real_start = staticmethod(platform_tools.start_player)


class ChangingWhilePlayingTests(Case):
    def playing(self, at):
        sound = playback.play(self.clip)
        sound.began = time.monotonic() - at
        return sound

    def test_new_level_continues_from_where_it_was(self):
        sound = self.playing(1.0)
        first = sound.process
        playback.set_volume(50)
        playback.follow_volume()
        self.assertEqual(first.returncode, -15)
        self.assertIsNone(sound.poll())                                            # หน้าที่เปิดอยู่เห็นว่ายังเล่น
        path, level, _process = self.started[-1]
        self.assertEqual(level, 0.5)
        self.assertLess(abs(len(self.written()) - RATE * 3), RATE // 10)           # เหลืออีก 3 วินาทีจาก 4
        self.assertLess(abs(sound.position() - 1.0), 0.1)

    def test_changing_twice_keeps_counting_from_the_clip(self):
        sound = self.playing(1.0)
        playback.set_volume(50)
        playback.follow_volume()
        sound.began -= 1.0
        playback.set_volume(200)
        playback.follow_volume()
        self.assertLess(abs(len(self.written()) - RATE * 2), RATE // 10)

    def test_stop_ends_the_new_player(self):
        sound = self.playing(1.0)
        playback.set_volume(50)
        playback.follow_volume()
        sound.terminate()
        self.assertEqual(sound.poll(), -15)
        self.assertEqual(self.started[-1][2].returncode, -15)

    def test_finished_sound_is_left_alone(self):
        sound = self.playing(1.0)
        sound.process.returncode = 0
        playback.set_volume(50)
        playback.follow_volume()
        self.assertEqual(len(self.started), 1)
        self.assertEqual(sound.returncode, 0)                                      # Loop selection ดูค่านี้

    def test_past_the_end_starts_nothing(self):
        self.playing(10.0)
        playback.set_volume(50)
        playback.follow_volume()
        self.assertEqual(len(self.started), 1)

    def test_file_that_cannot_be_cut_keeps_playing(self):
        broken = self.temp / "reference.mp3"
        broken.write_bytes(b"not audio")
        sound = playback.play(broken)
        playback.set_volume(50)
        playback.follow_volume()
        self.assertIsNone(sound.poll())
        self.assertEqual(len(self.started), 1)


class ControlTests(Case):
    def setUp(self):
        super().setUp()
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.window = tk.Toplevel(self.root)
        self.first, self.second = playback.control(self.window), playback.control(self.window)
        self.first.pack()
        self.second.pack()
        self.root.update()

    def parts(self, frame):
        names = [child.winfo_class() for child in frame.winfo_children()]
        self.assertEqual(names, ["TLabel", "TButton", "TScale", "TButton", "TLabel", "TButton"])
        return frame.winfo_children()

    def settle(self):
        self.root._playback_level._settled()

    def test_buttons_step_and_stop_at_the_ends(self):
        _label, minus, _scale, plus, percent, _normal = self.parts(self.first)
        plus.invoke()
        self.assertEqual((playback.volume(), percent.cget("textvariable") and self.root._playback_level.text.get()),
                         (110, "110%"))
        for _ in range(30):
            plus.invoke()
        self.assertEqual(playback.volume(), 300)
        for _ in range(40):
            minus.invoke()
        self.assertEqual(playback.volume(), 0)

    def test_normal_returns_to_full_level_in_every_window_and_while_playing(self):
        sound = playback.play(self.clip)
        sound.began = time.monotonic() - 1.0
        for _ in range(5):
            self.parts(self.first)[3].invoke()
        self.assertEqual(playback.volume(), 150)
        self.parts(self.second)[5].invoke()
        self.assertEqual(playback.volume(), 100)
        self.assertEqual(self.root._playback_level.text.get(), "100%")
        self.assertEqual(float(self.parts(self.first)[2].get()), 100.0)
        time.sleep((playback.SETTLE_MS + 60) / 1000)
        self.root.update()
        self.assertEqual(self.saved, {playback.SETTING: 100})
        self.assertEqual(self.started[-1][1], 1.0)
        self.assertTrue(RATE * 2 < len(self.written()) < RATE * 3)                 # เล่นต่อจากจุดเดิม ไม่เริ่มใหม่

    def test_every_window_shows_the_same_level(self):
        self.parts(self.first)[3].invoke()
        self.assertEqual(float(self.parts(self.second)[2].get()), 110.0)

    def test_level_is_saved_and_applied_once_after_settling(self):
        sound = playback.play(self.clip)
        sound.began = time.monotonic() - 1.0
        plus = self.parts(self.first)[3]
        plus.invoke()
        plus.invoke()
        self.assertEqual(self.saved, {})
        time.sleep((playback.SETTLE_MS + 60) / 1000)
        self.root.update()
        self.assertEqual(self.saved, {playback.SETTING: 120})
        self.assertEqual(len(self.started), 2)

    def test_clicking_does_not_take_the_keyboard_from_the_clip_list(self):
        import tkinter as tk
        keeper = tk.Entry(self.window)
        keeper.pack()
        keeper.focus_force()
        self.root.update()
        scale = self.parts(self.first)[2]
        scale.event_generate("<ButtonPress-1>", x=40, y=5)
        scale.event_generate("<ButtonRelease-1>", x=40, y=5)
        self.parts(self.first)[3].invoke()
        self.root.update()
        self.assertIs(self.window.focus_get(), keeper)


class ReviewTabTests(unittest.TestCase):
    def test_review_buttons_fit_the_default_window(self):
        import tkinter as tk
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
                    patch.object(g.App, "_start", lambda self, *a, **k: None):
                app = g.App(root)
                root.createcommand("::tk::mac::Quit", lambda: None)
                root.update_idletasks()

                def walk(widget):
                    yield widget
                    for child in widget.winfo_children():
                        yield from walk(child)

                volume = next(w for w in walk(root) if w.winfo_class() == "TLabel" and str(w.cget("text")) == "Vol")
                room = 914                                                         # พื้นที่จริงของแถวเมื่อหน้าต่างกว้าง 1000
                self.assertLessEqual(volume.master.master.winfo_reqwidth(), room)
                names = [app.nb.tab(tab, "text") for tab in app.nb.tabs()]
                self.assertEqual(names, ["Settings", "Review clips", "Merge clips"])   # เรียงตามลำดับงาน
                texts = [str(w.cget("text")) for w in walk(root) if w.winfo_class() == "TButton"]
                for gone in ("▶  Analyze", "File…", "Folder…", "Large view…"):         # ซ้ำกับปุ่มอื่น
                    self.assertNotIn(gone, texts)
                self.assertEqual(texts.count("Add Log folder…"), 1)
                self.assertIs(app.run_btn, app.quick_start_btn)
                lookups = app.second_opinion_btn.master
                self.assertLessEqual(lookups.winfo_reqwidth() + int(str(app.checklist_label.cget("wraplength"))), room)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
