"""ปุ่ม Analyze บนแถบบนทางเดียว: ค่าที่พิมพ์ใน Settings ยังแทนค่าในไฟล์ได้ และปุ่มภาพกรอบ BirdNET ในแท็บ Review"""

from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import birdnet_gui as g  # noqa: E402


def text_var(value=""):
    return SimpleNamespace(get=lambda: value)


class AnalyzeTests(unittest.TestCase):
    def details(self, overrides):
        """กด Start analysis ในหน้า Review recording details -> อาร์กิวเมนต์ที่ส่งให้ run"""
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        self.addCleanup(root.destroy)
        with patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                patch.object(g.App, "_start", lambda *_args, **_kw: None):
            app = g.App(root)
        root.createcommand("::tk::mac::Quit", lambda: None)
        app._overrides = overrides
        app.quick_files = [Path("a.wav")]
        context = {"datetime": None, "datetime_source": None, "has_time": True, "lat": 1.0, "lon": 2.0,
                   "coords_source": "embedded metadata", "place": None}
        calls = []
        app.run = lambda **kwargs: calls.append(kwargs)
        app.date.set("20261030")
        app._quick_details([(Path("a.wav"), context)], [], [])
        dialog = next(w for w in root.winfo_children() if w.winfo_class() == "Toplevel")

        def walk(widget):
            yield widget
            for child in widget.winfo_children():
                yield from walk(child)

        labels = [str(w.cget("text")) for w in walk(dialog) if w.winfo_class() == "TLabel"]
        next(w for w in walk(dialog) if w.winfo_class() == "TButton" and str(w.cget("text")) == "Start analysis").invoke()
        return calls[0], labels

    def test_values_typed_in_settings_replace_what_the_file_says(self):
        call, labels = self.details(True)
        self.assertEqual((call["fill_missing"], call["confirmed_batch"]), (False, False))
        self.assertTrue(any("replace these details" in text for text in labels))

    def test_a_new_file_only_fills_what_is_missing(self):
        call, labels = self.details(False)
        self.assertEqual((call["fill_missing"], call["confirmed_batch"]), (True, True))
        self.assertFalse(any("replace these details" in text for text in labels))

    def test_analyze_notes_typed_values_before_asking(self):
        fake = SimpleNamespace(quick_busy=False, audio=text_var(__file__), date=text_var("20261030"),
                               start_time=text_var(), coords=text_var(), quick_file_btn=SimpleNamespace(configure=lambda **_k: None),
                               quick_folder_btn=SimpleNamespace(configure=lambda **_k: None),
                               quick_start_btn=SimpleNamespace(configure=lambda **_k: None),
                               quick_probe=None, root=SimpleNamespace(after=lambda *_a: None), status=SimpleNamespace(set=lambda _t: None),
                               progress=SimpleNamespace(start=lambda *_a: None), quick_q=None, _quick_probe_worker=lambda _p: None,
                               _quick_poll=lambda: None, quick_metadata=SimpleNamespace(set=lambda _t: None))
        g.App._quick_start_selected(fake)
        self.assertTrue(fake._overrides)


class ButtonTests(unittest.TestCase):
    def test_analyze_can_always_be_pressed_and_asks_for_audio_first(self):
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as temp, patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                patch.object(g.App, "_start", lambda *_args, **_kw: None):
            app = g.App(root)
            root.createcommand("::tk::mac::Quit", lambda: None)
            self.assertEqual(str(app.quick_start_btn.cget("state")), "normal")
            self.assertEqual(str(app.quick_start_btn.cget("text")), "▶ Analyze")
            sounds = Path(temp) / "20261102" / "Sounds"
            (sounds / "Raw").mkdir(parents=True)
            app.audio.set("")
            app.out.set(str(sounds / "Done"))
            with patch.object(g.filedialog, "askdirectory", return_value="") as chooser, \
                    patch.object(g.messagebox, "showwarning") as warned:
                app.quick_start_btn.invoke()
            self.assertEqual(chooser.call_args.kwargs["initialdir"], str(sounds / "Raw"))   # เปิดที่ Raw ของวันนั้น
            warned.assert_not_called()
            self.assertFalse(app.quick_busy)                                                # ยกเลิก = ไม่เริ่มงาน


class FinishTests(unittest.TestCase):
    def test_a_folder_run_opens_the_first_result_and_lists_them_all(self):
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as temp, patch.object(g, "save_settings", lambda *_args, **_kw: None), \
                patch.object(g, "load_settings", lambda *_args, **_kw: {}), \
                patch.object(g.App, "_start", lambda *_args, **_kw: None):
            app = g.App(root)
            root.createcommand("::tk::mac::Quit", lambda: None)
            day = Path(temp) / "2026.11.06"
            summaries = [day / name / "summary.xlsx" for name in ("1658_v3preview", "1318_v3preview", "1451_v3preview")]
            opened, refreshed = [], []

            def load(quiet=False, keep_selection=False):
                opened.append(app.review_summary.get())
                app.review_rows = {"2": {}}

            with patch.object(app, "review_load", load), patch.object(app, "refresh_results", lambda: refreshed.append(1)):
                app.current_job, app._active_is_analyze = {"title": "Analyze"}, True
                app._generated_summaries, app._clip_count, app._already_done = summaries, 144, False
                app._finish(0)
            self.assertEqual(opened, [str(summaries[1])])                                  # 13:18 = ชุดแรกของวัน
            self.assertEqual(refreshed, [1])
            self.assertEqual(app.nb.select(), str(app.review_tab))
            self.assertIn("3 results", app.status.get())


class PictureTests(unittest.TestCase):
    def test_boxes_button_opens_the_clips_picture(self):
        with tempfile.TemporaryDirectory() as temp:
            clip = Path(temp) / "clip.wav"
            png = clip.with_suffix(".png")
            app = SimpleNamespace(_review_audio_path=lambda: clip)
            with patch.object(g, "open_path") as opened, patch.object(g.messagebox, "showinfo") as told:
                g.App.review_boxes_picture(app)
                told.assert_called_once()
                opened.assert_not_called()
                png.write_bytes(b"x")
                g.App.review_boxes_picture(app)
                opened.assert_called_once_with(png)


if __name__ == "__main__":
    unittest.main()
