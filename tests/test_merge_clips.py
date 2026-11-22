"""แท็บ Merge clips: ชื่อจากชนิดของคลิป เก็บข้างคลิปหรือเลือกโฟลเดอร์เอง ไม่ทับไฟล์เดิม"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import merge_clips  # noqa: E402

READY = "/day/Ready/Asian Tit/2026.10.25_0909_v3preview_987321e5b7eb_Parus.cinereus_Call_R4.wav"
WITH = "/day/Ready/Asian Tit/with-Oriental-Magpie__2026.10.25_0909_v3preview_aa_Parus.cinereus_Song_R3.wav"
RESULT = "/day/0909_v3preview/Swinhoe's White-eye/2026.10.25_0909_0914_Zosterops.simplex_7a6b38c1d8a5_001_R0.wav"
OTHER = "/Downloads/XC12345 bird.wav"


class NameTests(unittest.TestCase):
    def test_bird_name_comes_from_the_species_folder(self):
        self.assertEqual(merge_clips.suggested_name([READY, WITH]), "Asian Tit merged")
        self.assertEqual(merge_clips.suggested_name([RESULT]), "Swinhoe's White-eye merged")

    def test_other_files_and_mixed_species(self):
        self.assertEqual(merge_clips.suggested_name([OTHER, OTHER]), "XC12345 bird merged")
        self.assertEqual(merge_clips.suggested_name([READY, RESULT]), "Asian Tit and Swinhoe's White-eye merged")
        self.assertEqual(merge_clips.species_in([READY, OTHER, WITH, RESULT]), ["Asian Tit", "Swinhoe's White-eye"])
        self.assertEqual(merge_clips.suggested_name([]), "")

    def test_existing_file_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            first = merge_clips.free_path(folder, "Asian Tit merged")
            self.assertEqual(first.name, "Asian Tit merged.wav")
            first.write_bytes(b"x")
            self.assertEqual(merge_clips.free_path(folder, "Asian Tit merged.WAV").name, "Asian Tit merged 2.wav")
            (folder / "Asian Tit merged 2.wav").write_bytes(b"x")
            self.assertEqual(merge_clips.free_path(folder, "Asian Tit merged").name, "Asian Tit merged 3.wav")

    def test_name_cannot_leave_the_folder(self):
        self.assertEqual(merge_clips.free_path("/x", "../a/b:c").parent, Path("/x"))
        self.assertEqual(merge_clips.free_path("/x", "   ").name, "Merged.wav")


class TabTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import birdnet_gui as g
        self.g = g
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.jobs = []
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        self.root.withdraw()
        self.root.createcommand("::tk::mac::Quit", lambda: None)
        self.addCleanup(self.root.destroy)
        jobs = self.jobs
        for patcher in (patch.object(g, "save_settings", lambda *_args, **_kw: None),
                        patch.object(g, "load_settings", lambda *_args, **_kw: {}),
                        patch.object(g.App, "_start", lambda _self, *args, **kwargs: jobs.append((args, kwargs)))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = g.App(self.root)
        self.root.createcommand("::tk::mac::Quit", lambda: None)
        self.folder = self.temp / "Ready" / "Asian Tit"
        self.folder.mkdir(parents=True)
        self.clips = [str(self.folder / f"2026.10.25_0909_v3preview_{n}_Parus.cinereus_Call_R4.wav") for n in "ab"]

    def add(self, *clips):
        with patch.object(self.g.filedialog, "askopenfilenames", return_value=clips):
            self.app.merge_add()

    def test_adding_clips_fills_the_name_and_uses_their_folder(self):
        self.add(*self.clips)
        self.assertEqual(self.app.merge_name.get(), "Asian Tit merged")
        self.assertEqual(self.app.merge_where.get(), str(self.folder))
        self.app.merge_run()
        args, kwargs = self.jobs[0]
        self.assertEqual(args[0][-3:-1], [self.clips[1], "-o"])
        self.assertEqual(Path(args[0][-1]), self.folder / "Asian Tit merged.wav")
        kwargs["on_done"](0, [])
        self.assertEqual(self.app.merge_where.get(), f"Saved: {self.folder / 'Asian Tit merged.wav'}")

    def test_a_name_you_typed_stays_when_clips_change(self):
        self.add(self.clips[0])
        self.app.merge_name.set("Dawn duet")
        self.add(self.clips[1])
        self.assertEqual(self.app.merge_name.get(), "Dawn duet")
        self.app.merge_clear()
        self.assertEqual(self.app.merge_name.get(), "Dawn duet")

    def test_choosing_a_folder_sends_the_file_there_and_going_back_works(self):
        self.add(*self.clips)
        other = self.temp / "Upload"
        other.mkdir()
        with patch.object(self.g.filedialog, "askdirectory", return_value=str(other)):
            self.app.merge_choose_folder()
        self.assertEqual((self.app.merge_place.get(), self.app.merge_where.get()), ("other", str(other)))
        self.app.merge_run()
        self.assertEqual(Path(self.jobs[0][0][0][-1]), other / "Asian Tit merged.wav")
        self.app.merge_place.set("same")
        self.app._merge_show_place()
        self.app.merge_run()
        self.assertEqual(Path(self.jobs[1][0][0][-1]).parent, self.folder)

    def test_cancelling_the_folder_dialog_keeps_the_clip_folder(self):
        self.add(*self.clips)
        self.app.merge_place.set("other")
        with patch.object(self.g.filedialog, "askdirectory", return_value=""):
            self.app._merge_other()
        self.assertEqual(self.app.merge_place.get(), "same")
        self.assertEqual(self.app.merge_where.get(), str(self.folder))

    def test_different_species_ask_first(self):
        self.add(self.clips[0], RESULT)
        with patch.object(self.g.messagebox, "askokcancel", return_value=False) as ask:
            self.app.merge_run()
        self.assertIn("2 species", ask.call_args.args[1])
        self.assertEqual(self.jobs, [])

    def test_one_clip_is_not_enough(self):
        self.add(self.clips[0])
        with patch.object(self.g.messagebox, "showwarning") as warned:
            self.app.merge_run()
        warned.assert_called_once()
        self.assertEqual(self.jobs, [])


if __name__ == "__main__":
    unittest.main()
