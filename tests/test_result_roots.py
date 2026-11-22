"""ทุกโฟลเดอร์ผลลัพธ์: ผลแยกตามวันใน Sounds/Done กับผลเก่าที่ ~/BirdNET_eBird ต้องเห็นรวมกัน"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import app_settings  # noqa: E402
import field_log  # noqa: E402
import result_roots  # noqa: E402
from audio_io import collect_audio  # noqa: E402


def result(root, day, session, approved=0):
    folder = Path(root) / day / session
    folder.mkdir(parents=True)
    pd.DataFrame([{"File": f"{n}.wav", "Source file": "a.wav", "Review status": "Approved" if n < approved else None}
                  for n in range(2)]).to_excel(folder / "summary.xlsx", index=False)
    return folder / "summary.xlsx"


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class RootsCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name).resolve()
        self.default = self.temp / "BirdNET_eBird"
        self.default.mkdir()
        for patcher in (patch.object(result_roots, "DEFAULT", self.default),
                        patch.object(app_settings, "SETTINGS_FILE", self.temp / "settings.json"),
                        patch.object(field_log, "STORE", self.temp / "field-logs.json")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def day_done(self, day="20261030", month="Oct"):
        done = self.temp / "Birding" / day[:4] / month / day / "Sounds" / "Done"
        done.mkdir(parents=True)
        return done

    def test_done_folder_of_a_day(self):
        sounds = Path("/x/Birding/2026/Oct/20261030/Sounds")
        for path in (sounds, sounds / "Raw", sounds / "Raw/261030_063421/261030_063421_TrMic.WAV",
                     sounds / "Done/2026.10.30"):
            self.assertEqual(result_roots.done_folder(path), sounds / "Done")
        self.assertIsNone(result_roots.done_folder("/x/Birding/2026/Oct"))
        self.assertIsNone(result_roots.done_folder("/x/Sounds/Raw"))          # ไม่ใช่โฟลเดอร์วัน
        self.assertIsNone(result_roots.done_folder(""))

    def test_remembered_keeps_each_folder_once(self):
        one = self.temp / "one"
        app_settings.save_settings({result_roots.SETTING: result_roots.remembered(one)})
        app_settings.save_settings({result_roots.SETTING: result_roots.remembered(one)})
        self.assertEqual(result_roots.remembered(self.temp / "two"), [str(one), str(self.temp / "two")])

    def test_known_roots_finds_every_place(self):
        done = self.day_done()
        other_day = self.day_done("20261025")
        remembered = self.temp / "external"
        remembered.mkdir()
        opened = self.temp / "opened"
        last = result(opened, "2026.09.01", "0700_v3preview")
        current = self.temp / "current"
        current.mkdir()
        app_settings.save_settings({result_roots.SETTING: [str(remembered), str(self.temp / "unplugged")],
                                    "last_summary": str(last)})
        field_log._save({"roots": [str(self.temp / "Birding")]})
        self.assertEqual(result_roots.known_roots(current)[:2], [current, self.default])
        self.assertEqual(set(result_roots.known_roots(current)), {current, self.default, remembered, opened, done, other_day})
        self.assertEqual(len(result_roots.known_roots(self.default)), 5)     # ไม่ซ้ำ

    def test_all_results_joins_folders_newest_first(self):
        field_log._save({"roots": [str(self.temp / "Birding")]})
        old = result(self.default, "2026.10.25", "0909_v3preview", approved=1)
        new = result(self.day_done(), "2026.10.30", "0634_v3pruned", approved=2)
        found = result_roots.all_results(self.default)
        self.assertEqual([info["path"] for info in found], [new, old])
        self.assertEqual([info["approved"] for info in found], [2, 1])

    def test_nothing_inside_done_is_analyzed_again(self):
        sounds = self.temp / "20261030" / "Sounds"
        (sounds / "Raw/261030_063421").mkdir(parents=True)
        recording = sounds / "Raw/261030_063421/261030_063421_TrMic.WAV"
        recording.write_bytes(b"x")
        unfinished = sounds / "Done/2026.10.30/0808_v3pruned/Test bird"     # วิเคราะห์ค้าง: ยังไม่มี summary.xlsx
        unfinished.mkdir(parents=True)
        (unfinished / "clip.wav").write_bytes(b"x")
        self.assertEqual(collect_audio(sounds), [recording])
        self.assertEqual(collect_audio(sounds / "Done"), [unfinished / "clip.wav"])     # เลือก Done เองตรง ๆ ยังเห็น

    def test_output_follows_the_day_of_the_audio(self):
        import birdnet_gui
        app = SimpleNamespace(audio=Var("/x/2026/Oct/20261030/Sounds/Raw/a.WAV"), out=Var("/home/BirdNET_eBird"))
        birdnet_gui.App._follow_audio(app)
        self.assertEqual(app.out.get(), "/x/2026/Oct/20261030/Sounds/Done")
        app.out.set("/my/own/folder")                     # เลือก Output เองหลังเลือกเสียง: ไม่ถูกทับจนกว่าจะเปลี่ยนเสียง
        app.audio.set("/somewhere/else/a.wav")
        birdnet_gui.App._follow_audio(app)
        self.assertEqual(app.out.get(), "/my/own/folder")


if __name__ == "__main__":
    unittest.main()
