"""ไฟล์เสียงหนึ่งไฟล์อยู่ checklist เดียว: เดาจากเวลา เลือกเองจาก drop down และคลิกทุกจุดในไฟล์ได้ checklist เดียวกัน"""

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import field_log  # noqa: E402
import field_log_window  # noqa: E402
import manual_clips  # noqa: E402
from test_field_log import write_log  # noqa: E402

DAY = date(2026, 10, 30)
MORNING = {"id": "S2", "date": "2026-10-30T06:36", "location": "Jiaoziya", "species": [], "lat": 29.117096, "lon": 110.479191}
NOON = {"id": "S1", "date": "2026-10-30T11:28", "location": "Chongshan", "species": [], "lat": 29.025113, "lon": 110.512244}


def at(clock):
    return datetime.fromisoformat(f"2026-10-30T{clock}")


def recording(first, last, name="a.wav"):
    return {"start": at(first), "end": at(last), "file": name}


class Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.store = self.temp / "field-logs.json"
        self.store.write_text(json.dumps({"checklists": {"S1": NOON, "S2": MORNING}}), encoding="utf-8")
        patcher = patch.object(field_log, "STORE", self.store)
        patcher.start()
        self.addCleanup(patcher.stop)


class ChoosingTests(Case):
    def test_without_a_choice_the_start_time_decides(self):
        for clock, expected in (("06:23:49", "S2"),            # เปิดเครื่องอัดก่อนเริ่ม checklist
                                ("09:15:11", "S2"), ("12:22:53", "S1"),
                                ("05:00:00", "S2")):           # ก่อนทุก checklist
            self.assertEqual(field_log.for_recording(at(clock))[0]["id"], expected, clock)

    def test_a_choice_covers_every_moment_of_that_recording_only(self):
        field_log.choose(recording("09:15:11", "09:29:35"), "S1")
        self.assertEqual(field_log.for_recording(at("09:15:11"))[0]["id"], "S1")
        self.assertEqual(field_log.for_recording(at("09:29:35"))[0]["id"], "S1")
        self.assertEqual(field_log.for_recording(at("09:20:03"))[0]["id"], "S1")   # หน้า Compare รู้แค่เวลาคลิป
        self.assertEqual(field_log.for_recording(at("09:30:00"))[0]["id"], "S2")   # ไฟล์ถัดไปไม่เกี่ยว

    def test_no_checklist_is_kept_apart_from_not_chosen(self):
        field_log.choose(recording("06:23:49", "06:31:08"), field_log.NONE)
        self.assertEqual(field_log.for_recording(at("06:25:00")), [])
        self.assertIsNone(field_log.picked(at("06:34:21")))

    def test_a_checklist_that_is_gone_falls_back_to_the_time(self):
        field_log.choose(recording("06:34:21", "08:07:32"), "S999")
        self.assertEqual(field_log.for_recording(at("06:40:00"))[0]["id"], "S2")

    def test_adding_another_log_keeps_the_choices(self):
        field_log.choose(recording("06:34:21", "08:07:32"), "S1")
        log = write_log(self.temp / "Birding", DAY, "S3", [("x", "Asian Tit", "Parus cinereus", "1", 0)])
        field_log.add(log)
        self.assertEqual(field_log.picked(at("07:00:00")), "S1")

    def test_review_says_which_checklist_and_how_it_was_chosen(self):
        self.assertEqual(field_log.describe_recording(at("09:15:11")), "eBird checklist S2 30 Oct 06:36 Jiaoziya (by time)")
        field_log.choose(recording("09:15:11", "09:29:35"), "S1")
        self.assertEqual(field_log.describe_recording(at("09:20:00")),
                         "eBird checklist S1 30 Oct 11:28 Chongshan (your choice)")
        field_log.choose(recording("06:34:21", "08:07:32"), field_log.NONE)
        self.assertIn("none for this recording", field_log.describe_recording(at("06:40:00")))
        self.assertEqual(field_log.describe_recording(datetime(2026, 1, 1, 8, 0)), "")
        self.assertEqual(field_log.describe_recording(None), "")

    def test_nothing_for_an_unknown_time(self):
        self.assertEqual(field_log.for_recording(None), [])
        self.assertEqual(field_log.for_recording(datetime(2026, 1, 1, 8, 0)), [])


class CoordinateTests(Case):
    def test_every_file_of_a_checklist_gets_its_coordinates(self):
        spot = field_log.spot_for(at("06:34:21"))
        self.assertEqual((spot["lat"], spot["lon"], spot["checklist"], spot["how"]),
                         (29.117096, 110.479191, "S2", "by time"))
        self.assertEqual(field_log.spot_for(at("12:22:53"))["checklist"], "S1")

    def test_a_recording_far_from_any_checklist_is_not_given_a_place(self):
        self.assertIsNone(field_log.spot_for(at("05:00:00")))                      # ก่อน checklist แรก
        self.assertIsNone(field_log.spot_for(at("18:00:00")))                      # เย็น คนละที่ได้

    def test_choosing_the_checklist_overrides_the_time_limit(self):
        field_log.choose(recording("18:00:00", "18:30:00"), "S1")
        spot = field_log.spot_for(at("18:00:00"))
        self.assertEqual((spot["checklist"], spot["how"]), ("S1", "your choice"))

    def test_no_checklist_means_no_coordinates(self):
        field_log.choose(recording("06:34:21", "08:07:32"), field_log.NONE)
        self.assertIsNone(field_log.spot_for(at("06:34:21")))

    def test_old_log_without_coordinates_gives_none(self):
        self.store.write_text(json.dumps({"checklists": {"S9": {"id": "S9", "date": "2026-10-30T06:36", "species": []}}}),
                              encoding="utf-8")
        self.assertIsNone(field_log.spot_for(at("06:34:21")))


class ListingTests(Case):
    def sounds(self, *names):
        for name in names:
            folder = self.temp / "Birding/2026/Oct/20261030/Sounds/Raw" / name[:13]
            folder.mkdir(parents=True)
            sf.write(folder / name, np.zeros(4800, np.float32), 48000)            # 0.1 วินาที

    def stored(self):
        return patch.object(field_log, "stored", return_value=([NOON], [str(self.temp / "Birding")]))

    def result(self, source, start_offset, end_offset, clock):
        folder = self.temp / "out/2026.10.30/0900_v3pruned"
        folder.mkdir(parents=True)
        pd.DataFrame([{"File": "a.wav", "Source path": source, "Start offset (s)": start_offset,
                       "End offset (s)": end_offset, "Clock time": clock,
                       "Analysis settings": json.dumps({"date": "2026-10-30"})}]).to_excel(folder / "summary.xlsx",
                                                                                          index=False)
        return {"folder": "2026.10.30_0900_v3pruned", "path": folder / "summary.xlsx"}

    def test_recordings_not_yet_analyzed_are_listed(self):
        self.sounds("261030_063421_TrMic.WAV", "261030_080733_TrMic.WAV")
        with self.stored(), patch.object(field_log_window, "all_results", return_value=[]):
            found = field_log_window.recordings_on(DAY)
        self.assertEqual([item["file"] for item in found], ["261030_063421_TrMic.WAV", "261030_080733_TrMic.WAV"])
        self.assertEqual(found[0]["start"], at("06:34:21"))
        self.assertEqual(found[0]["end"], at("06:34:21") + timedelta(seconds=0.1))

    def test_result_of_a_file_elsewhere_is_listed_once(self):
        self.sounds("261030_063421_TrMic.WAV")
        same = self.result(str(self.temp / "gone/261030_063421_TrMic.WAV"), 100.0, 103.0, "06:36:01")
        with self.stored(), patch.object(field_log_window, "all_results", return_value=[same]):
            self.assertEqual(len(field_log_window.recordings_on(DAY)), 1)

    def test_recording_known_only_from_a_result_ends_at_its_last_clip(self):
        moved = self.result("/gone/moved.wav", 100.0, 103.0, "06:36:01")
        with self.stored(), patch.object(field_log_window, "all_results", return_value=[moved]):
            found = field_log_window.recordings_on(DAY)
        self.assertEqual((found[0]["file"], found[0]["start"], found[0]["end"]),
                         ("moved.wav", at("06:34:21"), at("06:36:04")))


class WindowTests(Case):
    def setUp(self):
        super().setUp()
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        reader = patch.object(field_log_window, "recordings_on", return_value=[
            recording("06:34:21", "08:07:32", "first.wav"), recording("12:22:53", "13:06:49", "second.wav")])
        reader.start()
        self.addCleanup(reader.stop)
        self.changed = []
        self.window = field_log_window.RecordingChecklists(self.root, None, DAY, changed=lambda: self.changed.append(1))
        self.window.fill(DAY, field_log_window.recordings_on(DAY))
        self.root.update()

    def boxes(self):
        return [child for child in self.window.rows.winfo_children() if child.winfo_class() == "TCombobox"]

    def test_each_recording_gets_a_drop_down_of_the_days_checklists(self):
        first, second = self.boxes()
        self.assertEqual(first.get(), "S2 30 Oct 06:36 Jiaoziya")
        self.assertEqual(second.get(), "S1 30 Oct 11:28 Chongshan")
        self.assertEqual(list(first.cget("values")), ["S2 30 Oct 06:36 Jiaoziya", "S1 30 Oct 11:28 Chongshan",
                                                      field_log_window.NO_CHECKLIST])
        self.assertEqual([choice[1].get() for choice in self.window._kept], ["by time", "by time"])

    def test_each_row_shows_the_coordinates_it_will_use(self):
        self.assertEqual([kept[2].get() for kept in self.window._kept], ["29.117096, 110.479191", "29.025113, 110.512244"])
        box = self.boxes()[1]
        box.set(field_log_window.NO_CHECKLIST)
        box.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertEqual(self.window._kept[1][2].get(), "no coordinates")

    def test_picking_saves_at_once_and_tells_the_page_behind(self):
        box = self.boxes()[0]
        box.set("S1 30 Oct 11:28 Chongshan")
        box.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertEqual(field_log.for_recording(at("07:30:00"))[0]["id"], "S1")
        self.assertEqual(self.changed, [1])
        self.assertEqual(self.window._kept[0][1].get(), "")                        # เลือกเองแล้ว ไม่ใช่ by time
        self.assertEqual(self.window._kept[1][1].get(), "by time")

    def test_scrolling_over_a_drop_down_does_not_change_it(self):
        box = self.boxes()[0]
        before = box.get()
        self.root.update()
        box.event_generate("<MouseWheel>", delta=-120)
        box.event_generate("<MouseWheel>", delta=120)
        box.event_generate("<MouseWheel>", delta=120)
        self.root.update()
        self.assertEqual(box.get(), before)
        self.assertEqual(self.changed, [])

    def test_days_that_have_a_checklist_can_be_chosen(self):
        self.assertEqual(self.window.days, [DAY])
        self.assertEqual(self.window.day.get(), "30 Oct 2026")

    def test_closing_while_reading_leaves_no_timer(self):
        self.window.read()
        self.assertIsNotNone(self.window._poll_job)
        self.window.win.destroy()
        self.assertIsNone(self.window._poll_job)
        self.root.update()

    def test_no_logs_yet_says_so(self):
        self.store.write_text("{}", encoding="utf-8")
        window = field_log_window.RecordingChecklists(self.root, None)
        self.assertEqual(window.state.get(), "Add a field log first")
        window.win.destroy()


if __name__ == "__main__":
    unittest.main()
