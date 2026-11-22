"""จังหวะในตัวเทียบเสียง: ความเร็วโน้ต ความสม่ำเสมอ ระดับเสียงขึ้นลง โน้ตท้ายวลี และการรัวภายในโน้ต"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import frequency_compare as fc  # noqa: E402

SR = 22050


def note(start, dur=0.1, peak=4000.0, score=10.0, pulses=None):
    return {"start": start, "end": start + dur, "dur": dur, "peak": peak, "score": score, "pulses": pulses}


def tone(seconds, hz=4000.0, wobble=None):
    t = np.arange(int(SR * seconds)) / SR
    sound = np.sin(2 * np.pi * hz * t)
    if wobble:
        sound *= 0.5 + 0.5 * np.sin(2 * np.pi * wobble * t)
    return (0.5 * sound).astype(np.float32)


class PhraseTests(unittest.TestCase):
    def test_notes_close_together_form_a_phrase(self):
        notes = [note(t) for t in (0.0, 0.25, 0.5)] + [note(3.0), note(3.3)] + [note(t) for t in (6.0, 6.2, 6.4, 6.6)]
        self.assertEqual([len(p) for p in fc.phrases(notes)], [3, 4])          # 2 โน้ตไม่นับเป็นวลี

    def test_overlapping_notes_keep_the_stronger(self):
        notes = [note(0.0, score=5), note(0.05, score=9), note(0.3), note(0.6)]
        first = fc.phrases(notes)[0][0]
        self.assertEqual(first["score"], 9)


class RhythmTests(unittest.TestCase):
    def test_speed_steadiness_pitch_step_and_last_note(self):
        notes = [note(0.0, peak=4000), note(0.25, peak=4000 * 2 ** (1 / 12)), note(0.5, peak=4000),
                 note(0.75, dur=0.2, peak=4000 * 2 ** (2 / 12))]
        r = fc.rhythm(notes)
        self.assertAlmostEqual(r["rate"], 4.0, places=3)                     # ต้นโน้ตห่างกัน 0.25 วิ
        self.assertAlmostEqual(r["steady"], 0.0, places=6)                   # เท่ากันทุกโน้ต
        self.assertAlmostEqual(r["step"], 1.0, places=2)                     # 1, 1, 2 semitone
        self.assertAlmostEqual(r["final"], 2.0, places=3)                    # โน้ตท้ายยาวเป็น 2 เท่า

    def test_one_long_trilled_note_still_gives_a_trill(self):
        r = fc.rhythm([note(0.0, dur=1.0, pulses=14.0)])
        self.assertIsNone(r["rate"])
        self.assertEqual(r["trill"], {"share": 1.0, "rate": 14.0})

    def test_nothing_to_measure(self):
        self.assertIsNone(fc.rhythm([note(0.0, dur=0.05)]))


class PulseTests(unittest.TestCase):
    def test_trilled_note_gives_its_pulse_rate(self):
        rate = fc.pulse_rate(tone(1.0, wobble=15.0), SR, 0.0, 1.0, 3500.0, 4500.0)
        self.assertAlmostEqual(rate, 15.0, delta=1.0)

    def test_steady_whistle_and_short_note_are_not_trills(self):
        self.assertIsNone(fc.pulse_rate(tone(1.0), SR, 0.0, 1.0, 3500.0, 4500.0))
        self.assertIsNone(fc.pulse_rate(tone(0.1, wobble=15.0), SR, 0.0, 0.1, 3500.0, 4500.0))


class FileTests(unittest.TestCase):
    def test_strongest_sounds_are_the_same_as_before_and_all_notes_are_kept(self):
        rng = np.random.default_rng(1)
        audio = np.zeros(SR * 12, np.float32)
        for n, start in enumerate(np.arange(0.5, 11.0, 0.5)):
            audio[int(start * SR):int(start * SR) + 2205] += tone(0.1, hz=3000 + 40 * n) * rng.uniform(0.2, 1.0)
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"):
            path = Path(temp) / "XC1 - Bird - Avis testus.wav"
            sf.write(path, audio, SR)
            top = fc.measure_samples(audio, SR)                              # วิธีเดิม: 12 เสียงที่เด่นที่สุด
            events = fc.measure_file(path)
            notes = fc.file_notes(path)
        self.assertEqual([(e["start"], e["peak"]) for e in events], [(e["start"], e["peak"]) for e in top])
        self.assertGreater(len(notes), len(events))
        self.assertGreater(fc.rhythm(notes)["rate"], 1.5)


class VerdictTests(unittest.TestCase):
    def species(self, rate, trilled, files=10):
        return {"files": files, "rate": rate, "rate_spread": 0.5, "steady": 0.2, "step": 1.0, "final": 1.2,
                "trill_files": files, "trilled": trilled, "trill_rate": 14.0 if trilled else None}

    def test_trill_points_to_one_species_and_speed_is_only_shown(self):
        yours = {"rate": 2.4, "steady": 0.3, "step": 1.0, "final": 1.4, "trill": {"share": 0.8, "rate": 14.0}}
        text = fc.rhythm_verdict(yours, {"Pale-legged": self.species(2.5, 3), "Sakhalin": self.species(6.0, 0)})
        self.assertNotIn("speed closer", text)                           # ทดสอบกับคลิปจริงแล้วผิดครึ่งหนึ่ง
        self.assertIn("Sakhalin 6.0 notes/s", text)
        self.assertIn("trill fits Pale-legged", text)
        self.assertIn("trill 14/s in 80% of notes", text)

    def test_no_trill_hint_when_both_species_trill_sometimes(self):
        yours = {"rate": None, "steady": None, "step": None, "final": None, "trill": {"share": 1.0, "rate": 14.0}}
        text = fc.rhythm_verdict(yours, {"A": self.species(2.5, 3), "B": self.species(2.4, 2)})
        self.assertNotIn("trill fits", text)

    def test_no_hint_when_both_species_behave_alike(self):
        yours = {"rate": 2.4, "steady": 0.3, "step": 1.0, "final": None, "trill": {"share": 0.0, "rate": None}}
        text = fc.rhythm_verdict(yours, {"A": self.species(2.5, 0), "B": self.species(2.4, 0)})
        self.assertNotIn("→", text)
        self.assertIn("Compare the same sound type", text)

    def test_asks_for_notes_when_yours_cannot_be_measured(self):
        self.assertIn("choose at least 3 notes", fc.rhythm_verdict(None, {"A": self.species(2.5, 0)}))
        self.assertEqual(fc.rhythm_verdict(None, {}), "")

    def test_files_without_a_note_series_still_pool_their_trill(self):
        pooled = fc.pooled_rhythm([{"rate": None, "steady": None, "step": None, "final": None,
                                    "trill": {"share": 1.0, "rate": 13.0}},
                                   {"rate": 2.0, "steady": 0.2, "step": 1.0, "final": 1.1,
                                    "trill": {"share": 0.0, "rate": None}}])
        self.assertEqual((pooled["rate"], pooled["trilled"], pooled["trill_files"], pooled["trill_rate"]),
                         (2.0, 1, 2, 13.0))
        self.assertEqual(fc._species_rhythm_cell(pooled), "2.0/s · trill 1/2")


if __name__ == "__main__":
    unittest.main()
