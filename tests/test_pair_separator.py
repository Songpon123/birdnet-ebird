"""Train on references (pair_separator.py) — โมเดลจำลองแทน BirdNET (เวกเตอร์ = พลังงานในแต่ละย่านความถี่) ไม่ต่อเน็ต"""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import frequency_compare as fc  # noqa: E402
import pair_separator  # noqa: E402

SR = 32000


def calls(path, hz, seed):
    """เสียง "ซี่" 5 ครั้งที่ราว hz Hz ในเสียงรบกวนเบา ๆ"""
    rng = np.random.default_rng(seed)
    t = np.arange(SR * 8) / SR
    x = rng.normal(scale=2e-3, size=len(t))
    for start in (0.8, 2.2, 3.6, 5.0, 6.4):
        mask = (t >= start) & (t < start + 0.25)
        x[mask] += 0.3 * np.sin(2 * np.pi * (hz + rng.normal(scale=60)) * t[mask])
    sf.write(path, x.astype(np.float32), SR)
    return path


class FakeBirdNET:
    """encode_arrays แบบ BirdNET: เวกเตอร์ 16 มิติ = พลังงานในย่าน 2–10 kHz ทีละ 500 Hz"""

    def encode_arrays(self, arrays, **_kwargs):
        rows = []
        for number, (samples, sr) in enumerate(arrays):
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1 / sr)
            rows.append((number, 0.0, [spectrum[(freqs >= low) & (freqs < low + 500)].sum()
                                       for low in range(2000, 10000, 500)]))
        found = np.array(rows, dtype=[("input", "i4"), ("start_time", "f4"), ("embedding", "f4", (16,))])
        return types.SimpleNamespace(to_structured_array=lambda: found)


class PairSeparatorTests(unittest.TestCase):
    def spec(self, temp, per_species=5):
        species = []
        for number, (common, scientific, hz) in enumerate((("Chestnut-flanked White-eye", "Zosterops erythropleurus", 5850),
                                                           ("Swinhoe's White-eye", "Zosterops simplex", 4700))):
            files = []
            for n in range(per_species):
                path = calls(Path(temp) / f"{number}-{n}.wav", hz, seed=10 * number + n)
                events = fc.main_calls(fc.measure_file(path))
                files.append({"name": path.name, "path": str(path),
                              "centers": [(e["start"] + e["end"]) / 2 for e in events]})
            species.append({"common": common, "scientific": scientific, "files": files})
        mine = calls(Path(temp) / "mine.wav", 5790, seed=99)
        return {"species": species, "yours": {"path": str(mine), "centers": [0.9, 2.3, 3.7]}}

    def test_trains_tests_each_left_out_file_and_scores_your_sound(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"), \
                patch.object(pair_separator, "CACHE", Path(temp) / "vectors"), \
                patch.dict(sys.modules, {"birdnet": types.SimpleNamespace(load=lambda *a, **k: FakeBirdNET())}):
            spec = self.spec(temp)
            self.assertEqual([len(s["files"]) for s in spec["species"]], [5, 5])
            logged = []
            result = pair_separator.run(spec, log=logged.append)
            self.assertEqual(result["files"], [5, 5])
            self.assertEqual(result["calls"], [25, 25])
            self.assertEqual((result["test"]["files_right"], result["test"]["files"]), (10, 10))
            self.assertEqual(result["yours"]["windows"], 3)
            self.assertGreater(result["yours"]["mean"][0], 0.65)                   # เอียงไปชนิดแรก
            text = pair_separator.describe(result)
            self.assertIn("10 of 10 right (100%)", text)
            self.assertIn("leans Chestnut-flanked White-eye (3 of 3 windows).", text)
            self.assertNotIn("⚠", text)
            self.assertEqual(len(list((Path(temp) / "vectors").glob("*.npy"))), 10)   # เวกเตอร์ของไฟล์อ้างอิงเก็บไว้
            self.assertTrue(any("Reference sounds 10/10" in line for line in logged))

    def test_refuses_with_too_few_references_and_warns_when_they_do_not_separate(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(fc, "CACHE", Path(temp) / "cache"):
            spec = self.spec(temp, per_species=4)
            result = pair_separator.run(spec)
            self.assertIn("needs at least 5 reference recordings per species", result["error"])
            self.assertTrue(pair_separator.describe(result).startswith("Separator not trained"))
        weak = {"species": ["A", "B"], "files": [5, 5], "calls": [20, 20],
                "test": {"files_right": 6, "files": 10, "calls_right": 24, "calls": 40, "per_species": [[3, 5], [3, 5]]},
                "yours": {"windows": 4, "mean": [0.55, 0.45], "votes": [2, 2]}}
        text = pair_separator.describe(weak)
        self.assertIn("— undecided.", text)
        self.assertIn("⚠ the references do not separate these species well", text)


if __name__ == "__main__":
    unittest.main()
