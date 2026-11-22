"""ตัวเทียบเสียงกับเสียงอ้างอิงของหลายชนิด (เช่น Pale-legged กับ Sakhalin Leaf Warbler ที่ต้องแยกด้วยเสียง)
เทียบ 3 อย่างของแต่ละเสียง: ความถี่ที่ดังสุด (Fppeak), ย่านเสียง (ขอบล่าง–บนที่ตัวตรวจเสียงหาได้), ความยาว
แหล่งเสียงอ้างอิง: ไฟล์ของคุณเอง (data/reference-sounds/<Genus species>/; ไฟล์ชื่อแบบ xeno-canto
"XC123 - ชื่อ - Genus species.mp3" เข้าชนิดนั้นเอง) และ xeno-canto (ไฟล์ที่เคยโหลดไว้ใช้ได้ทันที)

วัดแบบเดียวกันทั้งเสียงของคุณและเสียงอ้างอิง: หาเสียงในกรอบ/ไฟล์ด้วย signal_events แล้ววัดแต่ละเสียงในกรอบเวลา+ความถี่ของ
เสียงนั้น (prism_view.analysis_stats) — ในแต่ละไฟล์ใช้ "เสียงหลัก" = กลุ่มเสียงที่ความถี่ใกล้กันมากที่สุดซึ่งร้องซ้ำ
(ตัดเสียงนกตัวอื่นที่ติดมาในไฟล์อ้างอิงหรือในกรอบของคุณ)
ความใกล้ = ระยะของความถี่ที่ดังสุดจาก median ของชนิดนั้น หารด้วยความกระจาย (robust z: 1.4826×MAD, อย่างน้อย 150 Hz)
ทดลอง 2026-10-25 (เสียง call): Pale-legged ในเครื่อง 6 ไฟล์ + XC1147749, Sakhalin 6 ไฟล์ + XC1075519 — Pale-legged ส่วนใหญ่
5.8–5.9 kHz (มองโกเลียใน มิ.ย. 6.9 kHz), Sakhalin 4.5–5.2 kHz; เสียงในคลิป REC-001 18 ต.ค. 5.77 kHz
ความยาวเสียงขึ้นกับระยะ (เสียงใกล้มีหางสะท้อนยาว ~0.35 วินาที เสียงไกลเหลือ ~0.05 วินาที) จึงใช้ประกอบเท่านั้น
ขั้นต่ำ MIN_FILES ไฟล์ต่อชนิดจาก MIN_RECORDISTS ผู้บันทึก: สุ่มเลือกไฟล์อ้างอิงทีละ n ไฟล์ (คลิป Pale-legged 18 ต.ค.)
ผลสรุปตรงกับตอนใช้ทุกไฟล์ 72% (1 ไฟล์), 75% (2), 90% (3), 100% (4 ขึ้นไป) — ต่ำกว่าขั้นต่ำ: ผลสรุปเป็น "Provisional"
และ Find references โหลดเพิ่มให้ครบเอง; ชนิดที่ "call" บน xeno-canto มีหลายแบบ (Swinhoe's White-eye: 2.6–6.8 kHz ต่อไฟล์)
ความถี่อย่างเดียวแยกไม่ได้ไม่ว่ากี่ไฟล์ -> Compare spectrograms + Train on references (pair_separator.py)
ความใสของเสียง (tone_purity): HKBWS บอกว่า call "tseeoo" ของ Chestnut-flanked ใสกว่า ไม่แหบ/สั่นแบบ Swinhoe's — วัดกับไฟล์ที่โหลดไว้
2026-10-25: Chestnut-flanked 8 ไฟล์ ค่ากลาง 0.67 (0.49–0.83), Swinhoe's 11 ไฟล์ 0.41 (0.30–0.67, 9 ไฟล์ ≤ 0.45)
เลือกเสียงของคุณในหน้าต่าง: กรอบเสียงของ BirdNET (แบบในภาพ spectrogram ของคลิป) หรือเสียงที่ตัวตรวจหาเจอ — คลิกกรอบ = ใช้/ไม่ใช้,
Auto = กรอบของเสียงหลักที่ร้องซ้ำ, ลากเอง = กรอบเวลา+ความถี่ของคุณ (วัดทุกเสียงในกรอบแล้วใช้เสียงหลัก)
จังหวะ (rhythm, ตั้งแต่ 2026-11-04): โน้ตห่างกันไม่เกิน PHRASE_GAP = วลี -> โน้ตต่อวินาที ความสม่ำเสมอ ระดับเสียงขึ้นลงระหว่างโน้ต
โน้ตท้ายวลี และการรัวภายในโน้ต (pulse_rate) — ทดลองกับไฟล์จริง: ความเร็วโน้ตของ call/song ทั้งสองคู่แทบเท่ากัน (~2.5/s)
แต่ song ของ Pale-legged รัว 3/10 ไฟล์ (~12/s) ส่วน Sakhalin 0/8; call ไม่รัว
xeno-canto วัดจริง: ค้นหา 3–7 วินาที, mp3 สั้น ~4 วินาทีต่อไฟล์ แต่ WAV 1–5 MB ใช้ 30–120 วินาที จึงโหลด mp3/สั้นก่อน
"""

import hashlib
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

import paths
import playback

LIBRARY = paths.SETTINGS_FILE.parent / "reference-sounds"   # ไฟล์อ้างอิงที่คุณเพิ่มเอง (data/reference-sounds)
CACHE = paths.CACHE / "frequency-refs"
VERSION = 4                    # 2 = ขอบเสียง/ความยาว + ตัวตรวจไวขึ้นสำหรับเสียงสั้นแบบ "ติ๊ก", 3 = ความใสของเสียง
SEARCH_DAYS = 7
NOTES_PER_FILE = 300           # โน้ตทั้งหมดที่ตัวตรวจเจอ (เก็บไว้วัดจังหวะ) — 4 = เริ่มเก็บ
PHRASE_GAP = 0.6               # โน้ตห่างกันไม่เกินเท่านี้วินาที = วลีเดียวกัน
PHRASE_NOTES = 3               # วลีที่มีโน้ตน้อยกว่านี้ไม่ใช้วัดจังหวะ
NOTE_BAND = 1.25               # โน้ตของไฟล์อ้างอิงที่นับ: ความถี่ที่ดังสุดอยู่ในช่วงเสียงหลัก ×/÷ เท่านี้
RATE_SPREAD = 0.5              # ความกระจายต่ำสุดของความเร็ว (โน้ตต่อวินาที)
PULSE_RANGE = (6.0, 80.0)      # การรัวภายในโน้ต (ครั้งต่อวินาที) ที่หา
TRILL_SECONDS = 0.15           # โน้ตสั้นกว่านี้ดูการรัวไม่ได้
PULSE_DEPTH = 0.15             # ความดังต้องขึ้นลงอย่างน้อยเท่านี้ของความดังเฉลี่ย จึงนับว่ารัว
TRILLED = 0.5                  # โน้ตรัวตั้งแต่ครึ่งหนึ่ง = ไฟล์/เสียงนั้นรัว
TRILL_SPECIES = 0.25           # ชนิดที่ไฟล์รัวตั้งแต่เท่านี้ = รัวได้ (ไฟล์อ้างอิงมักมีเสียงอื่นปน), ไม่เกิน NO_TRILL = ไม่รัว
NO_TRILL = 0.1
EVENTS_PER_FILE = 12           # วัดเสียงที่เด่นที่สุดเท่านี้ต่อไฟล์
MAX_SECONDS = 90.0             # อ่านแค่ช่วงต้นของไฟล์ยาว (เสียงอ้างอิงส่วนใหญ่สั้นกว่านี้)
BAND_MARGIN = 200.0            # วัดแต่ละเสียงในขอบล่าง–บนที่ตัวตรวจหาได้ ± เท่านี้ Hz
CLUSTER_WIDTH = 0.08           # เสียงหลัก: ความถี่ที่ดังสุดห่างกันไม่เกิน ±8%
MIN_SPREAD = 150.0             # ความกระจายต่ำสุดที่ใช้หาร (ความละเอียด FFT 256 ≈ 190 Hz)
MARGIN = 0.05                  # inside/near: เผื่อ ±5% ของความถี่
PURITY_FFT = 1024              # ความใส: FFT ละเอียดกว่า (~43 Hz ต่อช่องที่ 44.1 kHz) ให้เห็นเสียงแหบข้างเส้นหลัก
PURITY_SPREAD = 0.05           # ความกระจายต่ำสุดของความใสที่ใช้หาร
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff"}
SOUND_TYPES = ("call", "song", "any")
MIN_FILES = 5                  # ไฟล์อ้างอิงที่มีเสียงชัดต่อชนิดอย่างน้อยเท่านี้ ผลสรุปถึงจะไม่ใช่ Provisional
MIN_RECORDISTS = 3             # จากผู้บันทึกอย่างน้อยเท่านี้คน (ไฟล์ของคนเดียวกันอาจเป็นนกตัวเดียวกัน)
MAX_TOP_UP = 8                 # Find references โหลดเพิ่มให้ครบขั้นต่ำได้ไม่เกินเท่านี้ไฟล์ต่อชนิด
YOUR_SOURCES = ("BirdNET boxes", "Sound detector")
BOX_CONTEXT = 1.0              # หาเสียงในกรอบของ BirdNET โดยดูพื้นหลังรอบกรอบ ± เท่านี้วินาที
XC_NAME = re.compile(r"^(XC\d+)\s*-\s*(.+?)\s*-\s*([A-Z][a-z]+ [a-z-]+)")


# ----------------------------- วัด (ไม่แตะ Tk) -----------------------------
def _detect_params(sr, low=250.0, high=12000.0):
    import signal_events
    return signal_events.DetectionParams(low=max(0.0, low), high=min(high, sr / 2 - 1), minimum=0.02, maximum=3.0,
                                         gap=0.05, excess_db=10.0)


def tone_purity(samples, sr, start, end, low, high):
    """ความใสของเสียงหนึ่ง (0–1): ในเฟรมครึ่งที่ดังกว่า สัดส่วนพลังงานในย่าน low–high (± BAND_MARGIN) ที่อยู่ใน ±1 ช่อง
    ของเส้นที่ดังที่สุด แล้วเอาค่ากลาง — นกหวีดใสเส้นเดียวสูง, เสียงแหบ/สั่นต่ำ (เสียงไกล/มีเสียงรบกวนก็ต่ำลง)"""
    part = np.asarray(samples, np.float32)[max(0, int(start * sr)):int(end * sr)]
    if len(part) < PURITY_FFT * 2:
        part = np.pad(part, (0, PURITY_FFT * 2 - len(part)))
    frames = np.lib.stride_tricks.sliding_window_view(part, PURITY_FFT)[::PURITY_FFT // 4]
    power = np.abs(np.fft.rfft(frames * np.hanning(PURITY_FFT).astype(np.float32), axis=1)) ** 2
    freqs = np.fft.rfftfreq(PURITY_FFT, 1 / sr)
    power = power[:, (freqs >= max(0.0, low - BAND_MARGIN)) & (freqs <= min(sr / 2, high + BAND_MARGIN))]
    if power.shape[1] < 3:
        return None
    total = power.sum(axis=1)
    loud = total >= np.median(total)
    power, total = power[loud], total[loud]
    rows, peak = np.arange(len(power)), power.argmax(axis=1)
    padded = np.pad(power, ((0, 0), (1, 1)))
    near = padded[rows, peak] + padded[rows, peak + 1] + padded[rows, peak + 2]
    return float(np.median(near / np.maximum(total, 1e-20)))


def pulse_rate(samples, sr, start, end, low, high):
    """การรัวภายในโน้ตหนึ่ง (ครั้งต่อวินาที) จากความดังที่ขึ้นลงเป็นจังหวะในย่าน low–high -> float หรือ None (ไม่รัว)
    ตัวตรวจโน้ตรวมเสียงรัวเร็ว (ช่องว่าง < 50 ms) เป็นโน้ตเดียว จึงต้องดูในโน้ตเอง — ทดลอง 2026-11-04: song ของ
    Pale-legged Leaf Warbler รัว 6/10 ไฟล์ (~14 ครั้ง/วินาที) Sakhalin 1/8 ไฟล์, call ทั้งสองชนิดแทบไม่รัว"""
    from scipy.signal import butter, sosfiltfilt
    part = np.asarray(samples, np.float64)[max(0, int(start * sr)):int(end * sr)]
    if len(part) < TRILL_SECONDS * sr or high <= low:
        return None
    sos = butter(4, [max(100.0, low * 0.9), min(sr / 2 - 100.0, high * 1.1)], btype="band", fs=sr, output="sos")
    step = max(1, int(sr // 1000))                     # ความดังทุก 1 ms
    envelope = np.abs(sosfiltfilt(sos, part))
    envelope = envelope[:len(envelope) // step * step].reshape(-1, step).mean(axis=1)
    level = envelope.mean()
    envelope -= level
    if level <= 0 or envelope.std() < PULSE_DEPTH * level:   # ความดังแทบไม่ขึ้นลง = เสียงหวีดเรียบ ไม่ใช่เสียงรัว
        return None
    ac = np.correlate(envelope, envelope, "full")[len(envelope) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac)) / 1000.0
    usable = np.flatnonzero((lags >= 1 / PULSE_RANGE[1]) & (lags <= 1 / PULSE_RANGE[0]) & (lags < (end - start) / 3))
    peaks = [i for i in usable[1:-1] if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] - ac[:i].min() > 0.2]
    best = max(peaks, key=lambda i: ac[i], default=None)
    return None if best is None or ac[best] <= 0.3 else float(1 / lags[best])


def measure_samples(samples, sr, offset=0.0, band=None, top=EVENTS_PER_FILE):
    """เสียงที่เด่นที่สุดในเสียงช่วงนี้ (band = (low, high) Hz จำกัดการหา) ->
    [{"start", "end", "dur", "low", "high", "peak", "fmin", "fmax", "fmean", "score", "purity"}]
    (วินาที, Hz; low/high = ขอบเสียง, purity = tone_purity)"""
    import prism_view as kv
    import signal_events
    samples = np.asarray(samples, np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    low, high = band or (250.0, 12000.0)
    if len(samples) < kv.FFT_SIZE * 2:
        return []
    events = signal_events.detect_samples(samples, sr, _detect_params(sr, low, high))
    events = sorted(events, key=lambda e: -e["score"])[:top]
    for rank, event in enumerate(events):     # อันดับความเด่น: เลือก EVENTS_PER_FILE เสียงจากโน้ตทั้งหมดได้ตรงเดิม
        event["rank"] = rank
    db = kv.spectrum_db(samples) if events else None
    found = []
    for event in sorted(events, key=lambda e: e["start"]):
        first = max(0, int(event["start"] * sr / kv.FFT_HOP))
        last = min(len(db), max(first + 2, int(event["end"] * sr / kv.FFT_HOP)))
        stats = kv.analysis_stats(db[first:last], sr, max(0.0, event["low"] - BAND_MARGIN),
                                  min(sr / 2, event["high"] + BAND_MARGIN))
        purity = tone_purity(samples, sr, event["start"], event["end"], event["low"], event["high"])
        pulses = pulse_rate(samples, sr, event["start"], event["end"], event["low"], event["high"])
        found.append({"start": round(offset + event["start"], 3), "end": round(offset + event["end"], 3),
                      "dur": round(event["end"] - event["start"], 3), "low": round(event["low"]),
                      "high": round(event["high"]), "peak": round(stats["Fppeak"]), "fmin": round(stats["Fpmin"]),
                      "fmax": round(stats["Fpmax"]), "fmean": round(stats["Fpmean"]), "score": round(event["score"], 1),
                      "purity": None if purity is None else round(purity, 3), "rank": event["rank"],
                      "pulses": None if pulses is None else round(pulses, 1)})
    return found


def _cache_file(path):
    stat = Path(path).stat()
    key = hashlib.sha256(f"{VERSION}|{Path(path).name}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:20]
    return CACHE / f"{key}.json"


def measure_file(path, top=EVENTS_PER_FILE):
    """measure_samples ของไฟล์อ้างอิงทั้งไฟล์ (MAX_SECONDS แรก) -> top เสียงที่เด่นที่สุด
    เก็บ cache วัดครั้งเดียว พร้อมโน้ตทั้งหมดไว้วัดจังหวะ (file_notes)"""
    from audio_io import audio_duration, read_mono
    target = _cache_file(path)
    try:
        notes = json.loads(target.read_text(encoding="utf-8"))["notes"]
    except (OSError, ValueError, KeyError):
        duration = audio_duration(path) or MAX_SECONDS
        samples, sr = read_mono(path, 0.0, min(duration, MAX_SECONDS))
        notes = measure_samples(samples, sr, top=NOTES_PER_FILE)
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"file": str(path), "notes": notes}), encoding="utf-8")
        temporary.replace(target)
    return [note for note in notes if note["rank"] < top]


def file_notes(path):
    """โน้ตทั้งหมดของไฟล์อ้างอิงที่วัดไว้แล้ว (ไม่วัดใหม่) -> list หรือ None"""
    try:
        return json.loads(_cache_file(path).read_text(encoding="utf-8"))["notes"]
    except (OSError, ValueError, KeyError):
        return None


def phrases(notes, gap=PHRASE_GAP):
    """โน้ตเรียงตามเวลา -> วลี (โน้ตห่างกันไม่เกิน gap วินาที) ที่มีอย่างน้อย PHRASE_NOTES โน้ต
    โน้ตที่เวลาทับกัน (ตัวตรวจเจอซ้ำ หรือนกสองตัวร้องพร้อมกัน) เก็บเฉพาะเสียงที่เด่นกว่า"""
    ordered = []
    for note in sorted((n for n in notes if n.get("peak", 0) > 0), key=lambda n: n["start"]):
        if ordered and note["start"] < ordered[-1]["end"]:
            if note.get("score", 0) > ordered[-1].get("score", 0):
                ordered[-1] = note
            continue
        ordered.append(note)
    groups = []
    for note in ordered:
        if groups and note["start"] - groups[-1][-1]["end"] <= gap:
            groups[-1].append(note)
        else:
            groups.append([note])
    return [group for group in groups if len(group) >= PHRASE_NOTES]


def trill(notes):
    """โน้ตที่รัว -> {"share" สัดส่วนโน้ตที่รัว, "rate" ครั้งต่อวินาที} หรือ None (ไม่มีโน้ตยาวพอให้ดู)"""
    long = [note for note in notes if note.get("dur", 0) >= TRILL_SECONDS and note.get("peak", 0) > 0]
    if not long:
        return None
    rates = [note["pulses"] for note in long if note.get("pulses")]
    return {"share": len(rates) / len(long), "rate": float(np.median(rates)) if rates else None}


def rhythm(notes):
    """จังหวะของโน้ต -> {"phrases", "notes", "rate", "steady", "step", "final", "trill"} หรือ None (วัดอะไรไม่ได้)
    rate = โน้ตต่อวินาที, steady = ช่วงห่างระหว่างต้นโน้ตแปรปรวนกี่ส่วน (0 = เท่ากันทุกโน้ต),
    step = ระดับเสียงเปลี่ยนระหว่างโน้ตติดกัน (semitone), final = โน้ตท้ายวลียาวเป็นกี่เท่าของโน้ตอื่น (ค่ากลางของทุกวลี)
    ค่าของวลี = None เมื่อไม่มีวลี ≥ PHRASE_NOTES โน้ต (เช่นเสียงรัวยาวเสียงเดียว), trill ดูได้แม้มีโน้ตเดียว"""
    rates, steady, steps, finals, used = [], [], [], [], []
    for phrase in phrases(notes):
        onsets = np.diff([note["start"] for note in phrase])
        if onsets.min() <= 0:
            continue
        used.append(phrase)
        rates.append(1 / float(np.median(onsets)))
        steady.append(float(onsets.std() / onsets.mean()))
        steps += [abs(12 * math.log2(b["peak"] / a["peak"])) for a, b in zip(phrase, phrase[1:])]
        others = float(np.median([note["dur"] for note in phrase[:-1]]))
        if others > 0:
            finals.append(phrase[-1]["dur"] / others)
    trilled = trill(notes)
    if not used and trilled is None:
        return None
    middle = lambda values: float(np.median(values)) if values else None   # noqa: E731
    return {"phrases": len(used), "notes": sum(map(len, used)), "rate": middle(rates), "steady": middle(steady),
            "step": middle(steps), "final": middle(finals), "trill": trilled}


def file_rhythm(notes, summary):
    """จังหวะของไฟล์อ้างอิง: เฉพาะโน้ตในช่วงความถี่ของเสียงหลัก (ตัดนกตัวอื่นที่ติดมาในไฟล์)"""
    if not notes or not summary:
        return None
    low, high = summary["low"] / NOTE_BAND, summary["high"] * NOTE_BAND
    return rhythm([note for note in notes if low <= note["peak"] <= high])


def pooled_rhythm(rhythms):
    """จังหวะของหลายไฟล์ของชนิดเดียวกัน -> ค่ากลางของแต่ละค่า, ความกระจายของความเร็ว, ไฟล์ที่รัว หรือ None"""
    found = [r for r in rhythms if r]
    if not found:
        return None
    middle = lambda name: (float(np.median([r[name] for r in found if r[name] is not None]))   # noqa: E731
                           if any(r[name] is not None for r in found) else None)
    rates = np.array([r["rate"] for r in found if r["rate"] is not None])
    trills = [r["trill"] for r in found if r["trill"]]
    trilled = [t for t in trills if t["share"] >= TRILLED]
    return {"files": len(found), "rate": middle("rate"), "steady": middle("steady"), "step": middle("step"),
            "final": middle("final"),
            "rate_spread": max(RATE_SPREAD, 1.4826 * float(np.median(np.abs(rates - np.median(rates))))) if len(rates)
            else RATE_SPREAD,
            "trill_files": len(trills), "trilled": len(trilled),
            "trill_rate": float(np.median([t["rate"] for t in trilled])) if trilled else None}


def describe_rhythm(r):
    parts = []
    if r.get("rate") is not None:
        parts.append(f"{r['rate']:.1f} notes/s, timing varies {r['steady']:.0%}, pitch step {r['step']:.1f} semitones")
        if r.get("final") is not None:
            parts.append(f"last note {r['final']:.1f}× longer")
    if r.get("trill"):
        share, rate = r["trill"]["share"], r["trill"]["rate"]
        parts.append(f"trill {rate:.0f}/s in {share:.0%} of notes" if rate else "no trill")
    return ", ".join(parts)


def _describe_species_rhythm(r):
    text = f"{r['rate']:.1f} notes/s" if r.get("rate") is not None else "no note series"
    if r["trill_files"]:
        text += f", trill in {r['trilled']}/{r['trill_files']} files"
        if r["trill_rate"]:
            text += f" ({r['trill_rate']:.0f}/s)"
    return text


def rhythm_verdict(yours, pooled):
    """จังหวะของเสียงคุณเทียบแต่ละชนิด -> ข้อความ ('' = ไม่มีอะไรให้เทียบ)
    ความเร็วแสดงเป็นตัวเลขเท่านั้น: ทดสอบกับคลิปที่ตรวจแล้ว 18 คลิป (2026-11-04) สรุปจากความเร็ว 2 ครั้ง ผิด 1
    การรัว: เสียงคุณรัว และมีชนิดเดียวที่ไฟล์รัว ≥ TRILL_SPECIES ส่วนชนิดอื่น ≤ NO_TRILL (ไม่รัวไม่บอก: เพลงหลายชนิดมีทั้งสองแบบ)"""
    found = {name: r for name, r in pooled.items() if r}
    if not yours and not found:
        return ""
    if not yours:
        return f"Rhythm: choose at least {PHRASE_NOTES} notes in a row, or one long note, of your sound to compare it"
    text = f"Rhythm: yours {describe_rhythm(yours)}"
    if not found:
        return text
    text += " — " + " · ".join(f"{name} {_describe_species_rhythm(r)}" for name, r in found.items())
    hints = []
    share = {name: r["trilled"] / r["trill_files"] for name, r in found.items() if r["trill_files"]}
    trilling = [name for name, value in share.items() if value >= TRILL_SPECIES]
    if (yours.get("trill") and yours["trill"]["share"] >= TRILLED and len(share) > 1 and len(trilling) == 1
            and all(value <= NO_TRILL for name, value in share.items() if name not in trilling)):
        hints.append(f"trill fits {trilling[0]}")
    if hints:
        text += " → " + ", ".join(hints)
    return text + ". Compare the same sound type: songs and calls have different rhythms."


def main_calls(events, width=CLUSTER_WIDTH):
    """เสียงหลักที่ร้องซ้ำ: กลุ่มเสียงที่ความถี่ที่ดังสุดใกล้กันมากที่สุด (±width) ถ้ามีอย่างน้อย 3 เสียงหรือ 30% ของทั้งหมด
    ไม่งั้นคืนทุกเสียง (ไฟล์มีเสียงเดียว/หลายแบบปนกัน)"""
    usable = [e for e in events if e["peak"] > 0]
    if len(usable) < 3:
        return usable
    peaks = np.array([e["peak"] for e in usable], float)
    scores = np.array([e.get("score", 0.0) for e in usable], float)
    near = np.abs(peaks[:, None] - peaks[None, :]) <= width * peaks[:, None]
    count = near.sum(axis=1)
    best = int(np.lexsort((-(near * scores[None, :]).sum(axis=1), -count))[0])
    cluster = [e for e, keep in zip(usable, near[best]) if keep]
    return cluster if len(cluster) >= max(3, 0.3 * len(usable)) else usable


def summarize(events):
    """เสียงหลายเสียง -> ความถี่ที่ดังสุดแบบ box plot (median, ครึ่งกลาง q1–q3, ต่ำสุด–สูงสุด) + ขอบเสียงล่าง/บน และความยาว
    (median) — None = ไม่มีเสียงชัด"""
    usable = [e for e in events if e["peak"] > 0]
    if not usable:
        return None
    peaks = np.array([e["peak"] for e in usable], float)
    purity = np.array([e["purity"] for e in usable if e.get("purity") is not None], float)
    return {"calls": len(usable), "peak": float(np.median(peaks)), "q1": float(np.percentile(peaks, 25)),
            "q3": float(np.percentile(peaks, 75)), "low": float(peaks.min()), "high": float(peaks.max()),
            "spread": max(MIN_SPREAD, 1.4826 * float(np.median(np.abs(peaks - np.median(peaks))))),
            "lo_edge": float(np.median([e.get("low", e["peak"]) for e in usable])),
            "hi_edge": float(np.median([e.get("high", e["peak"]) for e in usable])),
            "dur": float(np.median([e.get("dur", e.get("end", 0) - e.get("start", 0)) for e in usable])),
            "fmean": float(np.median([e["fmean"] for e in usable])),
            "purity": float(np.median(purity)) if len(purity) else None,
            "purity_spread": max(PURITY_SPREAD, 1.4826 * float(np.median(np.abs(purity - np.median(purity)))))
            if len(purity) else PURITY_SPREAD}


def your_peak(yours):
    """ความถี่ที่ดังสุดของเสียงคุณ: summary (median ของทุกเสียงในกรอบ) หรือค่าเดียวจาก Prism Analysis"""
    if not yours:
        return 0.0
    return float(yours.get("peak") or yours.get("Fppeak") or 0.0)


def matches(yours, summary):
    """ความถี่ที่ดังสุดของคุณเทียบกับเสียงอ้างอิง -> "inside" (ครึ่งกลาง ±5%), "near" (ต่ำสุด–สูงสุด ±5%), "outside", None"""
    peak = your_peak(yours)
    if not summary or peak <= 0:
        return None
    if summary["q1"] * (1 - MARGIN) <= peak <= summary["q3"] * (1 + MARGIN):
        return "inside"
    if summary["low"] * (1 - MARGIN) <= peak <= summary["high"] * (1 + MARGIN):
        return "near"
    return "outside"


def distance(yours, summary):
    """ความไกลของความถี่ที่ดังสุดจาก median ของชนิดนั้น หน่วยความกระจาย (robust z) — None = ไม่มีข้อมูล"""
    peak = your_peak(yours)
    if not summary or peak <= 0:
        return None
    return abs(peak - summary["peak"]) / summary["spread"]


def band_overlap(yours, summary):
    """ย่านเสียงของคุณ (ขอบล่าง–บน) ซ้อนกับของชนิดนั้นกี่ส่วน (0–1) — None = ไม่มีข้อมูล"""
    if not summary or not yours or not yours.get("hi_edge"):
        return None
    low, high = max(yours["lo_edge"], summary["lo_edge"]), min(yours["hi_edge"], summary["hi_edge"])
    width = max(yours["hi_edge"] - yours["lo_edge"], 1.0)
    return max(0.0, high - low) / width


def verdict(yours, pooled):
    """pooled = {ชื่อชนิด: summary ของเสียงหลักทุกไฟล์} -> (ข้อความ, ชื่อชนิดที่ใกล้สุด หรือ None)"""
    scored = sorted(((distance(yours, s), name) for name, s in pooled.items() if s and distance(yours, s) is not None))
    if not scored:
        return "", None
    best_z, best = scored[0]
    if best_z > 2.5:
        return f"Your peak is far from every species compared (closest {best}, {best_z:.1f}× its spread)", None
    if len(scored) == 1:
        return f"Your peak is {best_z:.1f}× the spread from {best}'s median", best if best_z <= 1.5 else None
    second_z, second = scored[1]
    if best_z <= 1.5 and second_z - best_z >= 1.0:
        return f"Closer to {best} ({best_z:.1f}× spread) than {second} ({second_z:.1f}×)", best
    return f"Inconclusive: {best} {best_z:.1f}× vs {second} {second_z:.1f}× its spread", None


def purity_verdict(yours, pooled):
    """ความใสของเสียงคุณเทียบแต่ละชนิด -> ข้อความ ('' ถ้าไม่มีค่า) — ใกล้ชนิดหนึ่ง ≤ 1.5 เท่าของความกระจาย และห่างชนิดอื่น
    มากกว่าอย่างน้อย 1 เท่า = "closer to" (เกณฑ์เดียวกับความถี่)"""
    if not yours or yours.get("purity") is None:
        return ""
    found = {name: s for name, s in pooled.items() if s and s.get("purity") is not None}
    if not found:
        return ""
    away = {name: abs(yours["purity"] - s["purity"]) / s["purity_spread"] for name, s in found.items()}
    order = sorted(away, key=away.get)
    text = (f"Tone purity (high = clear whistle, low = buzzy): yours {yours['purity']:.2f} — "
            + " · ".join(f"{name} {found[name]['purity']:.2f}" for name in order))
    if len(order) > 1 and away[order[0]] <= 1.5 and away[order[1]] - away[order[0]] >= 1.0:
        text += f" → closer to {order[0]}"
    return text + " (distant or noisy calls look buzzier than they are)."


def _rhythm_cell(r):
    """ช่อง Rhythm ของไฟล์อ้างอิงหนึ่งไฟล์: '2.4/s · trill 14/s'"""
    if not r:
        return ""
    parts = [f"{r['rate']:.1f}/s"] if r.get("rate") is not None else []
    if r.get("trill") and r["trill"]["share"] >= TRILLED and r["trill"]["rate"]:
        parts.append(f"trill {r['trill']['rate']:.0f}/s")
    return " · ".join(parts)


def _species_rhythm_cell(r):
    """ช่อง Rhythm ของชนิดหนึ่ง: '2.4/s · trill 6/10'"""
    if not r:
        return ""
    parts = [f"{r['rate']:.1f}/s"] if r.get("rate") is not None else []
    if r["trill_files"]:
        parts.append(f"trill {r['trilled']}/{r['trill_files']}")
    return " · ".join(parts)


def overall(rows):
    """เสียงหลักของทุกไฟล์ที่วัดแล้วรวมกัน แบบเดียวกับ summarize"""
    return summarize([e for row in rows for e in main_calls(row.get("events") or [])])


def parse_xc_name(path):
    """'XC1075519 - Sakhalin Leaf Warbler - Phylloscopus borealoides.mp3' -> ("XC1075519", common, scientific) หรือ None"""
    match = XC_NAME.match(Path(path).stem)
    return (match.group(1), match.group(2), match.group(3)) if match else None


# ----------------------------- แหล่งเสียงอ้างอิง -----------------------------
def species_folder(scientific):
    return LIBRARY / re.sub(r"[^A-Za-z0-9 _-]", "", " ".join(str(scientific).split()[:2])).strip()


def library_files(scientific):
    folder = species_folder(scientific)
    if not scientific or not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def add_to_library(scientific, files):
    """คัดลอกไฟล์ที่เลือกเข้า data/reference-sounds/<Genus species>/ (ชื่อซ้ำ: เติมเลขต่อท้าย) -> [Path]"""
    folder = species_folder(scientific)
    folder.mkdir(parents=True, exist_ok=True)
    added = []
    for source in map(Path, files):
        if source.suffix.lower() not in AUDIO_EXTS:
            continue
        target, number = folder / source.name, 2
        while target.exists():
            target = folder / f"{source.stem}-{number}{source.suffix}"
            number += 1
        shutil.copy2(source, target)
        added.append(target)
    return added


def _search_file(scientific, sound_type):
    slug = re.sub(r"[^a-z0-9]+", "-", f"{scientific} {sound_type}".lower()).strip("-")
    return CACHE / f"search-{slug}.json"


def search(scientific, common, key, sound_type="any", lat=None, lon=None, refresh=False):
    """รายการเสียงของชนิดนี้บน xeno-canto (เก็บผลไว้ SEARCH_DAYS วัน) -> list ของ dict จาก xeno_canto.simplify"""
    import xeno_canto
    target = _search_file(scientific, sound_type)
    if not refresh:
        try:
            saved = json.loads(target.read_text(encoding="utf-8"))
            if time.time() - saved["made"] < SEARCH_DAYS * 86400:
                return saved["recordings"]
        except (OSError, ValueError, KeyError):
            pass
    recordings, _total = xeno_canto.search(scientific, key, "" if sound_type == "any" else sound_type,
                                           lat=lat, lon=lon, limit=40, common=common)
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"made": time.time(), "recordings": recordings}), encoding="utf-8")
    temporary.replace(target)
    return recordings


def cached_path(recording):
    import xeno_canto
    path = xeno_canto.CACHE / f"{recording['id']}{recording.get('ext') or '.mp3'}"
    return path if path.is_file() and path.stat().st_size > 0 else None


def length_seconds(text):
    try:
        parts = [float(p) for p in str(text).split(":")]
    except ValueError:
        return 1e9
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def download_order(recordings):
    """ไฟล์ที่ยังไม่มี เรียงให้โหลดเร็วก่อน: คุณภาพ A/B, mp3 (เล็กกว่า WAV มาก), สั้นก่อน"""
    import xeno_canto
    todo = [r for r in recordings if cached_path(r) is None and r.get("file")]
    return sorted(todo, key=lambda r: (xeno_canto.QUALITY_RANK.get(r.get("quality"), 9) > 1, r.get("ext") != ".mp3",
                                       length_seconds(r.get("length"))))


# ----------------------------- หน้าต่าง -----------------------------
SPECIES_COLORS = ("#4fc3f7", "#ffb74d", "#ba68c8", "#aed581")
TEXT_COLORS = {   # สีตัวหนังสือที่ตั้งเอง: พื้นสว่าง / พื้นเข้ม (macOS Dark Mode)
    False: {"verdict": "#2e7d32", "log": "#00695c", "purity": "#6a1b9a", "separation": "#1565c0", "state": "#1f6fb2",
            "hint": "#757575", "match": "#2e7d32", "near": "#a26316", "nomatch": "#c62828", "waiting": "#808080"},
    True: {"verdict": "#81c784", "log": "#4db6ac", "purity": "#ce93d8", "separation": "#64b5f6", "state": "#64b5f6",
           "hint": "#a0a0a0", "match": "#81c784", "near": "#ffb74d", "nomatch": "#ef9a9a", "waiting": "#9e9e9e"}}


def dark_mode(widget):
    """macOS ใช้ Dark Mode อยู่ไหม (สีตัวหนังสือที่ตั้งเองต้องอ่านได้บนพื้นเข้ม) — ระบบอื่น/อ่านไม่ได้ = ไม่ใช่"""
    try:
        return bool(int(widget.tk.call("tk::unsupported::MacWindowStyle", "isdark", widget.winfo_toplevel())))
    except (tk.TclError, ValueError):
        return False


class ScrollFrame(ttk.Frame):
    """เนื้อหาที่เลื่อนขึ้นลงได้เมื่อหน้าต่างเตี้ยกว่าเนื้อหา (ไม่มีอะไรตกขอบ) — ใส่ชิ้นต่าง ๆ ใน .inner
    หน้าต่างสูงพอ: .inner สูงเท่าหน้าต่าง ชิ้นที่ pack expand=True ได้ที่ที่เหลือ และแถบเลื่อนซ่อน"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, yscrollincrement=8,
                                background=self.winfo_toplevel().cget("background"))
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._item = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _event: self._fit())
        self.canvas.bind("<Configure>", lambda _event: self._fit())
        self.winfo_toplevel().bind("<MouseWheel>", self._wheel, add="+")

    def _fit(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        needed = self.inner.winfo_reqheight()
        self.canvas.itemconfigure(self._item, width=width, height=max(needed, height))
        self.canvas.configure(scrollregion=(0, 0, width, max(needed, height)))
        if needed > height + 1 and not self.bar.winfo_ismapped():
            self.bar.pack(side="right", fill="y", before=self.canvas)
        elif needed <= height + 1 and self.bar.winfo_ismapped():
            self.bar.pack_forget()
            self.canvas.yview_moveto(0)

    def _wheel(self, event):
        """ปัดสองนิ้ว/ล้อเมาส์ในหน้าต่างนี้ = เลื่อนเนื้อหา (ยกเว้นบนตาราง/รายการที่เลื่อนเองได้)"""
        try:
            if (not self.bar.winfo_ismapped() or not str(event.widget).startswith(str(self))
                    or event.widget.winfo_class() in ("Treeview", "Listbox", "Text", "TCombobox")):
                return
        except (tk.TclError, AttributeError):
            return
        self.canvas.yview_scroll(-event.delta if sys.platform == "darwin" else -event.delta // 120, "units")


class FlowFrame(ttk.Frame):
    """แถวปุ่ม/ช่องที่ขึ้นบรรทัดใหม่เองเมื่อหน้าต่างแคบ (ไม่ล้นจนปุ่มทางขวาหายไป) — สร้างชิ้นด้วย master = แถวนี้ แล้ว add()"""

    def __init__(self, master, gap=6, **kwargs):
        super().__init__(master, **kwargs)
        self.gap, self.items, self._job = gap, [], None
        self.bind("<Configure>", lambda _event: self._arrange())

    def add(self, widget, space=0):
        """space = ระยะเพิ่มก่อนชิ้นนี้ (แบ่งกลุ่ม)"""
        self.items.append((widget, space))
        if self._job is None:
            self._job = self.after_idle(self._arrange)
        return widget

    def destroy(self):
        if self._job is not None:            # จัดแถวที่รอทำอยู่: ยกเลิกก่อนปิด
            self.after_cancel(self._job)
            self._job = None
        super().destroy()

    def _arrange(self):
        self._job = None
        try:
            width = self.winfo_width()
        except tk.TclError:                  # ปิดหน้าต่างไปแล้ว
            return
        width = width if width > 1 else self.winfo_reqwidth()
        lines, x = [[]], 0
        for widget, space in self.items:
            w = widget.winfo_reqwidth()
            if lines[-1] and x + space + w > width:
                lines.append([])
                x = 0
            elif lines[-1]:
                x += space
            lines[-1].append((widget, x))
            x += w + self.gap
        y = 0
        for line in lines:
            height = max((widget.winfo_reqheight() for widget, _x in line), default=0)
            for widget, x in line:                     # ชิ้นที่เตี้ยกว่าอยู่กลางบรรทัด (ปุ่มกับป้ายตรงกัน)
                widget.place(x=x, y=y + (height - widget.winfo_reqheight()) // 2)
            y += height + 4
        y = max(0, y - 4)
        if int(str(self.cget("height") or 0)) != y:
            self.configure(height=y)


def your_summary(yours):
    """ค่าเดียวจาก Prism Analysis (ไม่มีเสียงให้วัด) -> summary"""
    peak = float((yours or {}).get("Fppeak") or 0.0)
    if peak <= 0:
        return None
    low, high = float(yours.get("Fpmin") or peak), float(yours.get("Fpmax") or peak)
    return {"calls": 1, "peak": peak, "q1": peak, "q3": peak, "low": peak, "high": peak, "spread": MIN_SPREAD,
            "lo_edge": low, "hi_edge": high, "dur": 0.0, "fmean": float(yours.get("Fpmean") or peak)}


def drawn_events(samples, sr, region):
    """กรอบที่ลากเอง (a, b, low, high): วัดทุกเสียงในกรอบแล้วใช้เสียงหลัก (ตัดเสียงนกตัวอื่นที่ติดมาในกรอบ)"""
    a, b, low, high = region
    part = np.asarray(samples)[max(0, int(a * sr)):int(b * sr)]
    return main_calls(measure_samples(part, sr, offset=a, band=(max(150.0, low), min(high, sr / 2 - 1)), top=40))


def box_events(samples, sr, box):
    """กรอบเสียงของ BirdNET (t0, t1, low, high) วัดแบบเดียวกับเสียงอ้างอิง (ตัวตรวจเสียงดูพื้นหลังรอบกรอบ)
    ตัวตรวจไม่เจอ (เสียงเบาเกินเกณฑ์) -> วัดทั้งกรอบเป็นเสียงเดียว"""
    import prism_view as kv
    t0, t1, low, high = (float(v) for v in box)
    samples = np.asarray(samples)
    band = (max(150.0, low - BAND_MARGIN), min(sr / 2 - 1, high + BAND_MARGIN))
    first = max(0.0, t0 - BOX_CONTEXT)
    events = [e for e in measure_samples(samples[int(first * sr):int((t1 + BOX_CONTEXT) * sr)], sr, offset=first,
                                         band=band, top=EVENTS_PER_FILE)
              if e["end"] > t0 and e["start"] < t1]
    if events:
        return events
    part = samples[max(0, int(t0 * sr)):int(t1 * sr)]
    if len(part) < kv.FFT_SIZE * 2:
        return []
    stats = kv.analysis_stats(kv.spectrum_db(part), sr, *band)
    if stats["Fppeak"] <= 0:
        return []
    purity = tone_purity(samples, sr, t0, t1, low, high)
    return [{"start": round(t0, 3), "end": round(t1, 3), "dur": round(t1 - t0, 3), "low": round(low),
             "high": round(high), "peak": round(stats["Fppeak"]), "fmin": round(stats["Fpmin"]),
             "fmax": round(stats["Fpmax"]), "fmean": round(stats["Fpmean"]), "score": 0.0,
             "purity": None if purity is None else round(purity, 3)}]


def clip_candidates(samples, sr, boxes, source):
    """กรอบเสียงที่เลือกได้ -> [{"t0", "t1", "low", "high", "kind", "events", "on"}] (ไม่แตะ Tk)
    source: "BirdNET boxes" = กรอบเสียงใน Detection boxes, "Sound detector" = เสียงที่ตัวตรวจหาเจอทั้งคลิป"""
    if source == YOUR_SOURCES[0]:
        return [{"t0": float(t0), "t1": float(t1), "low": float(low), "high": float(high), "kind": "birdnet",
                 "events": box_events(samples, sr, (t0, t1, low, high)), "on": False}
                for t0, t1, low, high in (boxes or {}).get("sounds", ())]
    return [{"t0": e["start"], "t1": e["end"], "low": float(e["low"]), "high": float(e["high"]), "kind": "detector",
             "events": [e], "on": False}
            for e in measure_samples(samples, sr, band=(250.0, min(12000.0, sr / 2 - 1)), top=40)]


def auto_groups(candidates, width=CLUSTER_WIDTH):
    """กลุ่มกรอบที่ความถี่ที่ดังสุดใกล้กัน (±width) เรียงจากร้องซ้ำมากสุด -> [[ลำดับกรอบ, ...], ...]
    (ไม่รวมกรอบที่ลากเอง) — นกตัวอื่นอาจร้องซ้ำมากกว่าชนิดที่สนใจ จึงให้เลื่อนไปกลุ่มถัดไปได้"""
    left = []
    for number, candidate in enumerate(candidates):
        peaks = [e["peak"] for e in candidate["events"] if e["peak"] > 0]
        if candidate["kind"] != "drawn" and peaks:
            left.append((number, float(np.median(peaks)), max(e.get("score", 0.0) for e in candidate["events"])))
    groups = []
    while left:
        peaks = np.array([peak for _n, peak, _s in left])
        scores = np.array([score for _n, _p, score in left])
        near = np.abs(peaks[:, None] - peaks[None, :]) <= width * peaks[:, None]
        best = int(np.lexsort((-(near * scores[None, :]).sum(axis=1), -near.sum(axis=1)))[0])
        groups.append([n for (n, _p, _s), keep in zip(left, near[best]) if keep])
        left = [item for item, keep in zip(left, near[best]) if not keep]
    return groups


def auto_choice(candidates):
    """Auto: กรอบของเสียงที่ร้องซ้ำมากที่สุด -> set ของลำดับกรอบ"""
    groups = auto_groups(candidates)
    return set(groups[0]) if groups else set()


def chosen_events(candidates):
    """เสียงจากกรอบที่เลือก — กรอบซ้อนกันวัดเสียงเดียวกันได้ค่าต่างกันเล็กน้อย: เวลาทับกันและความถี่ห่าง ≤ MARGIN = เสียงเดียว"""
    events = []
    for candidate in candidates:
        for event in candidate["events"] if candidate["on"] else ():
            if not any(event["start"] < other["end"] and other["start"] < event["end"]
                       and abs(event["peak"] - other["peak"]) <= MARGIN * max(event["peak"], other["peak"])
                       for other in events):
                events.append(event)
    return sorted(events, key=lambda e: e["start"])


def shift_boxes(boxes, origin):
    """Detection boxes (วินาทีนับจากต้นไฟล์/คลิป) -> นับจาก origin (ต้นช่วงที่โหลดใน Listen tools)"""
    if not boxes:
        return None
    return {"windows": [[a - origin, b - origin, *rest] for a, b, *rest in boxes.get("windows", ())],
            "sounds": [[a - origin, b - origin, low, high] for a, b, low, high in boxes.get("sounds", ())
                       if b > origin]}


class FrequencyCompareWindow:
    """เทียบเสียงของคุณกับเสียงอ้างอิงของหลายชนิด: yours = ค่าจาก Prism Analysis (ใช้เมื่อไม่มีเสียง),
    audio = (samples, sr, กรอบที่ลากไว้หรือ None) -> แสดงภาพเสียงให้เลือกกรอบเสียงเอง, boxes = Detection boxes ของคลิป
    (วินาทีนับจากต้น samples), db = spectrum_db ของ samples ถ้ามีแล้ว, species = (common, scientific) ชนิดแรก"""
    COLORS = {"bg": "#12151c", "fg": "#d7dde8", "yours": "#ff6b6b", "peak": "#7fdbff"}
    PICK_H = 130                               # ความสูงภาพเสียงของคุณ (รวมแกนเวลา)
    PLAY_MINE_MAX = 20.0                       # เล่นเสียงของคุณ: ช่วงกรอบที่เลือก (± 0.3 วินาที) ไม่เกินเท่านี้วินาที

    def __init__(self, master, yours, species=None, location=None, describe="", audio=None, boxes=None, db=None,
                 when=None):
        import field_log
        import prism_view as kv
        self.kv = kv
        self.location, self.describe, self.when = location, describe, when    # when = เวลาเริ่มของเสียงคุณ (หา field log)
        self.checklists = field_log.for_recording(when)
        self.yours, self.your_events = your_summary(yours), []
        self.candidates = []                   # กรอบเสียงของคุณที่เลือกได้ (clip_candidates + กรอบที่ลากเอง)
        self.player, self._playhead, self._playhead_job = None, None, None   # เสียงที่กำลังเล่นในหน้าต่างนี้
        self.spectrograms, self._spectro_job, self._spectro_rebuild = None, None, False   # Compare spectrograms
        self._separator = None                 # งาน Train on references ที่กำลังทำ (Popen)
        self.clip, region = None, None
        if audio is not None:
            samples, sr, region = audio
            samples = np.asarray(samples, np.float32)
            self.clip = {"samples": samples, "sr": sr, "duration": len(samples) / sr, "db": db, "boxes": boxes}
            self._lut, self._drag, self._clip_generation, self._auto_index = kv.palette(False), None, 0, 0
        self.species = []                      # [{"common", "scientific", "search": [...]}]
        self.rows = {}                         # iid -> {"species", "label", "path", "recording", "events", "where"}
        self.q = queue.Queue()
        self._stop = threading.Event()
        self._busy = False
        win = self.win = tk.Toplevel(master)
        win.title("Compare with reference sounds")
        win.geometry(f"1000x{min(800 if self.clip is None else 960, win.winfo_screenheight() - 70)}")
        win.minsize(640, 520)
        win.update_idletasks()
        colors = self.text_colors = TEXT_COLORS[dark_mode(win)]
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        bottom = ttk.Frame(body)                       # วางก่อน = จองที่ล่างสุดไว้ ปุ่มไม่ตกขอบเมื่อจอเตี้ย
        bottom.pack(side="bottom", fill="x", pady=(6, 0))
        self.credit = tk.StringVar()
        ttk.Label(bottom, textvariable=self.credit, foreground=colors["hint"], wraplength=960).pack(
            side="bottom", anchor="w", pady=(4, 0))
        ttk.Button(bottom, text="Close", command=self.close).pack(side="right", anchor="n")
        buttons = self.button_bar = FlowFrame(bottom)
        buttons.pack(side="left", fill="x", expand=True)
        buttons.add(ttk.Button(buttons, text="▶ Play reference", command=self.play_selected))
        buttons.add(ttk.Button(buttons, text="■ Stop", command=self.stop))
        buttons.add(playback.control(buttons))
        buttons.add(ttk.Button(buttons, text="Open in Listen tools", command=self.open_selected))
        buttons.add(ttk.Button(buttons, text="Compare spectrograms…", command=self.open_spectrograms), space=10)
        scroller = self.scroller = ScrollFrame(body)  # เนื้อหาเลื่อนได้เมื่อจอเตี้ย (ไม่มีอะไรตกขอบ)
        scroller.pack(fill="both", expand=True)
        content = scroller.inner
        # ---- เสียงของคุณ ----
        self.your_text = tk.StringVar(value=self._describe_yours())
        ttk.Label(content, textvariable=self.your_text, font=("Helvetica", 12, "bold"), wraplength=960,
                  justify="left").pack(anchor="w")
        log_row = ttk.Frame(content)                      # eBird checklist ของไฟล์เสียงนี้ (field_log.py)
        log_row.pack(fill="x", pady=(2, 0))
        self.log_text = tk.StringVar(value=self._field_log_text())
        ttk.Label(log_row, textvariable=self.log_text, foreground=colors["log"], wraplength=960,
                  justify="left").pack(side="left", fill="x", expand=True)
        if self.clip is not None:
            self._build_picker(content)
        # ---- ชนิดที่เทียบ + เสียงอ้างอิง (แคบแล้วขึ้นบรรทัดใหม่เอง) ----
        ttk.Separator(content).pack(fill="x", pady=(8, 4))
        add = self.species_bar = FlowFrame(content)
        add.pack(fill="x")
        add.add(ttk.Label(add, text="Add species"))
        self.common, self.scientific = tk.StringVar(), tk.StringVar()
        entry = add.add(ttk.Entry(add, textvariable=self.common, width=26))
        add.add(ttk.Entry(add, textvariable=self.scientific, width=24))
        from species_picker import SpeciesPicker
        self.picker = SpeciesPicker(entry, self._country, lambda c, s: (self.common.set(c), self.scientific.set(s)),
                                    on_exact=self.scientific.set)
        add.add(ttk.Button(add, text="✚ Add", command=self.add_species))
        add.add(ttk.Button(add, text="Remove selected species", command=self.remove_species))
        refs = self.reference_bar = FlowFrame(content)
        refs.pack(fill="x", pady=(4, 0))
        refs.add(ttk.Label(refs, text="References · sound type"))
        self.sound_type = tk.StringVar(value="call")
        refs.add(ttk.Combobox(refs, textvariable=self.sound_type, values=SOUND_TYPES, state="readonly", width=6))
        self.find_btn = refs.add(ttk.Button(refs, text="Find references", command=self.find), space=4)
        self.more_btn = refs.add(ttk.Button(refs, text="Download 3 more each", command=lambda: self.download(3)))
        refs.add(ttk.Button(refs, text="Add my own files…", command=self.add_files))
        self.train_btn = refs.add(ttk.Button(refs, text="Train on references", command=self.train_separator), space=10)
        columns = ("species", "files", "peak", "middle", "band", "dur", "purity", "rate", "result")
        self.species_tree = ttk.Treeview(content, columns=columns, show="headings", height=2)   # สูงตามจำนวนชนิด
        for key, label, width in (("species", "Species", 190), ("files", "References", 76), ("peak", "Peak kHz", 66),
                                  ("middle", "Middle half", 86), ("band", "Band kHz", 86), ("dur", "Length", 60),
                                  ("purity", "Purity", 56), ("rate", "Rhythm", 120), ("result", "Your sound", 200)):
            self.species_tree.heading(key, text=label)
            self.species_tree.column(key, width=width, stretch=key == "result")
        self.species_tree.tag_configure("best", font=("Helvetica", 12, "bold"))
        self.species_tree.pack(fill="x", pady=(6, 0))
        self.verdict = tk.StringVar()
        ttk.Label(content, textvariable=self.verdict, font=("Helvetica", 12, "bold"), foreground=colors["verdict"],
                  wraplength=960, justify="left").pack(anchor="w", pady=(6, 0))
        self.purity_text = tk.StringVar()              # ความใสของเสียง (tone_purity) เทียบแต่ละชนิด
        ttk.Label(content, textvariable=self.purity_text, foreground=colors["purity"], wraplength=960,
                  justify="left").pack(anchor="w")
        self.rhythm_text = tk.StringVar()              # จังหวะ (rhythm) เทียบแต่ละชนิด
        ttk.Label(content, textvariable=self.rhythm_text, foreground=colors["purity"], wraplength=960,
                  justify="left").pack(anchor="w")
        self.separation = tk.StringVar()               # ผลของ Train on references
        ttk.Label(content, textvariable=self.separation, foreground=colors["separation"], wraplength=960,
                  justify="left").pack(anchor="w")
        self.state = tk.StringVar()
        ttk.Label(content, textvariable=self.state, foreground=colors["state"], wraplength=960,
                  justify="left").pack(anchor="w")
        # ---- ล่าง: ไฟล์อ้างอิง / กราฟ เป็นแท็บ ใช้ที่ที่เหลือทั้งหมด (จอเตี้ยก็ยังเห็นตาราง) ----
        self.lower = ttk.Notebook(content)
        self.lower.pack(fill="both", expand=True, pady=(6, 0))
        table, chart = ttk.Frame(self.lower), ttk.Frame(self.lower)
        self.lower.add(table, text="Reference files")
        self.lower.add(chart, text="Chart: peak and band")
        self.canvas = tk.Canvas(chart, bg=self.COLORS["bg"], highlightthickness=0, height=160)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        columns = ("n", "species", "source", "quality", "type", "where", "calls", "peak", "band", "dur", "match",
                   "purity", "rate")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=6)   # อย่างน้อย 6 แถวเสมอ
        for key, label, width in (("n", "#", 30), ("species", "Species", 130), ("source", "Reference", 130),
                                  ("quality", "Q", 26), ("type", "Type", 80), ("where", "Where", 150),
                                  ("calls", "Main/all", 64), ("peak", "Peak kHz", 64), ("band", "Band kHz", 84),
                                  ("dur", "Length", 56), ("match", "Your peak", 76), ("purity", "Purity", 52),
                                  ("rate", "Rhythm", 90)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=key in ("where", "type"))
        for tag in ("match", "near", "nomatch", "waiting"):
            self.tree.tag_configure(tag, foreground=colors[tag])
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self._wrapping = []                            # ข้อความยาว: ตัดบรรทัดตามความกว้างหน้าต่างจริง
        for parent in (content, bottom):
            stack = list(parent.winfo_children())
            while stack:
                widget = stack.pop()
                stack += widget.winfo_children()
                if isinstance(widget, ttk.Label) and str(widget.cget("wraplength")) in ("960", "840"):
                    self._wrapping.append((widget, 140 if str(widget.cget("wraplength")) == "840" else 20))
        body.bind("<Configure>", lambda event: self._rewrap(event.width), add="+")
        win.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_job = self.win.after(100, self._poll)
        self._find_job = None
        if species and len(" ".join(str(species[1] or "").split()[:2]).split()) == 2:
            self._add(species[0] or species[1], species[1])
            self._find_job = self.win.after(150, self.find)
        else:
            self.state.set("Add the species to compare with (a few letters, then pick from the list) — add a second "
                           "species that sounds similar to tell them apart")
        if self.clip is not None:              # กรอบที่ลากไว้ใน Listen tools = เลือกไว้แล้ว ไม่งั้นให้ Auto เลือก
            if region:
                self._add_drawn(tuple(region))
            self._load_candidates(auto=not region)

    def _rewrap(self, width):
        for label, margin in self._wrapping:
            label.configure(wraplength=max(200, width - margin))

    def _describe_yours(self):
        s = self.yours
        where = f" — {self.describe}" if self.describe else ""
        if self.clip is not None:
            chosen = sum(candidate["on"] for candidate in self.candidates)
            where += f" — {chosen} box(es) chosen" if chosen else ""
        if not s:
            if self.clip is not None:
                return (f"Your sound{where}: nothing chosen yet — click the boxes in the picture, press Auto, "
                        "or drag your own box around the calls")
            return f"Your sound{where}: no clear sound found — drag a box around the calls (time and frequency) first"
        text = (f"Your sound{where}: {s['calls']} sound(s), peak {s['peak'] / 1000:.2f} kHz"
                + (f" (middle half {s['q1'] / 1000:.2f}–{s['q3'] / 1000:.2f})" if s["calls"] > 1 else "")
                + f", band {s['lo_edge'] / 1000:.2f}–{s['hi_edge'] / 1000:.2f} kHz"
                + (f", length {s['dur'] * 1000:.0f} ms" if s["dur"] else "")
                + (f", purity {s['purity']:.2f}" if s.get("purity") is not None else ""))
        if s["calls"] > 1 and s["q3"] - s["q1"] > 0.15 * s["peak"]:
            text += "  ⚠ the chosen sounds differ in pitch — choose one bird's calls only"
        return text

    # ---------- บันทึกภาคสนาม (eBird checklist ของคุณ) ----------
    def _field_log_text(self):
        import field_log
        if self.when is None:
            return "Field log: recording date unknown"
        if not self.checklists:
            if field_log.for_day(self.when):
                return "Field log: this recording is set to No checklist (Settings › Match recordings…)"
            return f"Field log: no eBird checklist of yours for {self.when:%d %b %Y} (Settings › Field logs)"
        names = [field_log.describe_species(self.checklists, self.species[i]["scientific"], self.species[i]["common"])
                 for i in self._active()]
        return f"Your checklist {field_log.describe_day(self.checklists)}" + (": " + "; ".join(names) if names else "")

    # ---------- เสียงของคุณ: เลือกกรอบในภาพ ----------
    def _build_picker(self, body):
        has_boxes = bool((self.clip["boxes"] or {}).get("sounds"))
        row = self.picker_bar = FlowFrame(body)
        row.pack(fill="x", pady=(6, 0))
        row.add(ttk.Label(row, text="Your sounds from"))
        self.source = tk.StringVar(master=self.win, value=YOUR_SOURCES[0] if has_boxes else YOUR_SOURCES[1])
        source = row.add(ttk.Combobox(row, textvariable=self.source,
                                      values=YOUR_SOURCES if has_boxes else YOUR_SOURCES[1:], state="readonly", width=14))
        source.bind("<<ComboboxSelected>>", lambda _event: self._load_candidates(auto=True))
        row.add(ttk.Button(row, text="Auto (repeated sound) ▸", command=lambda: self.auto_pick(step=True)), space=4)
        row.add(ttk.Button(row, text="All boxes", command=self.pick_all))
        row.add(ttk.Button(row, text="Clear", command=self.clear_pick))
        row.add(ttk.Button(row, text="▶ Play my sound", command=self.play_mine), space=10)
        row.add(ttk.Button(row, text="■ Stop", command=self.stop))
        canvas = self.clip_canvas = tk.Canvas(body, bg=self.COLORS["bg"], highlightthickness=0, height=self.PICK_H,
                                              cursor="crosshair")
        canvas.pack(fill="x", pady=(4, 0))
        canvas.bind("<Configure>", lambda _event: self._draw_clip())
        canvas.bind("<ButtonPress-1>", self._clip_press)
        canvas.bind("<B1-Motion>", self._clip_drag)
        canvas.bind("<ButtonRelease-1>", self._clip_release)

    def _load_candidates(self, auto):
        """หากรอบเสียงในคลิป (เบื้องหลัง) — auto = ให้ Auto เลือกเมื่อเสร็จ"""
        self._clip_generation += 1
        clip, source, results, generation, kv = self.clip, self.source.get(), self.q, self._clip_generation, self.kv
        self.state.set("Finding the sounds in your clip …")

        def work():   # ไม่ถือ self
            try:
                db = clip["db"] if clip["db"] is not None else kv.spectrum_db(clip["samples"])
                found = clip_candidates(clip["samples"], clip["sr"], clip["boxes"], source)
                results.put(("clip", generation, db, found, auto))
            except Exception as exc:  # noqa: BLE001  ไฟล์เสีย: แจ้งในหน้าต่าง
                results.put(("state", f"Could not read your sound: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def _clip_loaded(self, generation, db, found, auto):
        if generation != self._clip_generation:
            return
        self.clip["db"] = db
        self.candidates = found + [c for c in self.candidates if c["kind"] == "drawn"]
        self.state.set(f"{len(found)} sound box(es) in your clip ({self.source.get()}) — click a box to use or skip it, "
                       "drag in the picture to draw your own" if found else
                       "No clear sound box in your clip — drag a box around the calls")
        if auto:
            self.auto_pick()
        else:
            self._update_yours()

    def auto_pick(self, step=False):
        """เลือกกลุ่มเสียงที่ร้องซ้ำมากสุด — กดซ้ำ (step) = กลุ่มถัดไป (ถ้ายังไม่ได้เปลี่ยนที่เลือกเอง)"""
        groups = auto_groups(self.candidates)
        now = {n for n, c in enumerate(self.candidates) if c["on"] and c["kind"] != "drawn"}
        if step and groups and now == set(groups[self._auto_index % len(groups)]):
            self._auto_index = (self._auto_index + 1) % len(groups)
        else:
            self._auto_index = 0
        chosen = set(groups[self._auto_index]) if groups else set()
        for number, candidate in enumerate(self.candidates):
            candidate["on"] = number in chosen
        self._update_yours()
        if groups and self.yours:
            self.state.set(f"Auto: sound group {self._auto_index + 1} of {len(groups)} — {len(chosen)} box(es) around "
                           f"{self.yours['peak'] / 1000:.2f} kHz"
                           + (" · press Auto again for the next group, or click boxes" if len(groups) > 1 else ""))

    def pick_all(self):
        for candidate in self.candidates:
            if candidate["kind"] != "drawn":
                candidate["on"] = True
        self._update_yours()

    def clear_pick(self):
        self.candidates = [c for c in self.candidates if c["kind"] != "drawn"]
        for candidate in self.candidates:
            candidate["on"] = False
        self._update_yours()

    def _add_drawn(self, region):
        a, b, low, high = (float(v) for v in region)
        events = drawn_events(self.clip["samples"], self.clip["sr"], (a, b, low, high))
        self.candidates.append({"t0": a, "t1": b, "low": low, "high": high, "kind": "drawn", "events": events,
                                "on": True})
        if not events:
            self.state.set("No clear sound inside your box — draw it around the calls")
        self._update_yours()

    def _update_yours(self):
        self.your_events = chosen_events(self.candidates)
        self.yours = summarize(self.your_events)
        self.your_text.set(self._describe_yours())
        self._draw_clip()
        self._refresh()

    def _clip_geometry(self):
        """(ซ้ายของภาพ, กว้าง, สูง (ไม่รวมแกนเวลา), ความถี่บนสุด)"""
        width = max(200, self.clip_canvas.winfo_width())
        return self.kv.GUTTER, max(50, width - self.kv.GUTTER - 6), self.PICK_H - 16, min(12000.0, self.clip["sr"] / 2)

    def _draw_clip(self):
        """ภาพเสียงของคุณแบบ Prism + แถบบน = ช่วงที่ BirdNET เจอ, กรอบฟ้าประ = เลือกได้, เหลือง = ใช้อยู่,
        แดง = กรอบที่ลากเอง (แบบภาพ spectrogram ที่มีกรอบของ BirdNET)"""
        if self.clip is None:
            return
        kv, canvas, clip = self.kv, self.clip_canvas, self.clip
        canvas.delete("all")
        x0, plot_w, plot_h, fmax = self._clip_geometry()
        duration = max(clip["duration"], 1e-6)

        def x_of(t):
            return x0 + t / duration * plot_w

        def y_of(hz):
            return plot_h * (1 - min(max(hz, 0.0), fmax) / fmax)
        if clip["db"] is not None and len(clip["db"]):
            rgb = kv.spectrogram_rgb(kv.column_pool(clip["db"], plot_w), np.linspace(fmax, 0, plot_h), clip["sr"],
                                     self._lut)
            self._clip_photo = ImageTk.PhotoImage(Image.fromarray(rgb, "RGB").resize((plot_w, plot_h),
                                                                                    Image.BILINEAR), master=self.win)
            canvas.create_image(x0, 0, anchor="nw", image=self._clip_photo)
        else:
            canvas.create_text(x0 + plot_w / 2, plot_h / 2, text="Reading your sound …", fill="#9e9e9e",
                               font=kv.RULER_FONT)
        fg = self.COLORS["fg"]
        step = kv.nice_step(fmax, plot_h, 24)
        for hz in np.arange(step, fmax + 1, step):
            y = y_of(hz)
            if y >= 8:
                canvas.create_line(x0 - 6, y, x0, y, fill=fg)
                canvas.create_text(x0 - 8, y, text=kv.freq_label(hz), anchor="e", fill=fg, font=kv.RULER_FONT)
        tick = kv.nice_step(duration, plot_w, 70)
        for t in np.arange(0.0, duration + 1e-9, tick):
            canvas.create_line(x_of(t), plot_h, x_of(t), plot_h + 4, fill=fg)
            canvas.create_text(x_of(t) + 2, plot_h + 1, text=kv.time_label(t), anchor="nw", fill=fg,
                               font=("Helvetica", 10))
        for t0, t1, *rest in (clip["boxes"] or {}).get("windows", ()):
            canvas.create_rectangle(x_of(t0), 0, x_of(t1), 12, outline=fg)
            if rest:
                canvas.create_text(x_of(t0) + 3, 1, text=f"BirdNET {float(rest[0]):.2f}", anchor="nw", fill=fg,
                                   font=("Helvetica", 9))
        for candidate in self.candidates:
            drawn, on = candidate["kind"] == "drawn", candidate["on"]
            options = {"outline": "#e53935" if drawn else ("#ffeb3b" if on else "#00e5ff"), "width": 2 if on else 1}
            if not on:
                options["dash"] = (3, 3)
            left = x_of(candidate["t0"])
            canvas.create_rectangle(left, y_of(candidate["high"] * 1.05), max(x_of(candidate["t1"]), left + 3),
                                    y_of(candidate["low"] * 0.95), **options)
        placed = []                                    # ความใสของแต่ละกรอบ — กรอบที่เลือกก่อน ป้ายที่จะทับกันข้ามไป
        for candidate in sorted(self.candidates, key=lambda c: not c["on"]):
            purity = [e["purity"] for e in candidate["events"] if e.get("purity") is not None]
            if not purity:
                continue
            color = "#e53935" if candidate["kind"] == "drawn" else ("#ffeb3b" if candidate["on"] else "#00e5ff")
            item = canvas.create_text(x_of(candidate["t0"]), y_of(candidate["high"] * 1.05) - 1,
                                      text=f"{np.median(purity):.2f}", anchor="sw", fill=color, font=("Helvetica", 9))
            box = canvas.bbox(item)
            if box and any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                canvas.delete(item)
            elif box:
                placed.append(box)

    def _clip_press(self, event):
        self._drag = (event.x, event.y)

    def _clip_drag(self, event):
        if self._drag:
            self.clip_canvas.delete("rubber")
            self.clip_canvas.create_rectangle(*self._drag, event.x, event.y, outline="#e53935", dash=(4, 2),
                                              tags="rubber")

    def _clip_release(self, event):
        start, self._drag = self._drag, None
        self.clip_canvas.delete("rubber")
        if start is None:
            return
        x0, plot_w, plot_h, fmax = self._clip_geometry()
        duration = self.clip["duration"]

        def t_of(x):
            return min(duration, max(0.0, (x - x0) / plot_w * duration))

        def hz_of(y):
            return min(fmax, max(0.0, (1 - y / plot_h) * fmax))
        if abs(event.x - start[0]) < 5 and abs(event.y - start[1]) < 5:      # คลิก = ใช้/ไม่ใช้กรอบนั้น
            t, hz, slack = t_of(event.x), hz_of(event.y), 4 / plot_w * duration
            hits = [c for c in self.candidates
                    if c["t0"] - slack <= t <= c["t1"] + slack and c["low"] * 0.95 <= hz <= c["high"] * 1.05]
            if hits:
                hit = min(hits, key=lambda c: (c["t1"] - c["t0"] + slack) * (c["high"] - c["low"] + 1))
                hit["on"] = not hit["on"]
                self._update_yours()
            return
        a, b = sorted((t_of(start[0]), t_of(event.x)))
        low, high = sorted((hz_of(start[1]), hz_of(event.y)))
        if b - a >= 0.02 and high - low >= 100:
            self._add_drawn((a, b, low, high))

    def _country(self):
        import country_checklist
        lat, lon = self.location or (None, None)
        return country_checklist.country_at(lat, lon) if lat is not None else None

    # ---------- ชนิด ----------
    def _index_of(self, scientific):
        key = " ".join(str(scientific).split()[:2]).casefold()
        return next((n for n, s in enumerate(self.species) if s["scientific"].casefold() == key), None)

    def _add(self, common, scientific):
        scientific = " ".join(str(scientific).split()[:2])
        index = self._index_of(scientific)
        if index is not None:
            return index
        self.species.append({"common": common or scientific, "scientific": scientific, "search": []})
        index = len(self.species) - 1
        self.species_tree.insert("", "end", iid=f"s{index}", values=(common or scientific, 0, "", "", "", "", "", ""))
        self.species_tree.configure(height=max(2, min(4, len(self._active()))))
        self.log_text.set(self._field_log_text())
        self._spectrograms_changed(rebuild=True)
        return index

    def add_species(self):
        scientific = " ".join(self.scientific.get().split()[:2])
        if len(scientific.split()) < 2:
            self.state.set("Pick the species from the list so its scientific name is filled in")
            return
        self._add(self.common.get().strip(), scientific)
        self.common.set("")
        self.scientific.set("")
        self.find()

    def remove_species(self):
        chosen = self.species_tree.selection()
        if not chosen:
            return
        index = int(chosen[0][1:])
        for iid in [iid for iid, row in self.rows.items() if row["species"] == index]:
            self.tree.delete(iid)
            del self.rows[iid]
        self.species[index]["removed"] = True
        self.species_tree.delete(chosen[0])
        self.species_tree.configure(height=max(2, min(4, len(self._active()))))
        self._spectrograms_changed(rebuild=True)
        self._refresh()

    def _active(self):
        return [n for n, s in enumerate(self.species) if not s.get("removed")]

    # ---------- หาเสียงอ้างอิง ----------
    def find(self):
        if self._busy or not self._active():
            return
        for index in self._active():
            for path in library_files(self.species[index]["scientific"]):
                self._add_row(index, {"label": f"{path.name} (yours)", "path": path, "recording": None,
                                      "where": "your file"})
        self._work(self._find_work, list(self._active()), self.sound_type.get())

    def _work(self, function, *args):
        self._busy = True
        self.find_btn.configure(state="disabled")
        self.more_btn.configure(state="disabled")
        threading.Thread(target=function, args=args, daemon=True).start()

    def _find_work(self, indexes, sound_type):
        import app_settings
        try:
            good = {index: 0 for index in indexes}            # ไฟล์ที่มีเสียงชัดต่อชนิด (ไว้เทียบกับ MIN_FILES)
            for iid, row in list(self.rows.items()):           # ไฟล์ของคุณ: วัดก่อน (ไม่ต้องต่อเน็ต)
                if row.get("recording"):                       # ไฟล์ xeno-canto นับในรอบค้นหาด้านล่าง
                    continue
                events = row["events"] if row["events"] is not None else measure_file(row["path"])
                if row["events"] is None:
                    self.q.put(("measured", iid, events))
                if row["species"] in good and summarize(main_calls(events)):
                    good[row["species"]] += 1
            key = app_settings.load_settings().get("xeno_canto_key", "")
            if not key:
                self.q.put(("state", "No xeno-canto key (Settings tab): only your own reference files are used"))
                self.q.put(("done", None))
                return
            lat, lon = self.location or (None, None)
            waiting = 0
            for index in indexes:
                if self._stop.is_set():
                    return
                species = self.species[index]
                self.q.put(("state", f"Asking xeno-canto for {species['common']} ({sound_type}) …"))
                recordings = search(species["scientific"], species["common"], key, sound_type, lat, lon)
                self.q.put(("search", index, recordings))
                ready = [r for r in recordings if cached_path(r)]
                for recording in ready:
                    if self._stop.is_set():
                        return
                    self.q.put(("recording", index, recording))
                    events = measure_file(cached_path(recording))
                    self.q.put(("measured", recording["id"], events))
                    good[index] += bool(summarize(main_calls(events)))
                if good[index] < MIN_FILES and recordings:      # ไม่ถึงขั้นต่ำ: โหลดไฟล์ที่เร็วที่สุดเพิ่มจนครบ
                    good[index] += self._download_species(index, recordings, MAX_TOP_UP, enough=MIN_FILES - good[index])
                waiting += len(download_order(recordings))
            self.q.put(("state", "References measured." + (f" {waiting} more on xeno-canto — Download more fetches "
                                                           "them in the background." if waiting else "")))
        except Exception as exc:  # noqa: BLE001  เน็ต/ไฟล์เสีย: แจ้งในหน้าต่าง
            self.q.put(("state", f"Could not get references: {exc}"))
        self.q.put(("done", None))

    def _download_species(self, index, recordings, count, enough=None):
        """โหลดไฟล์ที่ยังไม่มีไม่เกิน count ไฟล์ (enough = หยุดเมื่อได้ไฟล์ที่มีเสียงชัดครบเท่านี้) -> จำนวนไฟล์ที่มีเสียงชัด"""
        import xeno_canto
        todo, good = download_order(recordings)[:count], 0
        for number, recording in enumerate(todo, 1):
            if self._stop.is_set() or (enough is not None and good >= enough):
                break
            self.q.put(("state", f"Downloading {self.species[index]['common']} {recording['id']} "
                                 f"({recording.get('length') or '?'}, {number} of {len(todo)}) — mp3 a few seconds, "
                                 "WAV up to 2 minutes …"))
            path = xeno_canto.download(recording)
            events = measure_file(path)
            self.q.put(("recording", index, recording))
            self.q.put(("measured", recording["id"], events))
            good += bool(summarize(main_calls(events)))
        return good

    def download(self, count):
        if self._busy:
            return
        targets = [n for n in self._active() if self.species[n]["search"]]
        if not targets:
            self.state.set("Click Find references first")
            return

        def work():
            try:
                for index in targets:
                    self._download_species(index, self.species[index]["search"], count)
                self.q.put(("state", "Downloads finished"))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("state", f"Download stopped: {exc}"))
            self.q.put(("done", None))
        self._work(work)

    def add_files(self):
        files = filedialog.askopenfilenames(parent=self.win, title="Reference sounds (xeno-canto file names are "
                                            "recognised: 'XC123 - Name - Genus species.mp3')",
                                            filetypes=[("Audio", " ".join(f"*{e} *{e.upper()}" for e in sorted(AUDIO_EXTS))),
                                                       ("All", "*.*")])
        if files:
            self.add_paths(files)

    def add_paths(self, files):
        """ไฟล์ที่ชื่อแบบ xeno-canto เข้าชนิดในชื่อ (เพิ่มชนิดให้ถ้ายังไม่มี) ที่เหลือเข้าชนิดที่เลือกอยู่ -> จำนวนที่เพิ่ม"""
        chosen = self.species_tree.selection()
        fallback = int(chosen[0][1:]) if chosen else (self._active()[0] if self._active() else None)
        added = 0
        for path in map(Path, files):
            parsed = parse_xc_name(path)
            index = self._add(parsed[1], parsed[2]) if parsed else fallback
            if index is None:
                messagebox.showinfo("Add reference files", f"Add a species first for {path.name}", parent=self.win)
                continue
            for target in add_to_library(self.species[index]["scientific"], [path]):
                iid = self._add_row(index, {"label": f"{target.name} (yours)", "path": target, "recording": None,
                                            "where": "your file"})
                self.rows[iid]["events"] = measure_file(target)
                self._update_row(iid)
                added += 1
        self._refresh()
        self.state.set(f"Added {added} file(s) to {LIBRARY} — used every time you compare these species, offline")
        return added

    # ---------- ตาราง ----------
    def _add_row(self, index, row):
        iid = row["recording"]["id"] if row.get("recording") else f"local-{row['path']}"
        if iid in self.rows:
            return iid
        self.rows[iid] = dict(row, species=index, events=None)
        self.tree.insert("", "end", iid=iid, tags=("waiting",), values=(
            len(self.rows), self.species[index]["common"], row["label"], (row["recording"] or {}).get("quality", ""),
            (row["recording"] or {}).get("type", ""), row.get("where", ""), "…", "", "", "", "", "", ""))
        return iid

    def _recording_row(self, index, recording):
        where = recording.get("country") or ""
        if recording.get("distance_km") is not None:
            where += f" · {recording['distance_km']:,} km"
        if recording.get("date"):
            where += f" · {recording['date'][:7]}"
        return self._add_row(index, {"label": recording["id"], "path": cached_path(recording), "recording": recording,
                                     "where": where})

    def _update_row(self, iid):
        row = self.rows[iid]
        main = main_calls(row["events"] or [])
        summary = row["summary"] = summarize(main)
        rhythm_now = row["rhythm"] = file_rhythm(file_notes(row["path"]), summary)
        match = matches(self.yours, summary)
        values = list(self.tree.item(iid, "values"))
        if summary:
            values[6:] = [f"{len(main)}/{len(row['events'])}", f"{summary['peak'] / 1000:.2f}",
                          f"{summary['lo_edge'] / 1000:.2f}–{summary['hi_edge'] / 1000:.2f}",
                          f"{summary['dur'] * 1000:.0f} ms",
                          {"inside": "inside ✓", "near": "near ~", "outside": "outside ✗", None: ""}[match],
                          "" if summary.get("purity") is None else f"{summary['purity']:.2f}",
                          _rhythm_cell(rhythm_now)]
        else:
            values[6:] = ["0", "", "", "", "no clear sound", "", ""]
        self.tree.item(iid, values=values, tags=({"inside": "match", "near": "near", "outside": "nomatch"}
                                                 .get(match, "waiting"),))

    def pooled(self):
        """{ชื่อชนิด: summary ของเสียงหลักทุกไฟล์ของชนิดนั้น}"""
        found = {}
        for index in self._active():
            rows = [row for row in self.rows.values() if row["species"] == index and row.get("events")]
            found[self.species[index]["common"]] = overall(rows) if rows else None
        return found

    def pooled_rhythms(self):
        """{ชื่อชนิด: จังหวะรวมของไฟล์อ้างอิงของชนิดนั้น}"""
        return {self.species[index]["common"]: pooled_rhythm(
            row.get("rhythm") for row in self.rows.values() if row["species"] == index and row.get("summary"))
            for index in self._active()}

    def reference_counts(self):
        """{ลำดับชนิด: (ไฟล์ที่มีเสียงชัด, จำนวนผู้บันทึก)} — ไฟล์ของคุณที่ไม่รู้ผู้บันทึกนับไฟล์ละคน"""
        counts = {}
        for index in self._active():
            rows = [r for r in self.rows.values() if r["species"] == index and r.get("summary")]
            people = {(r.get("recording") or {}).get("recordist") or f"file:{r['label']}" for r in rows}
            counts[index] = (len(rows), len(people))
        return counts

    def _short_of_minimum(self):
        return [f"{self.species[index]['common']} has {files} file(s) from {people} recordist(s)"
                for index, (files, people) in self.reference_counts().items()
                if files < MIN_FILES or people < MIN_RECORDISTS]

    def _refresh(self):
        pooled, rhythms = self.pooled(), self.pooled_rhythms()
        text, best = verdict(self.yours, pooled) if self.yours else ("", None)
        short = self._short_of_minimum()
        if text and short:                     # ไฟล์อ้างอิงน้อย: ผลยังเปลี่ยนได้ (ดู MIN_FILES)
            text = (f"Provisional — needs {MIN_FILES} reference recordings per species from {MIN_RECORDISTS} "
                    f"recordists ({'; '.join(short)}): {text}")
            best = None
        for index in self._active():
            species = self.species[index]
            summary = pooled.get(species["common"])
            files = sum(1 for row in self.rows.values() if row["species"] == index and row.get("summary"))
            if summary:
                z, overlap = distance(self.yours, summary), band_overlap(self.yours, summary)
                result = ("" if z is None else f"peak {z:.1f}× spread away · {matches(self.yours, summary) or ''}"
                          + ("" if overlap is None else f" · band overlap {overlap:.0%}"))
                values = (species["common"], files, f"{summary['peak'] / 1000:.2f}",
                          f"{summary['q1'] / 1000:.2f}–{summary['q3'] / 1000:.2f}",
                          f"{summary['lo_edge'] / 1000:.2f}–{summary['hi_edge'] / 1000:.2f}",
                          f"{summary['dur'] * 1000:.0f} ms",
                          "" if summary.get("purity") is None else f"{summary['purity']:.2f}",
                          _species_rhythm_cell(rhythms.get(species["common"])),
                          result)
            else:
                values = (species["common"], files, "", "", "", "", "", "", "no references measured yet")
            self.species_tree.item(f"s{index}", values=values, tags=("best",) if species["common"] == best else ())
        if text:
            lengths = [s["dur"] for s in pooled.values() if s]
            if self.yours and self.yours.get("dur") and lengths and self.yours["dur"] < 0.4 * min(lengths):
                text += (" — your sounds are much shorter than the references (distant birds lose the echo tail), so "
                         "length is not compared")
            text += ". A frequency match supports an ID but does not prove it — listen as well."
        self.verdict.set(text)
        self.purity_text.set(purity_verdict(self.yours, pooled))
        self.rhythm_text.set(rhythm_verdict(rhythm(self.your_events), rhythms) if self.your_events or any(rhythms.values())
                             else "")
        self.log_text.set(self._field_log_text())
        self.draw()
        self._spectrograms_changed()

    # ---------- ฝึกตัวแยกจากเสียงอ้างอิง ----------
    def separator_spec(self):
        """ไฟล์อ้างอิงที่วัดแล้วของทุกชนิด (จุดกลางของเสียงหลัก) + เสียงของคุณ (จุดกลางของกรอบที่เลือก) -> SPEC ของ pair_separator"""
        import soundfile as sf
        species = []
        for index in self._active():
            files = []
            for row in self.rows.values():
                calls = main_calls(row.get("events") or []) if row["species"] == index and row.get("summary") else []
                path = row.get("path") or (cached_path(row["recording"]) if row.get("recording") else None)
                if calls and path:
                    files.append({"name": str(row["label"]), "path": str(path),
                                  "centers": [round((e["start"] + e["end"]) / 2, 3) for e in calls]})
            species.append({"common": self.species[index]["common"], "scientific": self.species[index]["scientific"],
                            "files": files})
        spec = {"species": species}
        chosen = [c for c in self.candidates if c["on"]]
        if self.clip is not None and chosen:
            target = CACHE / "separator-yours.wav"
            CACHE.mkdir(parents=True, exist_ok=True)
            sf.write(str(target), self.clip["samples"], self.clip["sr"])
            spec["yours"] = {"path": str(target), "centers": [round((c["t0"] + c["t1"]) / 2, 3) for c in chosen]}
        return spec

    def train_separator(self):
        """Train on references: ให้ BirdNET 3.0 ฟังเสียงหลักของไฟล์อ้างอิง แล้วฝึกตัวแยกชนิด (งานเบื้องหลังแยก process)"""
        import pair_separator
        import platform_tools
        if self._separator is not None:
            return
        spec = self.separator_spec()
        short = [f"{s['common']} {len(s['files'])}" for s in spec["species"] if len(s["files"]) < pair_separator.MIN_FILES]
        if len(spec["species"]) < 2 or short:
            self.separation.set(f"Train on references needs two or more species with at least {pair_separator.MIN_FILES} "
                                f"measured references each" + (f" ({', '.join(short)})" if short else "")
                                + " — Find references or Download more first")
            return
        target = CACHE / "separator-spec.json"
        target.write_text(json.dumps(spec), encoding="utf-8")
        script = Path(__file__).resolve().parent / "field_audio_to_ebird.py"
        process = self._separator = platform_tools.start_child([platform_tools.child_python(), str(script),
                                                                 "--separate", str(target)])
        self.train_btn.configure(state="disabled")
        self.separation.set("Training on your references … (BirdNET listens to each reference once, then it is cached)")
        results = self.q

        def work():   # ไม่ถือ self
            found = None
            for line in process.stdout:
                if line.startswith("SEPARATE "):
                    found = json.loads(line[len("SEPARATE "):])
                elif line.strip():
                    results.put(("state", line.strip()))
            process.wait()
            results.put(("separated", found or {"error": f"the training job stopped (exit {process.returncode})"}))
        threading.Thread(target=work, daemon=True).start()

    # ---------- ภาพเสียงเทียบกัน ----------
    def open_spectrograms(self):
        if self.spectrograms is not None and self.spectrograms.exists():
            self.spectrograms.win.lift()
            self.spectrograms.build()
            return
        self.spectrograms = SpectrogramCompareWindow(self)

    def _spectrograms_changed(self, rebuild=False):
        """เสียงที่เลือก/เสียงอ้างอิงเปลี่ยน -> วาดภาพเทียบใหม่ (รวบหลายครั้งที่มาติดกันเป็นครั้งเดียว)"""
        if self.spectrograms is None or not self.spectrograms.exists():
            return
        self._spectro_rebuild = self._spectro_rebuild or rebuild
        if self._spectro_job:
            self.win.after_cancel(self._spectro_job)
        self._spectro_job = self.win.after(300, self._spectrograms_now)

    def _spectrograms_now(self):
        self._spectro_job = None
        rebuild, self._spectro_rebuild = self._spectro_rebuild, False
        if self.spectrograms is not None and self.spectrograms.exists():
            self.spectrograms.build() if rebuild else self.spectrograms.refresh()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "state":
                    self.state.set(payload[0])
                elif kind == "search":
                    index, recordings = payload
                    self.species[index]["search"] = recordings
                elif kind == "recording":
                    self._recording_row(*payload)
                elif kind == "measured":
                    iid, events = payload
                    if iid in self.rows:
                        self.rows[iid]["events"] = events
                        if self.rows[iid].get("recording") and self.rows[iid]["path"] is None:
                            self.rows[iid]["path"] = cached_path(self.rows[iid]["recording"])
                        self._update_row(iid)
                        self._refresh()
                elif kind == "clip":
                    self._clip_loaded(*payload)
                elif kind == "separated":
                    import pair_separator
                    self._separator = None
                    self.train_btn.configure(state="normal")
                    self.separation.set(pair_separator.describe(payload[0]))
                    self.state.set("")
                elif kind == "done":
                    self._busy = False
                    self.find_btn.configure(state="normal")
                    self.more_btn.configure(state="normal")
        except queue.Empty:
            pass
        self._poll_job = self.win.after(150, self._poll)

    # ---------- กราฟ ----------
    def draw(self):
        """แกนตั้ง = ความถี่ (ซูมเฉพาะย่านที่มีเสียง), คอลัมน์ = เสียงคุณ (แดง) แล้วแต่ละชนิด: box plot ของความถี่ที่ดังสุด
        ต่อไฟล์ + กล่อง "all" ของทั้งชนิด, แถบบาง = ย่านเสียง (ขอบล่าง–บน), เส้นแดง = ความถี่ที่ดังสุดของคุณ"""
        canvas, colors, kv = self.canvas, self.COLORS, self.kv
        canvas.delete("all")
        width, height = max(300, canvas.winfo_width()), max(140, canvas.winfo_height())
        plot_h, axis_h = height - 30, 30
        groups = []                                         # (ชื่อ, สี, [summary ต่อไฟล์], summary รวม)
        pooled = self.pooled()
        for number, index in enumerate(self._active()):
            files = [row["summary"] for row in self.rows.values() if row["species"] == index and row.get("summary")]
            groups.append((self.species[index]["common"], SPECIES_COLORS[number % len(SPECIES_COLORS)], files,
                           pooled.get(self.species[index]["common"])))
        summaries = [s for _n, _c, files, total in groups for s in files + ([total] if total else [])]
        if self.yours:
            summaries.append(self.yours)
        if not summaries:
            canvas.create_text(width / 2, plot_h / 2, fill="#9e9e9e", font=kv.RULER_FONT,
                               text="Reference sounds appear here once measured")
            return
        low = min(min(s["q1"], s["lo_edge"]) for s in summaries) * 0.9
        high = max(max(s["q3"], s["hi_edge"]) for s in summaries) * 1.1
        step = kv.nice_step(high - low, plot_h, 26)
        low, high = np.floor(low / step) * step, np.ceil(high / step) * step

        def y_of(hz):
            return plot_h * (1 - (hz - low) / (high - low))
        for hz in np.arange(low, high + 1, step):
            y = y_of(hz)
            if 10 <= y <= plot_h:
                canvas.create_line(0, y, kv.GUTTER - 1, y, fill=colors["fg"])
                canvas.create_text(kv.GUTTER - 4, y - 1, text=kv.freq_label(hz), anchor="se", fill=colors["fg"],
                                   font=kv.RULER_FONT)
        slots = 1 + sum(len(files) + 1 for _n, _c, files, _t in groups)
        slot = (width - kv.GUTTER - 10) / max(slots, 1)

        def box(x, summary, color, wide=False):
            half = slot * (0.32 if wide else 0.2)
            canvas.create_line(x + half + 3, y_of(summary["hi_edge"]), x + half + 3, y_of(summary["lo_edge"]),
                               fill=color, width=3)                                   # ย่านเสียง
            canvas.create_line(x, y_of(summary["high"]), x, y_of(summary["low"]), fill=color)
            canvas.create_rectangle(x - half, y_of(summary["q3"]), x + half, y_of(summary["q1"]), outline=color,
                                    width=2 if wide else 1)
            canvas.create_line(x - half - 2, y_of(summary["peak"]), x + half + 2, y_of(summary["peak"]),
                               fill=colors["peak"], width=2)
        x = kv.GUTTER + slot / 2
        if self.yours:
            box(x, self.yours, colors["yours"], wide=True)
            for event in self.your_events:
                if low <= event["peak"] <= high:
                    canvas.create_oval(x - 2, y_of(event["peak"]) - 2, x + 2, y_of(event["peak"]) + 2,
                                       outline=colors["yours"])
            canvas.create_text(x, plot_h + 4, text="yours", anchor="n", fill=colors["yours"], font=kv.RULER_FONT)
            canvas.create_line(kv.GUTTER, y_of(self.yours["peak"]), width - 6, y_of(self.yours["peak"]),
                               fill=colors["yours"], dash=(5, 4))
        for name, color, files, total in groups:
            first = x + slot
            for summary in files:
                x += slot
                box(x, summary, color)
            if total:
                x += slot
                box(x, total, color, wide=True)
                canvas.create_text(x, plot_h + 4, text="all", anchor="n", fill=color, font=kv.RULER_FONT)
            canvas.create_text((first + x) / 2, plot_h + 16, text=name, anchor="n", fill=color, font=kv.RULER_FONT)
            canvas.create_line(x + slot / 2, 4, x + slot / 2, plot_h, fill="#424242")

    # ---------- ฟัง/เปิด ----------
    def _chosen(self):
        chosen = self.tree.selection()
        row = self.rows.get(chosen[0]) if chosen else None
        if row and row.get("path") is None and row.get("recording"):
            row["path"] = cached_path(row["recording"])
        return row if row and row.get("path") else None

    def _selected(self):
        row = self._chosen()
        if row and row.get("recording"):
            r = row["recording"]
            self.credit.set(f"{r['id']} by {r.get('recordist') or '?'} · {r.get('location') or ''} {r.get('date') or ''} "
                            f"· {r.get('license') or ''} · {r.get('page') or ''}")
        elif row:
            self.credit.set(str(row["path"]))

    def _play(self, samples, rate, name):
        """เล่นเสียงตัวอย่าง (หยุดเสียงเดิมของหน้าต่างนี้ก่อน)"""
        import platform_tools
        from audio_io import write_preview
        self.stop()
        target = CACHE / name
        write_preview(samples, rate, target)
        self.player = platform_tools.play_audio(target)

    def play_selected(self):
        row = self._chosen()
        if not row:
            self.state.set("Select a measured reference first")
            return
        from audio_io import read_mono
        main = main_calls(row.get("events") or [])
        first = max(0.0, (main[0]["start"] - 0.5) if main else 0.0)
        samples, rate = read_mono(row["path"], first, first + 15.0)
        self._play(samples, rate, "preview.wav")

    def play_mine(self):
        """เสียงของคุณ: ช่วงของกรอบที่เลือก (ไม่ได้เลือก = ทั้งคลิป) พร้อมเส้นตำแหน่งบนภาพ"""
        if self.clip is None:
            self.state.set("No recording to play here — open Compare Hz from the clip or Listen tools")
            return
        clip, chosen = self.clip, [c for c in self.candidates if c["on"]]
        if chosen:
            first = max(0.0, min(c["t0"] for c in chosen) - 0.3)
            last = min(clip["duration"], max(c["t1"] for c in chosen) + 0.3)
        else:
            first, last = 0.0, clip["duration"]
        last = min(last, first + self.PLAY_MINE_MAX)
        sr = clip["sr"]
        self._play(clip["samples"][int(first * sr):int(last * sr)], sr, "preview-mine.wav")
        self._playhead = (first, last, time.monotonic())
        self._tick_playhead()

    def _tick_playhead(self):
        self._playhead_job = None
        self.clip_canvas.delete("playhead")
        if self._playhead is None or self.player is None or self.player.poll() is not None:
            self._playhead = None
            return
        first, last, started = self._playhead
        at = first + time.monotonic() - started
        if at <= last:
            x0, plot_w, plot_h, _fmax = self._clip_geometry()
            x = x0 + at / max(self.clip["duration"], 1e-6) * plot_w
            self.clip_canvas.create_line(x, 0, x, plot_h, fill="#ffffff", width=2, tags="playhead")
        self._playhead_job = self.win.after(40, self._tick_playhead)

    def stop(self):
        if self.player is not None and self.player.poll() is None:
            self.player.terminate()
        self.player, self._playhead = None, None
        if self._playhead_job:
            self.win.after_cancel(self._playhead_job)
            self._playhead_job = None
        if self.clip is not None:
            self.clip_canvas.delete("playhead")

    def open_selected(self):
        row = self._chosen()
        if not row:
            return
        from audio_io import audio_duration
        from listening_tools import ListeningToolsWindow
        species = self.species[row["species"]]
        ListeningToolsWindow(self.win, row["path"], start=0.0, end=min(600.0, audio_duration(row["path"]) or 600.0),
                             species=(species["common"], species["scientific"]))

    def close(self):
        self._stop.set()
        if self._separator is not None and self._separator.poll() is None:
            self._separator.terminate()
        try:
            self.stop()
            for job in (self._poll_job, self._find_job, self._spectro_job):
                if job:
                    self.win.after_cancel(job)
            self.win.destroy()
        except tk.TclError:
            pass


class SpectrogramCompareWindow:
    """ภาพเสียงเรียงกันทีละแถว: เสียงของคุณ แล้วเสียงอ้างอิงของแต่ละชนิด — สเกลเวลาและช่วงความถี่เดียวกันทุกแถว
    ไว้ดูรูปร่างเสียงของชนิดที่ความถี่ใกล้กันมาก (เช่น Chestnut-flanked กับ Swinhoe's White-eye)
    แต่ละแถวปรับความสว่างเองตามย่านของเสียงที่วัด (เสียงไกล/ใกล้ดังไม่เท่ากัน และเสียงแมลงย่านอื่นไม่ทำให้เสียงนกมืด),
    กรอบ = เสียงที่วัด, เส้นประแดง = ความถี่ที่ดังสุดของเสียงคุณ,
    เส้นประสีของชนิด = ค่ากลางของชนิดนั้น; ไฟล์อ้างอิงเรียงจากใกล้ค่ากลางของชนิดนั้นที่สุด (ไม่ได้เรียงตามเสียงของคุณ)"""
    SECONDS = ("Auto", "1", "2", "4", "8")
    AUTO_SECONDS = (2.0, 5.0)                  # Auto: ช่วงของกรอบที่เลือก (+0.6 วินาที) แต่ไม่ต่ำ/เกินเท่านี้
    RANGES = ("Fit to the sounds", "0–12 kHz")
    ROW_H = 150
    LEVEL_DB = 70.0                            # ช่วงความดังที่แสดงใต้จุดที่ดังที่สุดของแถว

    def __init__(self, compare):
        self.compare, self.kv, self.rows = compare, compare.kv, []
        self._lut = compare.kv.palette(False)
        win = self.win = tk.Toplevel(compare.win)
        win.title("Compare spectrograms")
        win.geometry(f"1000x{min(110 + 200 * (1 + len(compare._active())), win.winfo_screenheight() - 80)}")
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        top = FlowFrame(body)
        top.pack(fill="x")
        top.add(ttk.Label(top, text="Seconds shown"))
        self.seconds = tk.StringVar(master=win, value=self.SECONDS[0])
        self.range = tk.StringVar(master=win, value=self.RANGES[0])
        for variable, values, width, label in ((self.seconds, self.SECONDS, 5, None),
                                               (self.range, self.RANGES, 15, "Frequency")):
            if label:
                top.add(ttk.Label(top, text=label), space=6)
            box = top.add(ttk.Combobox(top, textvariable=variable, values=values, state="readonly", width=width))
            box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        top.add(ttk.Button(top, text="■ Stop", command=compare.stop), space=6)
        top.add(playback.control(top), space=6)
        ttk.Label(body, text="Same time and frequency scale in every row · red dashed line = your peak · "
                  "boxes = the sounds that were measured", foreground="#808080", wraplength=960,
                  justify="left").pack(anchor="w", pady=(4, 0))
        self.body = ttk.Frame(body)
        self.body.pack(fill="both", expand=True, pady=(6, 0))
        win.protocol("WM_DELETE_WINDOW", self.close)
        self.build()

    def exists(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        if self.exists():
            self.win.destroy()

    # ---------- แถว ----------
    def build(self):
        """แถวตามชนิดที่เทียบอยู่ (เพิ่ม/ลบชนิดแล้วสร้างใหม่ โดยคงไฟล์อ้างอิงที่เลือกไว้)"""
        kept = {row["index"]: (row["ref"], row["call"]) for row in self.rows if row["kind"] == "species"}
        for widget in self.body.winfo_children():
            widget.destroy()
        self.rows = []
        self._add_row("yours", None, self.compare.COLORS["yours"])
        for number, index in enumerate(self.compare._active()):
            row = self._add_row("species", index, SPECIES_COLORS[number % len(SPECIES_COLORS)])
            row["ref"], row["call"] = kept.get(index, (None, 0))
        self.refresh()

    def _add_row(self, kind, index, color):
        frame = ttk.Frame(self.body)
        frame.pack(fill="both", expand=True, pady=(0, 6))
        head = ttk.Frame(frame)
        head.pack(fill="x")
        row = {"kind": kind, "index": index, "color": color, "ref": None, "call": 0, "audio": None, "boxes": [],
               "focus": None, "refs": [], "text": tk.StringVar(master=self.win)}
        ttk.Label(head, text="■", foreground=color).pack(side="left")
        ttk.Label(head, textvariable=row["text"]).pack(side="left", padx=4)
        ttk.Button(head, text="▶ Play", command=lambda: self.play(row)).pack(side="right")
        if kind == "species":
            ttk.Button(head, text="Next call ▸", command=lambda: self.step(row, 1)).pack(side="right", padx=4)
            ttk.Button(head, text="◂", width=2, command=lambda: self.step(row, -1)).pack(side="right")
            row["choice"] = tk.StringVar(master=self.win)
            row["box"] = ttk.Combobox(head, textvariable=row["choice"], state="readonly", width=30)
            row["box"].pack(side="right", padx=6)
            row["box"].bind("<<ComboboxSelected>>", lambda _event: self._chose(row))
        row["canvas"] = tk.Canvas(frame, bg="#000000", highlightthickness=0, height=self.ROW_H)
        row["canvas"].pack(fill="both", expand=True)
        row["canvas"].bind("<Configure>", lambda _event: self.draw(row))
        self.rows.append(row)
        return row

    def _seconds(self):
        if self.seconds.get() != "Auto":
            return float(self.seconds.get())
        chosen = [c for c in self.compare.candidates if c["on"]]
        shortest, longest = self.AUTO_SECONDS
        if chosen:
            span = max(c["t1"] for c in chosen) - min(c["t0"] for c in chosen) + 0.6
            return float(min(longest, max(shortest, span)))
        return 4.0

    def _summaries(self):
        found = {"yours": self.compare.yours}
        for index in self.compare._active():
            found[index] = overall([r for r in self.compare.rows.values() if r["species"] == index and r.get("events")])
        return found

    def _range(self):
        """(ต่ำ, สูง) Hz ที่แสดง — Fit: ครอบย่านเสียงของคุณและทุกชนิด"""
        sums = [s for s in self._summaries().values() if s]
        if self.range.get() != self.RANGES[0] or not sums:
            return 0.0, 12000.0
        low = min(min(s["lo_edge"], s["q1"]) for s in sums) * 0.6
        high = max(max(s["hi_edge"], s["q3"]) for s in sums) * 1.4
        low, high = np.floor(low / 500) * 500, np.ceil(high / 500) * 500
        if high - low < 3000:
            low, high = max(0.0, (low + high) / 2 - 1500), (low + high) / 2 + 1500
        return float(max(0.0, low)), float(high)

    def _references(self, index):
        """ไฟล์อ้างอิงที่วัดแล้วของชนิดนั้น เรียงจากใกล้ค่ากลางของชนิดที่สุด (ตัวอย่างที่ปกติที่สุดก่อน)"""
        rows = [(iid, r) for iid, r in self.compare.rows.items()
                if r["species"] == index and r.get("summary") and (r.get("path") or r.get("recording"))]
        pooled = overall([r for _iid, r in rows])
        middle = pooled["peak"] if pooled else 0.0
        return sorted(rows, key=lambda item: abs(item[1]["summary"]["peak"] - middle))

    def refresh(self):
        seconds = self._seconds()
        for row in self.rows:
            row["audio"], row["boxes"], row["focus"] = None, [], None
            try:
                (self._load_yours if row["kind"] == "yours" else self._load_species)(row, seconds)
            except Exception as exc:  # noqa: BLE001  ไฟล์เสีย/อ่านไม่ได้: แจ้งในแถวนั้น
                row["text"].set(f"Cannot read this sound: {exc}")
            self.draw(row)

    def _load_yours(self, row, seconds):
        compare, clip = self.compare, self.compare.clip
        if clip is None:
            row["text"].set("Your sound: no recording here — open Compare Hz from the clip or Listen tools")
            return
        chosen = [c for c in compare.candidates if c["on"]]
        center = (min(c["t0"] for c in chosen) + max(c["t1"] for c in chosen)) / 2 if chosen else seconds / 2
        start = min(max(0.0, center - seconds / 2), max(0.0, clip["duration"] - seconds))
        sr = clip["sr"]
        row["audio"] = (clip["samples"][int(start * sr):int((start + seconds) * sr)], sr)
        row["boxes"] = [(c["t0"] - start, c["t1"] - start, c["low"], c["high"]) for c in chosen]
        s = compare.yours
        row["focus"] = (s["lo_edge"], s["hi_edge"]) if s else None
        row["text"].set(f"Your sound — {compare.describe} · {start:.1f}–{start + seconds:.1f} s"
                        + (f" · peak {s['peak'] / 1000:.2f} kHz" if s else " · no box chosen (showing the start)")
                        + (f" · purity {s['purity']:.2f}" if s and s.get("purity") is not None else ""))

    def _load_species(self, row, seconds):
        from audio_io import read_mono
        species = self.compare.species[row["index"]]
        refs = row["refs"] = self._references(row["index"])
        labels = [f"{r['label']} · {r['summary']['peak'] / 1000:.2f} kHz" for _iid, r in refs]
        row["box"].configure(values=labels)
        if not refs:
            row["choice"].set("")
            row["text"].set(f"{species['common']}: no measured reference yet — Find references or Add my own files")
            return
        iids = [iid for iid, _r in refs]
        if row["ref"] not in iids:
            row["ref"], row["call"] = iids[0], 0
        position = iids.index(row["ref"])
        row["choice"].set(labels[position])
        ref = refs[position][1]
        calls = main_calls(ref["events"] or []) or ref["events"] or []
        row["call"] %= max(1, len(calls))
        call = calls[row["call"]] if calls else None
        start = max(0.0, ((call["start"] + call["end"]) / 2 if call else seconds / 2) - seconds / 2)
        samples, sr = read_mono(ref.get("path") or cached_path(ref["recording"]), start, start + seconds)
        row["audio"] = (samples, sr)
        row["boxes"] = [(e["start"] - start, e["end"] - start, e["low"], e["high"]) for e in calls
                        if e["end"] > start and e["start"] < start + seconds]
        row["focus"] = (ref["summary"]["lo_edge"], ref["summary"]["hi_edge"])
        recording = ref.get("recording") or {}
        kind = ", ".join(str(v) for v in (recording.get("type"), recording.get("country")) if v)
        row["text"].set(f"{species['common']}: {ref['label']}" + (f" ({kind})" if kind else "")
                        + f" · call {row['call'] + 1} of {len(calls)} · peak {ref['summary']['peak'] / 1000:.2f} kHz"
                        + (f" · purity {ref['summary']['purity']:.2f}" if ref["summary"].get("purity") is not None else ""))

    def _chose(self, row):
        labels = list(row["box"].cget("values"))
        if row["choice"].get() in labels:
            row["ref"], row["call"] = row["refs"][labels.index(row["choice"].get())][0], 0
            self._reload(row)

    def step(self, row, step):
        row["call"] += step
        self._reload(row)

    def _reload(self, row):
        row["audio"], row["boxes"], row["focus"] = None, [], None
        try:
            self._load_species(row, self._seconds())
        except Exception as exc:  # noqa: BLE001
            row["text"].set(f"Cannot read this sound: {exc}")
        self.draw(row)

    def play(self, row):
        if row["audio"] is not None and len(row["audio"][0]):
            self.compare._play(*row["audio"], f"preview-row{self.rows.index(row)}.wav")

    # ---------- วาด ----------
    def draw(self, row):
        kv, canvas = self.kv, row["canvas"]
        canvas.delete("all")
        width, height = max(200, canvas.winfo_width()), max(80, canvas.winfo_height())
        x0, plot_w, plot_h = kv.GUTTER, max(50, width - kv.GUTTER - 6), height - 16
        low, high = self._range()
        seconds = self._seconds()

        def x_of(t):
            return x0 + t / seconds * plot_w

        def y_of(hz):
            return plot_h * (1 - (hz - low) / (high - low))
        if row["audio"] is not None and len(row["audio"][0]) >= kv.FFT_SIZE * 2:
            samples, sr = row["audio"]
            db = kv.spectrum_db(samples)
            freqs = np.arange(db.shape[1]) * sr / kv.FFT_SIZE
            band = (freqs >= low) & (freqs <= min(high, sr / 2))
            if row["focus"]:                                         # ความสว่างตามย่านของเสียงที่วัด
                focus = band & (freqs >= row["focus"][0] * 0.8) & (freqs <= row["focus"][1] * 1.2)
                band = focus if focus.any() else band
            loudest = float(np.percentile(db[:, band], 99.8)) if band.any() else 0.0
            rows_hz = np.linspace(high, low, plot_h)
            columns = max(1, int(round(plot_w * min(1.0, len(samples) / sr / seconds))))
            rgb = kv.spectrogram_rgb(kv.column_pool(db, columns), rows_hz, sr, self._lut, brightness=-loudest,
                                     floor=-self.LEVEL_DB)
            rgb[rows_hz > sr / 2] = 0                                # เหนือ Nyquist ของไฟล์นั้น = ไม่มีข้อมูล
            row["photo"] = ImageTk.PhotoImage(Image.fromarray(rgb, "RGB").resize((columns, plot_h), Image.BILINEAR),
                                              master=self.win)
            canvas.create_image(x0, 0, anchor="nw", image=row["photo"])
        fg = "#ffffff"
        step = kv.nice_step(high - low, plot_h, 24)
        for hz in np.arange(np.ceil(low / step) * step, high + 1, step):
            y = y_of(hz)
            if 8 <= y <= plot_h:
                canvas.create_line(x0 - 6, y, x0, y, fill=fg)
                canvas.create_text(x0 - 8, y, text=kv.freq_label(hz), anchor="e", fill=fg, font=kv.RULER_FONT)
        tick = kv.nice_step(seconds, plot_w, 70)
        for t in np.arange(0.0, seconds + 1e-9, tick):
            canvas.create_line(x_of(t), plot_h, x_of(t), plot_h + 4, fill=fg)
            canvas.create_text(x_of(t) + 2, plot_h + 1, text=kv.time_label(t), anchor="nw", fill=fg,
                               font=("Helvetica", 10))
        for t0, t1, lo, hi in row["boxes"]:
            if t1 <= 0 or t0 >= seconds:                             # อยู่นอกช่วงที่แสดง
                continue
            canvas.create_rectangle(x_of(max(0.0, t0)), y_of(min(high, hi * 1.05)), x_of(min(seconds, t1)),
                                    y_of(max(low, lo * 0.95)), outline="#ffeb3b" if row["kind"] == "yours" else row["color"],
                                    dash=(3, 3))
        summary = self._summaries().get(row["index"]) if row["kind"] == "species" else None
        if summary and low <= summary["peak"] <= high:
            canvas.create_line(x0, y_of(summary["peak"]), x0 + plot_w, y_of(summary["peak"]), fill=row["color"],
                               dash=(2, 4))
        yours = self.compare.yours
        if yours and low <= yours["peak"] <= high:
            canvas.create_line(x0, y_of(yours["peak"]), x0 + plot_w, y_of(yours["peak"]),
                               fill=self.compare.COLORS["yours"], dash=(6, 4))
