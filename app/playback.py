"""เล่นเสียงและระดับเสียงของแอป: ค่าเดียวใช้ทุกหน้า จำไว้ใน settings.json ปรับระหว่างเล่นได้ (เฉพาะเสียงที่ฟัง ไม่แตะไฟล์คลิป)

ไม่เกิน 100% = ลดด้วยตัวเล่น (afplay / ffplay) เกิน 100% = ขยายสัญญาณใส่ไฟล์ชั่วคราวแล้วค่อย ๆ อัดส่วนที่ดังเกินไม่ให้แตก
afplay/ffplay เปลี่ยนเสียงกลางคันไม่ได้และกระโดดไปกลางไฟล์ไม่ได้: ปรับแล้วตัดไฟล์ตั้งแต่จุดที่เล่นถึง และเริ่มตัวเล่นใหม่ต่อจากตรงนั้น
"""

import itertools
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import paths
import platform_tools
from app_settings import load_settings, save_settings

SETTING = "playback_volume"
DEFAULT, MAXIMUM, STEP = 100, 300, 10        # เปอร์เซ็นต์
SETTLE_MS = 200                              # หยุดลากแถบเท่านี้แล้วค่อยเล่นต่อที่ระดับใหม่และบันทึก
KNEE = 0.8                                   # ต่ำกว่านี้เสียงไม่เปลี่ยนรูป สูงกว่านี้อัดเข้าหา 1
FOLDER = paths.CACHE / "playback"

_percent = None                              # None = ยังไม่ได้อ่านจาก settings
_playing = []
_numbers = itertools.count(1)


def volume():
    """ระดับเสียงปัจจุบัน (เปอร์เซ็นต์)"""
    global _percent
    if _percent is None:
        try:
            _percent = max(0, min(MAXIMUM, int(load_settings().get(SETTING, DEFAULT))))
        except (TypeError, ValueError):
            _percent = DEFAULT
    return _percent


def set_volume(percent):
    global _percent
    _percent = max(0, min(MAXIMUM, int(percent)))


def _limited(samples):
    """ส่วนที่เกิน KNEE ค่อย ๆ อัดเข้าหา 1 ไม่ให้แตก ส่วนที่เบากว่านั้นคงเดิม"""
    import numpy as np
    level = np.abs(samples)
    room = 1 - KNEE
    return np.where(level > KNEE, np.sign(samples) * (KNEE + room * np.tanh((level - KNEE) / room)), samples)


def _rendered(path, start, percent):
    """เสียงตั้งแต่ start วินาที ขยายถ้าเกิน 100% -> ไฟล์ชั่วคราว (None = ไม่มีเสียงเหลือ)"""
    import soundfile as sf
    with sf.SoundFile(str(path)) as audio:
        rate = audio.samplerate
        audio.seek(min(int(start * rate), audio.frames))
        samples = audio.read(dtype="float32")
    if not len(samples):
        return None
    if percent > 100:
        samples = _limited(samples * percent / 100)
    FOLDER.mkdir(parents=True, exist_ok=True)
    for old in FOLDER.glob("play-*.wav"):        # รอบก่อน (Windows: ลบไม่ได้ถ้ายังเล่นอยู่ — ข้าม)
        try:
            old.unlink()
        except OSError:
            pass
    target = FOLDER / f"play-{next(_numbers)}.wav"
    sf.write(str(target), samples, rate, subtype="FLOAT")
    return target


class Playback:
    """เสียงที่กำลังเล่น ใช้แทน Popen ได้ (poll, terminate, returncode) เปลี่ยนตัวเล่นข้างในได้โดยหน้าที่เปิดอยู่ไม่รู้ตัว"""

    def __init__(self, path):
        self.path = Path(path)
        percent = volume()
        target = self.path
        if percent > 100:
            try:
                target = _rendered(self.path, 0.0, percent) or self.path
            except (OSError, RuntimeError, ValueError):  # ไฟล์ที่อ่านเองไม่ได้: เล่นตามปกติ
                pass
        self.process = platform_tools.start_player(target, min(percent, 100) / 100)
        self.offset, self.began = 0.0, time.monotonic()

    def poll(self):
        return self.process.poll()

    def terminate(self):
        self.process.terminate()

    @property
    def returncode(self):
        return self.process.returncode

    def position(self):
        return self.offset + time.monotonic() - self.began

    def follow_volume(self):
        """เล่นต่อจากจุดเดิมที่ระดับเสียงปัจจุบัน — ตัดไฟล์ไม่ได้ก็เล่นต่อแบบเดิม ระดับใหม่ใช้กับเสียงครั้งหน้า"""
        if self.poll() is not None:
            return
        at, percent = self.position(), volume()
        try:
            target = _rendered(self.path, at, percent)
        except (OSError, RuntimeError, ValueError):
            return
        if target is None:
            return
        self.process.terminate()
        self.process = platform_tools.start_player(target, min(percent, 100) / 100)
        self.offset, self.began = at, time.monotonic()


def play(path):
    """เล่นเสียงเบื้องหลัง -> Playback"""
    _playing[:] = [sound for sound in _playing if sound.poll() is None]
    sound = Playback(path)
    _playing.append(sound)
    return sound


def follow_volume():
    for sound in _playing:
        sound.follow_volume()


class _Level:
    """ระดับเสียงของหน้าต่างทุกบาน: แถบทุกอันผูกตัวแปรเดียวกัน จึงขยับตามกันเอง"""

    def __init__(self, root):
        self.root, self._job = root, None
        self.value = tk.DoubleVar(master=root, value=volume())
        self.text = tk.StringVar(master=root, value=f"{volume()}%")
        self.value.trace_add("write", self._changed)

    def step(self, change):
        self.value.set(max(0, min(MAXIMUM, round(self.value.get() / STEP) * STEP + change)))

    def _changed(self, *_args):
        set_volume(round(self.value.get()))
        self.text.set(f"{volume()}%")
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(SETTLE_MS, self._settled)

    def _settled(self):
        self._job = None
        follow_volume()
        save_settings({SETTING: volume()})


def control(master, length=64):
    """Vol  −  ━━●━━  +  120%  Normal — ใช้ค่าเดียวกันทุกหน้า ไม่ดึง focus (ปุ่มขึ้น/ลงในรายการคลิปยังใช้ได้)"""
    root = master.winfo_toplevel()._root()
    level = getattr(root, "_playback_level", None)
    if level is None:
        level = root._playback_level = _Level(root)
    frame = ttk.Frame(master)
    ttk.Label(frame, text="Vol").pack(side="left", padx=(0, 4))
    ttk.Button(frame, text="−", width=1, takefocus=0, command=lambda: level.step(-STEP)).pack(side="left")
    ttk.Scale(frame, from_=0, to=MAXIMUM, length=length, variable=level.value, takefocus=0).pack(side="left", padx=2)
    ttk.Button(frame, text="+", width=1, takefocus=0, command=lambda: level.step(STEP)).pack(side="left")
    ttk.Label(frame, textvariable=level.text, width=5).pack(side="left", padx=(4, 0))
    ttk.Button(frame, text="Normal", width=6, takefocus=0, command=lambda: level.value.set(DEFAULT)).pack(side="left")
    return frame
