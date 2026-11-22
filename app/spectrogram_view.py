"""ตัวดูคลิปแบบ Macaulay Library ในแท็บ Review: spectrogram ขาวดำ ความถี่เชิงเส้น 0-12 kHz เห็นทีละ ~6 วินาที
(ย่อ/ขยาย 1.5-60 วิ) + แถบภาพรวมทั้งคลิป (คลิก/ลากเพื่อเลื่อน) + แถบความคืบหน้า + เวลา
ขณะเล่น เส้นตำแหน่งอยู่กลางภาพ ภาพเลื่อนผ่านเส้น

หักเสียงพื้นหลังของแต่ละความถี่ออกก่อน (ลม รถ เสียงความถี่ต่ำ) ให้เสียงนกเด่นบนพื้นขาว
ภาพ .png ของคลิป (มีกรอบของ BirdNET) เปิดได้จากปุ่ม BirdNET boxes ในแท็บ Review
"""

import math
import queue
import threading
import time
from datetime import timedelta

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from manual_clips import format_offset
from manual_window import read_mono

WINDOW_S = 6.0
FMAX = 12000
LEFT, AXIS_H = 34, 18
FLOOR_DB = 7.0           # ดังกว่าพื้นหลังน้อยกว่านี้ = ขาว
MIN_TOP_DB = 24.0        # ดังกว่าพื้นหลังเท่านี้ (หรือมากกว่า) = ดำ
ZOOM_LEVELS = (60.0, 30.0, 15.0, 10.0, 6.0, 3.0, 1.5)   # วินาทีที่เห็นในภาพ: ปุ่ม − ไปซ้าย (เห็นมากขึ้น), + ไปขวา
BAR_H = 6                # แถบความคืบหน้าใต้ภาพรวม (ฟ้า = เล่นไปแล้ว)
FRAME_MS = 40


def clip_db(samples, sr, fmax=FMAX):
    """spectrogram ทั้งคลิป -> (dB แถวบน = ความถี่สูง, วินาทีต่อคอลัมน์, ความถี่สูงสุดในภาพ)"""
    n_fft = 1024 if sr <= 48000 else 2048     # ~21 ms ที่ 48 kHz: โน้ตเร็ว ๆ ไม่เบลอตามเวลา
    hop = n_fft // 4
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    count = 1 + (samples.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    keep = np.fft.rfftfreq(n_fft, 1 / sr) <= min(fmax, sr / 2)
    columns = []
    for first in range(0, count, 512):
        index = np.arange(first, min(count, first + 512))[:, None] * hop + np.arange(n_fft)
        spectrum = np.abs(np.fft.rfft(samples[index] * window, axis=1))[:, keep]
        columns.append(20 * np.log10(spectrum.astype(np.float32) + 1e-9))
    db = np.concatenate(columns).T[::-1]
    return db, hop / sr, min(fmax, sr / 2)


def _smooth(db):
    """เฉลี่ย 3x3 จุด: ลดจุดเม็ดฝุ่นของเสียงพื้นหลัง เสียงนกยังคมอยู่"""
    padded = np.pad(db, 1, mode="edge")
    rows, cols = db.shape
    return sum(padded[i:i + rows, j:j + cols] for i in range(3) for j in range(3)) / 9


def whiten(db):
    """ดังกว่าพื้นหลังของความถี่นั้นกี่ dB (พื้นหลัง = ค่ากลางของแต่ละแถว) + ค่าที่จะให้เป็นสีดำ"""
    db = _smooth(db)
    relative = db - np.median(db, axis=1, keepdims=True)
    top = max(MIN_TOP_DB, float(np.percentile(relative, 99.7)))
    return relative, top


def gray_image(relative, top, width, height, pool=False):
    """dB เหนือพื้นหลัง -> ภาพขาวดำ (เสียงดัง = ดำ); pool = ย่อคอลัมน์แบบเก็บค่าดังสุด (แถบภาพรวม)"""
    if pool and relative.shape[1] > width:
        edges = np.linspace(0, relative.shape[1], width, endpoint=False).astype(int)
        relative = np.maximum.reduceat(relative, edges, axis=1)
    level = np.clip((relative - FLOOR_DB) / (top - FLOOR_DB), 0, 1) ** 0.8
    pixels = (255 * (1 - level)).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(pixels), "L").resize((width, height), Image.Resampling.BILINEAR)


def gray_pixels(relative, top):
    """dB เหนือพื้นหลัง -> ค่าเทา (0 = ดำ = ดัง) ทั้งคลิป คำนวณครั้งเดียวตอนโหลด"""
    level = np.clip((relative - FLOOR_DB) / (top - FLOOR_DB), 0, 1) ** 0.8
    return (255 * (1 - level)).astype(np.uint8)


def render(pixels, width, height):
    """ค่าเทา (แถว = ความถี่, คอลัมน์ = เวลา) -> ภาพ width x height; คอลัมน์มากกว่าภาพ = เก็บจุดเข้มสุด
    (เสียงสั้น ๆ ไม่หายตอนย่อ)"""
    if pixels.shape[1] > width:
        edges = np.linspace(0, pixels.shape[1], width, endpoint=False).astype(int)
        pixels = np.minimum.reduceat(pixels, edges, axis=1)
    return Image.fromarray(np.ascontiguousarray(pixels), "L").resize((width, height), Image.Resampling.BILINEAR)


def merge_spans(spans, gap=0.5):
    """ช่วงที่ BirdNET เจอ (หน้าต่าง 3 วิ ซ้อนกัน) -> ช่วงต่อเนื่องที่ไม่ซ้อนกัน [(t0, t1)] เรียงตามเวลา"""
    merged = []
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def time_steps(window):
    """วินาทีที่เห็น -> (ระยะตัวเลขเวลา, ระยะขีดย่อย) แบบ Macaulay: ตัวเลขทุกวินาทีเมื่อเห็น 2.5-15 วิ"""
    if window < 2.5:
        return 0.5, 0.125
    for step, minor in ((1, 0.25), (2, 0.5), (5, 1), (10, 2), (30, 10)):
        if window / step <= 15:
            return step, minor
    return 60, 10


class SpectrogramView(ttk.Frame):
    """on_seek(seconds) เรียกเมื่อคลิกบนภาพหรือแถบความคืบหน้า (ให้แอปเล่นจากจุดนั้น)
    ขณะเล่น: เส้นตำแหน่งอยู่กลางภาพ ภาพเลื่อนผ่านเส้น (แบบ Macaulay); ต้นและท้ายคลิปเส้นเดินแทน"""

    def __init__(self, master, width=840, height=165, overview_height=30, on_seek=None, only_heard=None,
                 app=None, on_select=None, listening_tools=True):
        """only_heard = BooleanVar ที่ใช้ร่วมกันระหว่างตัวดู (แท็บ Review กับหน้าอื่น): เปิด = เห็นและเล่น
        เฉพาะช่วงที่ BirdNET ได้ยินชนิดนี้ ต่อกัน (เส้นแดง = รอยต่อ)"""
        super().__init__(master)
        self.width, self.height, self.overview_height = width, height, overview_height
        self.on_seek = on_seek
        self.app, self.on_select = app, on_select
        self.source_path = None
        self.source_start, self.source_end = 0.0, 3600.0
        self._selection = self._drag_origin = None
        self.q = queue.Queue()
        self._load_id = 0
        self._data = None             # ภาพที่แสดง (ค่าเทา, วินาทีต่อคอลัมน์, fmax, ความยาวที่แสดง)
        self._full = None             # ทั้งคลิป รูปแบบเดียวกัน
        self._segments = []           # [(วินาทีจริงเริ่ม, จบ, วินาทีในภาพที่เริ่ม)] — ทั้งคลิป = ช่วงเดียว
        self.view_start = 0.0
        self.clock_start = None
        self.spans = []
        self.flags = []               # flag ที่กดตอนอัด: วินาทีนับจากต้นเสียงที่แสดง (recorder_flags.py)
        self._zoom = ZOOM_LEVELS.index(WINDOW_S)
        self._position = 0.0          # ตำแหน่งที่เล่นถึง (แถบฟ้า + เวลา)
        self._photo = self._overview_dim = self._overview_bright = self._overview_photo = None
        self._window_photo = None
        self._play = None             # (ตำแหน่งเริ่ม, เวลาเครื่องตอนเริ่ม, ตัวเล่นเสียง)
        self._tick_job = self._poll_job = None

        # แถวล่าง: เวลา · ย่อ/ขยาย · รายละเอียด
        row = ttk.Frame(self)
        row.pack(side="bottom", fill="x", pady=(3, 0))
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(row, textvariable=self.time_var, width=11).pack(side="left")
        ttk.Label(row, text="Zoom").pack(side="left", padx=(8, 2))
        ttk.Button(row, text="−", width=2, command=lambda: self.zoom(-1)).pack(side="left")
        self.zoom_var = tk.DoubleVar(value=self._zoom)
        ttk.Scale(row, from_=0, to=len(ZOOM_LEVELS) - 1, variable=self.zoom_var, length=90,
                  command=self._zoom_slid).pack(side="left", padx=3)
        ttk.Button(row, text="+", width=2, command=lambda: self.zoom(1)).pack(side="left")
        self.only_heard = only_heard if only_heard is not None else tk.BooleanVar(master=self, value=False)
        self._only_box = ttk.Checkbutton(row, text="BirdNET parts only", variable=self.only_heard)
        self._only_box.pack(side="left", padx=(10, 0))
        self._trace = self.only_heard.trace_add("write", lambda *_args: self._mode_changed())
        self.info = tk.StringVar()
        ttk.Label(row, textvariable=self.info, foreground="#808080").pack(side="left", padx=(10, 0))
        if listening_tools:
            ttk.Button(row, text="Listen tools…", command=self.open_listening_tools).pack(side="right", padx=4)
            ttk.Button(row, text="Compare Hz…", command=self.open_compare).pack(side="right")
        # แถบความคืบหน้า (ฟ้า = เล่นไปแล้ว) และภาพรวมทั้งคลิป (ช่วงที่เห็นอยู่สว่าง ที่เหลือเทา)
        self.progress = tk.Canvas(self, width=LEFT + width, height=BAR_H, bg="white", highlightthickness=0,
                                  cursor="hand2")
        self.progress.pack(side="bottom", fill="x")
        self.overview = tk.Canvas(self, width=LEFT + width, height=overview_height + 4, bg="white",
                                  highlightthickness=0, cursor="sb_h_double_arrow")
        self.overview.pack(side="bottom", fill="x", pady=(2, 0))
        # ยืดหดตามพื้นที่ (หน้าต่างใหญ่ = ภาพใหญ่): ขอขนาดเล็กไว้ก่อน แล้ววาดใหม่ตามขนาดจริง
        self.main = tk.Canvas(self, width=LEFT + width, height=AXIS_H + 90, bg="white",
                              highlightthickness=0, cursor="hand2")
        self.main.pack(side="top", fill="both", expand=True)
        self.main.bind("<Configure>", self._resized)
        self._resize_job = None
        if on_select:
            self.main.bind("<Button-1>", self._select_press)
            self.main.bind("<B1-Motion>", self._select_drag)
            self.main.bind("<ButtonRelease-1>", self._select_release)
        else:
            self.main.bind("<Button-1>", self._clicked)
        self.main.bind("<Motion>", self._hover)
        self.main.bind("<Leave>", lambda _event: self._describe())
        self.main.bind("<Left>", lambda _event: self.page(-1))
        self.main.bind("<Right>", lambda _event: self.page(1))
        for key, step in (("<minus>", -1), ("<equal>", 1), ("<plus>", 1)):
            self.main.bind(key, lambda _event, step=step: self.zoom(step))
        self.overview.bind("<Button-1>", self._overview_drag)
        self.overview.bind("<B1-Motion>", self._overview_drag)
        self.progress.bind("<Button-1>", self._progress_clicked)
        self.message("Select a clip to see its sound")
        self._poll_job = self.after(100, self._poll)

    def destroy(self):
        self.stop_playhead()
        for job in (self._poll_job, self._resize_job):
            if job:
                self.after_cancel(job)
        try:
            self.only_heard.trace_remove("write", self._trace)
        except (tk.TclError, ValueError):
            pass
        super().destroy()

    def _resized(self, event):
        width, height = max(200, event.width - LEFT - 2), max(60, event.height - AXIS_H)
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._redraw)

    def _redraw(self):
        self._resize_job = None
        if self._data:
            self._draw_overview()
            self._draw_view()

    # ---------- โหลด ----------
    def message(self, text):
        self.source_path = None
        self._load_id += 1
        self._data = self._full = None
        self._segments = []
        self.stop_playhead()
        for canvas in (self.main, self.overview, self.progress):
            canvas.delete("all")
        self.main.create_text(LEFT + self.width / 2, AXIS_H + self.height / 2, text=text, fill="#888")
        self.info.set("")
        self.time_var.set("0:00 / 0:00")

    def load(self, path, clock_start=None, spans=(), start=0.0, end=3600.0, channel="mix"):
        """คลิป (wav) + เวลาจริงของต้นคลิป + ช่วงที่ BirdNET เจอ [(t0, t1)] วินาทีนับจากต้นคลิป"""
        self.message("Loading …")
        self.source_path, self.source_start, self.source_end = path, start, end
        self._selection = None
        load_id, self.clock_start, self.spans, self.view_start = self._load_id, clock_start, list(spans), 0.0
        self._position = 0.0
        self._only_box.state(["!disabled"] if self.spans else ["disabled"])   # คลิปที่เพิ่มเอง: ไม่มีช่วงของ BirdNET

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            try:
                if channel == "mix" and start == 0 and end == 3600:
                    samples, sr = read_mono(path, 0.0, 3600.0)
                else:
                    from audio_io import read_channels
                    samples, sr = read_channels(path, start, end, channel)
                    if samples.ndim == 2:
                        samples = samples.mean(axis=1)
                db, seconds_per_column, fmax = clip_db(samples, sr)
                pixels = gray_pixels(*whiten(db))
                results.put((load_id, (pixels, seconds_per_column, fmax, len(samples) / sr), None))
            except Exception as exc:  # noqa: BLE001
                results.put((load_id, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                load_id, data, error = self.q.get_nowait()
                if load_id != self._load_id:
                    continue
                if error:
                    self.message(f"Cannot show this clip: {error}")
                    continue
                self._full = data
                self._build_display()
                self._draw_overview()
                self._draw_view()
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    # ---------- วาด ----------
    @property
    def duration(self):
        """ความยาวที่แสดง (เปิด Only where BirdNET heard it = รวมเฉพาะช่วงที่ได้ยิน)"""
        return self._data[3] if self._data else 0.0

    @property
    def real_duration(self):
        return self._full[3] if self._full else 0.0

    @property
    def compressed(self):
        return bool(self._full) and self._data is not self._full

    def _build_display(self):
        pixels, seconds_per_column, fmax, duration = self._full
        segments, pieces, at = [], [], 0.0
        for a, b in (merge_spans(self.spans) if self.only_heard.get() else []):
            first = max(0, int(a / seconds_per_column))
            last = min(pixels.shape[1], int(math.ceil(b / seconds_per_column)))
            if last > first:
                segments.append((first * seconds_per_column, last * seconds_per_column, at))
                pieces.append(pixels[:, first:last])
                at += (last - first) * seconds_per_column
        if segments:
            self._segments, self._data = segments, (np.concatenate(pieces, axis=1), seconds_per_column, fmax, at)
        else:
            self._segments, self._data = [(0.0, duration, 0.0)], self._full

    def to_real(self, shown):
        """วินาทีในภาพ -> วินาทีจริงในคลิป"""
        for start, end, at in reversed(self._segments):
            if shown >= at:
                return min(end, start + shown - at)
        return shown

    def to_shown(self, real):
        """วินาทีจริง -> วินาทีในภาพ หรือ None (ช่วงที่ตัดออก)"""
        for start, end, at in self._segments:
            if start - 1e-9 <= real <= end + 1e-9:
                return at + real - start
        return None

    def audio_plan(self, shown):
        """ช่วงของคลิปจริงที่ต้องเล่นต่อกัน ตั้งแต่ตำแหน่ง shown ในภาพ -> [(t0, t1)]"""
        return [(start + max(0.0, shown - at), end) for start, end, at in self._segments
                if shown < at + end - start]

    def _mode_changed(self):
        if not self._full:
            return
        if self._play is not None:                # เสียงที่เล่นอยู่ไม่ตรงกับภาพใหม่แล้ว: หยุด
            player = self._play[2]
            if player is not None and player.poll() is None:
                player.terminate()
            self.stop_playhead()
        middle = self.to_real(self.view_start + self.window / 2)
        position = self.to_real(self._position)
        self._build_display()
        shown = self.to_shown(middle)
        self._position = self.to_shown(position) or 0.0
        self.view_start = min(max(0.0, (shown if shown is not None else 0.0) - self.window / 2),
                              max(0.0, self.duration - self.window))
        self._draw_overview()
        self._draw_view()

    @property
    def window(self):
        return min(ZOOM_LEVELS[self._zoom], self.duration) or ZOOM_LEVELS[self._zoom]

    def _x(self, seconds):
        return LEFT + (seconds - self.view_start) / self.window * self.width

    def _draw_view(self):
        pixels, seconds_per_column, fmax, duration = self._data
        first = int(self.view_start / seconds_per_column)
        last = max(first + 1, int((self.view_start + self.window) / seconds_per_column))
        self._photo = ImageTk.PhotoImage(render(pixels[:, first:last], self.width, self.height), master=self)
        canvas = self.main
        canvas.delete("all")
        canvas.create_image(LEFT, AXIS_H, anchor="nw", image=self._photo)
        for khz in range(2, int(fmax / 1000) + 1, 2):                       # แกนความถี่ (kHz)
            y = AXIS_H + self.height - khz * 1000 / fmax * self.height
            canvas.create_line(LEFT - 5, y, LEFT, y, fill="#999")
            canvas.create_text(LEFT - 7, y, text=str(khz), anchor="e", fill="#555", font=("Helvetica", 10))
        canvas.create_text(LEFT - 7, AXIS_H + self.height - 2, text="kHz", anchor="se", fill="#777",
                           font=("Helvetica", 9))
        step, minor = time_steps(self.window)                    # แกนเวลา (เวลาจริงในคลิป): ตัวเลข + ขีดย่อย
        view_end = self.view_start + self.window
        for start, end, at in self._segments:
            low, high = max(self.view_start, at), min(view_end, at + end - start)
            if high < low:
                continue
            index = math.ceil((start + low - at) / minor - 1e-9)
            while index * minor <= start + high - at + 1e-9:
                tick = index * minor
                x = self._x(at + tick - start)
                if abs(tick / step - round(tick / step)) < 1e-6:
                    canvas.create_line(x, 0, x, AXIS_H - 2, fill="#999")
                    label = format_offset(tick, 1 if step < 1 and round(tick * 2) % 2 else 0)
                    canvas.create_text(x + 3, 1, text=label, anchor="nw", fill="#555", font=("Helvetica", 10))
                else:
                    canvas.create_line(x, AXIS_H - 6, x, AXIS_H - 2, fill="#bbb")
                index += 1
            if at > 0 and self.view_start <= at <= view_end:               # รอยต่อ: ตัดช่วงที่ไม่ได้ยินออก
                x = self._x(at)
                canvas.create_line(x, 0, x, AXIS_H + self.height, fill="#d9534f")
        for a, b in self.spans:                                              # ช่วงที่ BirdNET เจอ
            for start, end, at in self._segments:
                low, high = max(float(a), start), min(float(b), end)
                if high > low and at + high - start > self.view_start and at + low - start < view_end:
                    canvas.create_rectangle(max(LEFT, self._x(at + low - start)), AXIS_H - 4,
                                            min(LEFT + self.width, self._x(at + high - start)), AXIS_H - 1,
                                            fill="#5b8def", outline="")
        for flag in self.flags:
            shown = self.to_shown(flag)
            if shown is not None and self.view_start <= shown <= view_end:
                canvas.create_line(self._x(shown), AXIS_H, self._x(shown), AXIS_H + self.height, fill="#e53935",
                                   width=2, dash=(6, 3))
                canvas.create_text(self._x(shown) + 3, AXIS_H + 2, text="⚑", anchor="nw", fill="#e53935",
                                   font=("Helvetica", 13))
        self._draw_overview_window()
        self._draw_progress()
        self._draw_selection()
        if self._play is None:
            self._describe()

    def _draw_overview(self):
        pixels = self._data[0]
        bright = render(pixels, self.width, self.overview_height)
        self._overview_bright = bright
        self._overview_dim = Image.fromarray((np.asarray(bright) * 0.86).astype(np.uint8), "L")  # นอกช่วงที่เห็น: เทา
        self._overview_photo = ImageTk.PhotoImage(self._overview_dim, master=self)
        self.overview.delete("all")
        self.overview.create_image(LEFT, 2, anchor="nw", image=self._overview_photo)

    def _draw_overview_window(self):
        self.overview.delete("window")
        if not self.duration or self._overview_bright is None:
            return
        scale = self.width / self.duration
        x0 = int(round(self.view_start * scale))
        x1 = max(x0 + 1, int(round((self.view_start + self.window) * scale)))
        x1 = min(x1, self._overview_bright.width)
        self._window_photo = ImageTk.PhotoImage(self._overview_bright.crop((x0, 0, x1, self.overview_height)),
                                                master=self)
        self.overview.create_image(LEFT + x0, 2, anchor="nw", image=self._window_photo, tags="window")
        self.overview.create_rectangle(LEFT + x0, 1, LEFT + x1, self.overview_height + 3, outline="#1f5f8b",
                                       tags="window")
        self.overview.tag_raise("playhead")

    def _draw_progress(self):
        canvas = self.progress
        canvas.delete("all")
        if not self.duration:
            return
        played = LEFT + min(1.0, self._position / self.duration) * self.width
        canvas.create_rectangle(LEFT, 0, LEFT + self.width, BAR_H, fill="#8c8c8c", outline="")
        if played > LEFT:
            canvas.create_rectangle(LEFT, 0, played, BAR_H, fill="#a6d1ea", outline="")
        # ปัดลงเป็นวินาทีเต็ม (59.6 ไม่กลายเป็น 0:60)
        self.time_var.set(f"{format_offset(int(self.to_real(self._position)), 0)} / "
                          f"{format_offset(int(self.real_duration), 0)}")

    def _describe(self, offset=None):
        if not self.duration:
            return
        start, end = self.to_real(self.view_start), self.to_real(min(self.view_start + self.window, self.duration))
        text = f"{format_offset(start, 0)}–{format_offset(end, 0)} of {format_offset(self.real_duration, 0)}"
        if self.compressed:
            text += f" (heard {format_offset(self.duration, 0)})"
        real = None if offset is None else self.to_real(offset)
        if self.clock_start is not None:
            clock = self.clock_start + timedelta(seconds=start if real is None else real)
            text += f"  ·  clock {clock:%H:%M:%S}" + (f".{clock.microsecond // 100000}" if real is not None else "")
        if real is not None:
            text += f"  ·  cursor {format_offset(real)}"
        elif not self.compressed:                 # สั้นไว้: ช่วงที่ได้ยินยาวแค่ไหนสำคัญกว่าคำแนะนำ
            text += "  ·  click to play from there"
        self.info.set(text)

    # ---------- เลื่อน/ย่อขยาย/คลิก ----------
    def set_view(self, start):
        if not self._data:
            return
        start = min(max(0.0, start), max(0.0, self.duration - self.window))
        if abs(start - self.view_start) > 1e-3:
            self.view_start = start
            self._draw_view()

    def page(self, direction):
        self.set_view(self.view_start + direction * self.window * 0.8)

    def zoom(self, step):
        """step -1 = ย่อ (เห็นหลายวินาที), +1 = ขยาย"""
        self.set_zoom(self._zoom + step)

    def set_zoom(self, index):
        index = min(max(0, int(index)), len(ZOOM_LEVELS) - 1)
        self.zoom_var.set(index)
        if index == self._zoom:
            return
        # ขยายรอบเส้นตำแหน่งถ้ากำลังเล่น ไม่งั้นรอบกลางภาพ
        center = self._position if self._play else self.view_start + self.window / 2
        self._zoom = index
        if self._data:
            self.view_start = min(max(0.0, center - self.window / 2), max(0.0, self.duration - self.window))
            self._draw_view()

    def _zoom_slid(self, value):
        self.set_zoom(round(float(value)))

    def _overview_drag(self, event):
        if self._data:
            seconds = (event.x - LEFT) / self.width * self.duration
            self.set_view(seconds - self.window / 2)

    def _progress_clicked(self, event):
        if self._data and self.on_seek:
            self.on_seek(min(max(0.0, (event.x - LEFT) / self.width), 1.0) * self.duration)

    def _offset_at(self, x):
        return self.view_start + min(max(x - LEFT, 0), self.width) / self.width * self.window

    def _clicked(self, event):
        self.main.focus_set()
        if self._data and self.on_seek:
            self.on_seek(self._offset_at(event.x))

    def _frequency_at(self, y):
        return min(1.0, max(0.0, (AXIS_H + self.height - y) / self.height)) * self._data[2]

    def _select_press(self, event):
        self.main.focus_set()
        if self._data and event.x >= LEFT:
            self._drag_origin = (self._offset_at(event.x), self._frequency_at(event.y), event.x, event.y)

    def _select_drag(self, event):
        if self._drag_origin is None:
            return
        at, hz, _x, _y = self._drag_origin
        self._selection = (*sorted((at, self._offset_at(event.x))), *sorted((hz, self._frequency_at(event.y))))
        self._draw_selection()

    def _select_release(self, event):
        if self._drag_origin is None:
            return
        _at, _hz, x, y = self._drag_origin
        if abs(event.x - x) < 4 and abs(event.y - y) < 4:
            self._selection = None
            self._clicked(event)
        else:
            self._select_drag(event)
            a, b, low, high = self._selection
            if b - a > 0.01 and high - low > 1:
                self.on_select(self.to_real(a), self.to_real(b), low, high)
        self._drag_origin = None
        self._draw_selection()

    def clear_selection(self):
        self._selection = None
        self._draw_selection()

    def _draw_selection(self):
        self.main.delete("selection")
        if not self._selection or not self._data:
            return
        a, b, low, high = self._selection
        fmax = self._data[2]
        self.main.create_rectangle(self._x(a), AXIS_H + self.height * (1 - high / fmax),
                                   self._x(b), AXIS_H + self.height * (1 - low / fmax),
                                   outline="#e09d00", width=2, tags="selection")

    def open_listening_tools(self):
        if self.source_path is None:
            return
        from listening_tools import ListeningToolsWindow
        if self.app is not None:
            self.app.review_stop()
        self._listening_window = ListeningToolsWindow(self.winfo_toplevel(), self.source_path, app=self.app,
                                                     start=self.source_start,
                                                     end=min(self.source_end, self.source_start + self.real_duration),
                                                     clock_start=self.clock_start,
                                                     species=getattr(self, "species", None),
                                                     location=getattr(self, "location", None),
                                                     boxes=getattr(self, "boxes", None),
                                                     flags=[flag + self.source_start for flag in self.flags])

    def open_compare(self):
        """เทียบความถี่/ย่านเสียงของคลิปนี้กับเสียงอ้างอิง — เลือกกรอบเสียงของ BirdNET, Auto หรือลากกรอบเองในหน้าต่างนั้น"""
        if self.source_path is None:
            return
        from frequency_compare import FrequencyCompareWindow
        if self.app is not None:
            self.app.review_stop()
        start = self.source_start
        samples, sr = read_mono(self.source_path, start, min(self.source_end, start + self.real_duration))
        self._compare_window = FrequencyCompareWindow(self.winfo_toplevel(), None,
                                                      species=getattr(self, "species", None),
                                                      location=getattr(self, "location", None),
                                                      describe=self.source_path.name, audio=(samples, sr, None),
                                                      boxes=getattr(self, "boxes", None), when=self.clock_start)

    def _hover(self, event):
        if self._data and event.x >= LEFT and self._play is None:
            self._describe(self._offset_at(event.x))

    # ---------- ตำแหน่งขณะเล่น ----------
    def start_playhead(self, offset, player, speed=1.0):
        """player = Popen ที่กำลังเล่นตั้งแต่ offset วินาทีของคลิป"""
        self.stop_playhead()
        self._play = (offset, time.monotonic(), player)
        self._play_speed = speed
        self._tick()

    def stop_playhead(self):
        if getattr(self, "_tick_job", None):
            self.after_cancel(self._tick_job)
            self._tick_job = None
        self._play = None
        self.main.delete("playhead")
        self.overview.delete("playhead")

    def _tick(self):
        self._tick_job = None
        if self._play is None or not self._data:
            return
        offset, started, player = self._play
        now = offset + (time.monotonic() - started) * getattr(self, "_play_speed", 1.0)
        if now > self.duration or player is None or player.poll() is not None:
            self.stop_playhead()
            self._describe()
            return
        self._position = now
        before = self.view_start
        self.set_view(now - self.window / 2)          # เส้นอยู่กลาง ภาพเลื่อนผ่านเส้น
        if self.view_start == before:                 # ต้น/ท้ายคลิป: ภาพไม่เลื่อน แค่แถบความคืบหน้า
            self._draw_progress()
        self.main.delete("playhead")
        self.overview.delete("playhead")
        x = self._x(now)
        self.main.create_line(x, AXIS_H, x, AXIS_H + self.height, fill="#111", width=1, tags="playhead")
        ox = LEFT + now / self.duration * self.width
        self.overview.create_line(ox, 1, ox, self.overview_height + 3, fill="#111", tags="playhead")
        self._tick_job = self.after(FRAME_MS, self._tick)
