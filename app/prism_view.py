"""หน้าเสียง Prism (ใช้ใน Listen tools): ภาพเสียงอยู่บน คลื่นเสียงอยู่ล่าง แถบควบคุมอยู่บนสุด ปุ่มซูมอยู่ทางซ้ายของภาพ

- spectrogram: FFT 256 จุด เลื่อนทีละ 128 (Hann), 0 dB = full scale, ช่วงสี -96..0 dB
  พื้นเทาเข้ม เสียงเบาไปดัง = ม่วง ชมพู ส้ม เหลืองอ่อน; Invert = พื้นกระดาษ ไล่ไปน้ำเงินเข้ม
- คลื่นเสียงสีเขียวอมฟ้า ค่าตัวอย่าง ±32768 หรือ dB
- แกนซ้าย: ตัวเลขชิดขวาเหนือเส้นขีด, แกนเวลาล่าง (ดับเบิลคลิกแกนเวลา = เวลาจริงของนาฬิกาเครื่องอัด)
- ซูมแนวนอนเป็นขั้น 0.5–256 วินาทีต่อความกว้างภาพ (เริ่ม 8 วินาที), แนวตั้งเริ่ม 0–12 kHz
- ลากกรอบ = เลือกเวลา+ความถี่, คลิกครั้งเดียว = ยกเลิก, คลิกขวา = เส้นอ้างอิงความถี่,
  คลิกขวาในกรอบ = ซูมพอดีกรอบ, ปุ่มลูกศร ←/→ = เลื่อน 1/10 จอ, Shift = ทั้งจอ
- แถบควบคุม: Play/Stop, Speed, Channel, Invert, Wave dB, Spectrogram, Log Hz, Labels, Analysis, Brightness, Contrast, ช่วงก่อน/ถัดไป
- Prism Analysis (ปุ่ม Analysis): สเปกตรัมของกรอบที่เลือก (ไม่เลือก = ภาพที่เห็น) + ค่า Tstart Tend Fstart Fend dBmin dBmax
  dBmean Fpmin Fpmax Fpmean Fppeak; สูตรตรวจด้วยไฟล์ทดสอบแล้ว (2026-10-24):
  dBmin/dBmax/dBmean = พลังงานในย่านความถี่ของแต่ละเฟรม (sine แอมพลิจูด A = 20·log A, เสียงรบกวน RMS σ = 20·log σ)
  เฉลี่ย 2 เฟรมติดกัน, Fpmean = ความถี่เฉลี่ยถ่วงด้วยพลังงาน, Fppeak = จุดสูงสุดประมาณค่าแบบพาราโบลา,
  Fpmin/Fpmax = ไล่จาก Fppeak จนสเปกตรัมลงถึงค่ากลาง + 3.5 dB (ค่าประมาณ)
"""

import math
import queue
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk

from frequency_compare import FlowFrame

FFT_SIZE, FFT_HOP = 256, 128          # FFT 256 จุด เลื่อนทีละ 128
GUTTER = 62                           # แกนซ้าย (ตัวเลข)
LABEL_H, TIME_H = 18, 30              # แถบป้ายชนิดเหนือภาพ, แกนเวลาใต้ภาพ
SIDE_W = 58                           # คอลัมน์ปุ่มซูมทางซ้ายของภาพ
REF_COLOR = "#7fdbff"                 # เส้นความถี่อ้างอิง
ICON_FOLDER = Path(__file__).resolve().parent / "assets" / "prism-icons"   # <ชื่อ>.png แทนตัวหนังสือบนปุ่ม (ดู README.txt ในโฟลเดอร์)
ICON_SIZE = 20
X_ZOOMS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)   # วินาทีต่อความกว้างภาพ
DEFAULT_X_ZOOM = 8.0
DEFAULT_FMAX = 12000.0                # non-bat mode: 0–12 kHz
LOG_FMIN = 100.0
FULL_SCALE = 32768.0
DB_FLOOR = -100.0                     # oscillogram แบบ dB
SPEEDS = (("Normal", 1.0), ("1/2", 0.5), ("3/4", 0.75), ("1.25", 1.25), ("1.5", 1.5), ("2", 2.0))
RULER_FONT = ("Helvetica", 12)
READOUT_FONT = ("Helvetica", 12, "bold")
FRAME_MS = 40
# พื้นเทาเข้ม เสียงเบาไปดัง = ม่วง ชมพู ส้ม เหลืองอ่อน; กลับสี = พื้นกระดาษ ไล่ไปน้ำเงินเข้ม — สีแรกของ stops = สีพื้น
COLORS = {False: {"bg": "#12151c", "fg": "#d7dde8", "wave": (56, 198, 192), "stops": (
              (0.0, (18, 21, 28)), (0.18, (60, 30, 110)), (0.35, (170, 45, 120)), (0.5, (235, 90, 60)),
              (0.68, (250, 170, 60)), (0.85, (253, 225, 120)), (1.0, (252, 245, 190)))},
          True: {"bg": "#fbfaf6", "fg": "#1f2a44", "wave": (15, 118, 110), "stops": (
              (0.0, (251, 250, 246)), (0.2, (190, 205, 230)), (0.4, (110, 140, 200)), (0.6, (50, 80, 160)),
              (1.0, (20, 40, 100)))}}


# ----------------------------- คำนวณ (ไม่แตะ Tk) -----------------------------
def palette(inverse=False):
    """ตารางสี 256 ระดับ ไล่ตาม stops ใน COLORS"""
    stops = COLORS[inverse]["stops"]
    places = [place for place, _color in stops]
    t = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(t, places, [color[channel] for _place, color in stops]) for channel in range(3)],
                    axis=1).round().astype(np.uint8)


def spectrum_db(samples, chunk=32768):
    """FFT 256 จุด (Hann) เลื่อนทีละ 128 -> dB เทียบ full scale (คลื่น sine เต็มสเกล = 0 dB) แถว = เฟรม"""
    samples = np.asarray(samples, np.float32)
    if len(samples) < FFT_SIZE:
        samples = np.pad(samples, (0, FFT_SIZE - len(samples)))
    window = np.hanning(FFT_SIZE).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, FFT_SIZE)[::FFT_HOP]
    scale = 2.0 / window.sum()
    out = np.empty((len(frames), FFT_SIZE // 2 + 1), np.float16)
    for first in range(0, len(frames), chunk):
        block = np.abs(np.fft.rfft(frames[first:first + chunk] * window, axis=1)) * scale
        out[first:first + chunk] = 20 * np.log10(block + 1e-7)
    return out


def blocks(samples, size=FFT_HOP):
    """ค่าต่ำสุด/สูงสุด/RMS ทุก size ตัวอย่าง (วาด oscillogram ตอนซูมออกโดยไม่ต้องอ่านทุกตัวอย่าง)"""
    samples = np.asarray(samples, np.float32)
    n = len(samples) // size
    if n == 0:
        return np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32)
    part = samples[:n * size].reshape(n, size)
    return part.min(axis=1), part.max(axis=1), np.sqrt((part.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def nice_step(span, pixels, min_gap, steps=(1, 2, 5)):
    """ระยะขีดของแกนที่ห่างกันอย่างน้อย min_gap พิกเซล"""
    if span <= 0 or pixels <= 0:
        return 1.0
    raw = span * min_gap / pixels
    base = 10 ** math.floor(math.log10(raw))
    for multiple in steps + (10,):
        if base * multiple >= raw:
            return base * multiple
    return base * 10


def time_label(seconds):
    """'2s', '0.5s', '1m4s' บนแกนเวลา"""
    seconds = round(seconds, 3)
    if abs(seconds) >= 60:
        minutes, rest = divmod(seconds, 60)
        return f"{int(minutes)}m{rest:g}s"
    return f"{seconds:g}s"


def freq_label(hz):
    return f"{hz / 1000:g}kHz"


def column_pool(data, columns):
    """แถว (เฟรม) มากกว่าคอลัมน์ภาพ: เอาค่าสูงสุดของแต่ละกลุ่ม (เสียงสั้นยังเห็น); น้อยกว่า: คืนเดิม (ไปขยายทีหลัง)"""
    if len(data) <= columns:
        return data
    edges = np.linspace(0, len(data), columns + 1).astype(int)
    return np.maximum.reduceat(data, edges[:-1], axis=0)


def spectrogram_rgb(db, rows_hz, sr, lut, brightness=0.0, floor=-96.0):
    """dB (เฟรม x bin) -> สีที่ความละเอียด bin ก่อน แล้วค่อยประมาณค่าตามแถวภาพ
    rows_hz = ความถี่ของแต่ละแถวภาพ (บนลงล่าง) -> uint8 (แถว, เฟรม, 3)"""
    level = (db.astype(np.float32) + brightness - floor) / (-floor)
    index = np.clip(level * 255, 0, 255).astype(np.uint8)
    colored = lut[index].astype(np.float32)                       # (เฟรม, bin, 3)
    position = np.clip(np.asarray(rows_hz) / (sr / FFT_SIZE), 0, db.shape[1] - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, db.shape[1] - 1)
    weight = (position - lower)[None, :, None]
    rows = colored[:, lower] * (1 - weight) + colored[:, upper] * weight   # (เฟรม, แถว, 3)
    return np.ascontiguousarray(rows.transpose(1, 0, 2)).round().astype(np.uint8)


# ----------------------------- Save screen (ภาพของหน้าเสียง) -----------------------------
_ANCHORS = {"nw": "la", "n": "ma", "ne": "ra", "w": "lm", "center": "mm", "e": "rm", "sw": "ld", "s": "md", "se": "rd"}


def _font(spec):
    from PIL import ImageFont
    size = int(spec[1]) if isinstance(spec, (tuple, list)) else 12
    bold = isinstance(spec, (tuple, list)) and "bold" in spec[2:]
    for path, index in (("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas_image(canvas, images):
    """ภาพของ canvas จากรายการที่วาด (ภาพ, เส้น, กรอบ, ข้อความ) — images = {item: ภาพ PIL}"""
    out = Image.new("RGB", (max(1, canvas.winfo_width()), max(1, canvas.winfo_height())), canvas.cget("bg"))
    draw = ImageDraw.Draw(out)
    for item in canvas.find_all():
        if "cursor" in canvas.gettags(item):
            continue
        kind, coords = canvas.type(item), canvas.coords(item)
        if kind == "image" and item in images:
            out.paste(images[item], (int(round(coords[0])), int(round(coords[1]))))
        elif kind == "line" and len(coords) >= 4:
            width = max(1, int(round(float(canvas.itemcget(item, "width")))))
            draw.line(coords, fill=canvas.itemcget(item, "fill") or None, width=width)
        elif kind in ("rectangle", "oval"):
            fill, outline = canvas.itemcget(item, "fill") or None, canvas.itemcget(item, "outline") or None
            (draw.rectangle if kind == "rectangle" else draw.ellipse)(coords, fill=fill, outline=outline)
        elif kind == "text":
            font = canvas.itemcget(item, "font")
            spec = tuple(canvas.tk.splitlist(font)) if font else ("Helvetica", 12)
            draw.text(coords, canvas.itemcget(item, "text"), fill=canvas.itemcget(item, "fill"),
                      font=_font(spec), anchor=_ANCHORS.get(canvas.itemcget(item, "anchor"), "la"))
    return out


# ----------------------------- Prism Analysis -----------------------------
_HANN = np.hanning(FFT_SIZE)
POWER_SCALE = float(_HANN.sum() ** 2 / (FFT_SIZE * (_HANN ** 2).sum()))   # (2|X|/Σw)² -> พลังงาน ×2 ต่อ bin
FLOOR_MARGIN_DB = 3.5
STATS = (("Tstart", "Tend", "Fstart", "Fend"), ("dBmin", "dBmax", "dBmean", "Qual"),
         ("Fpmin", "N", "Dur", "TBC"), ("Fpmax", "Fmax", "Fmin", "Fmean"),
         ("Fpmean", "Fc", "Tc", "Sc"), ("Fppeak", "Fk", "Tk", "S1"))
ZERO_CROSSING = {"Qual", "N", "Dur", "TBC", "Fmax", "Fmin", "Fmean", "Fc", "Tc", "Sc", "Fk", "Tk", "S1"}


def freq_fraction(hz, low, high, log=False):
    """ตำแหน่งความถี่บนแกน 0 (ล่าง) ถึง 1 (บน) แบบเดียวกับ spectrogram (linear หรือ log)"""
    if log:
        low = max(LOG_FMIN, low)
        return (math.log(max(hz, low)) - math.log(low)) / (math.log(high) - math.log(low))
    return (hz - low) / (high - low)


def analysis_stats(db, sr, low, high):
    """ค่าของ Prism Analysis จากเฟรม dB (เฟรม x bin) ของช่วงที่วิเคราะห์ ในย่าน [low, high] Hz -> dict
    spectrum_db = สเปกตรัมพลังงานเฉลี่ยตามเวลา (ใช้วาดเส้นแดง)"""
    db = np.atleast_2d(np.asarray(db, np.float32))
    freqs = np.arange(db.shape[1]) * sr / FFT_SIZE
    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        band[int(np.argmin(np.abs(freqs - (low + high) / 2)))] = True
    power = 10 ** (db[:, band].astype(np.float64) / 10)
    frame = POWER_SCALE * power.sum(axis=1)
    smooth = (frame[1:] + frame[:-1]) / 2 if len(frame) > 1 else frame
    spectrum = power.mean(axis=0)
    spectrum_db = 10 * np.log10(spectrum + 1e-30)
    f = freqs[band]
    peak = int(np.argmax(spectrum_db))
    shift = 0.0
    if 0 < peak < len(f) - 1:                                     # ประมาณจุดยอดแบบพาราโบลาจาก 3 bin
        a, b, c = spectrum_db[peak - 1:peak + 2]
        if a - 2 * b + c < 0:
            shift = float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5))
    step = sr / FFT_SIZE
    floor = float(np.median(spectrum_db)) + FLOOR_MARGIN_DB
    if spectrum_db[peak] <= floor:                                  # สเปกตรัมเรียบ (เสียงรบกวน): หาขอบไม่ได้
        lo = hi = None
    else:
        lo = hi = peak
        while lo > 0 and spectrum_db[lo - 1] > floor:
            lo -= 1
        while hi < len(f) - 1 and spectrum_db[hi + 1] > floor:
            hi += 1
    return {"freqs": f, "spectrum_db": spectrum_db,
            "dBmin": 10 * math.log10(max(float(smooth.min()), 1e-30)),
            "dBmax": 10 * math.log10(max(float(smooth.max()), 1e-30)),
            "dBmean": 10 * math.log10(max(float(frame.mean()), 1e-30)),
            "Fppeak": float(f[peak] + shift * step), "Fpmean": float((f * spectrum).sum() / max(spectrum.sum(), 1e-30)),
            "Fpmin": float(f[lo]) if lo is not None else 0.0, "Fpmax": float(f[hi]) if hi is not None else 0.0}


class AnalysisWindow:
    """หน้าต่าง Prism Analysis: สเปกตรัม (แกนตั้ง = ความถี่ตาม spectrogram, แกนนอน = dB จากจุดสูงสุด 0 dB
    ไปทางขวา ยาวตามแถบ Contrast) + ตารางค่า — อัปเดตทันทีเมื่อเลือก/เลื่อน; คลิกขวาที่กราฟ = คัดลอกค่า/บันทึกภาพ"""

    def __init__(self, view):
        self.view = view
        self.result = None
        win = self.win = tk.Toplevel(view)
        win.title("Prism Analysis")
        win.geometry(f"640x470+{view.winfo_rootx() + 40}+{view.winfo_rooty() + 40}")
        self.canvas = tk.Canvas(win, bg=COLORS[False]["bg"], highlightthickness=0, height=280)
        self.canvas.pack(side="top", fill="both", expand=True)
        grid = ttk.Frame(win, padding=(4, 4, 8, 6))
        grid.pack(side="top", fill="x")
        self.values = {}
        for row, names in enumerate(STATS):
            for column, name in enumerate(names):
                ttk.Label(grid, text=f"{name}:").grid(row=row, column=column * 2, sticky="e", padx=(10, 3), pady=1)
                value = tk.StringVar(master=win)
                ttk.Label(grid, textvariable=value).grid(row=row, column=column * 2 + 1, sticky="w")
                self.values[name] = value
        actions = ttk.Frame(win, padding=(8, 0, 8, 8))
        actions.pack(side="top", fill="x")
        # เทียบความถี่ของกรอบนี้กับเสียงอ้างอิงของชนิดนั้น (xeno-canto / ไฟล์ของคุณ) — frequency_compare.py
        ttk.Button(actions, text="Compare with reference sounds (frequency and band)…", command=self.compare).pack(side="left")
        self.canvas.bind("<Configure>", lambda _event: self.update())
        for button in ("<ButtonPress-3>", "<ButtonPress-2>", "<Control-ButtonPress-1>"):
            self.canvas.bind(button, self._menu)
        win.protocol("WM_DELETE_WINDOW", self.close)

    def exists(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        if self.exists():
            self.win.destroy()

    def update(self):
        view = self.view
        if not view.data or not self.exists():
            return
        a, b, low, high = view.analysis_region()
        sr = view.data["sr"]
        first = max(0, int(a * sr / FFT_HOP))
        last = min(len(view.data["db"]), max(first + 1, int(math.ceil(b * sr / FFT_HOP))))
        stats = self.result = analysis_stats(view.data["db"][first:last], sr, low, high)
        origin = view.data["origin"]
        text = {"Tstart": f"{origin + a:.6f}s", "Tend": f"{origin + b:.6f}s",
                "Fstart": f"{low / 1000:.3f}kHz", "Fend": f"{high / 1000:.3f}kHz"}
        for name in ("dBmin", "dBmax", "dBmean"):
            text[name] = f"{stats[name]:.2f}dB"
        for name in ("Fpmin", "Fpmax", "Fpmean", "Fppeak"):
            text[name] = f"{stats[name] / 1000:.3f}kHz"
        for name, value in self.values.items():
            value.set(text.get(name, "—" if name in ZERO_CROSSING else ""))
        self._draw(stats, low, high)

    def _draw(self, stats, low, high):
        canvas, colors = self.canvas, COLORS[self.view.inverse]
        canvas.configure(bg=colors["bg"])
        canvas.delete("all")
        width = max(50, canvas.winfo_width() - GUTTER - 6)
        height = max(40, canvas.winfo_height() - TIME_H)
        fg, log = colors["fg"], self.view.log_freq
        top_hz, bottom_hz = self.view.f_hi, self.view.f_lo

        def y_of(hz):
            return height * (1 - freq_fraction(hz, bottom_hz, top_hz, log))
        span = max(1.0, -float(self.view.contrast.get()))           # ช่วง dB ตามแถบ Contrast
        # แกนความถี่ (ตาม spectrogram)
        if log:
            ticks = [f for f in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if max(LOG_FMIN, bottom_hz) <= f <= top_hz]
            step = 1000
        else:
            step = nice_step(top_hz - bottom_hz, height, 30)
            ticks = list(np.arange(math.ceil(bottom_hz / step) * step, top_hz + 1e-6, step))
        for hz in ticks:
            y = y_of(hz)
            if 13 <= y <= height + 1:
                canvas.create_line(0, y, GUTTER - 1, y, fill=fg)
                label = f"{hz / 1000:.1f}kHz" if step < 2000 else freq_label(hz)
                canvas.create_text(GUTTER - 4, y - 1, text=label, anchor="se", fill=fg, font=RULER_FONT)
        # แกน dB: ขีดทุก 5 dB ตัวเลขทุก 20 dB
        for level in range(0, int(span) + 1, 5):
            x = GUTTER + level / span * width
            major = level % 20 == 0
            canvas.create_line(x, height, x, height + (TIME_H - 2 if major else 14), fill=fg)
            if major:
                canvas.create_text(x + 2, height + TIME_H - 2, text=f"-{level}dB", anchor="sw", fill=fg, font=RULER_FONT)
        # เส้นแดง: สเปกตรัมเฉลี่ย ปรับให้จุดสูงสุด = 0 dB
        relative = stats["spectrum_db"] - stats["spectrum_db"].max()
        points = []
        for hz, level in zip(stats["freqs"], relative):
            y = y_of(hz)
            if -1 <= y <= height + 1:
                points += [GUTTER + min(span, -float(level)) / span * width, y]
        if len(points) >= 4:
            canvas.create_line(*points, fill="#%02x%02x%02x" % colors["wave"])

    def compare(self):
        if not self.result:
            return
        from frequency_compare import FrequencyCompareWindow, shift_boxes
        span = f"{self.values['Tstart'].get()}–{self.values['Tend'].get()}"
        data = self.view.data
        # เสียงทั้งช่วงที่โหลด + กรอบที่ลากไว้ (ไม่ได้ลาก = เลือกจากกรอบเสียงของ BirdNET / Auto ในหน้าต่างเทียบ)
        audio = (data["samples"].astype(np.float32) / FULL_SCALE, data["sr"], self.view.selection)
        self.compare_window = FrequencyCompareWindow(
            self.win, self.result, species=self.view.species, location=self.view.location, describe=span,
            audio=audio, boxes=shift_boxes(getattr(self.view, "boxes", None), data["origin"]), db=data["db"],
            when=self.view.clock_start)

    def text(self):
        return "\n".join("\t".join(f"{name}: {self.values[name].get()}" for name in names) for names in STATS)

    def _menu(self, event):
        menu = tk.Menu(self.win, tearoff=False)
        menu.add_command(label="Copy values", command=self.copy_values)
        menu.add_command(label="Save picture…", command=self.save_picture)
        menu.tk_popup(event.x_root, event.y_root)

    def copy_values(self):
        self.win.clipboard_clear()
        self.win.clipboard_append(self.text())

    def save_picture(self):
        from tkinter import filedialog
        target = filedialog.asksaveasfilename(parent=self.win, defaultextension=".png", filetypes=[("PNG", "*.png")],
                                              initialfile="prism-analysis.png")
        if target:
            canvas_image(self.canvas, {}).save(target)


# ----------------------------- หน้าเสียง -----------------------------
class PrismView(ttk.Frame):
    """on_select(a, b, low, high) วินาทีนับจากต้นช่วงที่โหลด + Hz; on_play() ปุ่ม Play/Space;
    speed_var/channel_var ใช้ร่วมกับหน้าต่าง Listen tools ("1.0", "left"/"right"/"mix")"""

    def __init__(self, master, *, on_select=None, on_play=None, on_stop=None, speed_var=None, channel_var=None,
                 on_channel=None, on_segment=None):
        super().__init__(master)
        self.on_select, self.on_play, self.on_stop = on_select, on_play, on_stop
        self.on_channel, self.on_segment = on_channel, on_segment
        self.speed_var = speed_var or tk.StringVar(master=self, value="1.0")
        self.channel_var = channel_var or tk.StringVar(master=self, value="left")
        self.q = queue.Queue()
        self.data = None                       # sr, db, samples, mins/maxs/rms, duration, origin
        self.clock_start = None
        self.labels = []                       # [(start, end, ชื่อ, คะแนน)] วินาทีในไฟล์
        self.flags = []                        # flag ที่กดตอนอัด (วินาทีในไฟล์ — recorder_flags.py)
        self.min_conf = 0.5
        self.t0, self.x_zoom = 0.0, DEFAULT_X_ZOOM
        self.f_lo, self.f_hi = 0.0, DEFAULT_FMAX
        self.osc_zoom, self.osc_center = 1.0, 0.0
        self.osc_db = False
        self.inverse = self.log_freq = False
        self.show_spectrum = self.show_ids = True
        self.time_mode = "elapsed"
        self.selection = None                  # (a, b, low, high)
        self.ref_lines = []
        self._drag = None
        self._load_id = 0
        self._play = None
        self._tick_job = None
        self._spec_photo = self._osc_photo = None
        self._pil = {}                         # {(canvas, item): ภาพ PIL} สำหรับ Save screen
        self.analysis = None                   # หน้าต่าง Prism Analysis
        self.species = self.location = None    # (common, scientific) / (lat, lon) ของเสียงนี้ ถ้ารู้ — ใช้เทียบความถี่
        self._analysis_job = None
        self._lut = {False: palette(False), True: palette(True)}
        self._icons = {}
        self._build()
        self._poll_job = self.after(100, self._poll)

    # ---------- หน้าตา ----------
    def _build(self):
        bar = self.bar = FlowFrame(self)               # แถบควบคุมอยู่บนสุด ขึ้นบรรทัดใหม่เองเมื่อหน้าต่างแคบ
        bar.pack(side="top", fill="x", pady=(0, 6))
        self.play_btn = bar.add(self._tip(ttk.Button(bar, command=self._play_pressed, **self._face("play", "▶ Play")),
                                          "Play the selection (or from the left edge)"))
        self.speed_text = tk.StringVar(master=self, value="Normal")
        bar.add(ttk.Label(bar, text="Speed"), space=4)
        speed = bar.add(ttk.Combobox(bar, textvariable=self.speed_text, values=[name for name, _value in SPEEDS],
                                     state="readonly", width=7))
        speed.bind("<<ComboboxSelected>>", self._speed_chosen)
        self.channel_text = tk.StringVar(master=self, value="Left")
        bar.add(ttk.Label(bar, text="Channel"), space=4)
        self.channel_box = bar.add(ttk.Combobox(bar, textvariable=self.channel_text, values=("Left", "Right"),
                                                state="readonly", width=6))
        self.channel_box.bind("<<ComboboxSelected>>", self._channel_chosen)
        for name, text, command, tip in (("invert", "Invert", self.toggle_inverse, "Invert colours"),
                                         ("wave-db", "Wave dB", self.toggle_osc, "Waveform: sample values / dB"),
                                         ("spectrogram", "Spectrogram", self.toggle_spectrum, "Show or hide the spectrogram"),
                                         ("log-hz", "Log Hz", self.toggle_log_freq, "Frequency scale: linear / log"),
                                         ("labels", "Labels", self.toggle_ids, "Show or hide species labels"),
                                         ("analysis", "Analysis", self.toggle_analysis, "Analyze the view or selection")):
            bar.add(self._tip(ttk.Button(bar, command=command, **self._face(name, text)), tip))
        self.brightness = tk.DoubleVar(master=self, value=0.0)
        self.contrast = tk.DoubleVar(master=self, value=-96.0)
        bar.add(self._slider(bar, "Brightness", self.brightness, -96, 96), space=6)
        bar.add(self._slider(bar, "Contrast", self.contrast, -96, -1), space=6)
        bar.add(self._tip(ttk.Button(bar, command=lambda: self.on_segment and self.on_segment(-1), **self._face("previous-span", "◀ Span")),
                          "Previous span of the recording"), space=6)
        bar.add(self._tip(ttk.Button(bar, command=lambda: self.on_segment and self.on_segment(1), **self._face("next-span", "Span ▶")),
                          "Next span of the recording"))
        bar.add(ttk.Button(bar, command=self.save_screen_as, **self._face("save-screen", "Save screen…")))

        # ภาพเสียงอยู่บน คลื่นเสียงอยู่ล่าง ปุ่มซูมอยู่ทางซ้ายของภาพ (คอลัมน์กว้างเท่ากัน เวลาของสองภาพจึงตรงกัน)
        self.panes = tk.PanedWindow(self, orient="vertical", sashwidth=7, sashrelief="flat", bg="#f4f4f4",
                                    bd=0, opaqueresize=False)
        self.panes.pack(side="top", fill="both", expand=True)
        spec = tk.Frame(self.panes, bg="#e6e6e6")
        osc = tk.Frame(self.panes, bg="#e6e6e6")
        self.panes.add(spec, minsize=120, stretch="always")
        self.panes.add(osc, height=110, minsize=60, stretch="never")
        self.spec = tk.Canvas(spec, bg=COLORS[False]["bg"], highlightthickness=0, cursor="crosshair", height=300)
        self.spec.grid(row=0, column=1, rowspan=4, sticky="nsew")
        spec.columnconfigure(0, minsize=SIDE_W)
        spec.columnconfigure(1, weight=1)
        spec.rowconfigure(3, weight=1)
        ttk.Button(spec, width=2, command=lambda: self.zoom_freq(0.5), **self._face("zoom-in", "+")).grid(row=0, column=0, sticky="ew")
        ttk.Button(spec, width=2, command=lambda: self.zoom_freq(2.0), **self._face("zoom-out", "−")).grid(row=1, column=0, sticky="ew")
        ttk.Button(spec, width=2, command=self.fit_freq, **self._face("fit", "Fit")).grid(row=2, column=0, sticky="ew")
        self.freq_bar = ttk.Scrollbar(spec, orient="vertical", command=self._freq_scrolled)
        self.freq_bar.grid(row=3, column=0, sticky="ns")
        self.osc = tk.Canvas(osc, bg=COLORS[False]["bg"], highlightthickness=0, cursor="crosshair", height=110)
        self.osc.grid(row=0, column=1, rowspan=3, sticky="nsew")
        osc.columnconfigure(0, minsize=SIDE_W)
        osc.columnconfigure(1, weight=1)
        osc.rowconfigure(2, weight=1)
        ttk.Button(osc, width=2, command=lambda: self.zoom_osc(2.0), **self._face("zoom-in", "+")).grid(row=0, column=0, sticky="ew")
        ttk.Button(osc, width=2, command=lambda: self.zoom_osc(0.5), **self._face("zoom-out", "−")).grid(row=1, column=0, sticky="ew")
        self.osc_bar = ttk.Scrollbar(osc, orient="vertical", command=self._osc_scrolled)
        self.osc_bar.grid(row=2, column=0, sticky="ns")

        bottom = ttk.Frame(self)
        bottom.pack(side="top", fill="x", pady=(2, 0))
        self.time_bar = ttk.Scrollbar(bottom, orient="horizontal", command=self._time_scrolled)
        self.time_bar.pack(side="left", fill="x", expand=True, padx=(SIDE_W, 6))
        ttk.Label(bottom, text="Time").pack(side="left", padx=(0, 4))
        ttk.Button(bottom, width=2, command=lambda: self.zoom_time(1), **self._face("zoom-out", "−")).pack(side="left")
        ttk.Button(bottom, width=2, command=lambda: self.zoom_time(-1), **self._face("zoom-in", "+")).pack(side="left")
        ttk.Button(bottom, width=2, command=self.fit_time, **self._face("fit", "Fit")).pack(side="left")
        self.info = tk.StringVar(master=self)
        ttk.Label(self, textvariable=self.info, foreground="#808080").pack(side="top", anchor="w", pady=(4, 0))

        for canvas in (self.osc, self.spec):
            canvas.bind("<Configure>", lambda _event: self._schedule_redraw())
            canvas.bind("<Motion>", self._hover)
            canvas.bind("<Leave>", lambda event: event.widget.delete("cursor"))
            canvas.bind("<ButtonPress-1>", self._press)
            canvas.bind("<B1-Motion>", self._dragging)
            canvas.bind("<ButtonRelease-1>", self._release)
            canvas.bind("<MouseWheel>", self._wheel)
            canvas.bind("<Shift-MouseWheel>", self._wheel)
            for key, action in (("<Left>", lambda: self.scroll_time(-0.1)), ("<Right>", lambda: self.scroll_time(0.1)),
                                ("<Shift-Left>", lambda: self.scroll_time(-1.0)),
                                ("<Shift-Right>", lambda: self.scroll_time(1.0)),
                                ("<space>", self._play_pressed), ("<plus>", lambda: self.zoom_time(-1)),
                                ("<z>", lambda: self.zoom_time(-1)), ("<minus>", lambda: self.zoom_time(1)),
                                ("<equal>", self.fit_time)):
                canvas.bind(key, lambda _event, action=action: (action(), "break")[1])
        for button in ("<ButtonPress-3>", "<ButtonPress-2>", "<Control-ButtonPress-1>"):
            self.spec.bind(button, self._right_click)
        for button in ("<Double-ButtonPress-3>", "<Double-ButtonPress-2>"):
            self.spec.bind(button, self._clear_refs)
        self.spec.bind("<Double-ButtonPress-1>", self._double)
        self._redraw_job = None
        self.message("Loading …")

    def _slider(self, parent, name, variable, low, high):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=name).pack(side="left", padx=(0, 4))
        ttk.Scale(frame, from_=low, to=high, variable=variable, length=90,
                  command=lambda _value: self._colors_changed()).pack(side="left")
        return frame

    def _face(self, name, text):
        """หน้าปุ่ม: ICON_FOLDER/<name>.png ถ้ามีและเปิดได้ ไม่งั้นใช้ตัวหนังสือ"""
        if name not in self._icons:
            try:
                image = Image.open(ICON_FOLDER / f"{name}.png").convert("RGBA")
                image.thumbnail((ICON_SIZE, ICON_SIZE))
                self._icons[name] = ImageTk.PhotoImage(image, master=self)
            except OSError:
                self._icons[name] = ""
        return {"image": self._icons[name], "text": "" if self._icons[name] else text}

    def _tip(self, button, tip):
        button.bind("<Enter>", lambda _event: self.info.set(tip))
        button.bind("<Leave>", lambda _event: self.info.set(""))
        return button

    # ---------- โหลด ----------
    def message(self, text):
        self.data = None
        for canvas in (self.osc, self.spec):
            canvas.delete("all")
        self.spec.create_text(GUTTER + 300, 120, text=text, fill=COLORS[False]["fg"], tags="message")

    def load(self, path, clock_start=None, start=0.0, end=None, channel="left", labels=None, min_conf=0.5):
        """ช่วง [start, end] ของไฟล์; channel = left/right/mix (ไฟล์ mono ใช้ mix เอง)"""
        from audio_io import audio_info, read_channels
        self._load_id += 1
        load_id, results = self._load_id, self.q
        self.clock_start, self.min_conf = clock_start, min_conf
        self.labels = list(labels or [])
        self.selection = None
        self.stop_playhead()
        self.message("Loading …")

        def work():  # ไม่ถือ self
            try:
                info = audio_info(path)
                stereo = info["channels"] > 1
                wanted = channel if stereo and channel in ("left", "right") else "mix"
                samples, sr = read_channels(path, start, end if end is not None else info["duration"], wanted)
                db = spectrum_db(samples)
                mins, maxs, rms = blocks(samples)
                results.put((load_id, dict(sr=sr, db=db, samples=np.clip(samples * FULL_SCALE, -32768, 32767)
                                           .astype(np.int16), mins=mins, maxs=maxs, rms=rms,
                                           duration=len(samples) / sr, origin=float(start), stereo=stereo,
                                           channel=wanted), None))
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
                    self.message(f"Cannot show this sound: {error}")
                    continue
                self.data = data
                self.channel_box.configure(values=("Left", "Right") if data["stereo"] else ("Mono",),
                                           state="readonly" if data["stereo"] else "disabled")
                self.channel_text.set({"left": "Left", "right": "Right"}.get(data["channel"], "Mono"))
                self.f_hi = min(self.f_hi if self.f_hi > self.f_lo else DEFAULT_FMAX, data["sr"] / 2)
                self.x_zoom = min(self.x_zoom, max(X_ZOOMS[0], data["duration"]))
                self.t0 = min(self.t0, max(0.0, data["duration"] - self.x_zoom))
                self.redraw()
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    def destroy(self):
        self.stop_playhead()
        if self.analysis is not None:
            self.analysis.close()
        for job in (self._poll_job, self._redraw_job, self._analysis_job):
            if job:
                self.after_cancel(job)
        super().destroy()

    # ---------- พิกัด ----------
    @property
    def nyquist(self):
        return self.data["sr"] / 2 if self.data else 24000.0

    def _area(self, canvas):
        width = max(50, canvas.winfo_width() - GUTTER)
        if canvas is self.spec:
            return GUTTER, LABEL_H, width, max(20, canvas.winfo_height() - LABEL_H - TIME_H)
        return GUTTER, 0, width, max(20, canvas.winfo_height())

    def x_of(self, seconds, canvas=None):
        x0, _y0, width, _h = self._area(canvas or self.spec)
        return x0 + (seconds - self.t0) / self.x_zoom * width

    def time_at(self, x, canvas=None):
        x0, _y0, width, _h = self._area(canvas or self.spec)
        seconds = self.t0 + (min(max(x, x0), x0 + width) - x0) / width * self.x_zoom
        return min(max(0.0, seconds), self.data["duration"] if self.data else seconds)

    def y_of(self, hz):
        _x0, y0, _w, height = self._area(self.spec)
        return y0 + height * (1 - freq_fraction(hz, self.f_lo, self.f_hi, self.log_freq))

    def freq_at(self, y):
        _x0, y0, _w, height = self._area(self.spec)
        fraction = min(1.0, max(0.0, 1 - (y - y0) / height))
        if self.log_freq:
            lo = max(LOG_FMIN, self.f_lo)
            return math.exp(math.log(lo) + fraction * (math.log(self.f_hi) - math.log(lo)))
        return self.f_lo + fraction * (self.f_hi - self.f_lo)

    def _row_freqs(self, height):
        """ความถี่ของแต่ละแถวภาพ บนลงล่าง"""
        fractions = 1 - (np.arange(height) + 0.5) / height
        if self.log_freq:
            lo = max(LOG_FMIN, self.f_lo)
            return np.exp(np.log(lo) + fractions * (np.log(self.f_hi) - np.log(lo)))
        return self.f_lo + fractions * (self.f_hi - self.f_lo)

    # ---------- วาด ----------
    def _schedule_redraw(self):
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(60, self.redraw)

    def redraw(self):
        self._redraw_job = None
        if not self.data:
            return
        self._draw_spectrogram()
        self._draw_oscillogram()
        self._update_bars()

    def _colors_changed(self):
        if self.data:
            self._draw_spectrogram()

    # ---------- Prism Analysis ----------
    def analysis_region(self):
        """ช่วงที่วิเคราะห์: กรอบที่เลือก หรือภาพที่เห็น -> (a, b, low, high) วินาทีในช่วงที่โหลด, Hz"""
        if self.selection:
            return self.selection
        return (self.t0, min(self.t0 + self.x_zoom, self.data["duration"]), self.f_lo, self.f_hi)

    def toggle_analysis(self):
        if self.analysis is not None and self.analysis.exists():
            self.analysis.close()
            self.analysis = None
            return
        self.analysis = AnalysisWindow(self)
        self.after(50, self.analysis.update)

    def _analysis_changed(self):
        """เลือก/เลื่อน/ซูม/ปรับ Contrast: อัปเดตหน้าต่าง Analysis (รวบหลายครั้งติดกันเป็นครั้งเดียว)"""
        if self.analysis is None or not self.analysis.exists():
            return
        if self._analysis_job:
            self.after_cancel(self._analysis_job)
        self._analysis_job = self.after(60, self._analysis_now)

    def _analysis_now(self):
        self._analysis_job = None
        if self.analysis is not None:
            self.analysis.update()

    def _visible_frames(self):
        sr = self.data["sr"]
        first = max(0, int(self.t0 * sr / FFT_HOP))
        last = min(len(self.data["db"]), int(math.ceil((self.t0 + self.x_zoom) * sr / FFT_HOP)) + 1)
        return first, max(first + 1, last)

    def _draw_spectrogram(self):
        canvas, colors = self.spec, COLORS[self.inverse]
        canvas.configure(bg=colors["bg"])
        canvas.delete("all")
        x0, y0, width, height = self._area(canvas)
        width, height = int(width), int(height)
        if self.show_spectrum:
            first, last = self._visible_frames()
            db = column_pool(self.data["db"][first:last], width * 2)
            rgb = spectrogram_rgb(db, self._row_freqs(height), self.data["sr"], self._lut[self.inverse],
                                  self.brightness.get(), self.contrast.get())
            image = Image.fromarray(rgb, "RGB")
            visible = (last - first) * FFT_HOP / self.data["sr"]          # วินาทีของเฟรมที่ใช้ (อาจเกินขอบเล็กน้อย)
            shown = int(round(width * min(1.0, visible / self.x_zoom)))
            image = image.resize((max(1, shown), height), Image.BILINEAR)
            self._spec_photo = ImageTk.PhotoImage(image, master=self)
            start = self.x_of(first * FFT_HOP / self.data["sr"])
            item = canvas.create_image(max(x0, start), y0, anchor="nw", image=self._spec_photo, tags="image")
            self._pil = {key: value for key, value in self._pil.items() if key[0] != "spec"}
            self._pil[("spec", item)] = image
        # แถบป้ายชนิดและพื้นแกน
        canvas.create_rectangle(0, 0, x0 + width, y0, fill=colors["bg"], outline="", tags="chrome")
        canvas.create_rectangle(0, y0 + height, x0 + width, y0 + height + TIME_H, fill=colors["bg"],
                                outline="", tags="chrome")
        canvas.create_rectangle(0, 0, x0 - 1, y0 + height + TIME_H, fill=colors["bg"], outline="", tags="chrome")
        self._freq_ruler()
        self._time_ruler()
        if self.show_ids:
            self._draw_labels()
        self._draw_refs()
        self._draw_flags()
        self._draw_selection()

    def _freq_ruler(self):
        canvas, fg = self.spec, COLORS[self.inverse]["fg"]
        _x0, y0, _width, height = self._area(canvas)
        if self.log_freq:
            lo = max(LOG_FMIN, self.f_lo)
            ticks = [f for f in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000) if lo <= f <= self.f_hi]
        else:
            step = nice_step(self.f_hi - self.f_lo, height, 30)
            first = math.ceil(self.f_lo / step) * step
            ticks = list(np.arange(first, self.f_hi + 1e-6, step))
        for hz in ticks:
            y = self.y_of(hz)
            if y < y0 + 13 or y > y0 + height + 1:        # ตัวเลขบนสุดจะทับแถบป้ายชนิด: ไม่แสดง
                continue
            canvas.create_line(0, y, GUTTER - 1, y, fill=fg, tags="ruler")
            canvas.create_text(GUTTER - 4, y - 1, text=freq_label(hz), anchor="se", fill=fg, font=RULER_FONT,
                               tags="ruler")

    def _time_ruler(self):
        canvas, fg = self.spec, COLORS[self.inverse]["fg"]
        x0, y0, width, height = self._area(canvas)
        top = y0 + height
        clock = self.time_mode == "clock" and self.clock_start is not None
        step = nice_step(self.x_zoom, width, 90, (1, 2, 5)) if not clock else \
            next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if s * width / self.x_zoom >= 90), 600)
        minor = step / 4
        if clock:
            origin = self.clock_start + timedelta(seconds=self.data["origin"])
            seconds0 = origin.hour * 3600 + origin.minute * 60 + origin.second + origin.microsecond / 1e6
            base = seconds0 + self.t0
        else:
            base = 0.0
        first = math.ceil(base / minor - 1e-9) * minor
        tick = first
        while tick <= base + self.x_zoom + 1e-9:
            x = x0 + (tick - base) / self.x_zoom * width
            major = abs(tick / step - round(tick / step)) < 1e-6
            canvas.create_line(x, top, x, top + (TIME_H - 2 if major else 14), fill=fg, tags="ruler")
            if major:
                if clock:
                    whole = int(round(tick)) % 86400
                    text = f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
                else:
                    text = time_label(tick)
                canvas.create_text(x + 2, top + TIME_H - 2, text=text, anchor="sw", fill=fg, font=RULER_FONT,
                                   tags="ruler")
            tick = round(tick + minor, 9)

    def _draw_labels(self):
        canvas, fg = self.spec, COLORS[self.inverse]["fg"]
        x0, _y0, width, _height = self._area(canvas)
        origin, last_end = self.data["origin"], -1e9
        for start, end, name, score, *_rest in self.labels:
            if score < self.min_conf:
                continue
            a, b = start - origin, end - origin
            if b < self.t0 or a > self.t0 + self.x_zoom:
                continue
            x = max(x0, self.x_of(a))
            if x < last_end + 6:
                continue
            item = canvas.create_text(x + 2, LABEL_H / 2 + 1, text=name, anchor="w", fill=fg, font=RULER_FONT,
                                      tags="labels")
            last_end = canvas.bbox(item)[2]

    def _draw_refs(self):
        canvas = self.spec
        x0, y0, width, height = self._area(canvas)
        for hz in self.ref_lines:
            y = self.y_of(hz)
            if y0 <= y <= y0 + height:
                canvas.create_line(x0, y, x0 + width, y, fill=REF_COLOR, dash=(6, 3), tags="refs")
                canvas.create_text(x0 + width - 4, y - 2, text=freq_label(round(hz)), anchor="se", fill=REF_COLOR,
                                   font=RULER_FONT, tags="refs")

    def _draw_flags(self):
        canvas = self.spec
        _x0, y0, _width, height = self._area(canvas)
        for number, flag in enumerate(sorted(self.flags), 1):
            at = flag - self.data["origin"]
            if self.t0 <= at <= self.t0 + self.x_zoom:
                canvas.create_line(self.x_of(at), y0, self.x_of(at), y0 + height, fill="#ff5252", width=2, dash=(6, 3),
                                   tags="flags")
                canvas.create_text(self.x_of(at) + 3, y0 + 2, text=f"Flag {number}", anchor="nw", fill="#ff5252",
                                   font=RULER_FONT, tags="flags")

    def _draw_selection(self):
        for canvas in (self.spec, self.osc):
            canvas.delete("selection")
        self._analysis_changed()
        if not self.selection or not self.data:
            return
        a, b, low, high = self.selection
        fg = COLORS[self.inverse]["fg"]
        x0, y0, width, height = self._area(self.spec)
        xa, xb = self.x_of(a), self.x_of(b)
        ya, yb = max(y0, self.y_of(high)), min(y0 + height, self.y_of(low))
        self.spec.create_rectangle(xa, ya, xb, yb, outline=fg, width=1, tags="selection")
        _ox, _oy, _ow, oheight = self._area(self.osc)
        self.osc.create_rectangle(self.x_of(a, self.osc), 0, self.x_of(b, self.osc), oheight, outline=fg,
                                  tags="selection")

    def _draw_oscillogram(self):
        canvas, colors = self.osc, COLORS[self.inverse]
        canvas.configure(bg=colors["bg"])
        canvas.delete("all")
        x0, _y0, width, height = self._area(canvas)
        width, height = int(width), int(height)
        sr = self.data["sr"]
        first = max(0, int(self.t0 * sr))
        last = min(len(self.data["samples"]), int((self.t0 + self.x_zoom) * sr) + 1)
        per_pixel = max(1e-9, (last - first) / width)
        wave = np.zeros((height, width, 3), np.uint8)
        wave[:] = self._lut[self.inverse][0]
        columns = int(round(width * min(1.0, (last - first) / sr / self.x_zoom)))
        if columns < 1 or last - first < 1:
            pass                                              # ภาพเลยท้ายเสียง: พื้นว่าง
        elif self.osc_db:
            rms = self._column_stat("rms", first, last, columns)
            level = 20 * np.log10(np.maximum(rms, 1e-9) / FULL_SCALE)
            top = ((1 - (level - DB_FLOOR) / -DB_FLOOR) * height).clip(0, height).astype(int)
            rows = np.arange(height)[:, None]
            wave[:, :columns][rows >= top[None, :]] = colors["wave"]
        elif per_pixel >= 2:
            lo, hi = self._column_stat("min", first, last, columns), self._column_stat("max", first, last, columns)
            y_hi, y_lo = self._osc_y(hi, height), self._osc_y(lo, height)
            rows = np.arange(height)[:, None]
            top, bottom = np.round(y_hi), np.maximum(np.round(y_lo), np.round(y_hi))   # อย่างน้อย 1 px แบบเส้นบาง
            wave[:, :columns][(rows >= top[None, :]) & (rows <= bottom[None, :])] = colors["wave"]
        image = Image.fromarray(wave, "RGB")
        self._osc_photo = ImageTk.PhotoImage(image, master=self)
        item = canvas.create_image(x0, 0, anchor="nw", image=self._osc_photo, tags="image")
        self._pil = {key: value for key, value in self._pil.items() if key[0] != "osc"}
        self._pil[("osc", item)] = image
        if not self.osc_db and per_pixel < 2 and last - first > 1:        # ซูมใกล้: เส้นคลื่นจริง
            values = self.data["samples"][first:last].astype(np.float32)
            xs = x0 + (np.arange(first, last) / sr - self.t0) / self.x_zoom * width
            points = np.column_stack([xs, self._osc_y(values, height)]).ravel().tolist()
            canvas.create_line(*points, fill="#%02x%02x%02x" % colors["wave"], tags="wave")
        self._osc_ruler(height)
        self._draw_selection()

    def _column_stat(self, kind, first, last, columns):
        """ค่าต่อคอลัมน์ภาพ: ใช้ค่าทุก 128 ตัวอย่างที่คำนวณไว้เมื่อซูมออก (เร็ว) ไม่งั้นใช้ตัวอย่างจริง"""
        columns = max(1, columns)
        per_column = (last - first) / columns
        if per_column >= FFT_HOP * 2:
            source = {"min": self.data["mins"], "max": self.data["maxs"], "rms": self.data["rms"]}[kind] * FULL_SCALE
            first, last = first // FFT_HOP, max(first // FFT_HOP + 1, last // FFT_HOP)
        else:
            source = self.data["samples"].astype(np.float32)
            if kind == "rms":
                source = np.abs(source)
        part = source[first:last]
        if len(part) == 0:
            return np.zeros(columns, np.float32)
        edges = np.linspace(0, len(part), columns + 1).astype(int)
        edges = np.minimum(edges[:-1], len(part) - 1)
        if kind == "min":
            return np.minimum.reduceat(part, edges)
        if kind == "max":
            return np.maximum.reduceat(part, edges)
        return np.sqrt(np.add.reduceat(part.astype(np.float64) ** 2, edges)
                       / np.maximum(np.diff(np.append(edges, len(part))), 1))

    def _osc_y(self, values, height):
        half = FULL_SCALE / self.osc_zoom
        center = self.osc_center * FULL_SCALE
        return (0.5 - (np.asarray(values, np.float32) - center) / (2 * half)) * height

    def _osc_ruler(self, height):
        canvas, fg = self.osc, COLORS[self.inverse]["fg"]
        if self.osc_db:
            for level in range(-20, int(DB_FLOOR) - 1, -20):
                y = (1 - (level - DB_FLOOR) / -DB_FLOOR) * height
                canvas.create_line(0, y, GUTTER - 1, y, fill=fg)
                canvas.create_text(GUTTER - 4, y - 1, text=f"{level}dB", anchor="se", fill=fg, font=RULER_FONT)
            return
        half = FULL_SCALE / self.osc_zoom
        center = self.osc_center * FULL_SCALE
        step = nice_step(2 * half, height, 22)
        value = math.ceil((center - half) / step) * step
        while value <= center + half:
            y = float(self._osc_y(value, height))
            if 0 < y < height:
                canvas.create_line(0, y, GUTTER - 1, y, fill=fg)
                canvas.create_text(GUTTER - 4, y - 1, text=f"{value:g}", anchor="se", fill=fg, font=RULER_FONT)
            value += step

    def _update_bars(self):
        duration = self.data["duration"]
        self.time_bar.set(self.t0 / duration, min(1.0, (self.t0 + self.x_zoom) / duration))
        ny = self.nyquist
        self.freq_bar.set(1 - self.f_hi / ny, 1 - self.f_lo / ny)
        half = 1 / self.osc_zoom
        self.osc_bar.set((1 - (self.osc_center + half)) / 2, (1 - (self.osc_center - half)) / 2)

    def screen_image(self):
        """oscillogram + spectrogram เป็นภาพเดียว"""
        osc = canvas_image(self.osc, {item: image for (name, item), image in self._pil.items() if name == "osc"})
        spec = canvas_image(self.spec, {item: image for (name, item), image in self._pil.items() if name == "spec"})
        out = Image.new("RGB", (max(osc.width, spec.width), osc.height + 7 + spec.height), "#e6e6e6")
        out.paste(spec, (0, 0))
        out.paste(osc, (0, spec.height + 7))
        return out

    def save_screen_as(self):
        from tkinter import filedialog
        if not self.data:
            return
        target = filedialog.asksaveasfilename(parent=self, defaultextension=".png", filetypes=[("PNG", "*.png")],
                                              initialfile="prism-view.png")
        if target:
            self.screen_image().save(target)
            self.info.set(f"Saved {target}")

    # ---------- ซูม/เลื่อน ----------
    def set_view(self, t0):
        if not self.data:
            return
        self.t0 = min(max(0.0, t0), max(0.0, self.data["duration"] - self.x_zoom))
        self.redraw()

    def scroll_time(self, screens):
        self.set_view(self.t0 + screens * self.x_zoom)

    def zoom_time(self, step, center=None):
        if not self.data:
            return
        center = self.t0 + self.x_zoom / 2 if center is None else center
        levels = [z for z in X_ZOOMS if z <= max(X_ZOOMS[0], self.data["duration"])] or [X_ZOOMS[0]]
        current = min(range(len(levels)), key=lambda i: abs(levels[i] - self.x_zoom))
        self.x_zoom = levels[min(max(current + step, 0), len(levels) - 1)]
        self.set_view(center - self.x_zoom / 2)

    def fit_time(self):
        if self.data:
            self.x_zoom, self.t0 = self.data["duration"], 0.0
            self.redraw()

    def zoom_freq(self, factor):
        span = min(self.nyquist, max(200.0, (self.f_hi - self.f_lo) * factor))
        self.f_hi = min(self.nyquist, self.f_lo + span)
        self.f_lo = max(0.0, self.f_hi - span)
        self.redraw()

    def fit_freq(self):
        self.f_lo, self.f_hi = 0.0, self.nyquist
        self.redraw()

    def zoom_osc(self, factor):
        self.osc_zoom = min(256.0, max(1.0, self.osc_zoom * factor))
        half = 1 / self.osc_zoom
        self.osc_center = min(max(self.osc_center, -1 + half), 1 - half)
        if self.data:
            self._draw_oscillogram()
            self._update_bars()

    def _scrolled(self, args, position, size, total):
        """ค่าจาก Scrollbar -> ตำแหน่งเริ่มใหม่ (moveto / scroll units/pages)"""
        if args[0] == "moveto":
            return float(args[1]) * total
        amount = float(args[1]) * (size if args[2] == "pages" else size / 10)
        return position + amount

    def _time_scrolled(self, *args):
        if self.data:
            self.set_view(self._scrolled(args, self.t0, self.x_zoom, self.data["duration"]))

    def _freq_scrolled(self, *args):
        if not self.data:
            return
        span, ny = self.f_hi - self.f_lo, self.nyquist
        top = ny - self._scrolled(args, ny - self.f_hi, span, ny)
        self.f_hi = min(ny, max(span, top))
        self.f_lo = self.f_hi - span
        self.redraw()

    def _osc_scrolled(self, *args):
        if not self.data:
            return
        half = 1 / self.osc_zoom
        top = 1 - 2 * self._scrolled(args, (1 - (self.osc_center + half)) / 2, half, 1.0)
        self.osc_center = min(max(top - half, -1 + half), 1 - half)
        self._draw_oscillogram()
        self._update_bars()

    def _wheel(self, event):
        if self.data:
            direction = -1 if event.delta > 0 else 1
            self.scroll_time(direction * 0.1)

    # ---------- ปุ่ม ----------
    def toggle_inverse(self):
        self.inverse = not self.inverse
        self.redraw()

    def toggle_osc(self):
        self.osc_db = not self.osc_db
        if self.data:
            self._draw_oscillogram()

    def toggle_spectrum(self):
        self.show_spectrum = not self.show_spectrum
        if self.data:
            self._draw_spectrogram()

    def toggle_log_freq(self):
        self.log_freq = not self.log_freq
        if self.data:
            self._draw_spectrogram()

    def toggle_ids(self):
        self.show_ids = not self.show_ids
        if self.data:
            self._draw_spectrogram()

    def _speed_chosen(self, _event=None):
        self.speed_var.set(str(dict(SPEEDS)[self.speed_text.get()]))
        if self.on_stop:
            self.on_stop()

    def _channel_chosen(self, _event=None):
        self.channel_var.set(self.channel_text.get().lower())
        if self.on_channel:
            self.on_channel()

    def _play_pressed(self):
        if self._play is not None and self.on_stop:
            self.on_stop()
        elif self.on_play:
            self.on_play()

    # ---------- เมาส์ ----------
    def _hover(self, event):
        canvas = event.widget
        canvas.delete("cursor")
        if not self.data or event.x < GUTTER:
            return
        fg = COLORS[self.inverse]["fg"]
        x0, y0, width, height = self._area(canvas)
        if not (y0 <= event.y <= y0 + height):
            return
        seconds = self.time_at(event.x, canvas)
        canvas.create_line(x0, event.y, x0 + width, event.y, fill=fg, tags="cursor")
        canvas.create_line(event.x, y0, event.x, y0 + height, fill=fg, tags="cursor")
        if self._drag and self.selection:
            a, b, low, high = self.selection
            text = (f"{self.data['origin'] + a:.6f}s {b - a:.6f}s {low / 1000:.3f}kHz {(high - low) / 1000:.3f}kHz "
                    f"{self.db_at(seconds, self.freq_at(event.y)):.1f}dB")
        elif canvas is self.spec:
            hz = self.freq_at(event.y)
            text = f"{self.data['origin'] + seconds:.6f}s {hz / 1000:.3f}kHz {self.db_at(seconds, hz):.1f}dB"
        else:
            index = min(len(self.data["samples"]) - 1, int(seconds * self.data["sr"]))
            value = int(self.data["samples"][index])
            level = 20 * math.log10(max(abs(value), 1) / FULL_SCALE)
            text = f"{self.data['origin'] + seconds:.6f}s " + (f"{level:.1f}dB" if self.osc_db else f"{value}")
        anchor = "sw" if event.x < x0 + width - 260 else "se"
        canvas.create_text(event.x + (4 if anchor == "sw" else -4), event.y - 3, text=text, anchor=anchor, fill=fg,
                           font=READOUT_FONT, tags="cursor")

    def db_at(self, seconds, hz):
        """dB ที่เวลา/ความถี่นี้ (เทียบ full scale) ตัวเลขที่แสดงตามเคอร์เซอร์"""
        frame = min(len(self.data["db"]) - 1, max(0, int(round(seconds * self.data["sr"] / FFT_HOP - 1))))
        column = min(self.data["db"].shape[1] - 1, int(round(hz / (self.data["sr"] / FFT_SIZE))))
        return float(self.data["db"][frame, column])

    def _press(self, event):
        event.widget.focus_set()
        if not self.data or event.x < GUTTER:
            return
        canvas = event.widget
        _x0, y0, _w, height = self._area(canvas)
        if canvas is self.spec and event.y > y0 + height:
            return                                           # แกนเวลา
        hz = self.freq_at(event.y) if canvas is self.spec else None
        self._drag = (canvas, self.time_at(event.x, canvas), hz, event.x, event.y)

    def _dragging(self, event):
        if not self._drag:
            return
        canvas, at, hz, _x, _y = self._drag
        now = self.time_at(event.x, canvas)
        if hz is None:                                        # oscillogram: ทุกความถี่
            low, high = 0.0, self.nyquist
        else:
            low, high = sorted((hz, self.freq_at(event.y)))
        self.selection = (*sorted((at, now)), low, high)
        self._draw_selection()
        self._hover(event)

    def _release(self, event):
        if not self._drag:
            return
        _canvas, _at, _hz, x, y = self._drag
        if abs(event.x - x) < 4 and abs(event.y - y) < 4:      # คลิกครั้งเดียว = ยกเลิกกรอบ
            self.clear_selection()
        else:
            self._dragging(event)
            a, b, low, high = self.selection
            if b - a > 0.005 and self.on_select:
                self.on_select(a, b, low, high)
        self._drag = None

    def clear_selection(self):
        self.selection = None
        self._draw_selection()

    def _right_click(self, event):
        if not self.data or event.x < GUTTER:
            return
        seconds, hz = self.time_at(event.x), self.freq_at(event.y)
        if self.selection:
            a, b, low, high = self.selection
            if a <= seconds <= b and low <= hz <= high:           # ซูมพอดีกรอบ
                self.x_zoom = max(0.05, b - a)
                self.f_lo, self.f_hi = low, max(high, low + 100)
                self.set_view(a)
                return
        self.ref_lines = (self.ref_lines + [hz])[-2:]
        self._draw_spectrogram()

    def _clear_refs(self, _event=None):
        self.ref_lines = []
        if self.data:
            self._draw_spectrogram()

    def _double(self, event):
        _x0, y0, _w, height = self._area(self.spec)
        if event.y > y0 + height and self.clock_start is not None:   # ดับเบิลคลิกแกนเวลา
            self.time_mode = "clock" if self.time_mode == "elapsed" else "elapsed"
            self._draw_spectrogram()

    # ---------- เส้นตำแหน่งขณะเล่น ----------
    def start_playhead(self, offset, player, speed=1.0):
        self.stop_playhead()
        self._play = (offset, time.monotonic(), player, speed)
        self.play_btn.configure(**self._face("stop", "■ Stop"))
        self._tick()

    def stop_playhead(self):
        if self._tick_job:
            self.after_cancel(self._tick_job)
            self._tick_job = None
        self._play = None
        self.play_btn.configure(**self._face("play", "▶ Play"))
        for canvas in (self.osc, self.spec):
            canvas.delete("wiper")

    def _tick(self):
        self._tick_job = None
        if self._play is None or not self.data:
            return
        offset, started, player, speed = self._play
        now = offset + (time.monotonic() - started) * speed
        if now > self.data["duration"] or player is None or player.poll() is not None:
            self.stop_playhead()
            return
        if not self.t0 <= now <= self.t0 + self.x_zoom:          # wiper ถึงขอบขวา: เลื่อนไปหน้าถัดไป
            self.set_view(now - self.x_zoom * 0.05)
        fg = COLORS[self.inverse]["fg"]
        for canvas in (self.osc, self.spec):
            canvas.delete("wiper")
            _x0, y0, _w, height = self._area(canvas)
            x = self.x_of(now, canvas)
            canvas.create_line(x, y0, x, y0 + height, fill=fg, width=2, tags="wiper")
        self._tick_job = self.after(FRAME_MS, self._tick)
