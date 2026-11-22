"""หน้าต่าง Add a clip yourself: เลื่อนดู spectrogram ของไฟล์ต้นฉบับ ลากเลือกช่วงเสียง ใส่ชนิดเอง
(หรือให้ BirdNET เดาเฉพาะช่วงนั้น) แล้วตัดเป็นคลิปเข้าแท็บ Review — สำหรับเสียงที่ BirdNET ไม่เจอ

ภาพและเสียงตัวอย่างในหน้าต่างนี้อ่านจากไฟล์ต้นฉบับทีละช่วง (ไฟล์ 1 ชม. ก็ไม่ต้องโหลดทั้งก้อน)
การตัดคลิปจริงทำใน process แยก (manual_clips.py) ผ่านคิวงานเดียวกับการวิเคราะห์
"""

import itertools
import json
import math
import queue
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import country_checklist
import ebird
import manual_clips
import paths
import platform_tools
import playback
import recorder_flags
import result_folders
from audio_io import PREVIEW_PEAK, audio_duration, read_mono, write_preview  # noqa: F401  ใช้ต่อจากที่อื่นได้
from entry_formats import DATE_HINT, TIME_HINT, parse_date_entry, parse_time_entry
from frequency_compare import FlowFrame
from manual_clips import SOUND_TYPES, format_offset, parse_clock, parse_offset
from review_store import load_reviews

PREVIEW_DIR = paths.CACHE / "manual-preview"
VIEW_CHOICES = ("10 s", "20 s", "30 s", "60 s")
LEFT, CANVAS_W, BAR_H, SPEC_H, AXIS_H = 46, 900, 18, 210, 34
FMAX = 12000
MAGMA = ((0.0, (0, 0, 4)), (0.25, (81, 18, 124)), (0.5, (183, 55, 121)),
         (0.75, (252, 137, 97)), (1.0, (252, 253, 191)))
BAR_COLORS = {"Approved": "#4caf50", "Rejected": "#777777", "Unsure": "#9c7fc4", "Pending": "#e0a800",
              "manual": "#00bcd4"}
LABEL_MIN = 0.2                # ชื่อที่ BirdNET ทายบนภาพ: แสดงตั้งแต่คะแนนนี้ (ต่ำกว่า min_conf = ตัวเทา)
AUDIO_TYPES = [("Audio", "*.wav *.WAV *.mp3 *.flac *.m4a *.ogg *.aif *.aiff"), ("All", "*.*")]


# ----------------------------- เสียง/ภาพ (ไม่ขึ้นกับ Tk) -----------------------------
def _palette():
    stops = np.array([stop for stop, _ in MAGMA])
    colors = np.array([color for _, color in MAGMA], dtype=float)
    levels = np.linspace(0, 1, 256)
    return np.stack([np.interp(levels, stops, colors[:, c]) for c in range(3)], axis=1).astype(np.uint8)


PALETTE = _palette()


def index_labels(path, sid=None, floor=LABEL_MIN):
    """ชนิดที่ BirdNET ทายอันดับ 1 ของทุกช่วง 3 วินาที (จากดัชนีเสียง)
    -> [(start, end, ชื่อ eBird, คะแนน, ชื่อวิทยาศาสตร์ eBird, เป็นนกไหม)]
    ช่วงติดกันที่เป็นชนิดเดียวกันรวมเป็นอันเดียว; ไม่มีดัชนี = []"""
    import sound_index
    import species_names
    try:
        if not sid:
            from field_audio_to_ebird import source_id
            sid = source_id(Path(path))
        index = sound_index.load(sid)
    except OSError:
        return []
    if index is None:
        return []
    spans = []
    for i, start in enumerate(index["starts"]):
        guesses = sound_index.window_guesses(index, i, top=1)
        if not guesses or guesses[0][2] < floor:
            continue
        scientific, common, score = guesses[0]
        names = species_names.to_ebird(scientific, common) or {"common": common, "scientific": scientific}
        name = names["common"]
        start = float(start)
        last = spans[-1] if spans else None
        if last and last[2] == name and start <= last[1] - sound_index.WINDOW + sound_index.HOP + 1e-6:
            spans[-1] = (last[0], start + sound_index.WINDOW, name, max(last[3], score)) + last[4:]
        else:
            spans.append((start, start + sound_index.WINDOW, name, score, names["scientific"],
                          species_names.is_bird_label(scientific, common)))
    return spans


def _expected_of(summary):
    """{ชื่อวิทยาศาสตร์ eBird: ความน่าจะเป็นสัปดาห์นี้} จาก Expected species ของผลชุด; ยังไม่เคยถาม = None"""
    import near_misses
    found = near_misses.load_expected(summary) if summary and Path(summary).is_file() else None
    return None if found is None else {item["scientific"]: item["week"] for item in found["species"]}


def spectrogram_image(samples, sr, width, height, fmax=FMAX):
    """spectrogram เชิงเส้น 0..fmax Hz -> ภาพ RGB ขนาด width x height (สีแบบ magma)"""
    n_fft = 1024 if sr <= 32000 else 2048
    hop = n_fft // 2
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    count = 1 + (samples.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    keep = np.fft.rfftfreq(n_fft, 1 / sr) <= min(fmax, sr / 2)
    columns = []
    for first in range(0, count, 512):                   # ทีละชุด กันใช้ RAM มากกับช่วง 60 วิ
        index = np.arange(first, min(count, first + 512))[:, None] * hop + np.arange(n_fft)
        spectrum = np.abs(np.fft.rfft(samples[index] * window, axis=1))[:, keep]
        columns.append(20 * np.log10(spectrum.astype(np.float32) + 1e-9))
    db = np.concatenate(columns)
    if len(db) > width:                                  # หลายเฟรมต่อหนึ่งพิกเซล: เก็บค่าที่ดังสุด (ไม่ให้เสียงสั้นหาย)
        db = np.maximum.reduceat(db, np.linspace(0, len(db), width, endpoint=False).astype(int), axis=0)
    low, high = np.percentile(db, 25), np.percentile(db, 99.9)
    level = np.clip((db - low) / max(high - low, 1.0), 0, 1)
    rgb = PALETTE[(level * 255).astype(np.uint8).T[::-1]]
    return Image.fromarray(np.ascontiguousarray(rgb), "RGB").resize((width, height), Image.Resampling.BILINEAR)


def species_matches(text, country, limit=40):
    """ชนิดในระบบชื่อ eBird ที่ชื่ออังกฤษ/วิทยาศาสตร์มีคำที่พิมพ์ -> [(common, scientific)]
    ชนิดที่พบตามปกติในประเทศที่บันทึก (จีน/ไทย) ขึ้นก่อน"""
    words = str(text).casefold().split()
    if not words:
        return []
    by_scientific = country_checklist._species()[0]
    found = []
    for scientific, row in by_scientific.items():
        name = f"{row['common']} {scientific}".casefold()
        if all(word in name for word in words):
            status = row.get(country, "") if country else ""
            rank = 0 if status in ("native", "endemic") else 1 if status and status not in country_checklist.CHECK else 2
            starts = 0 if name.startswith(words[0]) or f" {words[0]}" in f" {row['common'].casefold()}" else 1
            found.append((rank, starts, row["common"], scientific))
    found.sort()
    return [(common, scientific) for _rank, _starts, common, scientific in found[:limit]]


# ----------------------------- หน้าต่าง -----------------------------
class ManualClipWindow:
    def __init__(self, app, summary, rows, source=None, offset=None):
        self.app = app
        self.summary = Path(summary) if summary else None
        self.rows = list(rows)
        self.q = queue.Queue()
        self.sources = {}                 # ข้อความในเมนู -> Path
        self.context = None               # ไฟล์ที่เปิดอยู่: path, start, lat, lon, place, date_source, duration, summary
        self.view_start, self.view_len = 0.0, 30.0
        self.selection = None             # (t0, t1) วินาทีนับจากต้นไฟล์
        self._fmax = FMAX
        self._render_id = 0
        self._spec_photo = None
        self._drag_from = None
        self._play = None                 # (t0, t1, เวลาที่เริ่มเล่น)
        self._suggest_busy = False
        self._preview_number = itertools.count()
        self._pending_offset = offset
        self.labels = None                # ชนิดที่ BirdNET ทายต่อช่วง (index_labels) ของไฟล์ที่เปิดอยู่; None = ยังไม่มีดัชนี
        self.expected = {}                # ชนิดนก -> ความน่าจะเป็นที่จุด/สัปดาห์นี้ (Expected species ของผลชุด)
        self.flags = []                   # flag ที่กดตอนอัดในไฟล์ที่เปิดอยู่ (วินาทีจากต้นไฟล์)
        self._expected_asked = set()
        import near_misses
        self.min_conf = near_misses.min_conf_of(self.rows)

        win = self.win = tk.Toplevel(app.root)
        # จอเตี้ย (เช่นโน้ตบุ๊ก 768 px): ภาพ spectrogram เตี้ยลง ให้ส่วนอื่นยังพอดีหน้าต่าง
        self.spec_h = SPEC_H if win.winfo_screenheight() >= 900 else max(120, SPEC_H - (900 - win.winfo_screenheight()))
        win.title("Add a clip yourself")
        win.geometry(f"1010x{min(870, win.winfo_screenheight() - 70)}")
        win.minsize(980, 600)
        win.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(win, padding=(12, 8, 12, 10))
        body.pack(fill="both", expand=True)
        # ปุ่มเพิ่มคลิปอยู่ล่างสุดเสมอ (จอเล็ก: ส่วนกลางหดก่อน)
        ttk.Label(body, text="The clip is cut from the original recording at full quality and normalized like "
                  "BirdNET's clips, then listed in Review clips as Pending — rate and approve it there. "
                  "Re-analysing the recording never removes clips you added.",
                  foreground="#888", wraplength=960).pack(side="bottom", anchor="w", pady=(6, 0))
        add = ttk.Frame(body)
        add.pack(side="bottom", fill="x", pady=(8, 0))
        self.add_btn = ttk.Button(add, text="✚ Add clip to Review", style="QuickAction.TButton", command=self.add)
        self.add_btn.pack(side="left")
        ttk.Button(add, text="Close", command=self.close).pack(side="right")
        self.add_state = tk.StringVar()
        ttk.Label(add, textvariable=self.add_state, wraplength=640).pack(side="left", padx=10)
        ttk.Label(body, text="For sounds BirdNET missed: find the sound, drag across it on the spectrogram "
                  "(leave 1–3 s before and after), name the species, then add it to Review.",
                  foreground="#808080").pack(anchor="w", pady=(0, 6))

        rec = ttk.Frame(body)
        rec.pack(fill="x")
        ttk.Label(rec, text="Recording").pack(side="left")
        self.source_choice = tk.StringVar()
        self.source_box = ttk.Combobox(rec, textvariable=self.source_choice, state="readonly", width=52)
        self.source_box.pack(side="left", padx=6)
        self.source_box.bind("<<ComboboxSelected>>", lambda _event: self._source_selected())
        ttk.Button(rec, text="Other recording…", command=self.choose_other).pack(side="left")
        self.rec_info = tk.StringVar()
        self.rec_info_label = ttk.Label(body, textvariable=self.rec_info, foreground="#808080", wraplength=960)
        self.rec_info_label.pack(anchor="w", pady=(3, 0))

        # รายละเอียดของไฟล์ที่ยังไม่เคยวิเคราะห์ (ซ่อนไว้จนกว่าจะต้องใช้)
        self.details = ttk.Frame(body)
        self.date_text, self.time_text, self.coords_text = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for column, (label, var, width) in enumerate((
                (f"Date ({DATE_HINT})", self.date_text, 12), (f"Start time ({TIME_HINT})", self.time_text, 9),
                ("Coordinates (lat,lon or Google Maps link)", self.coords_text, 30))):
            ttk.Label(self.details, text=label).grid(row=0, column=column, sticky="w", padx=(0, 10))
            ttk.Entry(self.details, textvariable=var, width=width).grid(row=1, column=column, sticky="w", padx=(0, 10))
        ttk.Button(self.details, text="Use these details", command=self._apply_details).grid(row=1, column=3)

        nav = ttk.Frame(body)
        nav.pack(fill="x", pady=(6, 3))
        ttk.Label(nav, text="Go to").pack(side="left")
        self.goto = tk.StringVar()
        goto = ttk.Entry(nav, textvariable=self.goto, width=11)
        goto.pack(side="left", padx=4)
        goto.bind("<Return>", lambda _event: self.go())
        self.time_mode = tk.StringVar(value="file")
        ttk.Radiobutton(nav, text="position in file (mm:ss)", variable=self.time_mode, value="file",
                        command=self._mode_changed).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(nav, text="clock time (hh:mm:ss)", variable=self.time_mode, value="clock",
                        command=self._mode_changed).pack(side="left", padx=(4, 0))
        ttk.Button(nav, text="Go", width=4, command=self.go).pack(side="left", padx=4)
        self.view_choice = tk.StringVar(value="30 s")
        view_box = ttk.Combobox(nav, textvariable=self.view_choice, values=VIEW_CHOICES, state="readonly", width=6)
        view_box.pack(side="right")
        view_box.bind("<<ComboboxSelected>>", lambda _event: self._view_length_changed())
        ttk.Label(nav, text="View").pack(side="right", padx=(10, 4))
        for text, step in (("▶▶", 1.0), ("▶", 0.25), ("◀", -0.25), ("◀◀", -1.0)):
            ttk.Button(nav, text=text, width=5, command=lambda step=step: self.step(step)).pack(side="right", padx=1)

        self.canvas = tk.Canvas(body, width=LEFT + CANVAS_W + 8, height=BAR_H + self.spec_h + AXIS_H,
                                bg="#141414", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(anchor="w")
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Left>", lambda _event: self.step(-0.25))
        self.canvas.bind("<Right>", lambda _event: self.step(0.25))
        self.canvas.bind("<space>", lambda _event: self.play_selection(fallback_view=True))
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _event: self.cursor_info.set(""))
        info = ttk.Frame(body)
        info.pack(fill="x", pady=(2, 0))
        # ชื่อที่ BirdNET ทายทุก 3 วินาที (จากดัชนีเสียง) บนภาพ
        self.show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(info, text="BirdNET labels", variable=self.show_labels,
                        command=self._draw_overlay).pack(side="left", padx=(0, 8))
        self.view_info = tk.StringVar()
        ttk.Label(info, textvariable=self.view_info, foreground="#808080").pack(side="left")
        # เวลาจริงตรงเมาส์ (ทศนิยมวินาที) ไว้เทียบกับเวลาในภาพถ่าย
        self.cursor_info = tk.StringVar()
        ttk.Label(info, textvariable=self.cursor_info, foreground="#3a87c8").pack(side="right")

        sel = ttk.Frame(body)
        sel.pack(fill="x", pady=(4, 0))
        ttk.Label(sel, text="Selection").pack(side="left")
        self.sel_from, self.sel_to = tk.StringVar(), tk.StringVar()
        for var, label in ((self.sel_from, "from"), (self.sel_to, "to")):
            ttk.Label(sel, text=label).pack(side="left", padx=(8, 3))
            entry = ttk.Entry(sel, textvariable=var, width=11)
            entry.pack(side="left")
            entry.bind("<Return>", lambda _event: self._selection_typed())
        ttk.Button(sel, text="Set", width=4, command=self._selection_typed).pack(side="left", padx=4)
        self.sel_info = tk.StringVar(value="Drag across the spectrogram to select the sound")
        ttk.Label(sel, textvariable=self.sel_info, foreground="#808080").pack(side="left", padx=8)

        play = FlowFrame(body)
        play.pack(fill="x", pady=(4, 0))
        play.add(ttk.Button(play, text="▶ Play selection", command=self.play_selection))
        play.add(ttk.Button(play, text="▶ Play view", command=self.play_view))
        play.add(ttk.Button(play, text="Listen tools…", command=self.listening_tools))
        play.add(ttk.Button(play, text="■ Stop", command=self.stop))
        play.add(playback.control(play), space=4)
        self.ask_btn = play.add(ttk.Button(play, text="Ask BirdNET about the selection", command=self.ask_birdnet), space=10)
        # เสียงแบบเดียวกันที่อื่น (ดัชนีเสียง): ไฟล์นี้ ผลชุดนี้ หรือทุกผลชุด
        play.add(ttk.Button(play, text="Find similar…", command=self.find_similar))
        self.ask_state = tk.StringVar()
        play.add(ttk.Label(play, textvariable=self.ask_state, foreground="#808080"), space=2)

        columns = ("species", "score", "area", "checklist")
        self.suggest_tree = ttk.Treeview(body, columns=columns, show="headings", height=3)
        for key, label, width in (("species", "BirdNET guess for the selection (click to use)", 380),
                                  ("score", "Score", 60), ("area", "This place & week", 190),
                                  ("checklist", "Country checklist", 330)):
            self.suggest_tree.heading(key, text=label)
            self.suggest_tree.column(key, width=width, stretch=(key == "checklist"))
        self.suggest_tree.pack(fill="x", pady=(4, 0))
        self.suggest_tree.bind("<<TreeviewSelect>>", lambda _event: self._use_suggestion())
        self._suggestions = {}

        species = ttk.LabelFrame(body, text="Species", padding=(8, 4, 8, 6))
        species.pack(fill="x", pady=(6, 0))
        species.columnconfigure(1, weight=1)
        self.common, self.scientific = tk.StringVar(), tk.StringVar()
        ttk.Label(species, text="Common name").grid(row=0, column=0, sticky="w")
        common = ttk.Entry(species, textvariable=self.common)
        common.grid(row=0, column=1, sticky="ew", padx=6)
        common.bind("<KeyRelease>", self._typed_name)
        common.bind("<Down>", lambda _event: self._focus_matches())
        ttk.Label(species, text="Scientific name").grid(row=0, column=2, sticky="w")
        scientific = ttk.Entry(species, textvariable=self.scientific, width=28)
        scientific.grid(row=0, column=3, sticky="w", padx=6)
        scientific.bind("<KeyRelease>", lambda _event: self._update_checklist())
        self.matches = tk.Listbox(species, height=3, activestyle="dotbox", exportselection=False)
        self.matches.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=(4, 0))
        self.matches.bind("<<ListboxSelect>>", lambda _event: self._use_match())
        self.matches.bind("<Return>", lambda _event: self._use_match())
        self._match_names = []
        self.checklist_info = tk.StringVar()
        self.sound_type, self.notes = tk.StringVar(), tk.StringVar()
        ttk.Label(species, text="Sound type").grid(row=2, column=0, sticky="w", pady=(6, 0))
        extra = ttk.Frame(species)
        extra.grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Combobox(extra, textvariable=self.sound_type, values=SOUND_TYPES, state="readonly",
                     width=12).pack(side="left")
        self.checklist_label = ttk.Label(extra, textvariable=self.checklist_info, wraplength=520)
        self.checklist_label.pack(side="left", padx=(12, 0))
        ttk.Label(species, text="Notes").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(species, textvariable=self.notes, width=28).grid(row=2, column=3, sticky="w", padx=6, pady=(6, 0))
        # eBird: ชนิดนี้เคยมีรายงานในมณฑลนี้ช่วงเดียวกันของปีก่อน ๆ ไหม (ต้องมี eBird key)
        self.season_info = tk.StringVar()
        self.season_label = ttk.Label(species, textvariable=self.season_info, wraplength=900)
        self.season_label.grid(row=3, column=1, columnspan=3, sticky="w", padx=6, pady=(4, 0))
        self._season = None               # (ไฟล์, ผล season_reports, ชื่อมณฑล)


        app._install_clipboard_fix()
        self._fill_sources(source)
        self.win.after(100, self._poll)

    # ---------- ไฟล์ต้นฉบับ ----------
    def _fill_sources(self, preferred=None):
        self.sources = {}
        for path in manual_clips.sources_in(self.rows):
            context = manual_clips.source_context(self.rows, path)
            label = path.name + (f"  —  started {context['start']:%Y-%m-%d %H:%M:%S}"
                                 if context and context["start"] else "")
            self.sources[label] = path
        self.source_box.configure(values=list(self.sources))
        choice = next((label for label, path in self.sources.items() if str(path) == str(preferred)), None)
        choice = choice or next(iter(self.sources), None)
        if choice:
            self.source_choice.set(choice)
            self._source_selected()
        else:
            self.rec_info.set("No recording in the current result. Choose one with Other recording…")

    def _source_selected(self):
        path = self.sources.get(self.source_choice.get())
        if path is None:
            return
        context = manual_clips.source_context(self.rows, path)
        self.details.pack_forget()
        if not path.is_file():
            self.context = None
            self.rec_info.set(f"Recording not found: {path}. If it moved, choose it with Other recording…")
            return
        self._open(path, dict(context, summary=self.summary))

    def choose_other(self):
        name = filedialog.askopenfilename(parent=self.win, title="Choose a recording", filetypes=AUDIO_TYPES)
        if not name:
            return
        path = Path(name).resolve()
        known = next((p for p in self.sources.values() if str(p) == str(path)), None)
        if known is not None:                     # อยู่ในผลชุดนี้แล้ว
            self.source_choice.set(next(label for label, p in self.sources.items() if p == known))
            self._source_selected()
            return
        self.rec_info.set(f"Reading {path.name} …")
        results = list(self.app._results.values())

        out_q = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            # วิเคราะห์ไว้แล้วในผลชุดอื่น: ใช้ชุดนั้น (เวลาเริ่ม/พิกัดตรงกับคลิปของ BirdNET)
            for summary in results:
                try:
                    rows = load_reviews(Path(summary))
                except Exception:  # noqa: BLE001  ไฟล์เสีย/กำลังเขียน
                    continue
                if any(str(row.get("Source path") or "") == str(path) for row in rows):
                    out_q.put(("other-known", path, Path(summary), rows))
                    return
            try:
                from field_audio_to_ebird import recording_context_preview
                out_q.put(("other", path, recording_context_preview(path), None))
            except Exception as exc:  # noqa: BLE001
                out_q.put(("other", path, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _other_known(self, path, summary, rows):
        self.summary, self.rows = summary, rows
        self._fill_sources(path)

    def _other_ready(self, path, found, error):
        if error:
            self.rec_info.set(f"Cannot read {path.name}: {error}")
            return
        start = found["datetime"] if found["datetime"] and found["has_time"] else None
        context = {"start": start, "lat": found["lat"], "lon": found["lon"], "place": found.get("place") or "",
                   "date_source": found["datetime_source"] or "", "habitat": [], "summary": None}
        self.source_choice.set(path.name + "  (not analyzed in this result)")
        self.date_text.set(found["datetime"].strftime("%Y%m%d") if found["datetime"] else "")
        self.time_text.set(start.strftime("%H%M%S") if start else "")
        previous = self.context or {}
        if found["lat"] is not None and found["lon"] is not None:
            self.coords_text.set(f"{found['lat']:.6f},{found['lon']:.6f}")
        elif previous.get("lat") is not None:          # ไม่มี geotag: เสนอพิกัดของไฟล์ที่เปิดอยู่ (ให้ตรวจก่อนใช้)
            self.coords_text.set(f"{previous['lat']:.6f},{previous['lon']:.6f}")
        else:
            self.coords_text.set("")
        self._other_path = path
        if start is None or context["lat"] is None or context["lon"] is None:
            self.details.pack(fill="x", pady=(6, 0), after=self.rec_info_label)
            self.rec_info.set(f"{path.name} is not in any result yet: check the recording details "
                              "(coordinates are copied from the recording you had open), then click Use these details")
            self.context = None
            return
        self.details.pack_forget()
        self._open(path, context)

    def _apply_details(self):
        path = getattr(self, "_other_path", None)
        if path is None:
            return
        try:
            day = parse_date_entry(self.date_text.get())
            clock = parse_time_entry(self.time_text.get())
        except ValueError:
            messagebox.showwarning("Recording start", f"Enter the date as {DATE_HINT} and the start time as {TIME_HINT}",
                                   parent=self.win)
            return
        from field_audio_to_ebird import parse_coords
        coords = parse_coords(self.coords_text.get())
        if coords is None:
            messagebox.showwarning("Location", "Enter coordinates, e.g. 30.5928,114.3055", parent=self.win)
            return
        start = datetime.combine(day.date() if isinstance(day, datetime) else day, clock)
        self.details.pack_forget()
        self._open(path, {"start": start, "lat": coords[0], "lon": coords[1], "place": "",
                          "date_source": "entered when adding a clip", "habitat": [], "summary": None})

    def _open(self, path, context):
        context = dict(context, path=path, duration=None)
        if context.get("summary") is None and context.get("start"):
            # ไฟล์ที่ยังไม่มีผลในชุดนี้: ใช้โฟลเดอร์เดียวกับที่การวิเคราะห์ BirdNET 3.0 จะสร้าง
            out = Path(self.app.out.get().strip() or paths.PROGRAM).expanduser()
            context["summary"] = result_folders.folder_for(out, context["start"], "_v3preview") / "summary.xlsx"
        self.context = context
        self.selection = None
        self.labels = None
        self.flags = []
        self._clear_suggestions()
        self.rec_info.set(f"Reading {path.name} …")

        results = self.q
        sid = next((str(row["Source ID"]) for row in self.rows
                    if str(row.get("Source path") or "") == str(path) and row.get("Source ID")), None)
        summary = context.get("summary")

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            results.put(("duration", path, audio_duration(path)))
            results.put(("flags", path, recorder_flags.flags_of(sid, path)))
            results.put(("labels", path, index_labels(path, sid), _expected_of(summary)))
        threading.Thread(target=work, daemon=True).start()

    def _describe_recording(self):
        context = self.context
        parts = [f"{format_offset(context['duration'], 0)} long"]
        if context["start"]:
            parts.insert(0, f"started {context['start']:%Y-%m-%d %H:%M:%S}")
        if context["lat"] is not None and context["lon"] is not None:
            parts.append(f"{context['lat']:.5f}, {context['lon']:.5f}")
        if context.get("summary"):
            parts.append(f"new clips go to {Path(context['summary']).parent.name}")
        self.rec_info.set(f"{context['path'].name}:  " + "  ·  ".join(parts))

    # ---------- เลื่อน/วาด ----------
    def _clamp_view(self):
        duration = (self.context or {}).get("duration") or 0.0
        self.view_start = min(max(0.0, self.view_start), max(0.0, duration - self.view_len))

    def step(self, fraction):
        if not self.context or self.context.get("duration") is None:
            return
        self.view_start += fraction * self.view_len
        self._clamp_view()
        self.render()

    def _view_length_changed(self):
        middle = self.view_start + self.view_len / 2
        self.view_len = float(self.view_choice.get().split()[0])
        self.view_start = middle - self.view_len / 2
        self._clamp_view()
        self.render()

    def _to_offset(self, text):
        if self.time_mode.get() == "clock":
            if not self.context or not self.context.get("start"):
                raise ValueError("The recording start time is unknown; use the position in the file")
            return parse_clock(text, self.context["start"])
        return parse_offset(text)

    def _label(self, offset, decimals=0):
        if self.time_mode.get() == "clock" and self.context and self.context.get("start"):
            return self._clock(offset, decimals)
        return format_offset(offset, decimals)

    def go(self):
        if not self.context or self.context.get("duration") is None:
            return
        try:
            offset = self._to_offset(self.goto.get())
        except ValueError as exc:
            messagebox.showwarning("Go to", str(exc), parent=self.win)
            return
        if not 0 <= offset <= self.context["duration"]:
            messagebox.showwarning("Go to", "That time is outside the recording", parent=self.win)
            return
        self.view_start = offset - self.view_len / 4
        self._clamp_view()
        self.render()

    def _mode_changed(self):
        self._draw_overlay()
        self._update_selection_fields()

    def render(self):
        context = self.context
        if not context or context.get("duration") is None:
            return
        self._render_id += 1
        render_id, start, length = self._render_id, self.view_start, self.view_len
        self.canvas.delete("loading")
        self.canvas.create_text(LEFT + CANVAS_W / 2, BAR_H + self.spec_h / 2, text="Loading …", fill="#bbb",
                                tags="loading")

        results = self.q
        spec_height = self.spec_h

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            try:
                samples, sr = read_mono(context["path"], start, start + length)
                image = spectrogram_image(samples, sr, CANVAS_W, spec_height)
                results.put(("view", render_id, image, min(FMAX, sr / 2), None))
            except Exception as exc:  # noqa: BLE001
                results.put(("view", render_id, None, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self._draw_overlay()

    def _x(self, offset):
        return LEFT + (offset - self.view_start) / self.view_len * CANVAS_W

    def _offset_at(self, x):
        x = min(max(x, LEFT), LEFT + CANVAS_W)
        return self.view_start + (x - LEFT) / CANVAS_W * self.view_len

    def _draw_overlay(self):
        canvas = self.canvas
        canvas.delete("overlay")
        if not self.context:
            return
        top, bottom = BAR_H, BAR_H + self.spec_h
        for khz in range(2, int(self._fmax / 1000) + 1, 2):        # แกนความถี่ (kHz)
            y = bottom - khz * 1000 / self._fmax * self.spec_h
            canvas.create_line(LEFT - 4, y, LEFT, y, fill="#999", tags="overlay")
            canvas.create_text(LEFT - 6, y, text=f"{khz}k", anchor="e", fill="#bbb",
                               font=("Helvetica", 9), tags="overlay")
        # แกนเวลา 2 แถว: ตำแหน่งในไฟล์ (เทา) และเวลาจริงของนาฬิกาเครื่องอัด (เหลือง) ที่วินาทีเต็มของนาฬิกา
        step = {10: 1, 20: 2, 30: 5, 60: 5}.get(int(self.view_len), 5)
        start = self.context.get("start")
        canvas.create_text(LEFT - 6, bottom + 4, text="file", anchor="ne", fill="#999",
                           font=("Helvetica", 8), tags="overlay")
        if start:
            canvas.create_text(LEFT - 6, bottom + 18, text="clock", anchor="ne", fill="#ffd54f",
                               font=("Helvetica", 8), tags="overlay")
        tick = math.ceil(self.view_start / step - 1e-9) * step
        while tick <= self.view_start + self.view_len + 1e-6:
            x = self._x(tick)
            canvas.create_line(x, bottom, x, bottom + 4, fill="#999", tags="overlay")
            canvas.create_text(x, bottom + 4, text=format_offset(tick, 0), anchor="n", fill="#ccc",
                               font=("Helvetica", 9), tags="overlay")
            tick += step
        if start:
            tick = math.ceil((start.second + self.view_start) / step - 1e-9) * step - start.second
            while tick <= self.view_start + self.view_len + 1e-6:
                x = self._x(tick)
                canvas.create_text(x, bottom + 18, text=self._clock(tick, 0), anchor="n", fill="#ffd54f",
                                   font=("Helvetica", 9), tags="overlay")
                tick += step
        # คลิปที่มีอยู่แล้วของไฟล์นี้ (แถบด้านบน): BirdNET = เหลือง/เขียว/เทา ตามผลตรวจ, ที่เพิ่มเอง = ฟ้า
        for row in self.rows:
            if str(row.get("Source path") or "") != str(self.context["path"]):
                continue
            a, b = manual_clips._number(row.get("Start offset (s)")), manual_clips._number(row.get("End offset (s)"))
            if a is None or b is None or b < self.view_start or a > self.view_start + self.view_len:
                continue
            color = BAR_COLORS["manual"] if manual_clips.is_manual(row) else BAR_COLORS.get(
                str(row.get("Review status") or "Pending"), BAR_COLORS["Pending"])
            x0, x1 = max(LEFT, self._x(a)), min(LEFT + CANVAS_W, self._x(max(b, a + 0.3)))
            canvas.create_rectangle(x0, 2, x1, BAR_H - 3, fill=color, outline="", tags="overlay")
            canvas.create_text(x0 + 2, BAR_H / 2 - 1, text=str(row.get("Species (common)") or ""), anchor="w",
                               fill="#111", font=("Helvetica", 8), tags="overlay")
        if self.show_labels.get() and self.labels:
            self._draw_labels(top)
        for number, flag in enumerate(self.flags, 1):             # flag ที่กดตอนอัด
            if self.view_start <= flag <= self.view_start + self.view_len:
                canvas.create_line(self._x(flag), top, self._x(flag), bottom, fill="#ff5252", width=2, dash=(6, 3),
                                   tags="overlay")
                canvas.create_text(self._x(flag) + 3, bottom - 3, text=f"⚑ {number}", anchor="sw", fill="#ff5252",
                                   font=("Helvetica", 10), tags="overlay")
        if self.selection:
            a, b = self.selection
            if b >= self.view_start and a <= self.view_start + self.view_len:
                x0, x1 = max(LEFT, self._x(a)), min(LEFT + CANVAS_W, self._x(b))
                canvas.create_rectangle(x0, top, x1, bottom, outline="#ffeb3b", width=2, tags="overlay")
        end = min(self.view_start + self.view_len, self.context["duration"] or 0)
        labels = ("labels: BirdNET's top guess per 3 s (gray = below min_conf, ✗ = not expected here, "
                  "teal = not a bird)" if self.labels else "no BirdNET labels until the sound index is made")
        self.view_info.set(f"Showing {format_offset(self.view_start, 0)}–{format_offset(end, 0)} in the file"
                           + (f"  ·  clock {self._clock(self.view_start, 0)}–{self._clock(end, 0)}" if start else "")
                           + "  ·  bars: clips already made (cyan = yours)"
                           + (f"  ·  {labels}" if self.show_labels.get() else ""))
        canvas.tag_raise("playhead")

    def _ask_expected(self):
        """ยังไม่มีรายชื่อชนิดที่ควรพบของผลชุดนี้: ถามโมเดลพื้นที่เป็นงานเบื้องหลัง (~2 วิ) แล้ววาดป้ายใหม่"""
        summary = Path(self.context.get("summary") or "")
        if not summary.is_file() or summary in self._expected_asked:
            return
        self._expected_asked.add(summary)
        path = self.context["path"]

        def done(code, _lines):
            if code == 0 and self.win.winfo_exists() and self.context and self.context["path"] == path:
                self.expected = _expected_of(summary) or {}
                self._draw_overlay()
        self.app._start(self.app._launch() + ["--sound-index", "expected", str(summary)], None,
                        f"Expected species — {summary.parent.name}", is_analyze=False, on_done=done, background=True)

    def _draw_labels(self, top):
        """ป้ายชื่อที่ BirdNET ทายบนส่วนบนของภาพ (ความถี่สูง ซึ่งเสียงนกส่วนใหญ่ไม่ถึง) — 2 แถว ไม่ซ้อนกัน
        เส้นใต้ป้าย = ช่วงที่ทายชนิดนั้น; ขาว = ถึง min_conf, เทา = ต่ำกว่า, เทาเข้ม ✗ = นกที่ไม่ควรพบที่นี่สัปดาห์นี้,
        เขียวอมฟ้า = ไม่ใช่นก (แมลง กบ …)"""
        import near_misses
        canvas = self.canvas
        ends = [-1e9, -1e9]
        view_end = self.view_start + self.view_len
        for a, b, name, score, scientific, bird in self.labels:
            if b < self.view_start or a > view_end:
                continue
            x0, x1 = max(LEFT, self._x(a)), min(LEFT + CANVAS_W, self._x(b))
            unexpected = bird and self.expected and self.expected.get(scientific, 0.0) < near_misses.GEO_EXPECTED
            text = f"{name} {score:.2f}" + (" ✗" if unexpected else "")
            width = len(text) * 5.6 + 6
            row = next((r for r in range(len(ends)) if x0 >= ends[r] + 3), None)
            if row is None:
                continue                      # ช่วงนี้แน่นเกิน: ข้าม (ซูมเข้าเพื่อดู)
            ends[row] = max(x1, x0 + width)
            y = top + 2 + row * 17
            color = ("#80cbc4" if not bird else "#6d6d6d" if unexpected
                     else "#ffffff" if score >= self.min_conf else "#9e9e9e")
            canvas.create_rectangle(x0, y, min(LEFT + CANVAS_W, x0 + width), y + 12, fill="#000000",
                                    outline="", tags="overlay")
            canvas.create_text(x0 + 3, y + 6, text=text, anchor="w", fill=color, font=("Helvetica", 8),
                               tags="overlay")
            canvas.create_line(x0, y + 14, x1, y + 14, fill=color, width=2, tags="overlay")

    # ---------- เลือกช่วง ----------
    def _press(self, event):
        self.canvas.focus_set()
        if self.context and self.context.get("duration") is not None:
            self._drag_from = self._offset_at(event.x)

    def _drag(self, event):
        if self._drag_from is None:
            return
        here = self._offset_at(event.x)
        self.selection = (min(self._drag_from, here), max(self._drag_from, here))
        self._draw_overlay()
        self._update_selection_fields()

    def _release(self, event):
        if self._drag_from is None:
            return
        here = self._offset_at(event.x)
        if abs(here - self._drag_from) < self.view_len / 300:      # คลิกเฉย ๆ = ยกเลิกช่วงที่เลือก
            self.selection = None
        else:
            self.selection = (round(min(self._drag_from, here), 2), round(max(self._drag_from, here), 2))
        self._drag_from = None
        self._draw_overlay()
        self._update_selection_fields()

    def _hover(self, event):
        if not self.context or self.context.get("duration") is None or not LEFT <= event.x <= LEFT + CANVAS_W:
            self.cursor_info.set("")
            return
        offset = self._offset_at(event.x)
        text = f"⌖ {format_offset(offset)} in file"
        if self.context.get("start"):
            text += f"  ·  clock {self._clock(offset, 1, date=True)}"
        if BAR_H <= event.y <= BAR_H + self.spec_h:
            text += f"  ·  {(BAR_H + self.spec_h - event.y) / self.spec_h * self._fmax / 1000:.1f} kHz"
        self.cursor_info.set(text)

    def _update_selection_fields(self):
        if not self.selection:
            self.sel_from.set("")
            self.sel_to.set("")
            self.sel_info.set("Drag across the spectrogram to select the sound")
            return
        a, b = self.selection
        self.sel_from.set(self._label(a, 1))
        self.sel_to.set(self._label(b, 1))
        text = f"{b - a:.1f} s  ·  file {format_offset(a)}–{format_offset(b)}"
        if self.context.get("start"):
            text += f"  ·  clock {self._clock(a, 1, date=True)}–{self._clock(b, 1)}"
        self.sel_info.set(text)

    def _clock(self, offset, decimals=0, date=False):
        """เวลาจริง (นาฬิกาเครื่องอัด) ของตำแหน่งนี้: '14:54:47' / '14:54:47.3' / '2026-10-18 14:54:47.3'"""
        when = self.context["start"] + timedelta(seconds=round(offset, decimals))
        text = when.strftime("%Y-%m-%d %H:%M:%S" if date else "%H:%M:%S")
        return text + (f".{when.microsecond // 100000}" if decimals else "")

    def _selection_typed(self):
        if not self.context or self.context.get("duration") is None:
            return
        try:
            a, b = self._to_offset(self.sel_from.get()), self._to_offset(self.sel_to.get())
        except ValueError as exc:
            messagebox.showwarning("Selection", str(exc), parent=self.win)
            return
        a, b = max(0.0, min(a, b)), min(self.context["duration"], max(a, b))
        if b - a <= 0:
            return
        self.selection = (a, b)
        if a < self.view_start or b > self.view_start + self.view_len:
            self.view_start = a - max(1.0, (self.view_len - (b - a)) / 2)
            self._clamp_view()
            self.render()
        self._draw_overlay()
        self._update_selection_fields()

    # ---------- ฟังเสียง ----------
    def play_selection(self, fallback_view=False):
        if not self.selection:
            if fallback_view:
                self.play_view()
            else:
                self.sel_info.set("Select a span first (drag across the spectrogram)")
            return
        self._play_span(*self.selection)

    def play_view(self):
        if self.context and self.context.get("duration") is not None:
            self._play_span(self.view_start, min(self.view_start + self.view_len, self.context["duration"]))

    def listening_tools(self):
        if not self.context or not self.context.get("duration"):
            return
        from listening_tools import ListeningToolsWindow
        self.stop()
        a, b = self.selection or (self.view_start, min(self.view_start + self.view_len, self.context["duration"]))
        clock = self.context["start"] + timedelta(seconds=a) if self.context.get("start") else None
        scientific = " ".join(self.scientific.get().split()[:2])
        self._listening_window = ListeningToolsWindow(
            self.win, self.context["path"], app=self.app, start=a, end=b, clock_start=clock,
            species=(self.common.get().strip(), scientific) if len(scientific.split()) == 2 else None,
            location=(self.context["lat"], self.context["lon"]) if self.context.get("lat") is not None else None,
            flags=self.flags)

    def _play_span(self, a, b):
        path = self.context["path"]
        target = PREVIEW_DIR / f"preview-{next(self._preview_number)}.wav"

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            try:
                samples, sr = read_mono(path, a, b)
                write_preview(samples, sr, target)
                results.put(("play", target, a, b))
            except Exception as exc:  # noqa: BLE001
                results.put(("play-error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        self.app.review_stop()
        self._play = None
        self.canvas.delete("playhead")

    def _tick_playhead(self):
        if self._play is None or not self.win.winfo_exists():
            return
        a, b, started = self._play
        now = a + time.monotonic() - started
        player = self.app.audio_player
        if now > b or player is None or player.poll() is not None:
            self._play = None
            self.canvas.delete("playhead")
            return
        self.canvas.delete("playhead")
        if self.view_start <= now <= self.view_start + self.view_len:
            x = self._x(now)
            self.canvas.create_line(x, BAR_H, x, BAR_H + self.spec_h, fill="#ffffff", width=1, tags="playhead")
        self.win.after(40, self._tick_playhead)

    # ---------- ชนิด ----------
    def _country(self):
        if not self.context:
            return None
        return country_checklist.country_at(self.context.get("lat"), self.context.get("lon"))

    def _typed_name(self, event=None):
        if event is not None and event.keysym in ("Down", "Up", "Return", "Tab"):
            return
        self._match_names = species_matches(self.common.get(), self._country())
        self.matches.delete(0, "end")
        for common, scientific in self._match_names:
            self.matches.insert("end", f"{common}  —  {scientific}")
        exact = next((sci for common, sci in self._match_names
                      if common.casefold() == self.common.get().strip().casefold()), None)
        if exact:
            self.scientific.set(exact)
        self._update_checklist()

    def _focus_matches(self):
        if self._match_names:
            self.matches.focus_set()
            self.matches.selection_clear(0, "end")
            self.matches.selection_set(0)
            self.matches.activate(0)

    def _use_match(self):
        chosen = self.matches.curselection()
        if chosen and chosen[0] < len(self._match_names):
            common, scientific = self._match_names[chosen[0]]
            self.common.set(common)
            self.scientific.set(scientific)
            self._update_checklist()

    def _update_checklist(self):
        if not self.context:
            return
        result = country_checklist.check(self.scientific.get().strip(), self.context.get("lat"),
                                         self.context.get("lon"), self.common.get().strip())
        self.checklist_info.set(f"Checklist: {country_checklist.describe(result)}" if result else "")
        self.checklist_label.configure(
            foreground="#c77c10" if country_checklist.needs_check(result) else "#4a9a4f")
        self._update_season()

    def _start_season(self):
        """ดึงรายงาน eBird ช่วงเดียวกันของปีก่อน ๆ ของมณฑลนี้ (เบื้องหลัง ครั้งเดียวต่อไฟล์)"""
        context, key = self.context, self.app.ebird_key.get().strip()
        self._season = None
        if not key or not context.get("start") or context.get("lat") is None:
            self.season_info.set("")
            return
        self.season_info.set("eBird: checking the same dates in past years …")
        path = context["path"]

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            try:
                region = ebird.region_of(context["lat"], context["lon"], key)
                if region is None:
                    raise ebird.EBirdError("no eBird region for this place")
                season = ebird.season_reports(region, context["start"], key)
                ebird.taxonomy()                  # โหลดระบบชื่อไว้ก่อน: ค้นชื่อตอนพิมพ์ไม่ต้องรอเน็ต
                results.put(("season", path, season, ebird.region_name(region, key), None))
            except ebird.EBirdError as exc:
                results.put(("season", path, None, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _update_season(self):
        if not self._season or self._season[0] != self.context.get("path"):
            return
        _path, season, region_name = self._season
        scientific, common = self.scientific.get().strip(), self.common.get().strip()
        entry = ebird.species(scientific, common) if scientific or common else None
        if entry is None:
            self.season_info.set("")
            return
        from birdnet_gui import format_season
        text = format_season(season, entry["code"], region_name)
        self.season_info.set("eBird: " + text)
        self.season_label.configure(foreground="#c77c10" if text.startswith("⚠") else "#4a9a4f")

    # ---------- ถาม BirdNET ----------
    def _clear_suggestions(self):
        self._suggestions = {}
        self.suggest_tree.delete(*self.suggest_tree.get_children())
        self.ask_state.set("")

    def ask_birdnet(self):
        if self._suggest_busy:
            return
        problem = self._missing_for_birdnet()
        if problem:
            messagebox.showinfo("Ask BirdNET", problem, parent=self.win)
            return
        a, b = self.selection
        context = self.context
        cmd = self.app._launch() + ["--manual-clip", "suggest", "--source", str(context["path"]),
                                    "--start", f"{a:.3f}", "--end", f"{b:.3f}",
                                    "--recorded", context["start"].isoformat(),
                                    "--lat", str(context["lat"]), "--lon", str(context["lon"])]
        self._suggest_busy = True
        self.ask_btn.configure(state="disabled")
        self._clear_suggestions()
        self.ask_state.set("BirdNET is listening to the selection … (a few seconds)")

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            result, error = None, None
            try:
                proc = platform_tools.start_child(cmd)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line.startswith("SUGGEST "):
                        result = json.loads(line[8:])
                    elif line.startswith("ERROR "):
                        error = line[6:]
                proc.wait()
            except (OSError, ValueError) as exc:
                error = str(exc)
            results.put(("suggest", result, error or (None if result is not None else "BirdNET gave no answer")))
        threading.Thread(target=work, daemon=True).start()

    def find_similar(self):
        """หน้าต่าง Find similar sounds ด้วยช่วงที่เลือกเป็นตัวอย่าง"""
        if not self.context or self.context.get("duration") is None:
            messagebox.showinfo("Find similar", "Open a recording first", parent=self.win)
            return
        if not self.selection:
            messagebox.showinfo("Find similar", "Select the sound first (drag across the spectrogram)", parent=self.win)
            return
        from field_audio_to_ebird import source_id
        a, b = self.selection
        path = Path(self.context["path"])
        self.app.open_find_similar({
            "sid": self.context.get("source_id") or source_id(path), "source": path,
            "summary": self.context.get("summary"), "start": a, "end": b,
            "label": f"{path.name} {format_offset(a)}–{format_offset(b)}"
                     + (f" ({self.common.get().strip()})" if self.common.get().strip() else ""),
            "common": self.common.get().strip(), "scientific": self.scientific.get().strip()})

    def _missing_for_birdnet(self):
        if not self.context or self.context.get("duration") is None:
            return "Open a recording first"
        if not self.selection:
            return "Select the sound first (drag across the spectrogram)"
        if not self.context.get("start") or self.context.get("lat") is None:
            return "The recording date/time and location are needed"
        return None

    def _show_suggestions(self, ranking, error):
        self._suggest_busy = False
        if self.win.winfo_exists():
            self.ask_btn.configure(state="normal")
        if error:
            self.ask_state.set(f"BirdNET failed: {error}")
            return
        self._clear_suggestions()
        if not ranking:
            self.ask_state.set("BirdNET has no guess for this selection — identify it by ear")
            return
        for number, item in enumerate(ranking):
            area = {True: "expected ✓", False: "not expected ✗", None: "unknown"}[item.get("expected")]
            area += self._season_mark(item["scientific"], item["common"])
            iid = f"s{number}"
            self._suggestions[iid] = item
            self.suggest_tree.insert("", "end", iid=iid, values=(
                f"{item['common']}  ({item['scientific']})", f"{item['confidence']:.2f}", area,
                item.get("checklist") or ""))
        self.ask_state.set("Low scores are only hints — confirm by ear and with reference sounds")

    def _season_mark(self, scientific, common):
        """' · eBird ✓/✗' = มี/ไม่มีรายงานในมณฑลนี้ช่วงเดียวกันของปีก่อน ๆ ('' ถ้ายังไม่มีข้อมูล)"""
        if not self._season or self._season[0] != (self.context or {}).get("path"):
            return ""
        try:
            entry = ebird.species(scientific, common)
        except ebird.EBirdError:
            return ""
        if entry is None:
            return ""
        status = ebird.season_status(self._season[1], entry["code"])
        return {"reported": " · eBird ✓", "not reported": " · eBird ✗"}.get(status, "")

    def _use_suggestion(self):
        chosen = self.suggest_tree.selection()
        item = self._suggestions.get(chosen[0]) if chosen else None
        if item:
            self.common.set(item["common"])
            self.scientific.set(item["scientific"])
            self._typed_name()

    # ---------- เพิ่มคลิป ----------
    def add(self):
        problem = self._missing_for_birdnet()
        if problem:
            messagebox.showinfo("Add clip", problem, parent=self.win)
            return
        a, b = self.selection
        if b - a < manual_clips.MIN_SECONDS:
            messagebox.showinfo("Add clip", f"Select at least {manual_clips.MIN_SECONDS} s", parent=self.win)
            return
        common, scientific = self.common.get().strip(), " ".join(self.scientific.get().split())
        if not common or len(scientific.split()) < 2:
            messagebox.showinfo("Add clip", "Enter the common name and the scientific name "
                                "(type a few letters and pick the species from the list)", parent=self.win)
            return
        result = country_checklist.check(scientific, self.context["lat"], self.context["lon"], common)
        if country_checklist.needs_check(result) and not messagebox.askokcancel(
                "Check the species", f"{common}: {country_checklist.describe(result)}.\nAdd it anyway?",
                parent=self.win):
            return
        context = self.context
        cmd = self.app._launch() + [
            "--manual-clip", "add", "--summary", str(context["summary"]), "--source", str(context["path"]),
            "--start", f"{a:.3f}", "--end", f"{b:.3f}", "--recorded", context["start"].isoformat(),
            "--lat", str(context["lat"]), "--lon", str(context["lon"]),
            "--common", common, "--scientific", scientific, "--sound-type", self.sound_type.get(),
            "--notes", self.notes.get().strip(), "--place", context.get("place") or "",
            "--date-source", context.get("date_source") or ""]
        self.add_state.set("Adding … (waits for the running job to finish)" if self.app.job_running else "Adding …")
        self.app._start(cmd, self.add_btn, f"Add clip: {common} at {self._clock(a)}", is_analyze=False,
                        on_done=self._added)

    def _added(self, code, lines):
        added = next((line for line in reversed(lines) if line.startswith("MANUAL ")), None)
        error = next((line[6:] for line in reversed(lines) if line.startswith("ERROR ")), None)
        alive = self.win.winfo_exists()
        if code == 0 and added:
            info = json.loads(added[7:])
            self.app.show_manual_result(info["summary"], info["file"])
            if alive:
                self.summary = Path(info["summary"])
                self.context["summary"] = self.summary
                self.rows = list(self.app.review_rows.values())
                self.selection = None
                self._draw_overlay()
                self._update_selection_fields()
                self.add_state.set(f"Added ✓ {info['common']} {info['clock']} — it is in Review clips (Pending). "
                                   "Select another sound to add more.")
            return
        message = error or ("stopped" if code == 130 else f"finished with exit {code}")
        if alive:
            self.add_state.set(f"Could not add the clip: {message}")
        messagebox.showerror("Could not add the clip", message, parent=self.win if alive else None)

    # ---------- รับผลจาก thread ----------
    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "duration":
                    path, seconds = payload
                    if self.context and self.context["path"] == path:
                        if not seconds:
                            self.rec_info.set(f"Cannot read the length of {path.name}")
                            continue
                        self.context["duration"] = seconds
                        if self._pending_offset is not None:
                            self.view_start = self._pending_offset - 3
                            self._pending_offset = None
                        self._clamp_view()
                        self._describe_recording()
                        self._start_season()
                        self._update_checklist()
                        self.render()
                elif kind == "view":
                    render_id, image, fmax, error = payload
                    if render_id != self._render_id:
                        continue                      # ผู้ใช้เลื่อนไปแล้ว
                    self.canvas.delete("loading", "spec")
                    if error:
                        self.canvas.create_text(LEFT + CANVAS_W / 2, BAR_H + self.spec_h / 2, fill="#f88",
                                                text=f"Cannot draw: {error}", tags="spec")
                    else:
                        self._fmax = fmax
                        self._spec_photo = ImageTk.PhotoImage(image, master=self.win)
                        self.canvas.create_image(LEFT, BAR_H, anchor="nw", image=self._spec_photo, tags="spec")
                        self.canvas.tag_lower("spec")
                    self._draw_overlay()
                elif kind == "flags":
                    if self.context and self.context["path"] == payload[0]:
                        self.flags = payload[1]
                        self._draw_overlay()
                elif kind == "labels":
                    path, spans, expected = payload
                    if self.context and self.context["path"] == path:
                        self.labels, self.expected = spans or None, expected or {}
                        self._draw_overlay()
                        if spans and expected is None:
                            self._ask_expected()
                elif kind == "other":
                    self._other_ready(*payload)
                elif kind == "other-known":
                    self._other_known(*payload)
                elif kind == "play":
                    target, a, b = payload
                    self.app.review_stop()
                    try:
                        self.app.audio_player = platform_tools.play_audio(target)
                    except (OSError, RuntimeError) as exc:
                        self.sel_info.set(f"Cannot play: {exc}")
                        continue
                    self._play = (a, b, time.monotonic())
                    self._tick_playhead()
                elif kind == "play-error":
                    self.sel_info.set(f"Cannot play: {payload[0]}")
                elif kind == "suggest":
                    self._show_suggestions(*payload)
                elif kind == "season":
                    path, season, region_name, error = payload
                    if self.context and self.context["path"] == path:
                        if error:
                            self.season_info.set(f"eBird (past years): {error}")
                        else:
                            self._season = (path, season, region_name)
                            self.season_info.set("")
                            self._update_season()
        except queue.Empty:
            pass
        self.win.after(80, self._poll)

    def close(self):
        self.stop()
        self.win.destroy()
        self.app._manual_window = None
        shutil.rmtree(PREVIEW_DIR, ignore_errors=True)     # ไฟล์ที่ยังเล่นอยู่บน Windows ลบรอบหน้า
