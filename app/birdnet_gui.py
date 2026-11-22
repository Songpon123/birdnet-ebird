#!/usr/bin/env python3
"""Simple desktop UI for the BirdNET to eBird workflow."""

import os
import sys
import queue
import re
import threading
import json
import time
import webbrowser
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from entry_formats import DATE_HINT, TIME_HINT, parse_date_entry, parse_time_entry
from habitat import MISMATCH_MIN_CONF, SITE_HABITATS
import paths
import platform_tools
from platform_tools import open_path, play_audio, request_stop
from app_settings import load_settings, save_settings
from audio_io import collect_audio
from result_roots import SETTING as OUTPUT_FOLDERS, all_results, done_folder, known_roots, remembered
from review_store import SOUND_TYPES, load_reviews, save_review
import auto_checks
import country_checklist
import ebird
import clip_photos
import field_log
import merge_clips
import playback
import ready_folders
import recorder_flags
import manual_clips
import species_names
from spectrogram_view import SpectrogramView
import background_species
import upload_sheet
import second_opinion
import result_folders
import sound_type
import tk_threads
import xeno_canto

PYTHON = platform_tools.child_python()   # Windows: python.exe แทน pythonw.exe (อ่าน stdout ได้)
SCRIPT = Path(__file__).resolve().parent / "field_audio_to_ebird.py"
ICON = Path(__file__).resolve().parent / "assets" / "app_icon.png"


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


REVIEW_STATUS_FILTERS = ("All", "Pending", "Approved", "Rejected", "Not sure", "Check", "Near a flag")
# ปุ่มจริงบนคีย์บอร์ด (ไม่ขึ้นกับ layout ไทย/อังกฤษ): mac = virtual key code ใน byte บนสุดของ keycode,
# Windows = virtual-key code
_MAC_KEYS = {0: "a", 1: "s", 3: "f", 8: "c", 15: "r", 32: "u", 35: "p", 18: "1", 19: "2", 20: "3", 21: "4",
             23: "5", 36: "return", 76: "return", 49: "space"}
_WINDOWS_KEYS = {65: "a", 83: "s", 70: "f", 67: "c", 82: "r", 85: "u", 80: "p", 49: "1", 50: "2", 51: "3",
                 52: "4", 53: "5", 97: "1", 98: "2", 99: "3", 100: "4", 101: "5", 13: "return", 32: "space"}


def physical_key(event):
    """ปุ่มลัดของแท็บ Review จาก event -> 'a', 'r', '1' … 'return', 'space', 'up', 'down' หรือ None"""
    keysym = str(event.keysym).lower()
    if keysym in ("return", "kp_enter"):
        return "return"
    if keysym in ("space", "up", "down"):
        return keysym
    if len(keysym) == 1 and keysym.isascii():         # layout อังกฤษ: ใช้ตัวอักษรตรง ๆ
        return keysym if keysym in "asfcrup12345" else None
    if sys.platform == "darwin":
        return _MAC_KEYS.get(event.keycode >> 24)
    return _WINDOWS_KEYS.get(event.keycode) if os.name == "nt" else None


def second_look_note(row):
    """คลิปที่ BirdNET เจอตอนฟังซ้ำ (near_misses.py) -> ข้อความอธิบาย หรือ ''"""
    note = str(row.get("Found by") or "")
    return note if note.startswith("Second look") else ""


def review_status_shown(wanted, status, check, flagged=False):
    """คลิปสถานะนี้ (ข้อความในตาราง) แสดงเมื่อเลือก Show = wanted ไหม"""
    if wanted in ("", "All"):
        return True
    if wanted == "Check":
        return bool(check)
    if wanted == "Near a flag":
        return flagged
    return status == wanted or (wanted == "Approved" and status == "Ready file missing")


def result_label(info):
    """ผลวิเคราะห์หนึ่งชุด -> ข้อความในเมนู Results"""
    match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})(\d{2})", info["folder"])
    when = f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}" if match else info["folder"]
    sources = ", ".join(info["sources"][:2]) + (f" +{len(info['sources']) - 2}" if len(info["sources"]) > 2 else "")
    reviewed = info["approved"] + info["rejected"] + info.get("unsure", 0)
    model = ("" if info["folder"].endswith("_v3preview") else "  ·  BirdNET 3.0 pruned"
             if info["folder"].endswith("_v3pruned") else "  ·  BirdNET 2.4")
    return f"{when}  ·  {sources or 'unknown source'}  ·  {info['clips']} clips, {reviewed} reviewed{model}"


def species_links(code, region=None):
    """หน้าของชนิดใน eBird และ Macaulay Library (รูป/เสียงเรียงคะแนนสูงสุด; เสียงกรองตามมณฑลถ้ารู้)"""
    library = f"https://search.macaulaylibrary.org/catalog?taxonCode={quote(code)}&sort=rating_rank_desc"
    return {"species": f"https://ebird.org/species/{quote(code)}",
            "photo": library + "&mediaType=photo",
            "audio": library + "&mediaType=audio" + (f"&regionCode={quote(region)}" if region else "")}


def format_ebird(result):
    """ผลเช็ก eBird -> ข้อความสั้นใต้ภาพ spectrogram"""
    if not result.get("known"):
        return f"eBird: {result.get('scientific', '')} is not in the eBird taxonomy (check the name)"
    parts = [f"eBird: {result['common']}"]
    region = result.get("region_name") or result.get("region")
    if result.get("in_region") is True:
        parts.append(f"reported in {region} ✓")
    elif result.get("in_region") is False:
        parts.append(f"⚠ never reported in {region} on eBird — check carefully")
    if result.get("recent"):
        recent = result["recent"]
        parts.append(f"seen within {ebird.NEARBY_KM} km in the last {ebird.RECENT_DAYS} days "
                     f"(latest {recent['date']}{', ' + recent['where'] if recent['where'] else ''})")
    elif result.get("recent_checked"):
        parts.append(f"no report within {ebird.NEARBY_KM} km in the last {ebird.RECENT_DAYS} days")
    elif result.get("region"):
        parts.append("recent sightings not checked (recording older than 30 days)")
    if result.get("hotspot"):
        parts.append(f"nearest hotspot: {result['hotspot']['name']} ({result['hotspot']['km']} km)")
    return "  ·  ".join(parts)


def format_season(season, code, region_name=None):
    """รายงานใน eBird ช่วงเดียวกันของปีก่อน ๆ (มณฑลเดียวกัน) -> ข้อความสั้นใต้บรรทัด eBird"""
    where = region_name or season["region"]
    center = date.fromisoformat(season["center"])
    around = f"±{ebird.SEASON_HALF_WINDOW} days of {center:%b} {center.day}, {season['first_year']}–{season['last_year']}"
    status = ebird.season_status(season, code)
    if status == "reported":
        days = season["reported"][code]
        years = ", ".join(sorted({day[:4] for day in days}))
        return f"{where}, {around}: reported on {len(days)} of {season['days']} days ({years}) ✓"
    if status == "few reports":
        return (f"{where}, {around}: only {season['species_total']} species reported — "
                "too few eBird reports to judge")
    return (f"⚠ {where}, {around}: not reported ({season['species_total']} other species were) — "
            "unusual then, or just under-recorded; listen carefully")


def season_flags(rows, seasons):
    """คลิปที่ชนิดไม่มีรายงานในมณฑลช่วงเดียวกันของปีก่อน ๆ (มณฑลที่มีข้อมูลพอ) -> {(วันที่, ชื่อวิทยาศาสตร์)}
    seasons = {(วันที่ ISO, lat 2 ตำแหน่ง, lon 2 ตำแหน่ง): ผล season_reports}"""
    flagged = set()
    for row in rows:
        group = _season_group(row)
        season = seasons.get(group) if group else None
        scientific = row.get("Reviewed scientific") or row.get("Species (scientific)")
        entry = ebird.species(scientific, row.get("Reviewed common") or row.get("Species (common)") or "") \
            if season and scientific else None
        if entry and ebird.season_status(season, entry["code"]) == "not reported":
            flagged.add((group[0], str(scientific)))
    return flagged


def _clip_spans_and_clock(row):
    """ช่วงที่ BirdNET เจอ (วินาทีนับจากต้นคลิป) และเวลาจริงของต้นคลิป (None ถ้าไม่รู้)
    คลิปของ BirdNET เริ่มก่อนเสียงแรก windows[0][0] วินาที; คลิปที่เพิ่มเองเริ่มที่ Start offset"""
    try:
        windows = json.loads(row.get("Detection boxes") or "{}").get("windows", [])
    except (TypeError, ValueError, AttributeError):
        windows = []
    spans = [(float(a), float(b)) for a, b, *_rest in windows]
    start, offset = manual_clips.recording_start_of(row), _number(row.get("Start offset (s)"))
    clock = None
    if start is not None and offset is not None:
        from datetime import timedelta
        clock = start + timedelta(seconds=offset - (spans[0][0] if spans else 0.0))
    return spans, clock


def _clip_boxes(row):
    """Detection boxes ของคลิป (ช่วงที่ BirdNET เจอ + กรอบเสียง วินาทีนับจากต้นคลิป) -> dict หรือ None"""
    try:
        boxes = json.loads(row.get("Detection boxes") or "null")
    except (TypeError, ValueError):
        return None
    return boxes if isinstance(boxes, dict) else None


def _season_group(row):
    """(วันที่บันทึก, lat, lon) ของคลิป -> ใช้จัดกลุ่มคำถามช่วงฤดู (None ถ้าข้อมูลไม่ครบ)"""
    lat, lon = _number(row.get("Latitude")), _number(row.get("Longitude"))
    try:
        day = json.loads(row.get("Analysis settings") or "{}").get("date")
    except (TypeError, ValueError):
        day = None
    return (day, round(lat, 2), round(lon, 2)) if day and lat is not None and lon is not None else None


def clip_checklist(row, scientific=None, common=None):
    """สถานภาพของชนิดในคลิปตาม checklist ของประเทศที่บันทึก (จีน ฮ่องกง ไต้หวัน ไทย) ไม่ต้องต่อเน็ต"""
    return country_checklist.check(scientific or row.get("Species (scientific)"),
                                   _number(row.get("Latitude")), _number(row.get("Longitude")),
                                   common or row.get("Species (common)") or "")


def format_checklist(result):
    """ผล checklist ของประเทศ -> ข้อความสั้นใต้ภาพ spectrogram ('' = นอกจีน/ไทย หรือไม่ใช่ชนิดนก)"""
    if not result:
        return ""
    if country_checklist.needs_check(result):
        return f"Checklist: ⚠ {country_checklist.describe(result)}"
    return f"Checklist: {country_checklist.describe(result)} ✓"


def format_second_opinion(result):
    """ผลความเห็นที่สอง -> ข้อความสั้นใต้ภาพ spectrogram"""
    ranking = [r for r in result.get("ranking", []) if r.get("similarity") is not None]
    if not ranking:
        return "Second opinion: no usable xeno-canto references for these species"
    text = ("Second opinion (similarity to nearby xeno-canto recordings): " +
            "  ·  ".join(f"{r['common']} {r['similarity']:.2f}" for r in ranking))
    current = next((r for r in ranking if r["scientific"] == result.get("current")), None)
    top = ranking[0]
    if current is None:
        text += "\nNo usable references for the current species"
    elif top["scientific"] != current["scientific"]:
        text += (f"\n⚠ Closer to {top['common']} than to {current['common']} "
                 "— listen with Reference sounds… before approving")
    elif len(ranking) > 1:
        text += f"   ✓ agrees with {current['common']}"
    missing = [r["common"] for r in result.get("ranking", []) if r.get("similarity") is None]
    if missing:
        text += f"   (no references: {', '.join(missing)})"
    return text


def xeno_canto_url(scientific):
    """'Phylloscopus inornatus [ssp]' -> xeno-canto species page (None ถ้าไม่ใช่ชื่อสองคำ)"""
    parts = str(scientific or "").split()
    if len(parts) < 2 or not (parts[0].isalpha() and parts[1].isalpha()):
        return None
    return "https://xeno-canto.org/species/" + quote(f"{parts[0].capitalize()}-{parts[1].lower()}")

DEF_AUDIO = ""
DEF_OUT   = str(Path.home() / "BirdNET_eBird")
COORD_HINT = "e.g. 13.75633,100.50177"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("BirdNET → eBird clipper")
        # สูงตามจอ (แท็บ Review มีตาราง + ตัวดูเสียง + ช่องตรวจ) แต่ไม่เกินจอ
        root.geometry(f"1000x{max(700, min(920, root.winfo_screenheight() - 60))}")
        try:   # ไอคอนหน้าต่าง/Dock (บน mac แทนไอคอน Python)
            self._icon = tk.PhotoImage(file=str(ICON))
            root.iconphoto(True, self._icon)
        except tk.TclError:
            pass
        self.q: queue.Queue = queue.Queue()
        self.proc = None
        self.job_running = False
        self.quick_busy = False
        self._overrides = False
        self.quick_q: queue.Queue = queue.Queue()
        self.quick_files = []
        self.audio_player = None
        self._closing = False
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if platform_tools.IS_MAC:             # Cmd+Q / ระบบสั่งปิดเครื่อง: ผ่านขั้นตอนปิดปกติ (หยุดงานอย่างปลอดภัย)
            root.createcommand("::tk::mac::Quit", self._on_close)

        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        # ---- แถบปุ่มเลือกไฟล์ (บน) / แถบสถานะงาน (ล่าง เห็นตลอด) / แท็บ+รายละเอียด (กลาง ลากเส้นแบ่งได้) ----
        toolbar = ttk.Frame(outer)
        toolbar.pack(side="top", fill="x", pady=(0, 4))
        run_row = ttk.Frame(outer)
        run_row.pack(side="bottom", fill="x", pady=(6, 0))
        self.paned = ttk.Panedwindow(outer, orient="vertical")
        self.paned.pack(fill="both", expand=True)
        self.nb = ttk.Notebook(self.paned)
        self.paned.add(self.nb, weight=3)
        analyze = ttk.Frame(self.nb, padding=8)
        merge = ttk.Frame(self.nb, padding=8)
        review = ttk.Frame(self.nb, padding=8)
        self.nb.add(analyze, text="Settings")
        self.nb.add(review, text="Review clips")       # เรียงตามลำดับงาน: ตั้งค่า -> ตรวจ -> รวมคลิป
        self.nb.add(merge, text="Merge clips")
        self._build_analyze(analyze)
        self._build_merge(merge)
        self.root.bind("<<RecordingsMatched>>", self._recordings_matched)
        self._build_review(review)
        self._build_toolbar(toolbar)
        self.review_tab = review
        self.nb.select(analyze)

        # ---- แถบสถานะงาน: รายละเอียด / progress / สถานะ / Stop / Jobs ----
        run_row.columnconfigure(1, weight=1)
        # รายละเอียดถูกพับไว้ก่อน เพื่อให้หน้าเริ่มต้นอ่านง่าย
        self.log_visible = False
        self.log_toggle = ttk.Button(run_row, text="Show details ▴", command=self._toggle_log)
        self.log_toggle.grid(row=0, column=0, padx=(0, 10))
        self.progress = ttk.Progressbar(run_row, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(run_row, textvariable=self.status).grid(row=0, column=2, sticky="e")
        # งานที่รันอยู่ + คิว: หยุดได้ (ไฟล์ที่เสร็จแล้วยังอยู่) ดูรายการได้ที่ Jobs…
        self.stop_btn = ttk.Button(run_row, text="■ Stop", command=self.stop_job, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=(10, 0))
        self.jobs_btn = ttk.Button(run_row, text="Jobs…", command=self.show_jobs)
        self.jobs_btn.grid(row=0, column=4, padx=(6, 0))
        # ปิดเครื่องเมื่องานทั้งหมดเสร็จ (รวมงานเบื้องหลัง) — นับถอยหลัง 60 วิ กดยกเลิกได้; ครั้งเดียว ไม่จำไว้
        self.shutdown_after = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_row, text="Shut down when done", variable=self.shutdown_after,
                        command=self._shutdown_toggled).grid(row=0, column=5, padx=(10, 0))
        self._shutdown_idle = 0
        self._shutdown_window = None
        self.job_queue = []                   # งานที่รอ (dict: title, cmd, button, is_analyze, added)
        self.current_job = None               # งานที่กำลังรัน
        self._jobs_window = None

        logf = ttk.Frame(self.paned)          # ใส่ใน paned เมื่อกด Show details
        self.log_frame = logf
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log = tk.Text(logf, height=6, wrap="none", bg="#111", fg="#ddd",
                           insertbackground="#ddd", font=("Menlo" if sys.platform == "darwin" else "Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = sb.set

        # Ctrl+C/V/X/A + คลิกขวา ให้ทำงานทุก keyboard layout (เช่น ไทย)
        self._install_clipboard_fix()
        # โหลด pandas/numpy ล่วงหน้าตอนผู้ใช้ยังไม่กดอะไร -> กดเลือกไฟล์ครั้งแรกไม่ต้องรอ import
        root.after(500, lambda: threading.Thread(target=self._preload_pipeline, daemon=True).start())
        # ผลที่วิเคราะห์ไว้แล้ว: ใส่ในเมนู Results และเปิดชุดล่าสุดที่ดูค้างไว้ (ไม่ต้องวิเคราะห์ใหม่)
        root.after(300, lambda: self.refresh_results(auto_load=True))

    @staticmethod
    def _preload_pipeline():
        try:
            import field_audio_to_ebird  # noqa: F401
        except Exception:  # noqa: BLE001  ถ้าพังจะแจ้งตอนใช้งานจริง
            pass

    def _on_close(self):
        if self.job_running:
            waiting = (f"\n{len(self.job_queue)} queued job(s) will not run." if self.job_queue else "")
            # งานเบื้องหลังของแอปเอง (ตรวจอัตโนมัติ/วาดภาพ) หยุดได้เลย ครั้งหน้าทำต่อจากที่ค้าง
            only_background = (self.current_job or {}).get("background") and \
                all(job.get("background") for job in self.job_queue)
            if not only_background and not messagebox.askokcancel(
                    "Stop the current job?", "Closing now stops the analysis.\n"
                    "Finished files are saved; the next run continues with the remaining files." + waiting):
                return
            self.job_queue.clear()
            self._closing = True
            self._stop_sent = False
            self._close_deadline = time.monotonic() + 15
            self.status.set("Stopping…")
            self._stop_job_then_close()
            return
        self._close_now()

    def _stop_job_then_close(self):
        proc = self.proc
        if self.job_running and time.monotonic() < self._close_deadline:
            if proc is not None and proc.poll() is None and not self._stop_sent:
                request_stop(proc)   # CLI เปลี่ยนเป็น KeyboardInterrupt แล้ว rollback เอง
                self._stop_sent = True
            self.root.after(100, self._stop_job_then_close)
            return
        if proc is not None and proc.poll() is None:
            proc.kill()
        self._close_now()

    def _close_now(self):
        self.review_stop()
        self.root.destroy()

    def _build_toolbar(self, frm):
        style = ttk.Style(self.root)
        style.configure("Quick.TButton", font=("Helvetica", 14, "bold"), padding=(14, 5))
        style.configure("QuickAction.TButton", font=("Helvetica", 12), padding=(12, 5))

        buttons = ttk.Frame(frm)
        buttons.pack(fill="x")
        self.quick_file_btn = ttk.Button(buttons, text="Choose audio file",
                                         style="Quick.TButton", command=lambda: self._quick_choose(False))
        self.quick_file_btn.pack(side="left", padx=(0, 8))
        self.quick_folder_btn = ttk.Button(buttons, text="Choose audio folder",
                                           style="Quick.TButton", command=lambda: self._quick_choose(True))
        self.quick_folder_btn.pack(side="left", padx=(0, 8))
        self.quick_start_btn = self.run_btn = ttk.Button(buttons, text="▶ Analyze", style="Quick.TButton",
                                                         command=self._quick_start_selected)
        self.quick_start_btn.pack(side="left")
        ttk.Button(buttons, text="Open output folder", style="QuickAction.TButton",
                   command=self._open_output_folder).pack(side="right")

        info = ttk.Frame(frm)                     # แถวเดียว: ไฟล์ที่เลือก · ข้อมูลที่ตรวจเจอ · โมเดล
        info.pack(fill="x", pady=(4, 0))
        info.columnconfigure(1, weight=1)
        self.quick_selected = tk.StringVar(value="No audio selected yet")
        ttk.Label(info, textvariable=self.quick_selected, foreground="#808080",
                  wraplength=520, justify="left").grid(row=0, column=0, sticky="w")
        self.quick_metadata = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.quick_metadata,
                  wraplength=420, justify="left").grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.quick_model_hint = tk.StringVar()
        self._update_quick_model_hint()
        ttk.Label(info, textvariable=self.quick_model_hint, foreground="#a26316").grid(row=0, column=2, sticky="e")

    def _toggle_log(self):
        if self.log_visible:
            self.paned.forget(self.log_frame)
            self.log_toggle.configure(text="Show details ▴")
        else:
            self.paned.add(self.log_frame, weight=1)
            self.log_toggle.configure(text="Hide details ▾")
            # เริ่มที่ราว 40% ของพื้นที่ ลากเส้นแบ่งปรับเองได้
            self.root.update_idletasks()
            self.paned.sashpos(0, int(self.paned.winfo_height() * 0.6))
        self.log_visible = not self.log_visible

    def _show_log(self):
        if not self.log_visible:
            self._toggle_log()

    def _follow_audio(self):
        """เสียงอยู่ใน <YYYYMMDD>/Sounds: เก็บผลที่ Sounds/Done ของวันนั้น (เลือก Output เองทับได้หลังเลือกเสียง)"""
        done = done_folder(self.audio.get().strip())
        if done is not None:
            self.out.set(str(done))

    def _open_output_folder(self):
        path = Path(self.out.get().strip()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("No results yet", "The output folder is created after the first successful analysis.")
            return
        try:
            open_path(path)
        except Exception as exc:
            messagebox.showerror("Cannot open folder", str(exc))

    def _quick_choose(self, folder):
        if self.quick_busy:              # งานวิเคราะห์ที่รันอยู่ไม่บล็อก: งานใหม่ต่อคิว
            return
        path = self.browse_folder() if folder else self.browse_file()
        if path:
            self.analyze_path(path)

    def analyze_path(self, path):
        """เริ่มวิเคราะห์ไฟล์/โฟลเดอร์นี้แบบเดียวกับปุ่มเลือกไฟล์ (ถามรายละเอียดที่ขาดก่อน)"""
        if self.quick_busy:
            return
        self.audio.set(str(path))
        self.date.set("")
        self.coords.set("")
        self.start_time.set("")
        self.quick_selected.set(path)
        self.quick_metadata.set("")
        self.quick_start_btn.configure(state="normal")
        self._quick_start_selected()

    def _quick_start_selected(self):
        if self.quick_busy:              # งานวิเคราะห์ที่รันอยู่ไม่บล็อก: งานใหม่ต่อคิว
            return
        if not self.audio.get().strip():   # ยังไม่ได้เลือกเสียง: เลือกตอนนี้เลย
            self._quick_choose(True)
            return
        path = Path(self.audio.get().strip()).expanduser()
        if not path.exists():
            messagebox.showwarning("File not found", "Choose the audio file or folder again.")
            return
        # ช่อง Date/Start time/Coordinates ในแท็บ Settings ที่กรอกไว้ = ใช้แทนค่าในไฟล์ทุกไฟล์ (เลือกไฟล์ใหม่แล้วช่องจะว่าง)
        self._overrides = any(var.get().strip() for var in (self.date, self.start_time, self.coords))
        self.quick_busy = True
        self.quick_file_btn.configure(state="disabled")
        self.quick_folder_btn.configure(state="disabled")
        self.quick_start_btn.configure(state="disabled")
        self.status.set("Checking file details…")
        self.progress.start(12)
        threading.Thread(target=self._quick_probe_worker, args=(path,), daemon=True).start()
        self.root.after(100, self._quick_poll)

    def _quick_probe_worker(self, path):
        try:
            from field_audio_to_ebird import collect_inputs, recording_context_preview
            files = collect_inputs(path)
            if not files:
                raise ValueError("No audio files found at the selected location")
            missing_date = []
            missing_coords = []
            contexts = []
            for audio_file in files:
                context = recording_context_preview(audio_file)
                contexts.append((audio_file, context))
                if context["datetime"] is None:
                    missing_date.append(audio_file)
                if context["lat"] is None or context["lon"] is None:
                    missing_coords.append(audio_file)
            self.quick_q.put(("ready", files, contexts, missing_date, missing_coords))
        except Exception as exc:
            self.quick_q.put(("error", str(exc)))

    def _quick_poll(self):
        try:
            item = self.quick_q.get_nowait()
        except queue.Empty:
            self.root.after(100, self._quick_poll)
            return
        self.progress.stop()
        self.quick_busy = False
        self.quick_file_btn.configure(state="normal")
        self.quick_folder_btn.configure(state="normal")
        self.quick_start_btn.configure(state="normal")
        if item[0] == "error":
            self.status.set("Could not check the files")
            messagebox.showerror("Cannot open file", item[1])
            return
        _, self.quick_files, contexts, missing_date, missing_coords = item
        if len(contexts) == 1:
            _, context = contexts[0]
            dt = context["datetime"]
            when = dt.strftime("%Y-%m-%d %H:%M") if dt and context["has_time"] else (
                dt.strftime("%Y-%m-%d (time missing)") if dt else "date/time missing")
            where = (f"{context['lat']:.6f}, {context['lon']:.6f}"
                     if context["lat"] is not None and context["lon"] is not None
                     else "location missing")
            self.quick_metadata.set(f"Detected: {when}  •  {where}")
        else:
            self.quick_metadata.set(f"Detected details for {len(contexts)} files; review each file before analysis")
        self.status.set("Review recording details")
        self._quick_details(contexts, missing_date, missing_coords)

    def _quick_details(self, contexts, missing_date, missing_coords):
        dialog = tk.Toplevel(self.root)
        dialog.title("Recording details")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        body = ttk.Frame(dialog, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Review recording details",
                  font=("Helvetica", 17, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(body, text="Details found in the audio file, its filename or the recorder's file times are used automatically. "
                  "Enter only what is missing.", wraplength=690).pack(anchor="w", pady=(0, 12))
        if self._overrides:
            ttk.Label(body, text="The Date, Start time or Coordinates typed in Settings replace these details for "
                      "every file.", foreground="#c77c10", wraplength=690).pack(anchor="w", pady=(0, 10))

        def date_text(context):
            dt = context["datetime"]
            if not dt:
                return "Missing"
            value = dt.strftime("%Y-%m-%d %H:%M") if context["has_time"] else dt.strftime("%Y-%m-%d; time missing")
            return f"{value} ({context['datetime_source']})"

        def coords_text(context):
            if context["lat"] is None or context["lon"] is None:
                return "Missing"
            return f"{context['lat']:.6f}, {context['lon']:.6f} ({context['coords_source']})"

        if len(contexts) == 1:
            path, context = contexts[0]
            ttk.Label(body, text=path.name, font=("Helvetica", 13, "bold"),
                      wraplength=690).pack(anchor="w", pady=(0, 7))
            ttk.Label(body, text="Date/time: " + date_text(context),
                      wraplength=690).pack(anchor="w", pady=2)
            ttk.Label(body, text="Location: " + coords_text(context),
                      wraplength=690).pack(anchor="w", pady=2)
            if context["place"]:
                ttk.Label(body, text="Place: " + str(context["place"]),
                          wraplength=690).pack(anchor="w", pady=2)
        else:
            columns = ("file", "date", "location")
            table_frame = ttk.Frame(body)
            table_frame.pack(fill="x", pady=(0, 12))
            table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                 height=min(8, len(contexts)))
            for key, label, width in (("file", "File", 190),
                                      ("date", "Date/time and source", 285),
                                      ("location", "Location", 350)):
                table.heading(key, text=label)
                table.column(key, width=width, stretch=False)
            table.pack(side="left")
            bar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
            bar.pack(side="right", fill="y")
            table.configure(yscrollcommand=bar.set)
            for path, context in contexts:
                table.insert("", "end", values=(path.name, date_text(context), coords_text(context)))

        first_entry = None
        if missing_date:
            ttk.Label(body, text=f"Recording date for {len(missing_date)} file(s) without one ({DATE_HINT}) *").pack(anchor="w")
            entry = ttk.Entry(body, textvariable=self.date, width=30)
            entry.pack(fill="x", pady=(3, 10))
            first_entry = entry
        if missing_coords:
            ttk.Label(body, text=f"Location for {len(missing_coords)} file(s) without geotag (latitude,longitude) *").pack(anchor="w")
            entry = ttk.Entry(body, textvariable=self.coords, width=42)
            entry.pack(fill="x", pady=(3, 4))
            ttk.Label(body, text="e.g. 13.75633,100.50177, or paste a Google Maps link",
                      foreground="#808080").pack(anchor="w", pady=(0, 10))
            if first_entry is None:
                first_entry = entry
        if len(contexts) == 1 and not contexts[0][1]["has_time"]:
            ttk.Label(body, text=f"Recording start time, if known ({TIME_HINT})").pack(anchor="w")
            ttk.Entry(body, textvariable=self.start_time, width=20).pack(anchor="w", pady=(3, 4))
            ttk.Label(body, text="Enter it if known so the clip clock times are correct.",
                      foreground="#808080").pack(anchor="w", pady=(0, 9))
        elif len(contexts) > 1 and any(not context["has_time"] for _, context in contexts):
            ttk.Label(body, text="Some files have no start time. Analyze them individually if exact clock times matter.",
                      foreground="#c77c10", wraplength=690).pack(anchor="w", pady=(0, 9))
        if len(contexts) > 1 and (missing_date or missing_coords):
            ttk.Label(body, text="Values entered below fill missing details only. Embedded details in other files stay in use.",
                      foreground="#c77c10", wraplength=690).pack(anchor="w", pady=(0, 12))

        ttk.Label(body, text="Site habitat (optional; tick all that apply)").pack(anchor="w", pady=(4, 0))
        self._habitat_checkboxes(body).pack(anchor="w", pady=(3, 2))
        ttk.Label(body, text="Water birds then need a matching water habitat, or a score of at least "
                  f"{MISMATCH_MIN_CONF} in case they flew over. Leave all unticked to skip this check.",
                  foreground="#808080", wraplength=690).pack(anchor="w", pady=(0, 10))

        def submit(_event=None):
            from field_audio_to_ebird import parse_coords
            if missing_date:
                value = self.date.get().strip()
                try:
                    parse_date_entry(value)
                except ValueError:
                    messagebox.showwarning("Invalid date", f"Enter the date as {DATE_HINT}", parent=dialog)
                    return
            if missing_coords and parse_coords(self.coords.get()) is None:
                messagebox.showwarning("Invalid location", "Enter coordinates, e.g. 13.75633,100.50177", parent=dialog)
                return
            if len(self.quick_files) > 1 and (missing_date or missing_coords):
                count = len(set(missing_date) | set(missing_coords))
                if not messagebox.askokcancel("Use for all files?",
                                              f"Fill the missing details for {count} file(s)? Existing metadata stays in use.",
                                              parent=dialog):
                    return
            dialog.destroy()
            self.run(confirmed_batch=not self._overrides, button=self.quick_start_btn, fill_missing=not self._overrides)

        action = ttk.Frame(body)
        action.pack(fill="x", pady=(6, 0))
        ttk.Button(action, text="Start analysis", style="QuickAction.TButton",
                   command=submit).pack(side="right", padx=(8, 0))
        ttk.Button(action, text="Cancel", command=dialog.destroy).pack(side="right")
        dialog.bind("<Return>", submit)
        dialog.grab_set()
        self._install_clipboard_fix()
        if first_entry is not None:
            first_entry.focus_set()

    def _install_clipboard_fix(self):
        # ปัญหา (Win/Linux, Thai layout): Tk ผูก Ctrl+V กับ keysym "v" — พอ layout เป็นไทย keysym
        #   เปลี่ยน เลย paste ไม่ได้
        # ปัญหา (macOS): ต้องผูก Cmd แยกจาก Ctrl (คนละ modifier/state bit) ไม่งั้น Cmd+V ไม่ทำงาน
        #   และ trackpad "คลิกขวา" (two-finger / control-click) ส่ง <Button-2> ไม่ใช่ <Button-3>
        # แก้: ดู keysym ก่อน (layout อังกฤษ) แล้วค่อยดูปุ่มจริงซึ่งไม่ขึ้นกับ layout:
        #   Windows = virtual-key code (V=86 C=67 X=88 A=65); macOS Tk เก็บ Mac virtual key code
        #   ไว้ใน byte บนสุดของ keycode (V=9 C=8 X=7 A=0; 0xff = event จำลอง)
        #   ผูก Ctrl (Win/Linux) หรือ Command (mac) — Tk ยิง binding นี้เมื่อกด modifier นั้นอยู่แล้ว
        is_mac = sys.platform == "darwin"
        physical = {9: "v", 8: "c", 7: "x", 0: "a"} if is_mac else {86: "v", 67: "c", 88: "x", 65: "a"}

        def on_key(e):
            w = e.widget
            letter = e.keysym.lower()
            if letter not in ("v", "c", "x", "a"):
                letter = physical.get(e.keycode >> 24 if is_mac else e.keycode)
            if letter in ("v", "c", "x"):
                w.event_generate({"v": "<<Paste>>", "c": "<<Copy>>", "x": "<<Cut>>"}[letter])
                return "break"
            if letter == "a":                 # select all
                try:
                    w.select_range(0, "end"); w.icursor("end")
                except Exception:  # noqa: BLE001
                    pass
                return "break"
            return None

        self._menu_target = None
        menu = getattr(self, "_edit_menu", None)
        if menu is None:                      # สร้างครั้งเดียว (ฟังก์ชันนี้ถูกเรียกซ้ำตอนเปิด dialog)
            menu = self._edit_menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="Cut",
                             command=lambda: self._menu_gen("<<Cut>>"))
            menu.add_command(label="Copy",
                             command=lambda: self._menu_gen("<<Copy>>"))
            menu.add_command(label="Paste",
                             command=lambda: self._menu_gen("<<Paste>>"))

        def popup(e):
            self._menu_target = e.widget
            e.widget.focus_set()
            try:
                menu.tk_popup(e.x_root, e.y_root)
            finally:
                menu.grab_release()

        def apply(w):
            if isinstance(w, (ttk.Entry, tk.Entry)):
                # ผูกที่ตัว widget -> ทำงานก่อน class binding; บน mac Ctrl+A/… เป็นปุ่มแก้ไขแบบ emacs
                w.bind("<Command-KeyPress>" if is_mac else "<Control-KeyPress>", on_key)
                if is_mac:
                    w.bind("<Button-2>", popup)            # mac secondary-click บน Tk 8.6
                w.bind("<Button-3>", popup)                # right-click (Win/Linux และ mac บน Tk 9)
            for c in w.winfo_children():
                apply(c)
        apply(self.root)

    def _habitat_checkboxes(self, parent, label=None):
        frame = ttk.Frame(parent)
        column = 0
        if label:
            ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8))
            column = 1
        for key, text in SITE_HABITATS.items():
            ttk.Checkbutton(frame, text=text, variable=self.habitat_vars[key]).grid(
                row=0, column=column, sticky="w", padx=(0, 10))
            column += 1
        return frame

    def _menu_gen(self, ev):
        if self._menu_target is not None:
            self._menu_target.event_generate(ev)

    # =========================== แท็บ Analyze ===========================
    def _build_analyze(self, frm):
        pad = {"padx": 6, "pady": 3}
        frm.columnconfigure(1, weight=1)
        r = 0

        def section(text):
            nonlocal r
            ttk.Label(frm, text=text, font=("Helvetica", 12, "bold")).grid(row=r, column=0, columnspan=3, sticky="w",
                                                                          padx=6, pady=(8, 0))
            r += 1

        self.audio = tk.StringVar(value=DEF_AUDIO)     # เลือกไฟล์จากแถบบน
        self.audio.trace_add("write", lambda *_args: self._follow_audio())

        section("Field logs")
        # บันทึกภาคสนาม: eBird checklist ของคุณจากโฟลเดอร์ Log ของโปรแกรมจัดรูป (field_log.py) — พิกัด ตัวเทียบเสียง แท็บ Review ใช้
        logf = ttk.Frame(frm); logf.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(logf, text="eBird checklist link").pack(side="left", padx=(0, 6))
        self.checklist_link = tk.StringVar()
        link = ttk.Entry(logf, textvariable=self.checklist_link, width=34)
        link.pack(side="left", padx=(0, 6))
        link.bind("<Return>", lambda _event: self.add_checklist_links())
        self.link_btn = ttk.Button(logf, text="Add link", command=self.add_checklist_links)
        self.link_btn.pack(side="left", padx=(0, 14))
        ttk.Button(logf, text="Add Log folder…", command=self.add_field_log).pack(side="left", padx=(0, 6))
        ttk.Button(logf, text="Match recordings…", command=self.match_recordings).pack(side="left")
        r += 1
        self.field_log_text = tk.StringVar(value=self._field_log_summary())
        ttk.Label(frm, textvariable=self.field_log_text, foreground="#888", wraplength=890,
                  justify="left").grid(row=r, column=0, columnspan=3, sticky="w", padx=6)
        r += 1

        section("Recording")
        ttk.Label(frm, text="Output folder").grid(row=r, column=0, sticky="w", **pad)
        self.out = tk.StringVar(value=DEF_OUT)
        ttk.Entry(frm, textvariable=self.out).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Choose…", command=self.browse_output).grid(row=r, column=2, **pad)
        r += 1

        grid = ttk.Frame(frm); grid.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        self.coords = tk.StringVar(value="")
        self.date = tk.StringVar(value="")
        self.start_time = tk.StringVar(value="")
        self.place = tk.StringVar(value="")
        ttk.Label(grid, text="Fill these only to replace what the file and field log say").grid(
            row=2, column=0, columnspan=4, sticky="w")
        ttk.Label(grid, text="Coordinates or Google Maps link").grid(
            row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.coords, width=36).grid(row=0, column=1, padx=(2, 14))
        ttk.Label(grid, text=COORD_HINT, foreground="#888").grid(
            row=0, column=2, sticky="w", padx=(2, 0))
        ttk.Label(grid, text="Date if no metadata (YYYYMMDD)").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.date, width=14).grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(grid, text="Start time (HHMM)").grid(row=1, column=2, sticky="w")
        ttk.Entry(grid, textvariable=self.start_time, width=12).grid(row=1, column=3, padx=2)
        r += 1

        section("BirdNET")
        ttk.Button(frm, text="Analysis presets…", command=self.open_presets).grid(row=r - 1, column=2, sticky="e", padx=6)
        grid2 = ttk.Frame(frm); grid2.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(grid2, text="Place").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid2, textvariable=self.place, width=34).grid(row=0, column=1, padx=(2, 14))
        ttk.Label(grid2, text="min_conf").grid(row=0, column=2, sticky="w")
        self.min_conf = tk.DoubleVar(value=0.5)
        ttk.Spinbox(grid2, from_=0.0, to=1.0, increment=0.05, width=6,
                    textvariable=self.min_conf).grid(row=0, column=3, padx=2)
        ttk.Label(grid2, text="BirdNET model").grid(row=1, column=0, sticky="w")
        self.model = tk.StringVar(value="3.0-preview")
        model_box = ttk.Combobox(grid2, textvariable=self.model,
                                 values=("3.0-preview", "3.0-pruned"),   # pruned = รุ่นย่อ (แทน 2.4)
                                 state="readonly", width=13)
        model_box.grid(row=1, column=1, sticky="w", padx=(2, 14))
        model_box.bind("<<ComboboxSelected>>", self._on_model_changed)
        ttk.Label(grid2, text="3.0-pruned = the smaller BirdNET model (results go to a separate "
                  "_v3pruned folder). Both check location and date; verify every ID by ear.",
                  foreground="#a26316", wraplength=310,
                  justify="left").grid(row=1, column=2, columnspan=2, sticky="w")
        r += 1

        # ถิ่นอาศัยของจุดบันทึก (เลือกได้หลายแบบ เช่น ป่า+บึง) ใช้ร่วมกับหน้าต่าง Recording details
        self.habitat_vars = {key: tk.BooleanVar(value=False) for key in SITE_HABITATS}
        self._habitat_checkboxes(frm, label="Site habitat").grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        section("Clips")
        grid3 = ttk.Frame(frm); grid3.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        self.gap = tk.DoubleVar(value=5.0)
        self.lead = tk.DoubleVar(value=3.0)
        self.tail = tk.DoubleVar(value=3.0)
        self.dbfs = tk.DoubleVar(value=-3.0)
        for i, (lab, var) in enumerate([("occurrence gap (s)", self.gap), ("lead (s)", self.lead),
                                        ("tail (s)", self.tail), ("normalize dBFS", self.dbfs)]):
            ttk.Label(grid3, text=lab).grid(row=0, column=i * 2, sticky="w")
            ttk.Spinbox(grid3, from_=-60, to=60, increment=0.5, width=7,
                        textvariable=var).grid(row=0, column=i * 2 + 1, padx=(2, 14))
        self.overlap = tk.DoubleVar(value=1.5)
        self.keep_out_of_range = tk.BooleanVar(value=False)   # ปกติตัดชนิดนอกพื้นที่/ฤดูทิ้ง
        self.out_of_range_min_conf = tk.DoubleVar(value=0.7)
        ttk.Label(grid3, text="analysis overlap (s)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(grid3, from_=0, to=2.5, increment=0.5, width=7,
                    textvariable=self.overlap).grid(row=1, column=1, padx=(2, 14))
        ttk.Checkbutton(grid3, text="also keep out-of-area ≥", variable=self.keep_out_of_range).grid(
            row=1, column=2, sticky="w")
        ttk.Spinbox(grid3, from_=0, to=1, increment=0.05, width=7,
                    textvariable=self.out_of_range_min_conf).grid(row=1, column=3, padx=(2, 14))
        r += 1

        tg = ttk.Frame(frm); tg.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        self.mono = tk.BooleanVar(value=False)        # คลิปเป็น stereo ตามที่อัด (คู่มือ Macaulay)
        self.spec = tk.BooleanVar(value=True)
        self.alt = tk.BooleanVar(value=True)
        self.unknown = tk.BooleanVar(value=False)
        self.force = tk.BooleanVar(value=False)
        self.filetime = tk.BooleanVar(value=False)
        self.continuous = tk.BooleanVar(value=False)
        ttk.Checkbutton(tg, text="to mono", variable=self.mono).pack(side="left", padx=8)
        ttk.Checkbutton(tg, text="mel-spectrogram", variable=self.spec).pack(side="left", padx=8)
        ttk.Checkbutton(tg, text="alt species", variable=self.alt).pack(side="left", padx=8)
        ttk.Checkbutton(tg, text="cut unknown", variable=self.unknown).pack(side="left", padx=8)
        ttk.Checkbutton(tg, text="force redo", variable=self.force).pack(side="left", padx=8)
        ttk.Checkbutton(tg, text="use file time", variable=self.filetime).pack(side="left", padx=8)
        r += 1

        ttk.Checkbutton(frm, text="Keep first-to-last call continuous for each species (same individual only)",
                        variable=self.continuous).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1
        # เสียงที่ BirdNET เกือบเจอ: ฟังซ้ำแบบวางหน้าต่างตรงเสียง (3.0 เท่านั้น — near_misses.py)
        self.second_look = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Second look at near misses (3.0): listen again to sounds scoring 0.2 to min_conf "
                        "with the window centred on the sound", variable=self.second_look).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        section("API keys")   # key ของผู้ใช้เอง (ใช้กับ Reference sounds ในแท็บ Review)
        keyf = ttk.Frame(frm); keyf.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(keyf, text="xeno-canto API key").pack(side="left")
        self.xc_key = tk.StringVar(value=load_settings().get("xeno_canto_key", ""))
        ttk.Entry(keyf, textvariable=self.xc_key, width=34, show="•").pack(side="left", padx=6)
        ttk.Button(keyf, text="Save key", command=self._save_xc_key).pack(side="left")
        ttk.Label(keyf, text="  Get it at xeno-canto.org/account",
                  foreground="#888").pack(side="left")
        r += 1
        ebf = ttk.Frame(frm); ebf.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(ebf, text="eBird API key").pack(side="left")
        self.ebird_key = tk.StringVar(value=load_settings().get("ebird_key", ""))
        ttk.Entry(ebf, textvariable=self.ebird_key, width=34, show="•").pack(side="left", padx=(31, 6))
        ttk.Button(ebf, text="Save key", command=self._save_ebird_key).pack(side="left")
        ttk.Label(ebf, text="  Get it at ebird.org/api/keygen",
                  foreground="#888").pack(side="left")

    def _field_log_summary(self):
        checklists, roots = field_log.stored()
        if not checklists:
            return "Paste an eBird checklist link, or add the Log folder of your photo organizer (…/YYYYMMDD/Log)"
        latest = "; ".join(f"{c['id']} {datetime.fromisoformat(c['date']):%d %b %Y} {c['location']}" for c in checklists[-3:])
        where = f" · other days found automatically under {', '.join(roots)}" if roots else ""
        return f"{len(checklists)} checklist(s), latest: {latest}{where}"

    def add_checklist_links(self):
        """ลิงก์ checklist ที่วาง -> ดึงจาก eBird เบื้องหลัง แล้วให้เลือกว่าไฟล์เสียงไหนอยู่ checklist ไหน"""
        text, key = self.checklist_link.get(), self.ebird_key.get().strip()
        if not field_log.checklist_ids(text):
            messagebox.showinfo("eBird checklist link", "Paste a checklist link, like https://ebird.org/checklist/S123456789")
            return
        if not key:
            messagebox.showwarning("eBird API key", "Save your eBird API key first (API keys, lower on this page)")
            return
        self.link_btn.configure(state="disabled")
        self.status.set("Reading the checklist from eBird …")
        results = queue.Queue()
        threading.Thread(target=lambda: results.put(field_log.add_links(text, key)), daemon=True).start()
        self.root.after(100, self._links_poll, results)

    def _links_poll(self, results):
        try:
            added, problems = results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._links_poll, results)
            return
        self.link_btn.configure(state="normal")
        self.field_log_text.set(self._field_log_summary())
        if problems:
            messagebox.showwarning("eBird checklist link", "\n".join(problems))
        if not added:
            self.status.set("No checklist added")
            return
        self.checklist_link.set("")
        self.status.set("Checklist added: " + ", ".join(f"{c['id']} ({c['date'][:10]}, {len(c['species'])} species)" for c in added))
        self._field_log_changed()
        self.match_recordings(datetime.fromisoformat(max(c["date"] for c in added)).date())

    def add_field_log(self):
        """เลือกโฟลเดอร์ Log (หรือ checklist.json) ของโปรแกรมจัดรูป — จำไว้ และหา Log วันอื่นใต้โฟลเดอร์ Birding เดียวกันเอง"""
        _checklists, roots = field_log.stored()
        start = next((root for root in roots if Path(root).is_dir()), str(Path.home() / "Desktop"))
        folder = filedialog.askdirectory(title="Field log folder (…/YYYYMMDD/Log)", initialdir=start)
        if not folder:
            return
        added = field_log.add(folder)
        if not added:
            messagebox.showinfo("Add field log", "No eBird checklist (checklist.json) found in that folder")
            return
        self.field_log_text.set(self._field_log_summary())
        self.status.set("Field log added: " + ", ".join(f"{c['id']} ({c['date'][:10]})" for c in added))
        self._field_log_changed()
        self.match_recordings(datetime.fromisoformat(max(c["date"] for c in added)).date())

    def match_recordings(self, day=None):
        """หน้าต่างเลือก checklist ของแต่ละไฟล์เสียง (drop down) — เด้งเองหลัง Add field log"""
        from field_log_window import RecordingChecklists
        RecordingChecklists(self.root, self.out.get().strip() or None, day, changed=self._field_log_changed)

    def _field_log_changed(self):
        if self._selected_review()[0] is not None:   # บรรทัดรายละเอียดของคลิปที่เลือกอยู่
            self._review_select()

    def _recordings_matched(self, _event=None):
        """ปิดหน้าต่าง Match recordings: ย้ายคลิปใน Ready ตาม checklist ล่าสุด ผ่านคิวงาน (ไม่ชนงานที่เขียน summary)"""
        days, self.root._matched_days = getattr(self.root, "_matched_days", set()), set()
        output = self.out.get().strip() or None
        for day in sorted(days):
            if any(ready_folders.plan(ready_folders.day_summaries(day, output))[:2]):
                self._start(self._launch() + ["--sort-ready", day.isoformat()] + ([output] if output else []), None,
                            f"Sort Ready by checklist ({day:%d %b})", is_analyze=False, on_done=self._ready_sorted)

    def _ready_sorted(self, code, _lines):
        if code == 0:
            self.status.set("Ready folders sorted by checklist ✓")
            if self.review_summary.get().strip():
                self.review_load(quiet=True, keep_selection=True)   # คอลัมน์ Ready file เปลี่ยน

    def _on_model_changed(self, _event=None):
        # Update the suggested threshold only while it still equals the last
        # model's default; keep a threshold the user entered themselves.
        if self.model.get() == "2.4" and self.min_conf.get() == 0.5:
            self.min_conf.set(0.25)
        elif self.model.get().startswith("3.0") and self.min_conf.get() == 0.25:
            self.min_conf.set(0.5)
        if hasattr(self, "quick_model_hint"):
            self._update_quick_model_hint()

    def _update_quick_model_hint(self):
        name = {"3.0-preview": "BirdNET 3.0 preview", "3.0-pruned": "BirdNET 3.0 pruned"}.get(
            self.model.get(), "BirdNET 2.4")
        self.quick_model_hint.set(f"{name} · listen before approving")

    # ============================ แท็บ Merge ============================
    def _build_merge(self, frm):
        pad = {"padx": 6, "pady": 3}
        frm.columnconfigure(0, weight=1)

        warn = ("⚠  Only merge recordings of the SAME individual bird "
                "(eBird guideline).\n"
                "BirdNET identifies species, not individuals — confirm by ear first.")
        ttk.Label(frm, text=warn, foreground="#b36b00", justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(frm, text="Clips to merge (top → bottom order)").grid(
            row=1, column=0, sticky="w", **pad)

        listf = ttk.Frame(frm); listf.grid(row=2, column=0, sticky="nsew", **pad)
        frm.rowconfigure(2, weight=1)
        listf.columnconfigure(0, weight=1)
        self.merge_list = tk.Listbox(listf, height=6, selectmode="extended")
        self.merge_list.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(listf, command=self.merge_list.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        self.merge_list["yscrollcommand"] = lsb.set

        btns = ttk.Frame(frm); btns.grid(row=2, column=1, sticky="n", **pad)
        ttk.Button(btns, text="Add…", width=10, command=self.merge_add).pack(pady=2)
        ttk.Button(btns, text="Remove", width=10, command=self.merge_remove).pack(pady=2)
        ttk.Button(btns, text="Up", width=10, command=lambda: self.merge_move(-1)).pack(pady=2)
        ttk.Button(btns, text="Down", width=10, command=lambda: self.merge_move(1)).pack(pady=2)
        ttk.Button(btns, text="Clear", width=10, command=self.merge_clear).pack(pady=2)

        # ที่เก็บ: โฟลเดอร์เดียวกับคลิป หรือเลือกเอง — ชื่อเติมจากชนิดของคลิป แก้เองได้
        outf = ttk.Frame(frm); outf.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        outf.columnconfigure(1, weight=1)
        self.merge_place = tk.StringVar(value="same")
        self.merge_folder = tk.StringVar(value="")
        self.merge_where = tk.StringVar(value="Add clips to see where the file goes")
        self.merge_name = tk.StringVar(value="")
        self._merge_suggested = ""
        ttk.Label(outf, text="Save in").grid(row=0, column=0, sticky="w")
        place = ttk.Frame(outf); place.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Radiobutton(place, text="Same folder as the clips", variable=self.merge_place, value="same",
                        command=self._merge_show_place).pack(side="left")
        ttk.Radiobutton(place, text="Another folder", variable=self.merge_place, value="other",
                        command=self._merge_other).pack(side="left", padx=(14, 0))
        ttk.Button(place, text="Choose folder…", command=self.merge_choose_folder).pack(side="left", padx=(6, 0))
        ttk.Label(outf, textvariable=self.merge_where, foreground="#888", wraplength=800,
                  justify="left").grid(row=1, column=1, sticky="w", padx=6, pady=(0, 4))
        ttk.Label(outf, text="File name").grid(row=2, column=0, sticky="w")
        name = ttk.Frame(outf); name.grid(row=2, column=1, sticky="ew", padx=6)
        name.columnconfigure(0, weight=1)
        ttk.Entry(name, textvariable=self.merge_name).grid(row=0, column=0, sticky="ew")
        ttk.Label(name, text=".wav").grid(row=0, column=1, padx=(4, 0))

        self.merge_btn = ttk.Button(frm, text="🔗  Merge", command=self.merge_run)
        self.merge_btn.grid(row=4, column=0, sticky="w", **pad)

    def _build_review(self, frm):
        frm.columnconfigure(0, weight=1)
        # พื้นที่ไม่พอ: ตารางคลิป (เลื่อนดูได้) หดก่อน ตัวดูเสียงหดทีหลัง
        frm.rowconfigure(3, weight=1, minsize=95)
        frm.rowconfigure(4, weight=2, minsize=300)
        header = ttk.Frame(frm)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(header, text="Review clips", font=("Helvetica", 17, "bold")).pack(side="left")
        ttk.Label(header, text="Select a clip → play it → check the species → approve",
                  font=("Helvetica", 12), foreground="#808080").pack(side="left", padx=(14, 0), pady=(4, 0))
        # แสดงเฉพาะชนิดเดียว (เลือกเอง หรือดับเบิลคลิกชื่อชนิดใน Review progress)
        self.review_species_filter = tk.StringVar(value="All")
        self.review_species_count = tk.StringVar()
        ttk.Label(header, textvariable=self.review_species_count, foreground="#808080").pack(side="right", padx=(6, 0))
        self.review_species_box = ttk.Combobox(header, textvariable=self.review_species_filter, values=("All",),
                                               state="readonly", width=26)
        self.review_species_box.pack(side="right")
        self.review_species_box.bind("<<ComboboxSelected>>", lambda _event: self._fill_review_tree())
        ttk.Label(header, text="Species").pack(side="right", padx=(0, 6))
        # แสดงเฉพาะสถานะ (เช่น Pending = ยังไม่ได้ตรวจ, Check = มีคำเตือน)
        self.review_status_filter = tk.StringVar(value="All")
        status_box = ttk.Combobox(header, textvariable=self.review_status_filter, values=REVIEW_STATUS_FILTERS,
                                  state="readonly", width=9)
        status_box.pack(side="right", padx=(0, 12))
        status_box.bind("<<ComboboxSelected>>", lambda _event: self._fill_review_tree())
        ttk.Label(header, text="Show").pack(side="right", padx=(0, 6))
        top = ttk.Frame(frm)
        top.grid(row=2, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        self.review_summary = tk.StringVar()
        # ผลที่วิเคราะห์ไว้แล้วทุกชุดในโฟลเดอร์ผลลัพธ์ เลือกแล้วโหลดทันที (ไม่ต้องวิเคราะห์ใหม่)
        ttk.Label(top, text="Results").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.results_choice = tk.StringVar()
        self.results_box = ttk.Combobox(top, textvariable=self.results_choice, state="readonly")
        self.results_box.grid(row=0, column=1, sticky="ew")
        self.results_box.bind("<<ComboboxSelected>>", self._on_result_selected)
        ttk.Button(top, text="Refresh list", command=self.refresh_results).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Open other results…", command=self.review_browse).grid(row=0, column=3)
        # เสียงที่ BirdNET ไม่เจอ: เลือกช่วงในไฟล์ต้นฉบับแล้วใส่ชนิดเอง
        ttk.Button(top, text="✚ Add clip yourself…", command=self.open_manual).grid(row=0, column=4, padx=(10, 0))
        # ตรวจไปแล้วกี่คลิป/ชนิดไหน + ตัวกรองไหนจับ ID ผิดได้ (ข้ามผลทุกชุด)
        ttk.Button(top, text="Review progress…", command=self.open_progress).grid(row=0, column=5, padx=(6, 0))
        # ไฟล์ใน Ready ของวันนี้ + รายละเอียดที่ต้องกรอกตอนอัปโหลด (เสียงพื้นหลัง, ชนิดที่ต้องเพิ่มใน checklist ก่อน)
        ttk.Button(top, text="Upload sheet…", command=self.open_upload_sheet).grid(row=0, column=7, padx=(6, 0))
        # ดัชนีเสียง: เสียงที่คล้ายคลิปนี้ / กลุ่มเสียงที่ยังไม่มีคลิป / ตัวจำแนกของคุณ
        finder = ttk.Menubutton(top, text="Find sounds ▾")
        finder.grid(row=0, column=6, padx=(6, 0))
        menu = tk.Menu(finder, tearoff=False)
        menu.add_command(label="Similar to the selected clip…", command=self.open_find_similar_clip)
        menu.add_command(label="Sound groups in this result…", command=self.open_sound_groups)
        menu.add_command(label="Your recognizers…", command=self.open_recognizers)
        # เสียงที่ BirdNET เกือบเจอ (+ ฟังซ้ำ) / ชนิดที่ควรพบที่จุดและสัปดาห์นั้น (near_misses.py)
        menu.add_command(label="Near misses in this result…", command=self.open_near_misses)
        menu.add_command(label="Expected species here…", command=self.open_expected)
        menu.add_separator()
        menu.add_command(label="Recorder flags…", command=self.open_recorder_flags)
        menu.add_command(label="Find sound events (without BirdNET)…", command=self.open_signal_events)
        menu.add_command(label="Import detections CSV…", command=self.open_detections_import)
        finder["menu"] = menu
        self._index_checked = set()            # ผลชุดที่สั่งทำดัชนีเสียงเบื้องหลังแล้วในรอบนี้
        self._index_q = queue.Queue()
        self._manual_window = None
        self._progress_window = None
        self._spectrograms_checked = set()     # โฟลเดอร์ผลที่ตรวจภาพแล้วในรอบนี้
        self._season_checked = set()           # summary ที่ถาม eBird ช่วงฤดูแล้ว
        self._auto_checked = set()             # summary ที่สั่งตรวจอัตโนมัติแล้วในรอบนี้
        self._sound_guess = None               # คำเดาประเภทเสียงของคลิปที่เลือกอยู่
        self._sound_training = [False]         # ใช้ร่วมกับงานเบื้องหลัง (ไม่ให้งานถือ self)
        self._auto_q = queue.Queue()
        # ML model จากผลตรวจของผู้ใช้ (Review progress): เตือนคลิปที่ ID ของ BirdNET น่าจะผิด
        self._warner = None
        self._warner_reviewed = -1             # จำนวนผลตรวจตอนฝึกครั้งล่าสุด
        self._model_risk = {}                  # sha256 ของคลิป -> ความเสี่ยง (ผลชุดที่เปิดอยู่)
        self._model_busy = False
        self._model_q = queue.Queue()
        self._season_flagged = set()           # (วันที่, ชื่อวิทยาศาสตร์) ที่ไม่มีรายงานช่วงนี้ในปีก่อน ๆ
        self._season_q = queue.Queue()
        self._season_line_q = queue.Queue()
        self._spectrogram_q = queue.Queue()
        self._results = {}                    # ข้อความในเมนู -> path ของ summary.xlsx
        self._results_q = queue.Queue()

        columns = ("time", "file", "species", "confidence", "flag", "location", "status")
        self.review_tree = ttk.Treeview(frm, columns=columns, show="headings", height=6)
        self._review_sort = None              # (คอลัมน์, มากไปน้อย) — None = ลำดับใน summary
        self._review_headings = {}
        for key, label, width in (("time", "Time", 78), ("file", "Clip", 275),
                                  ("species", "Species (eBird name)", 190),
                                  ("confidence", "Confidence", 85), ("flag", "Flag", 62),
                                  ("location", "Check", 85), ("status", "Status", 120)):  # คำเตือนทุกแบบ
            self._review_headings[key] = label
            self.review_tree.heading(key, text=label, command=lambda key=key: self._sort_review(key))
            self.review_tree.column(key, width=width, stretch=(key == "file"))
        self.review_tree.grid(row=3, column=0, sticky="ew", pady=5)
        self.review_tree.bind("<<TreeviewSelect>>", self._review_select)
        self.review_tree.bind("<KeyPress>", self._review_key, add="+")
        self.review_rows = {}
        self._review_flags = {}               # Source ID -> flag ที่กดตอนอัด (recorder_flags.py)

        preview = ttk.Frame(frm)
        preview.grid(row=4, column=0, sticky="nsew", pady=(3, 7))
        preview.columnconfigure(0, weight=1)
        self.review_context = tk.StringVar(value="Select a clip to see its sound and recording details")
        ttk.Label(preview, textvariable=self.review_context, font=("Helvetica", 11),
                  wraplength=840).grid(
            row=0, column=0, sticky="w", pady=(0, 5))
        # ตัวดูแบบ Macaulay: ขาวดำ 0-12 kHz ทีละ 6 วินาที + แถบภาพรวม (คลิกภาพ = เล่นจากจุดนั้น)
        # เห็น/เล่นเฉพาะช่วงที่ BirdNET ได้ยินชนิดนี้: ใช้ร่วมกับตัวดูอื่นและจำไว้ครั้งหน้า
        self.only_heard = tk.BooleanVar(value=bool(load_settings().get("only_heard", False)))
        self.only_heard.trace_add("write", lambda *_args: save_settings({"only_heard": bool(self.only_heard.get())}))
        self.review_view = SpectrogramView(preview, width=840, height=150, overview_height=26,
                                           app=self,
                                           only_heard=self.only_heard,
                                           on_seek=self._review_seek)
        self.review_view.grid(row=1, column=0, sticky="nsew")
        self.review_view.main.bind("<KeyPress>", self._review_key, add="+")
        preview.rowconfigure(1, weight=1)
        self._review_preview_number = 0
        self.second_opinion = tk.StringVar()
        self._hide_when_empty(self.second_opinion, ttk.Label(preview, textvariable=self.second_opinion,
                                                             foreground="#3a87c8", wraplength=880, justify="left"),
                              row=2, pady=(4, 0))
        self._second_opinion_q = queue.Queue()
        self._second_opinion_busy = False
        self.ebird_info = tk.StringVar()
        self._hide_when_empty(self.ebird_info, ttk.Label(preview, textvariable=self.ebird_info,
                                                         foreground="#4a9a4f", wraplength=880, justify="left"),
                              row=3, pady=(2, 0))
        self._ebird_q = queue.Queue()
        self._ebird_request = self._ebird_answered = 0
        # ชนิดในรูปของคุณที่ถ่ายตอนได้ยินเสียงนี้ (clip_photos.py) — บรรทัดเดียว ไม่แสดงภาพ
        self.photo_text = tk.StringVar()
        self._hide_when_empty(self.photo_text, ttk.Label(preview, textvariable=self.photo_text,
                                                         foreground="#8a6d3b", wraplength=880, justify="left"),
                              row=4, pady=(2, 0))
        self._photo_q, self._photo_request, self._photo_polling = queue.Queue(), 0, False
        self._ebird_polling = False
        self._ebird_region = None              # มณฑล/รัฐของจุดบันทึก (จากผลเช็ก eBird ล่าสุด)
        self._ebird_text = ""
        self._season_line_polling = False
        self._lookup_q = queue.Queue()

        controls = ttk.Frame(frm)
        controls.grid(row=5, column=0, sticky="w", pady=(4, 7))
        buttons = ttk.Frame(controls)
        buttons.pack(anchor="w")
        ttk.Button(buttons, text="▶ Play", style="QuickAction.TButton",
                   command=self.review_play).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="■ Stop", style="QuickAction.TButton",
                   command=self.review_stop).pack(side="left", padx=(0, 8))
        playback.control(buttons, length=78).pack(side="left", padx=(0, 10))
        ttk.Button(buttons, text="BirdNET boxes ↗", command=self.review_boxes_picture).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Reference sounds…", command=self.review_references).pack(side="left")
        # หน้าของชนิดนี้ (ชื่อในช่อง Scientific name): รูป/เสียง/แผนที่ ใน eBird, Macaulay Library, xeno-canto
        lookups = ttk.Frame(controls)
        lookups.pack(anchor="w", pady=(6, 0))
        for text, kind in (("eBird ↗", "species"), ("Photos ↗", "photo"), ("Sounds ↗", "audio")):
            ttk.Button(lookups, text=text, command=lambda kind=kind: self.review_lookup(kind)).pack(
                side="left", padx=(0, 6))
        ttk.Button(lookups, text="xeno-canto ↗", command=self.review_xeno_canto).pack(side="left", padx=(0, 6))
        ttk.Button(lookups, text="Checklist ↗", command=self.review_checklist).pack(side="left", padx=(0, 6))
        self.second_opinion_btn = ttk.Button(lookups, text="Second opinion", command=self.review_second_opinion)
        self.second_opinion_btn.pack(side="left")
        # checklist ของประเทศ (จีน/ไทย) อยู่แถวเดียวกับปุ่ม ไม่เพิ่มความสูงของแท็บ
        self.checklist_info = tk.StringVar()
        self.checklist_label = ttk.Label(lookups, textvariable=self.checklist_info,
                                         wraplength=200, justify="left")
        self.checklist_label.pack(side="left", padx=(14, 0))

        fields = ttk.Frame(frm)
        fields.grid(row=6, column=0, sticky="ew", pady=5)
        fields.columnconfigure(1, weight=1)
        self.review_common = tk.StringVar()
        self.review_scientific = tk.StringVar()
        self.review_rating = tk.StringVar()
        self.review_notes = tk.StringVar()
        self.review_sound_type = tk.StringVar()
        self.sound_guess = tk.StringVar()
        ttk.Label(fields, text="Reviewed common name").grid(row=0, column=0, sticky="w")
        common_entry = ttk.Entry(fields, textvariable=self.review_common)
        common_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(fields, text="Scientific name").grid(row=0, column=2, sticky="w")
        scientific_entry = ttk.Entry(fields, textvariable=self.review_scientific, width=24)
        scientific_entry.grid(row=0, column=3, padx=5)
        # ชนิดที่ BirdNET ไม่ได้ทาย: พิมพ์บางส่วนของชื่อ แล้วเลือกจากรายชื่อ eBird ที่เด้งขึ้น (เติมทั้งสองช่อง)
        from species_picker import SpeciesPicker
        self.species_picker = SpeciesPicker(common_entry, self._review_country, self._review_species_picked,
                                            on_exact=lambda scientific: self._review_species_picked(None, scientific),
                                            candidates_of=self._review_candidates)
        self._species_refresh_job = None
        scientific_entry.bind("<KeyRelease>", lambda _event: self._review_species_edited(), add="+")
        ttk.Label(fields, text="Audio quality (1–5)").grid(row=1, column=0, sticky="w")
        quality = ttk.Frame(fields)
        quality.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Combobox(quality, textvariable=self.review_rating, values=("1", "2", "3", "4", "5"),
                     state="readonly", width=5).pack(side="left")
        # ประเภทเสียง (แท็กของ eBird): แอปเดา song/call ให้ — มั่นใจแล้วเติมให้ ไม่มั่นใจให้ฟังแล้วเลือกเอง
        ttk.Label(quality, text="Sound type").pack(side="left", padx=(12, 4))
        ttk.Combobox(quality, textvariable=self.review_sound_type, values=SOUND_TYPES,
                     state="readonly", width=10).pack(side="left")
        ttk.Label(quality, textvariable=self.sound_guess, foreground="#808080").pack(side="left", padx=(6, 0))
        ttk.Label(fields, text="Notes").grid(row=1, column=2, sticky="w")
        ttk.Entry(fields, textvariable=self.review_notes, width=24).grid(row=1, column=3, padx=5)
        # เสียงพื้นหลัง (ช่อง background species ตอนอัปโหลด) คั่นด้วย ; — คลิปที่ยังไม่ได้ตรวจ แอปเติมชนิดอื่นที่ BirdNET
        # เจอระหว่างคลิปนี้ให้ คลิกช่อง = ชนิดเหล่านั้นที่ยังไม่อยู่ในช่อง, พิมพ์ = ค้นชื่อ eBird
        self.review_background = tk.StringVar()
        self._background_auto = None           # ข้อความที่แอปเติมให้คลิปนี้ (None = ค่าที่บันทึกไว้)
        ttk.Label(fields, text="Background species").grid(row=2, column=0, sticky="w")
        self.review_background_entry = ttk.Entry(fields, textvariable=self.review_background)
        self.review_background_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=5)
        self.background_picker = SpeciesPicker(self.review_background_entry, self._review_country,
                                               lambda _common, _scientific: None, multiple=True,
                                               candidates_of=self._background_candidates)

        actions = ttk.Frame(frm)
        actions.grid(row=7, column=0, sticky="ew", pady=(6, 2))
        ttk.Button(actions, text="✓ Approve → Ready", style="QuickAction.TButton",
                   command=lambda: self.review_save("Approved")).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Reject", style="QuickAction.TButton",
                   command=lambda: self.review_save("Rejected")).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Not sure", style="QuickAction.TButton",
                   command=lambda: self.review_save("Unsure")).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Keep pending", style="QuickAction.TButton",
                   command=lambda: self.review_save("Pending")).pack(side="left")
        # บันทึกแล้วเลือกคลิปถัดไปที่ยังไม่ได้ตรวจเอง (ตามลำดับในตาราง) — จำไว้ครั้งหน้า
        self.auto_next = tk.BooleanVar(value=bool(load_settings().get("auto_next", True)))
        self.auto_next.trace_add("write", lambda *_args: save_settings({"auto_next": bool(self.auto_next.get())}))
        ttk.Checkbutton(actions, text="then next pending", variable=self.auto_next).pack(side="left", padx=(8, 0))
        # คำอธิบายอยู่ข้างปุ่ม (ไม่เพิ่มแถว: ให้ตัวดูเสียงมีที่มากขึ้น)
        ttk.Label(actions, text="Reject = wrong species. Poor sound but right species: quality 1. "
                  "Not sure = too unclear. Keys (click the list first): 1–5 quality, Enter/A approve, R reject, "
                  "U not sure, P pending, S/C/F song/call/flight call, Space play.",
                  wraplength=760, justify="left", foreground="#808080").pack(side="left", padx=(12, 0))

    def refresh_results(self, auto_load=False):
        """ค้นผลวิเคราะห์ทุกชุดจากทุกโฟลเดอร์ผลลัพธ์ (เบื้องหลัง) แล้วใส่ในเมนู Results"""
        root = self.out.get().strip() or DEF_OUT

        results_q = self._results_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            notes = []
            try:
                if auto_load:    # เปิดแอป (ยังไม่ได้เปิดผลชุดไหน): ย้ายผลแบบเก่าเข้าโฟลเดอร์วัน
                    try:
                        for folder in known_roots(root):
                            result_folders.organize(folder, log=notes.append)
                    except OSError as exc:
                        notes.append(f"Could not group old results by day: {exc}")
                results_q.put((all_results(root), auto_load, notes))
            except OSError:
                results_q.put(([], auto_load, notes))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._results_poll)

    def _results_poll(self):
        try:
            found, auto_load, notes = self._results_q.get_nowait()
        except queue.Empty:
            self.root.after(150, self._results_poll)
            return
        if notes:
            self.status.set(notes[-1])
        self._results = {}
        for info in found:
            label = result_label(info)
            if label in self._results:        # ชื่อซ้ำ (นาทีเดียวกัน) -> ต่อชื่อโฟลเดอร์
                label += f" · {info['folder']}"
            if label in self._results:        # ผลเวลาเดียวกันในคนละโฟลเดอร์ผลลัพธ์ -> ต่อที่อยู่
                label += f" · {Path(info['path']).parent.parent.parent}"
            self._results[label] = info["path"]
        self.results_box.configure(values=list(self._results))
        if auto_load and not self.review_summary.get():
            # เปิดแอปใหม่: กลับไปที่ผลชุดล่าสุดที่เปิดดู (หรือชุดใหม่สุด)
            last = Path(load_settings().get("last_summary") or "")
            target = last if last.is_file() else (found[0]["path"] if found else None)
            if target is not None:
                self.review_summary.set(str(target))
                self.review_load(quiet=True)
                if self.review_rows:
                    self.nb.select(self.review_tab)
        self._select_result_label()

    def _select_result_label(self):
        current = Path(self.review_summary.get()).expanduser()
        label = next((text for text, path in self._results.items() if Path(path) == current), "")
        self.results_choice.set(label or (str(current) if self.review_summary.get() else ""))

    def _on_result_selected(self, _event=None):
        path = self._results.get(self.results_choice.get())
        if path:
            self.review_summary.set(str(path))
            self.review_load()

    def review_browse(self):
        path = filedialog.askopenfilename(title="Choose summary.xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            self.review_summary.set(path)
            self.review_load()

    def review_load(self, quiet=False, keep_selection=False):
        try:
            summary = Path(self.review_summary.get()).expanduser()
            rows = load_reviews(summary)
            try:
                save_settings({"last_summary": str(summary.resolve()),    # เปิดแอปครั้งหน้าจะกลับมาที่นี่
                               OUTPUT_FOLDERS: remembered(result_folders.root_of(summary))})
            except OSError:
                pass
            if summary.resolve() not in {Path(p).resolve() for p in self._results.values()}:
                self.refresh_results()            # ผลชุดใหม่ (เพิ่งวิเคราะห์เสร็จ) -> เพิ่มในเมนู
            else:
                self._select_result_label()
            self.review_rows = {str(row["_row"]): row for row in rows}
            self._review_flags = recorder_flags.by_source(rows)
            self._check_spectrograms(summary)
            self._check_season(summary)
            self._check_auto(summary)
            self._check_model(summary)
            self._check_index(summary)
            species = sorted({species_names.row_names(row)[0] for row in rows}, key=str.casefold)
            self.review_species_box.configure(values=["All"] + species)
            wanted = self.review_species_filter.get()
            if wanted != "All" and wanted not in species:          # ผลชุดนี้ไม่มีชนิดนี้: แสดงทั้งหมด
                self.review_species_filter.set("All")
                self.status.set(f"No {wanted} clips in this result — showing all species")
            self._fill_review_tree(summary, keep_selection=keep_selection)
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Cannot open summary", str(exc))

    def _fill_review_tree(self, summary=None, keep_selection=True):
        """ใส่คลิปลงตาราง (เฉพาะชนิดที่เลือกในช่อง Species) แล้วเรียงตามคอลัมน์ที่เลือกไว้
        keep_selection = เลือกคลิปเดิมต่อ (เปลี่ยนตัวกรอง) — ผลชุดอื่นเลขแถวซ้ำกันได้ จึงไม่เก็บ"""
        summary = summary or Path(self.review_summary.get()).expanduser()
        wanted = self.review_species_filter.get()
        wanted_status = self.review_status_filter.get()
        selected = self.review_tree.selection() if keep_selection else ()
        self.review_tree.delete(*self.review_tree.get_children())
        shown = 0
        for key, row in self.review_rows.items():
            if wanted != "All" and species_names.row_names(row)[0] != wanted:
                continue
            status = row.get("Review status") or "Pending"
            ready = row.get("Ready file")
            if status == "Approved" and (not ready or
                not result_folders.ready_file(summary.parent, str(ready)).is_file()):
                status = "Missing Ready"
            display_status = {"Missing Ready": "Ready file missing", "Unsure": "Not sure"}.get(status, status)
            check = "Check" if (str(row.get("Expected by location/date") or "") not in ("", "Yes") or
                                str(row.get("Fits site habitat") or "").startswith("No") or
                                country_checklist.needs_check(clip_checklist(row)) or
                                self._season_flagged_row(row) or
                                self._auto_flagged_row(row) or
                                self._model_flagged_row(row)) else ""
            flag = self._row_flag(row)
            if not review_status_shown(wanted_status, display_status, check, flag is not None):
                continue
            shown += 1
            self.review_tree.insert("", "end", iid=key,
                                    values=(row.get("Clock time") or "", row.get("File") or "",
                                            species_names.row_names(row)[0],
                                            "added by you" if manual_clips.is_manual(row)
                                            else f"{row.get('Max confidence')} (2nd look)" if second_look_note(row)
                                            else row.get("Max confidence") or "",
                                            "" if flag is None else f"⚑ {flag[1]:+.0f} s" if flag[1] else "⚑ in clip",
                                            check, display_status))
        self._apply_review_sort()
        self.review_species_count.set("" if wanted == "All" and wanted_status == "All"
                                      else f"{shown} of {len(self.review_rows)} clips")
        kept = [key for key in selected if self.review_tree.exists(key)]
        if kept:
            self.review_tree.selection_set(kept)
            self.review_tree.see(kept[0])
        else:
            self._clear_review_preview()

    def _row_flag(self, row):
        """flag ที่กดตอนอัดใกล้คลิปนี้ -> (flag วินาทีในไฟล์ต้นฉบับ, ระยะ บวก = คลิปอยู่หลัง flag) หรือ None"""
        flags = self._review_flags.get(str(row.get("Source ID") or ""))
        first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
        if not flags or first is None:
            return None
        return recorder_flags.distance(flags, first, last if last is not None else first)

    def _clip_flags(self, row):
        """flag ในเวลาของไฟล์คลิป (0 = ต้นคลิป) ไว้วาดบนภาพเสียง"""
        offset = _number(row.get("Start offset (s)"))
        if offset is None:
            return []
        spans, _clock = _clip_spans_and_clock(row)
        first = offset - (spans[0][0] if spans else 0.0)      # คลิปของ BirdNET เริ่มก่อนเสียงแรก
        return [flag - first for flag in self._review_flags.get(str(row.get("Source ID") or ""), ())]

    def open_upload_sheet(self):
        """รายการอัปโหลดของวันนี้ -> <วัน>/Ready/upload-sheet.csv แล้วเปิดให้ดูคู่กับหน้า eBird"""
        summary = Path(self.review_summary.get().strip()).expanduser()
        if not summary.is_file():
            messagebox.showinfo("No result open", "Open a result first: the sheet lists that day's Ready clips.")
            return
        try:
            path, clips = upload_sheet.write(summary)
            if clips:
                open_path(path)
        except Exception as exc:
            messagebox.showerror("Cannot make the upload sheet", str(exc))
            return
        if not clips:
            messagebox.showinfo("Nothing to upload yet", "No approved clips in Ready for this day.")
            return
        self.status.set(f"Upload sheet: {len(clips)} clips, {sum(bool(c['Additional species']) for c in clips)} with "
                        f"additional species, {sum(bool(c['BirdNET also heard']) for c in clips)} to listen to again")

    def open_recorder_flags(self):
        from flags_window import RecorderFlagsWindow
        window = getattr(self, "_flags_window", None)
        if window is not None and window.win.winfo_exists():
            window.win.lift()
            window.refresh()
            return
        self._flags_window = RecorderFlagsWindow(self)

    def open_manual(self):
        """หน้าต่างเลือกช่วงเสียงและใส่ชนิดเอง เริ่มที่ไฟล์/ตำแหน่งของคลิปที่เลือกอยู่ (ถ้ามี)"""
        from manual_window import ManualClipWindow
        if self._manual_window is not None and self._manual_window.win.winfo_exists():
            self._manual_window.win.lift()
            return
        _key, row = self._selected_review()
        summary = self.review_summary.get().strip()
        self._manual_window = ManualClipWindow(
            self, Path(summary).expanduser() if summary else None, list(self.review_rows.values()),
            source=row.get("Source path") if row else None,
            offset=_number(row.get("Start offset (s)")) if row else None)

    def open_manual_at(self, summary, source, offset):
        """หน้าต่างเพิ่มคลิปเองที่ไฟล์/ตำแหน่งนี้ (จากหน้าต่างค้นเสียง)"""
        from manual_window import ManualClipWindow
        if self._manual_window is not None and self._manual_window.win.winfo_exists():
            self._manual_window.close()
        try:
            rows = load_reviews(Path(summary))
        except Exception:  # noqa: BLE001
            rows = list(self.review_rows.values())
        self._manual_window = ManualClipWindow(self, Path(summary), rows, source=str(source), offset=offset)

    def open_find_similar(self, query):
        from sound_search import FindSimilarWindow
        FindSimilarWindow(self, query)

    def open_find_similar_clip(self):
        """เสียงที่คล้ายคลิปที่เลือก (ใช้ช่วงที่ BirdNET ได้ยิน หรือช่วงที่คุณเลือกตอนเพิ่มคลิปเอง)"""
        key, row = self._selected_review()
        if row is None:
            messagebox.showinfo("Find similar", "Select a clip first")
            return
        first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
        if not row.get("Source ID") or first is None or not row.get("Source path"):
            messagebox.showinfo("Find similar", "This clip has no position in its original recording")
            return
        common = self.review_common.get().strip() or species_names.row_names(row)[0]
        scientific = self.review_scientific.get().strip() or species_names.row_names(row)[1]
        clock = row.get("Clock time") or ""
        self.open_find_similar({
            "sid": str(row["Source ID"]), "source": Path(str(row["Source path"])),
            "summary": Path(self.review_summary.get()).expanduser(), "start": first,
            "end": max(last if last is not None else first + 3.0, first + 0.5),
            "label": f"{common} {clock} ({row.get('Source file') or ''})", "common": common, "scientific": scientific})

    def open_sound_groups(self):
        summary = Path(self.review_summary.get()).expanduser()
        if not summary.is_file():
            messagebox.showinfo("Sound groups", "Open a result first")
            return
        from sound_search import SoundGroupsWindow
        SoundGroupsWindow(self, summary)

    def open_near_misses(self, species=None):
        summary = Path(self.review_summary.get()).expanduser()
        if not summary.is_file():
            messagebox.showinfo("Near misses", "Open a result first")
            return
        from sound_search import NearMissWindow
        window = getattr(self, "_near_window", None)
        if window is not None and window.win.winfo_exists() and window.summary == summary.resolve():
            window.win.lift()
            if species:
                window.select_species(species)
            return
        self._near_window = NearMissWindow(self, summary, species)

    def open_expected(self):
        summary = Path(self.review_summary.get()).expanduser()
        if not summary.is_file():
            messagebox.showinfo("Expected species", "Open a result first")
            return
        from sound_search import ExpectedSpeciesWindow
        ExpectedSpeciesWindow(self, summary)

    def open_signal_events(self):
        from event_window import SignalEventsWindow
        SignalEventsWindow(self)

    def open_detections_import(self):
        from import_window import DetectionsImportWindow
        DetectionsImportWindow(self)

    def open_presets(self):
        from workflow_presets import PresetsWindow
        PresetsWindow(self)

    def open_recognizers(self):
        from sound_search import RecognizersWindow
        window = getattr(self, "_recognizers_window", None)
        if window is not None and window.win.winfo_exists():
            window.win.lift()
            return
        self._recognizers_window = RecognizersWindow(self)

    def _check_index(self, summary):
        """ไฟล์ต้นฉบับของผลชุดนี้ที่ยังไม่มีดัชนีเสียง -> ทำเป็นงานเบื้องหลังครั้งเดียว (หลีกทางให้งานที่คุณสั่ง)"""
        summary = summary.resolve()
        if summary in self._index_checked:
            return
        self._index_checked.add(summary)
        rows, index_q = list(self.review_rows.values()), self._index_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            import sound_index
            try:
                index_q.put((summary, len(sound_index.missing(sound_index.sources(rows, summary)))))
            except Exception:  # noqa: BLE001  ตรวจไม่ได้: ค่อยทำตอนใช้ Find sounds
                pass
        threading.Thread(target=work, daemon=True).start()
        self.root.after(600, self._index_poll)

    def _index_poll(self):
        try:
            summary, count = self._index_q.get_nowait()
        except queue.Empty:
            self.root.after(600, self._index_poll)
            return
        if count:
            self._start(self._launch() + ["--sound-index", "build", str(summary)], None,
                        f"Sound index: {count} recording(s) — {summary.parent.name}", is_analyze=False,
                        on_done=lambda _code, _lines: None, background=True)

    def show_manual_result(self, summary, file):
        """หลังเพิ่มคลิปเอง: เปิดผลชุดนั้นในแท็บ Review แล้วเลือกคลิปใหม่"""
        self.show_clip(summary, file, "Clip added — rate and approve it in Review clips ✓")

    def show_clip(self, summary, file, status=""):
        """เปิดผลชุดนั้นในแท็บ Review (ถ้ายังไม่ได้เปิด) แล้วเลือกคลิปนี้"""
        current = Path(self.review_summary.get()).expanduser()
        if not (current.is_file() and current.resolve() == Path(summary).resolve() and self.review_rows):
            self.review_summary.set(str(summary))
            self.review_load()
        else:
            self._fill_review_tree()              # ตัวกรอง Species อาจเพิ่งเปลี่ยน
        key = next((key for key, row in self.review_rows.items() if str(row.get("File")) == file), None)
        if key is not None and not self.review_tree.exists(key):     # คลิปนี้ถูกตัวกรอง Species ซ่อน: แสดงทั้งหมด
            self.review_species_filter.set("All")
            self._fill_review_tree()
        if key is not None:
            self.review_tree.selection_set(key)
            self.review_tree.see(key)
            self._review_select()
        self.nb.select(self.review_tab)
        if status:
            self.status.set(status)

    def open_sound_types(self, group=None):
        """หน้าต่างดูคำเดา Song/Call ของทุกคลิป แล้วเลือกประเภทที่ถูก (ช่วยตัวเดาให้แม่นขึ้น)"""
        from sound_type_window import SoundTypeWindow
        window = getattr(self, "_sound_window", None)
        if window is not None and window.win.winfo_exists():
            window.win.lift()
            window.refresh()
            return
        self._sound_window = SoundTypeWindow(self, group)

    def open_progress(self):
        from review_progress import ReviewProgressWindow
        if self._progress_window is not None and self._progress_window.win.winfo_exists():
            self._progress_window.win.lift()
            self._progress_window.refresh()
            return
        self._progress_window = ReviewProgressWindow(self)

    def _season_flagged_row(self, row):
        group = _season_group(row)
        scientific = row.get("Reviewed scientific") or row.get("Species (scientific)")
        return bool(group) and (group[0], str(scientific)) in self._season_flagged

    def _check_season(self, summary):
        """ถาม eBird (key ของผู้ใช้) ว่าชนิดในผลชุดนี้เคยมีรายงานในมณฑลช่วงเดียวกันของปีก่อน ๆ ไหม -> ช่อง Check"""
        key = self.ebird_key.get().strip()
        summary = summary.resolve()
        if not key or summary in self._season_checked:
            return
        self._season_checked.add(summary)
        rows = list(self.review_rows.values())

        season_q = self._season_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            seasons = {}
            for group in {g for g in map(_season_group, rows) if g}:
                try:
                    region = ebird.region_of(group[1], group[2], key)
                    if region:
                        seasons[group] = ebird.season_reports(region, date.fromisoformat(group[0]), key)
                except (ebird.EBirdError, ValueError):
                    continue
            try:
                flagged = season_flags(rows, seasons)
            except ebird.EBirdError:                  # ไม่มีระบบชื่อ eBird ในเครื่องและต่อเน็ตไม่ได้
                flagged = set()
            season_q.put((summary, flagged))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(500, self._season_poll)

    def _season_poll(self):
        try:
            summary, flagged = self._season_q.get_nowait()
        except queue.Empty:
            self.root.after(500, self._season_poll)
            return
        self._season_flagged |= flagged
        current = Path(self.review_summary.get()).expanduser()
        if flagged and current.is_file() and current.resolve() == summary:
            for iid, row in self.review_rows.items():
                if self._season_flagged_row(row) and self.review_tree.exists(iid):
                    self.review_tree.set(iid, "location", "Check")

    def _check_auto(self, summary):
        """คลิปที่ยังไม่ได้ตรวจอัตโนมัติ (แมลง/กบดังแข่ง, second opinion ของคลิปคะแนนต่ำ) -> งานเบื้องหลังครั้งเดียว"""
        summary = summary.resolve()
        if summary in self._auto_checked:
            return
        self._auto_checked.add(summary)
        rows = list(self.review_rows.values())
        has_key = bool(self.xc_key.get().strip())

        auto_q = self._auto_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            count = sum(auto_checks.needs_check(row, has_key) for row in rows)
            auto_q.put((summary, count))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(400, self._auto_poll)

    def _auto_poll(self):
        try:
            summary, count = self._auto_q.get_nowait()
        except queue.Empty:
            self.root.after(400, self._auto_poll)
            return
        if count:
            if self.xc_key.get().strip() != load_settings().get("xeno_canto_key", ""):
                self._save_xc_key()               # งานเบื้องหลังอ่าน key จากไฟล์ตั้งค่า
            self._start(self._launch() + ["--auto-checks", str(summary)], None,
                        f"Auto-checks: {count} clip(s) — {summary.parent.name}", is_analyze=False,
                        on_done=lambda code, lines: self._auto_done(summary, lines), background=True)

    def _auto_done(self, summary, lines=()):
        """ผลตรวจอัตโนมัติมาแล้ว: ใส่ Check ในตาราง และแสดงผลของคลิปที่เลือกอยู่"""
        done = ("Auto-checks done — xeno-canto was slow, so reference downloads wait for next time"
                if any(str(line).startswith("SLOW ") for line in lines) else "Auto-checks done ✓")
        current = Path(self.review_summary.get()).expanduser()
        if not (current.is_file() and current.resolve() == summary):
            self.status.set(done)
            return
        for iid, row in self.review_rows.items():
            if self._auto_flagged_row(row) and self.review_tree.exists(iid):
                self.review_tree.set(iid, "location", "Check")
        _key, row = self._selected_review()
        if row is not None:
            self._show_sound_guess(row)
            self._show_review_preview(row)
            cached = second_opinion.cached_result(row.get("Generated sha256"))
            if cached and not self._second_opinion_busy:
                self.second_opinion.set(format_second_opinion(cached))
        if self._progress_window is not None and self._progress_window.win.winfo_exists():
            self._progress_window.refresh()
        self.status.set(done)

    RETRAIN_EVERY = 5                          # ฝึกใหม่เมื่อผลตรวจเปลี่ยนไปอย่างน้อยเท่านี้

    def _model_flagged_row(self, row):
        risk = self._model_risk.get(str(row.get("Generated sha256") or ""))
        return risk is not None and self._warner is not None and risk >= self._warner["threshold"]

    def _check_model(self, summary):
        """ฝึก/ใช้ ML model เบื้องหลัง แล้วใส่ Check ให้คลิปที่เสี่ยงในผลชุดนี้"""
        summary = summary.resolve()
        self._model_risk = {}
        if self._model_busy:
            return
        self._model_busy = True
        root, key = self.out.get().strip() or DEF_OUT, self.ebird_key.get().strip()
        # งานเบื้องหลังไม่ถือ self (ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก): ส่งผลกลับทางคิวอย่างเดียว
        warner, trained_on, results = self._warner, self._warner_reviewed, self._model_q
        retrain_every = self.RETRAIN_EVERY

        def work():
            import review_progress
            new_warner, reviewed, risks = warner, trained_on, {}
            try:
                summaries = {Path(info["path"]).resolve() for info in all_results(root)} | {summary}
                records = review_progress.collect(sorted(summaries))
                reviewed = sum(review_progress.decided(r) and not r["_manual"] for r in records)
                if (warner is None and reviewed != trained_on) or abs(reviewed - trained_on) >= retrain_every:
                    season = review_progress.season_flags_all(records, key)
                    new_warner, _result = review_progress.warning_model(records, season)
                else:
                    review_progress.season_flags_all(records, None)            # แค่ใส่ _season_key
                    reviewed = trained_on
                if new_warner is not None:
                    for row in records:
                        if row["_summary"].resolve() == summary and not row["_manual"]:
                            risks[str(row.get("Generated sha256") or "")] = review_progress.warning_risk(new_warner, row)
            except Exception as exc:  # noqa: BLE001  โมเดลพังไม่ควรรบกวนการตรวจ
                print(f"ML model check failed: {exc}", flush=True)
            results.put((summary, risks, new_warner, reviewed))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(500, self._model_poll)

    def _model_poll(self):
        try:
            summary, risks, warner, reviewed = self._model_q.get_nowait()
        except queue.Empty:
            self.root.after(500, self._model_poll)
            return
        self._model_busy = False
        self._warner, self._warner_reviewed = warner, reviewed
        current = Path(self.review_summary.get()).expanduser()
        if not (current.is_file() and current.resolve() == summary):
            if current.is_file():                 # ผลชุดเปลี่ยนระหว่างคำนวณ: ทำของชุดใหม่
                self._check_model(current)
            return
        self._model_risk = risks
        for iid, row in self.review_rows.items():
            if self._model_flagged_row(row) and self.review_tree.exists(iid):
                self.review_tree.set(iid, "location", "Check")
        _key, row = self._selected_review()
        if row is not None:
            self._show_review_preview(row)

    @staticmethod
    def _auto_flagged_row(row):
        if manual_clips.is_manual(row):
            return False
        found = auto_checks.flags_for(row)
        return bool(found["nonbird"] or found["second"])

    def _check_spectrograms(self, summary):
        """ภาพของผลชุดนี้ขาด/วาดแบบเก่า (เช่น ก่อนมีแกนเวลาจริง) -> วาดใหม่เป็นงานเบื้องหลังครั้งเดียว"""
        folder = summary.resolve().parent
        if folder in self._spectrograms_checked:
            return
        self._spectrograms_checked.add(folder)

        spectrogram_q = self._spectrogram_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            try:
                from field_audio_to_ebird import spectrograms_to_update
                spectrogram_q.put((folder, spectrograms_to_update(folder)))
            except Exception:  # noqa: BLE001  ตรวจไม่ได้ก็ใช้ภาพเดิม
                pass
        threading.Thread(target=work, daemon=True).start()
        self.root.after(300, self._spectrogram_poll)

    def _spectrogram_poll(self):
        try:
            folder, count = self._spectrogram_q.get_nowait()
        except queue.Empty:
            self.root.after(300, self._spectrogram_poll)
            return
        if count:
            self._start(self._launch() + ["--gen-spectrograms", str(folder)], None,
                        f"Redraw {count} spectrogram(s) with clock time — {result_folders.tag(folder)}", is_analyze=False,
                        on_done=lambda code, _lines: self._spectrograms_redrawn(folder, code), background=True)

    def _spectrograms_redrawn(self, folder, code):
        self.status.set("Spectrograms now show clock time ✓" if code == 0 else "Some spectrograms could not be redrawn")
        current = Path(self.review_summary.get()).expanduser()
        _key, row = self._selected_review()
        if row is not None and current.is_file() and current.resolve().parent == folder:
            self._show_review_preview(row)

    def _sort_review(self, column):
        """คลิกหัวคอลัมน์: เรียงน้อยไปมาก คลิกซ้ำ = มากไปน้อย"""
        self._review_sort = (column, self._review_sort == (column, False))
        self._apply_review_sort()

    def _review_sort_key(self, iid, column):
        """ค่าที่ใช้เรียง หรือ None = ไม่มีค่า (ไว้ท้ายเสมอ)"""
        row = self.review_rows.get(iid, {})
        if column == "time":                      # เวลาจริงเต็ม (ข้ามเที่ยงคืนถูก) ถ้ารู้ ไม่งั้น Clock time
            span = manual_clips.heard_span(row)
            return span[0].isoformat() if span else (str(row.get("Clock time") or "") or None)
        if column == "confidence":                # คลิปที่เพิ่มเองไม่มีคะแนน
            return None if manual_clips.is_manual(row) else _number(row.get("Max confidence"))
        if column == "flag":                      # ใกล้ flag ที่สุดก่อน
            flag = self._row_flag(row)
            return None if flag is None else abs(flag[1])
        return str(self.review_tree.set(iid, column)).casefold() or None

    def _apply_review_sort(self):
        if not self._review_sort:
            return
        column, descending = self._review_sort
        keyed = [(self._review_sort_key(iid, column), iid) for iid in self.review_tree.get_children()]
        present = sorted((pair for pair in keyed if pair[0] is not None), key=lambda pair: pair[0],
                         reverse=descending)
        for index, (_key, iid) in enumerate(present + [pair for pair in keyed if pair[0] is None]):
            self.review_tree.move(iid, "", index)
        for key, label in self._review_headings.items():
            self.review_tree.heading(key, text=label + ((" ▼" if descending else " ▲") if key == column else ""))
        selected = self.review_tree.selection()
        if selected:
            self.review_tree.see(selected[0])

    def _selected_review(self):
        selected = self.review_tree.selection()
        return (selected[0], self.review_rows[selected[0]]) if selected else (None, None)

    def _review_step(self, step):
        """เลื่อนไปคลิปก่อนหน้า (step = -1) / ถัดไป (+1) ในตาราง"""
        tree = self.review_tree
        items = tree.get_children()
        if not items:
            return
        current = tree.selection()
        if current and current[0] in items:
            index = min(len(items) - 1, max(0, items.index(current[0]) + step))
        else:
            index = 0 if step > 0 else len(items) - 1
        tree.selection_set(items[index])
        tree.focus(items[index])
        tree.see(items[index])

    def _review_select(self, _event=None):
        selected = self.review_tree.selection()
        if selected and self.review_tree.focus() not in selected:   # เลือกจากโค้ด (เปิดแอป/คลิปที่แนะนำ): ให้ลูกศรเริ่มจากแถวนี้
            self.review_tree.focus(selected[0])
        _key, row = self._selected_review()
        if row is None:
            self._clear_review_preview()
            return
        # ชื่อตาม eBird (ผลรุ่นเก่ายังเป็นชื่อของ BirdNET เช่น Black-throated Bushtit = Black-throated Tit)
        common, scientific = species_names.row_names(row)
        self.review_common.set(row.get("Reviewed common") or common)
        self.review_scientific.set(row.get("Reviewed scientific") or scientific)
        rating = row.get("Quality rating")   # summary เก่าอาจเก็บเป็น 3.0
        self.review_rating.set(str(int(rating)) if isinstance(rating, (int, float)) and rating == rating else "")
        self.review_notes.set(row.get("Review notes") or "")
        self._fill_background(row)
        saved = str(row.get("Sound type") or "")
        self.review_sound_type.set(saved if saved in SOUND_TYPES else "")
        self._show_sound_guess(row)
        self._show_review_preview(row)
        self._show_review_photos(row)
        if not self._second_opinion_busy:     # ผลที่เคยคำนวณไว้ (เก็บตาม sha256 ของคลิป)
            cached = second_opinion.cached_result(row.get("Generated sha256"))
            self.second_opinion.set(format_second_opinion(cached) if cached else "")
        self.species_picker.hide()
        self.background_picker.hide()
        self._refresh_species_checks(row)

    @staticmethod
    def _hide_when_empty(variable, label, **grid):
        """บรรทัดข้อความใต้ภาพเสียงที่ว่างไม่กินที่ ภาพเสียงจึงได้ที่มากขึ้น"""
        label.grid(column=0, sticky="w", **grid)

        def follow(*_args):
            if variable.get():
                label.grid()
            else:
                label.grid_remove()
        variable.trace_add("write", follow)
        follow()

    def _show_review_photos(self, row):
        """ชนิดในรูปของคุณที่ถ่ายระหว่างหรือใกล้เสียงนี้ (clip_photos.py) — อ่านเวลาในรูปเบื้องหลัง"""
        self._photo_request += 1
        request, results = self._photo_request, self._photo_q
        summary = Path(self.review_summary.get()).expanduser()
        try:
            offset = float(load_settings().get("camera_minus_recorder_s") or 0.0)
        except (TypeError, ValueError):
            offset = 0.0

        def work():   # ไม่ถือ self
            try:
                results.put((request, clip_photos.describe(clip_photos.photos_at(row, summary, offset))))
            except Exception:  # noqa: BLE001  รูปเสีย/โฟลเดอร์หาย: ไม่แสดง
                results.put((request, ""))
        threading.Thread(target=work, daemon=True).start()
        if not self._photo_polling:
            self._photo_polling = True
            self.root.after(80, self._photos_poll)

    def _photos_poll(self):
        latest = None
        try:
            while True:
                item = self._photo_q.get_nowait()
                if item[0] == self._photo_request:
                    latest = item
        except queue.Empty:
            pass
        if latest is None:
            self.root.after(80, self._photos_poll)
            return
        self._photo_polling = False
        self.photo_text.set(latest[1])

    def _refresh_species_checks(self, row):
        """บรรทัด Checklist และ eBird ของชนิดในช่องตรวจ (เปลี่ยนตามชนิดที่แก้)"""
        checklist = clip_checklist(row, self.review_scientific.get().strip(), self.review_common.get().strip())
        self.checklist_info.set(format_checklist(checklist))
        self.checklist_label.configure(
            foreground="#c77c10" if country_checklist.needs_check(checklist) else "#4a9a4f")
        self._start_ebird_check(row)

    def _review_country(self):
        _key, row = self._selected_review()
        if row is None:
            return None
        return country_checklist.country_at(_number(row.get("Latitude")), _number(row.get("Longitude")))

    def _review_candidates(self):
        """ตัวเลือกตอนคลิกช่องชื่อ: second opinion (คล้ายเสียงอ้างอิงที่สุดก่อน) แล้วชนิดอื่นที่ BirdNET เดา"""
        _key, row = self._selected_review()
        if row is None:
            return []
        found, seen = [], set()

        def add(common, scientific, note):
            names = species_names.to_ebird(scientific, common) or {"common": common, "scientific": scientific}
            if names["scientific"] and names["scientific"] not in seen:
                seen.add(names["scientific"])
                found.append((names["common"], names["scientific"], note))

        result = second_opinion.cached_result(row.get("Generated sha256"))
        for item in (result or {}).get("ranking", []):
            if item.get("similarity") is not None:
                add(item.get("common") or "", item["scientific"], f"second opinion {item['similarity']:.2f}"
                    + (f", BirdNET {item['birdnet']:.2f}" if item.get("birdnet") else ""))
        common, scientific = species_names.row_names(row)
        score = _number(row.get("Max confidence"))
        add(common, scientific, "BirdNET's ID" + (f" {score:.2f}" if score is not None else ""))
        for n in (1, 2):
            other, confidence = row.get(f"Alt species {n}"), _number(row.get(f"Alt{n} conf"))
            if other:
                names = species_names.to_ebird("", other)
                if names:
                    add(names["common"], names["scientific"], "BirdNET's other guess"
                        + (f" {confidence:.2f}" if confidence is not None else ""))
        if not result:
            found.append((None, None, "No second opinion yet — press Second opinion, or type a name to search"))
        return found

    def _fill_background(self, row):
        """ช่อง Background species: ค่าที่บันทึกไว้ หรือถ้ายังไม่ได้ตรวจ = ชนิดอื่นที่ได้ยินระหว่างคลิปนี้
        (คลิปที่ตรวจไปแล้วแต่ช่องว่าง ไม่เติมให้: อาจลบออกเองเพราะฟังแล้วไม่มี — คลิกช่องเพื่อเลือกเพิ่มได้)"""
        saved = str(row.get("Background species") or "").strip()
        self._background_auto = None
        if saved or str(row.get("Review status") or "Pending") != "Pending":
            self.review_background.set(saved)
            return
        found = background_species.suggest(row, list(self.review_rows.values()), self.review_scientific.get().strip())
        self._background_auto = background_species.join(common for common, _scientific, _note in found)
        self.review_background.set(self._background_auto)

    def _background_candidates(self):
        """ตัวเลือกตอนคลิกช่อง Background species: ชนิดอื่นที่ได้ยินระหว่างคลิปนี้ ที่ยังไม่อยู่ในช่อง"""
        _key, row = self._selected_review()
        if row is None:
            return []
        listed = {name.casefold() for name in background_species.split(self.review_background.get())}
        return [item for item in background_species.suggest(row, list(self.review_rows.values()),
                                                            self.review_scientific.get().strip())
                if item[0].casefold() not in listed]

    def _review_species_picked(self, common, scientific):
        if common:
            self.review_common.set(common)
        self.review_scientific.set(scientific)
        self._review_species_edited(delay=0)

    def _review_species_edited(self, delay=600):
        """แก้ชื่อชนิดแล้ว: เช็ก checklist/eBird ของชนิดใหม่ (รอให้พิมพ์เสร็จก่อน)"""
        if self._species_refresh_job:
            self.root.after_cancel(self._species_refresh_job)

        def run():
            self._species_refresh_job = None
            _key, row = self._selected_review()
            if row is not None:
                # ชนิดหลักเปลี่ยน: ช่องพื้นหลังที่แอปเติมให้ (ยังไม่ได้แก้เอง) คำนวณใหม่ ไม่ให้ชนิดหลักซ้ำในนั้น
                if self._background_auto is not None and self.review_background.get() == self._background_auto:
                    self._fill_background(row)
                self._refresh_species_checks(row)
        self._species_refresh_job = self.root.after(delay, run)

    def _show_sound_guess(self, row):
        """คำเดา Song/Call ข้างช่อง Sound type; มั่นใจและยังไม่ได้เลือก = เติมให้"""
        found = sound_type.guess(row)
        self._sound_guess = found
        self.sound_guess.set(sound_type.describe(found))
        if found and found["sure"] and not self.review_sound_type.get() and not row.get("Sound type"):
            self.review_sound_type.set(found["type"])

    def _start_ebird_check(self, row):
        """เช็กชนิดนี้กับ eBird รอบจุดบันทึก (เบื้องหลัง, มี cache) + เติมชื่อตาม eBird ถ้ายังไม่ได้ตรวจ"""
        self._ebird_request += 1
        request = self._ebird_request
        scientific = self.review_scientific.get().strip()
        common = self.review_common.get().strip()
        if xeno_canto_url(scientific) is None:
            self._ebird_answered = request
            self.ebird_info.set("")
            return
        lat, lon = _number(row.get("Latitude")), _number(row.get("Longitude"))
        try:
            recorded = datetime.fromisoformat(json.loads(row.get("Analysis settings") or "{}").get("date"))
        except (TypeError, ValueError):
            recorded = None
        key = self.ebird_key.get().strip()
        self._ebird_region = None
        self.ebird_info.set("eBird: checking …" if key else "")
        prefill = not row.get("Reviewed common")

        ebird_q = self._ebird_q
        season_line_q = self._season_line_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            try:
                result = ebird.check(scientific, lat, lon, recorded, key, common)
                ebird_q.put((request, result, None, prefill, common))
            except ebird.EBirdError as exc:
                ebird_q.put((request, None, str(exc), prefill, common))
                return
            if key and recorded and result.get("region") and result.get("known"):
                try:
                    season = ebird.season_reports(result["region"], recorded, key)
                    season_line_q.put((request, format_season(season, result["code"], result["region_name"])))
                except ebird.EBirdError as exc:
                    season_line_q.put((request, f"Past years: {exc}"))
        threading.Thread(target=work, daemon=True).start()
        if not self._season_line_polling:
            self._season_line_polling = True
            self.root.after(300, self._season_line_poll)
        if not self._ebird_polling:
            self._ebird_polling = True
            self.root.after(150, self._ebird_poll)

    def _ebird_poll(self):
        """ตัวรับผลตัวเดียว: แสดงเฉพาะผลของคลิปล่าสุด แล้วหยุดเมื่อได้ผลนั้น"""
        try:
            while True:
                request, result, error, prefill, common = self._ebird_q.get_nowait()
                if request == self._ebird_request:
                    self._ebird_answered = request
                    self._show_ebird(result, error, prefill, common)
        except queue.Empty:
            pass
        if self._ebird_answered != self._ebird_request:
            self.root.after(150, self._ebird_poll)
        else:
            self._ebird_polling = False

    def _show_ebird(self, result, error, prefill, common):
        if result:
            self._ebird_region = result.get("region")
        if error:
            self._ebird_text = ""
            self.ebird_info.set(f"eBird: {error}" if self.ebird_key.get().strip() else "")
            return
        # ชื่อตามระบบ eBird (เติมเฉพาะเมื่อยังไม่เคยตรวจ และผู้ใช้ยังไม่ได้แก้ช่องนี้)
        if prefill and result.get("known") and self.review_common.get().strip() == common:
            self.review_common.set(result["common"])
        text = format_ebird(result) if self.ebird_key.get().strip() else ""
        if text and result.get("region") and result.get("known"):
            text += "\nChecking the same dates in past years …"
        self._ebird_text = text
        self.ebird_info.set(text)

    def _season_line_poll(self):
        """ผลช่วงเดียวกันของปีก่อน ๆ: แทนบรรทัด 'Checking …' เฉพาะเมื่อยังเป็นคลิปเดิม"""
        try:
            while True:
                request, line = self._season_line_q.get_nowait()
                if request == self._ebird_request and self._ebird_text:
                    self._ebird_text = self._ebird_text.split("\n")[0] + "\n" + line
                    self.ebird_info.set(self._ebird_text)
        except queue.Empty:
            pass
        # รอต่อเมื่อผลหลักของคลิปนี้ยังไม่มา หรือยังขึ้น 'Checking …' อยู่
        if self._ebird_answered != self._ebird_request or self._ebird_text.endswith("past years …"):
            self.root.after(300, self._season_line_poll)
        else:
            self._season_line_polling = False

    def _clear_review_preview(self):
        self.review_context.set("Select a clip to see its sound and recording details")
        self._photo_request += 1                  # ทิ้งรูปที่ยังอ่านค้าง
        self.photo_text.set("")
        self.review_view.message("Select a clip to see its sound")
        self._sound_guess = None
        self.sound_guess.set("")
        if not self._second_opinion_busy:
            self.second_opinion.set("")
        self.checklist_info.set("")
        self._ebird_request += 1                  # ทิ้งผลเช็ก eBird ที่ยังค้างอยู่
        self._ebird_answered = self._ebird_request
        self._ebird_text = ""
        self.ebird_info.set("")

    def _show_review_preview(self, row):
        parts = []
        if manual_clips.is_manual(row):
            parts.append("Added by you" + (f" ({row['Sound type']})" if row.get("Sound type") else ""))
        if species_names.birdnet_name(row):
            parts.append(f"BirdNET calls it {species_names.birdnet_name(row)}")
        if second_look_note(row):
            parts.append(second_look_note(row))
        if row.get("Source file"):
            parts.append(str(row["Source file"]))
        # เวลาจริงที่ได้ยินเสียง (นาฬิกาของเครื่องอัด) ไว้เทียบกับเวลาในภาพถ่าย
        span = manual_clips.heard_span(row)
        if span:
            parts.append(f"heard {span[0]:%Y-%m-%d %H:%M:%S}–{span[1]:%H:%M:%S}")
        elif row.get("Clock time"):
            parts.append(str(row["Clock time"]))
        flag = self._row_flag(row)
        if flag:
            at = manual_clips.format_offset(flag[0], 0)
            parts.append(f"⚑ your recorder flag at {at}" if not flag[1] else
                         f"⚑ {abs(flag[1]):.0f} s {'after' if flag[1] > 0 else 'before'} your recorder flag at {at}")
        started = manual_clips.recording_start_of(row)
        if field_log.describe_recording(started):                 # checklist ที่ไฟล์เสียงนี้ใช้ (เลือกเอง / เดาจากเวลา)
            parts.append(field_log.describe_recording(started))
        logged = field_log.for_recording(started)                 # eBird checklist ของไฟล์เสียงนี้
        if logged:
            reviewed = " ".join(str(row.get("Reviewed scientific") or "").split()[:2])
            common, scientific = (row.get("Reviewed common") or "", reviewed) if reviewed else species_names.row_names(row)
            parts.append(field_log.describe_species(logged, scientific, common))
        offset = _number(row.get("Start offset (s)"))
        if offset is not None:
            parts.append(f"{manual_clips.format_offset(offset, 0)} into the recording")
        if row.get("Duration (s)") is not None:
            parts.append(f"{row['Duration (s)']}s clip")
        area_status = str(row.get("Expected by location/date") or "")
        if area_status.startswith("No"):
            parts.append("Outside expected species list — verify by ear")
        elif area_status.startswith("Not checked"):
            parts.append("Location/date not checked — verify by ear")
        if str(row.get("Fits site habitat") or "").startswith("No"):
            parts.append(f"{row.get('Species habitat') or 'Water'} bird at this site habitat — "
                         "may be a flyover; verify by ear")
        if self._model_flagged_row(row):
            risk = self._model_risk[str(row.get("Generated sha256"))]
            parts.append(f"⚠ ML model: BirdNET IDs like this were often wrong in your reviews (risk {risk:.0%})")
        nonbird = auto_checks.describe_nonbird(row)
        if nonbird and not manual_clips.is_manual(row):
            parts.append(f"⚠ a non-bird sound scored as high here ({nonbird}) — may be an insect or frog")
        self.review_context.set("  •  ".join(parts) or "Selected clip")
        try:
            path = self._review_audio_path()
        except ValueError as exc:
            self.review_view.message(str(exc))
            return
        if not path.is_file():
            self.review_view.message("The clip file is missing")
            return
        spans, clock = _clip_spans_and_clock(row)
        # ชนิด/พิกัดของคลิปนี้ ให้ Listen tools › Compare frequencies เติมให้เอง (ชื่อที่ผู้ตรวจแก้มาก่อน)
        reviewed = " ".join(str(row.get("Reviewed scientific") or "").split()[:2])
        self.review_view.species = ((row.get("Reviewed common") or species_names.row_names(row)[0], reviewed)
                                    if reviewed else species_names.row_names(row))
        lat, lon = _number(row.get("Latitude")), _number(row.get("Longitude"))
        self.review_view.location = (lat, lon) if lat is not None and lon is not None else None
        self.review_view.boxes = _clip_boxes(row)      # กรอบเสียงของ BirdNET ให้เลือกใน Compare Hz
        self.review_view.flags = self._clip_flags(row)
        self.review_view.load(path, clock_start=clock, spans=spans)

    def _review_seek(self, seconds):
        """คลิกบนภาพ: เล่นคลิปตั้งแต่จุดนั้น"""
        try:
            self.play_from(self._review_audio_path(), seconds, self.review_view)
        except ValueError as exc:
            messagebox.showerror("Cannot play clip", str(exc))

    def play_from(self, path, seconds, view):
        """เล่นคลิปตั้งแต่ seconds (ตำแหน่งในภาพของ view) แล้วให้ view แสดงเส้นตำแหน่ง (แท็บ Review หรือหน้าอื่น)
        เปิด Only where BirdNET heard it: เล่นเฉพาะช่วงที่ได้ยินต่อกัน"""
        import numpy as np
        from manual_window import read_mono, write_preview
        try:
            pieces = [read_mono(path, a, b) for a, b in view.audio_plan(seconds)] or [read_mono(path, seconds, 3600.0)]
            samples, rate = np.concatenate([piece for piece, _rate in pieces]), pieces[0][1]
            self.review_stop()
            self._review_preview_number += 1
            target = paths.CACHE / "review-preview" / f"from-{self._review_preview_number}.wav"
            for old in target.parent.glob("from-*.wav"):     # ไฟล์ของรอบก่อน (Windows: ลบไม่ได้ถ้ายังเล่นอยู่ — ข้าม)
                try:
                    old.unlink()
                except OSError:
                    pass
            write_preview(samples, rate, target)
            self.audio_player = play_audio(target)
            view.start_playhead(seconds, self.audio_player)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot play clip", str(exc))

    def _review_audio_path(self):
        _key, row = self._selected_review()
        if row is None:
            raise ValueError("Select a clip first")
        parent = Path(self.review_summary.get()).expanduser().resolve().parent
        path = (parent / str(row["File"])).resolve()
        if not path.is_relative_to(parent):
            raise ValueError("Clip path is outside the output folder")
        return path

    def review_play(self):
        try:
            self.review_stop()
            path = self._review_audio_path()
            if not path.is_file():
                raise FileNotFoundError(path)
            if self.review_view.compressed:       # เฉพาะช่วงที่ BirdNET ได้ยิน
                self.play_from(path, 0.0, self.review_view)
                return
            self.audio_player = play_audio(path)
            if self._selected_review()[0] is not None:
                self.review_view.start_playhead(0.0, self.audio_player)
        except Exception as exc:
            messagebox.showerror("Cannot play clip", str(exc))

    def review_stop(self):
        if self.audio_player is not None and self.audio_player.poll() is None:
            self.audio_player.terminate()
        self.audio_player = None
        if hasattr(self, "review_view"):
            self.review_view.stop_playhead()

    def review_boxes_picture(self):
        """ภาพ .png ของคลิปนี้ที่มีกรอบเสียงของ BirdNET"""
        try:
            png = self._review_audio_path().with_suffix(".png")
        except ValueError as exc:
            messagebox.showwarning("No clip", str(exc))
            return
        if png.is_file():
            open_path(png)
        else:
            messagebox.showinfo("No picture", "This clip has no spectrogram picture yet")

    def review_checklist(self):
        """เปิด eBird checklist ที่ไฟล์เสียงของคลิปนี้ใช้"""
        _key, row = self._selected_review()
        if row is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        found = field_log.for_recording(manual_clips.recording_start_of(row))
        if not found:
            messagebox.showinfo("No checklist", "This recording has no eBird checklist yet: add a checklist link or Log "
                                "folder in Settings, or choose one in Match recordings…")
            return
        webbrowser.open(found[0]["url"] or f"https://ebird.org/checklist/{found[0]['id']}")

    def review_xeno_canto(self):
        _key, row = self._selected_review()
        if row is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        # ใช้ชื่อในช่อง Scientific name (ถ้าผู้ตรวจแก้ชนิด จะเทียบกับชนิดที่แก้)
        scientific = self.review_scientific.get().strip() or str(row.get("Species (scientific)") or "")
        url = xeno_canto_url(scientific)
        if url is None:
            messagebox.showinfo("No scientific name",
                                "Enter the scientific name first, e.g. Phylloscopus inornatus")
            return
        key = self.xc_key.get().strip()
        if not key:
            webbrowser.open(url)
            return
        common = self.review_common.get().strip() or str(row.get("Species (common)") or "")

        def work():   # xeno-canto อาจใช้สกุลอื่น (เช่น Pardaliparus): ถามหน้าที่ถูกจาก API ก่อน
            try:
                page = xeno_canto.species_page(scientific, key, common) or url
            except xeno_canto.XenoCantoError:
                page = url
            webbrowser.open(page)
        threading.Thread(target=work, daemon=True).start()

    def review_second_opinion(self):
        key, row = self._selected_review()
        if row is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        if self._second_opinion_busy:
            return
        if not self.xc_key.get().strip():
            messagebox.showinfo("xeno-canto API key needed",
                                "Second opinion compares the clip with xeno-canto recordings. "
                                "Paste your own API key (free, from xeno-canto.org/account) in the Settings tab.")
            return
        if self.xc_key.get().strip() != load_settings().get("xeno_canto_key", ""):
            self._save_xc_key()               # subprocess อ่าน key จากไฟล์ตั้งค่า ไม่ส่งผ่าน command line
        self._second_opinion_busy = True
        self.second_opinion_btn.configure(state="disabled")
        self.second_opinion.set("Second opinion: starting … (first run for a species downloads references)")
        cmd = self._launch() + ["--second-opinion", str(Path(self.review_summary.get()).expanduser()), str(key)]

        second_opinion_q = self._second_opinion_q

        def work():          # ไม่ถือ self: ตัวแปรของ Tk ต้องถูกเก็บกวาดใน thread หลัก
            try:
                proc = platform_tools.start_child(cmd)
                for line in proc.stdout:
                    second_opinion_q.put(line.rstrip())
                proc.wait()
            except OSError as exc:
                second_opinion_q.put(f"ERROR {exc}")
            second_opinion_q.put(None)
        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._second_opinion_poll)

    def _second_opinion_poll(self):
        try:
            while True:
                line = self._second_opinion_q.get_nowait()
                if line is None:
                    self._second_opinion_busy = False
                    self.second_opinion_btn.configure(state="normal")
                    return
                if line.startswith("RESULT "):
                    result = json.loads(line[7:])
                    _key, row = self._selected_review()
                    # ผู้ใช้อาจเลือกคลิปอื่นระหว่างรอ: แสดงเฉพาะเมื่อยังเป็นคลิปเดิม (ผลถูกเก็บไว้แล้ว)
                    same = row is not None and str(row.get("Generated sha256") or result["clip_sha"]) == result["clip_sha"]
                    self.second_opinion.set(format_second_opinion(result) if same else "")
                elif line.startswith("ERROR "):
                    self.second_opinion.set("Second opinion failed: " + line[6:])
                elif line and not line.startswith(("INFO", "WARNING")) and "Model loaded" not in line:
                    self.second_opinion.set("Second opinion: " + line.strip())
        except queue.Empty:
            pass
        self.root.after(200, self._second_opinion_poll)

    def review_lookup(self, kind):
        """เปิดหน้าชนิดนี้: eBird (รูป เสียง แผนที่) หรือ Macaulay Library เฉพาะรูป/เสียง"""
        _key, row = self._selected_review()
        if row is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        scientific = self.review_scientific.get().strip() or str(row.get("Species (scientific)") or "")
        common = self.review_common.get().strip() or str(row.get("Species (common)") or "")
        if xeno_canto_url(scientific) is None:
            messagebox.showinfo("No scientific name",
                                "Enter the scientific name first, e.g. Phylloscopus inornatus")
            return
        region = self._ebird_region
        self.status.set("Looking up the eBird species …")
        lookup_q = self._lookup_q

        def work():   # ครั้งแรกต้องโหลดระบบชื่อ eBird (~15 วินาที) หลังจากนั้นเก็บไว้ในเครื่อง
            try:
                entry = ebird.species(scientific, common)
            except ebird.EBirdError as exc:
                lookup_q.put(("lookup-error", str(exc)))
                return
            if entry is None:
                lookup_q.put(("lookup-error", f"{scientific} is not in the eBird taxonomy"))
                return
            webbrowser.open(species_links(entry["code"], region)[kind])
            lookup_q.put(("lookup-done", entry["common"]))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._lookup_poll)

    def _lookup_poll(self):
        try:
            kind, text = self._lookup_q.get_nowait()
        except queue.Empty:
            self.root.after(200, self._lookup_poll)
            return
        if kind == "lookup-error":
            self.status.set("")
            messagebox.showinfo("eBird lookup", text)
        else:
            self.status.set(f"Opened {text} in your browser")

    def _save_ebird_key(self):
        try:
            save_settings({"ebird_key": self.ebird_key.get().strip()})
            self.status.set("eBird key saved")
        except OSError as exc:
            messagebox.showerror("Cannot save key", str(exc))

    def _save_xc_key(self):
        try:
            save_settings({"xeno_canto_key": self.xc_key.get().strip()})
            self.status.set("xeno-canto key saved")
        except OSError as exc:
            messagebox.showerror("Cannot save key", str(exc))

    def review_references(self):
        """หน้าต่างเสียงตัวอย่างจาก xeno-canto ของชนิดในช่อง Scientific name: ฟังเทียบในแอป"""
        _key, row = self._selected_review()
        if row is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        scientific = (self.review_scientific.get().strip() or str(row.get("Species (scientific)") or "")).strip()
        if xeno_canto_url(scientific) is None:
            messagebox.showinfo("No scientific name",
                                "Enter the scientific name first, e.g. Phylloscopus inornatus")
            return
        if not self.xc_key.get().strip():
            if messagebox.askyesno("xeno-canto API key needed",
                                   "Reference sounds need your own xeno-canto API key (free, from "
                                   "xeno-canto.org/account). Paste it in the Settings tab.\n\n"
                                   "Open the species page in your browser instead?"):
                webbrowser.open(xeno_canto_url(scientific))
            return

        def number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        lat, lon = number(row.get("Latitude")), number(row.get("Longitude"))
        common = self.review_common.get().strip() or str(row.get("Species (common)") or "")
        ReferenceWindow(self, scientific, common, lat if lat == lat else None, lon if lon == lon else None)

    def review_save(self, status):
        key, row = self._selected_review()
        if key is None:
            messagebox.showwarning("No clip", "Select a clip first")
            return
        order = list(self.review_tree.get_children())   # ลำดับก่อนบันทึก (ตัวกรองอาจซ่อนคลิปนี้หลังบันทึก)
        try:
            rating = int(self.review_rating.get()) if self.review_rating.get() else None
            chosen = self.review_sound_type.get()
            ready = save_review(Path(self.review_summary.get()), int(key), status=status,
                                common=self.review_common.get(), scientific=self.review_scientific.get(),
                                rating=rating, notes=self.review_notes.get(),
                                expected_file=str(row["File"]), sound_type=chosen,
                                sound_type_guess=sound_type.guess_text(self._sound_guess) or None,
                                background=background_species.join(
                                    background_species.split(self.review_background.get())))
            if status == "Approved" and sound_type.reviewed_kind(chosen):
                self._retrain_sound_type()
            self.review_load()
            if not (self.auto_next.get() and self._select_next_pending(order, key)):
                if self.review_tree.exists(key):
                    self.review_tree.selection_set(key)
                    self.review_tree.see(key)
                    self._review_select()
            self.review_tree.focus_set()              # ปุ่มลัดใช้ต่อได้ทันที
            self._log(f"Review {status}: {ready or key}")
            if self._progress_window is not None and self._progress_window.win.winfo_exists():
                self._progress_window.refresh()
        except Exception as exc:
            messagebox.showerror("Could not save review", str(exc))

    def _select_next_pending(self, order, key):
        """เลือกคลิปถัดไป (ตามลำดับในตารางก่อนบันทึก วนกลับต้นตาราง) ที่ยังไม่ได้ตรวจ -> True ถ้าเจอ"""
        position = order.index(key) if key in order else -1
        for iid in order[position + 1:] + order[:max(position, 0)]:
            row = self.review_rows.get(iid)
            if row is not None and self.review_tree.exists(iid) and \
                    str(row.get("Review status") or "Pending") == "Pending":
                self.review_tree.selection_set(iid)
                self.review_tree.focus(iid)
                self.review_tree.see(iid)
                self._review_select()
                return True
        self.status.set("No pending clips left in this list ✓")
        return False

    def _review_key(self, event):
        """ปุ่มลัดในตารางคลิป/ตัวดูเสียง (ใช้ได้ทุก keyboard layout เช่น ไทย): 1–5 คุณภาพ, Enter/A Approve,
        R Reject, U Not sure, P Keep pending, S/C/F ประเภทเสียง, Space เล่น/หยุด, ↑/↓ คลิปก่อนหน้า/ถัดไป"""
        key = physical_key(event)
        # Ctrl (และ Cmd บน mac) + ปุ่ม = ไม่ใช่ปุ่มลัด — Windows ไม่ดู Mod1 เพราะ NumLock ตั้งบิตนั้น
        if key is None or event.state & (0x0C if sys.platform == "darwin" else 0x04):
            return None
        if key in ("up", "down"):        # ใช้ได้ทั้งในตารางและบนภาพเสียง (คลิกภาพเพื่อเล่นแล้วกดต่อได้)
            self._review_step(-1 if key == "up" else 1)
            return "break"
        if self._selected_review()[0] is None:
            return None
        if key in ("1", "2", "3", "4", "5"):
            self.review_rating.set(key)
            self.status.set(f"Audio quality {key} — Enter approves")
        elif key in ("return", "a"):
            self.review_save("Approved")
        elif key == "r":
            self.review_save("Rejected")
        elif key == "u":
            self.review_save("Unsure")
        elif key == "p":
            self.review_save("Pending")
        elif key in ("s", "c", "f"):
            self.review_sound_type.set({"s": "Song", "c": "Call", "f": "Flight call"}[key])
        elif key == "space":
            if self.audio_player is not None and self.audio_player.poll() is None:
                self.review_stop()
            else:
                self.review_play()
        else:
            return None
        return "break"

    def _retrain_sound_type(self):
        """ฝึกตัวเดาประเภทเสียงใหม่เบื้องหลัง ให้รวมคลิปที่เพิ่งเลือก Sound type (ถ้ากำลังฝึกอยู่ รอบหน้าค่อยรวม)"""
        busy = self._sound_training
        if busy[0]:
            return
        busy[0] = True
        root, current = self.out.get().strip() or DEF_OUT, Path(self.review_summary.get()).expanduser()

        def work():
            try:
                summaries = {Path(info["path"]).resolve() for info in all_results(root)}
                if current.is_file():
                    summaries.add(current.resolve())
                sound_type.train(sorted(summaries), log=lambda _text: None)
            except Exception as exc:  # noqa: BLE001  ฝึกไม่ได้ไม่ควรรบกวนการตรวจ
                print(f"sound type training failed: {exc}", flush=True)
            finally:
                busy[0] = False
        threading.Thread(target=work, daemon=True).start()

    # ---------- Analyze pickers ----------
    def _raw_folder(self):
        """Output อยู่ที่ <วัน>/Sounds/Done: เปิดหน้าเลือกเสียงที่ <วัน>/Sounds/Raw ของวันนั้น"""
        raw = Path(self.out.get().strip() or ".").expanduser().parent / "Raw"
        return str(raw) if raw.is_dir() else None

    def browse_file(self):
        p = filedialog.askopenfilename(
            title="Choose an audio file", initialdir=self._raw_folder(),
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aif *.aiff"), ("All", "*.*")])
        if p:
            self.audio.set(p)
        return p

    def browse_folder(self):
        p = filedialog.askdirectory(title="Choose a folder of recordings", initialdir=self._raw_folder())
        if p:
            self.audio.set(p)
        return p

    def browse_output(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.out.set(p)

    # ---------- Merge list ops ----------
    def merge_add(self):
        ps = filedialog.askopenfilenames(
            title="Choose clips of the same individual",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aif *.aiff"), ("All", "*.*")])
        for p in ps:
            self.merge_list.insert("end", p)
        self._merge_changed()

    def merge_remove(self):
        for i in reversed(self.merge_list.curselection()):
            self.merge_list.delete(i)
        self._merge_changed()

    def merge_clear(self):
        self.merge_list.delete(0, "end")
        self._merge_changed()

    def merge_move(self, delta):
        sel = self.merge_list.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if j < 0 or j >= self.merge_list.size():
            return
        text = self.merge_list.get(i)
        self.merge_list.delete(i)
        self.merge_list.insert(j, text)
        self.merge_list.selection_set(j)
        self._merge_changed()

    def _merge_changed(self):
        """รายการคลิปเปลี่ยน: ชื่อและที่เก็บตามคลิป — ชื่อที่พิมพ์เองไม่ถูกเขียนทับ"""
        suggested = merge_clips.suggested_name(list(self.merge_list.get(0, "end")))
        if self.merge_name.get().strip() in ("", self._merge_suggested):
            self.merge_name.set(suggested)
        self._merge_suggested = suggested
        self._merge_show_place()

    def _merge_target_folder(self):
        if self.merge_place.get() == "other":
            return Path(self.merge_folder.get()) if self.merge_folder.get() else None
        return Path(self.merge_list.get(0)).parent if self.merge_list.size() else None

    def _merge_show_place(self):
        folder = self._merge_target_folder()
        self.merge_where.set(str(folder) if folder else "Add clips to see where the file goes")

    def _merge_other(self):
        if self.merge_folder.get():
            self._merge_show_place()
        else:
            self.merge_choose_folder()

    def merge_choose_folder(self):
        folder = filedialog.askdirectory(title="Save the merged file in",
                                         initialdir=str(self._merge_target_folder() or Path.home() / "Desktop"))
        if folder:
            self.merge_folder.set(folder)
            self.merge_place.set("other")
        elif not self.merge_folder.get():
            self.merge_place.set("same")
        self._merge_show_place()

    # ---------- run helpers ----------
    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _launch(self):
        # frozen (.exe): เรียกตัว exe เองซ้ำ; dev/bundle: python + สคริปต์
        return [sys.executable] if getattr(sys, "frozen", False) else [PYTHON, str(SCRIPT)]

    def _start(self, cmd, btn, title, is_analyze=True, on_done=None, background=False):
        """เริ่มงาน หรือต่อคิวถ้ามีงานรันอยู่; on_done(code, บรรทัด MANUAL/ERROR) เรียกเมื่องานจบ
        background = งานที่แอปสั่งเอง (ตรวจอัตโนมัติ, วาดภาพใหม่): งานที่ผู้ใช้สั่งแซงคิวได้ และถ้างานเบื้องหลัง
        กำลังรัน จะถูกหยุดอย่างปลอดภัยแล้วต่อท้ายคิวเพื่อทำต่อทีหลัง"""
        if not SCRIPT.exists():
            messagebox.showerror("Script not found", str(SCRIPT))
            return
        job = {"title": title, "cmd": cmd, "button": btn, "is_analyze": is_analyze,
               "added": datetime.now(), "on_done": on_done, "lines": [], "background": background}
        # งานเบื้องหลังไม่แซงงานของผู้ใช้ที่รอเริ่มอยู่ (เช่น ช่วงพักระหว่างงานวิเคราะห์สองไฟล์)
        if self.job_running or (background and self.job_queue):
            position = len(self.job_queue) if background else next(
                (i for i, queued in enumerate(self.job_queue) if queued.get("background")), len(self.job_queue))
            self.job_queue.insert(position, job)
            if not background:
                self.status.set(f"Queued: {title} ({len(self.job_queue)} waiting)")
            self._yield_to_user_jobs()
            self._update_job_controls()
            if not self.job_running:              # กันคิวค้าง: ปกติงานที่จบไปแล้วตั้งเวลาเริ่มงานถัดไปไว้แล้ว
                self.root.after(1600, self._start_next_job)
            return
        self._launch_job(job)

    def _yield_to_user_jobs(self):
        """งานเบื้องหลังกำลังรันแต่มีงานของผู้ใช้รออยู่: ขอให้หยุดอย่างปลอดภัย แล้วต่อท้ายคิวทำต่อทีหลัง"""
        current = self.current_job
        if (self.job_running and current and current.get("background") and not current.get("stopping")
                and any(not queued.get("background") for queued in self.job_queue)):
            current["preempted"] = current["stopping"] = True
            if self.proc is not None and self.proc.poll() is None:
                request_stop(self.proc)

    def _launch_job(self, job):
        self.log.delete("1.0", "end")
        self._log("command: " + " ".join(f'"{a}"' if " " in a else a for a in job["cmd"]))
        self._active_btn = job["button"]
        self._active_is_analyze = job["is_analyze"]
        self._clip_count = None
        self._generated_summaries = []
        self._already_done = False
        job.update(started=time.monotonic(), total=None, index=0, file="", stopping=False)
        self.current_job = job
        self.job_running = True
        self.progress.start(12)
        self._update_job_controls()
        threading.Thread(target=self._worker, args=(job["cmd"],), daemon=True).start()
        self.root.after(100, self._poll)

    def _job_progress_text(self):
        job = self.current_job
        if job is None:
            return ""
        elapsed = int(time.monotonic() - job["started"])
        text = ("Analyzing " if job["is_analyze"] else "") + (job["file"] or job["title"])
        if job["total"] and job["total"] > 1:
            text += f" (file {job['index']}/{job['total']})"
        text += f"  ·  {elapsed // 60:02d}:{elapsed % 60:02d}"
        if job["stopping"]:
            text = "Stopping … " + text
        if self.job_queue:
            text += f"  ·  {len(self.job_queue)} queued"
        return text

    def _update_job_controls(self):
        self.stop_btn.configure(state="normal" if self.job_running else "disabled")
        self.jobs_btn.configure(text=f"Jobs ({len(self.job_queue)} queued)…" if self.job_queue else "Jobs…")
        if self._jobs_window is not None:
            self._jobs_window.refresh()

    SHUTDOWN_IDLE_S = 15                       # ไม่มีงานติดต่อกันเท่านี้ = เสร็จจริง (คิวถัดไปเริ่มใน ~2 วิ)
    SHUTDOWN_COUNTDOWN_S = 60

    def _shutdown_toggled(self):
        if self.shutdown_after.get():
            self._shutdown_idle = 0
            self.status.set("The computer will shut down when all jobs are done")
            self.root.after(5000, self._shutdown_watch)
        else:
            self.status.set("Shut down cancelled")

    def _shutdown_watch(self):
        if not self.shutdown_after.get() or self._closing:
            return
        busy = self.job_running or bool(self.job_queue)
        self._shutdown_idle = 0 if busy else self._shutdown_idle + 5
        if self._shutdown_idle >= self.SHUTDOWN_IDLE_S:
            self._shutdown_countdown()
            return
        self.root.after(5000, self._shutdown_watch)

    def _shutdown_countdown(self):
        """งานเสร็จหมดแล้ว: นับถอยหลังก่อนปิดเครื่อง (Cancel = ไม่ปิด)"""
        win = self._shutdown_window = tk.Toplevel(self.root)
        win.title("Shut down")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        body = ttk.Frame(win, padding=16)
        body.pack(fill="both", expand=True)
        text = tk.StringVar()
        ttk.Label(body, textvariable=text, font=("Helvetica", 13)).pack(anchor="w")
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(14, 0))
        left = [self.SHUTDOWN_COUNTDOWN_S]

        def cancel():
            left[0] = -1
            self.shutdown_after.set(False)
            self.status.set("Shut down cancelled")
            win.destroy()

        def now():
            left[0] = 0
            tick()

        def tick():
            if left[0] < 0 or not win.winfo_exists():
                return
            if self.job_running or self.job_queue:          # มีงานใหม่เข้ามา: รอต่อ
                win.destroy()
                self._shutdown_idle = 0
                self.root.after(5000, self._shutdown_watch)
                return
            if left[0] <= 0:
                left[0] = -1
                win.destroy()
                self.status.set("Shutting down …")
                ok, message = platform_tools.shut_down_computer()
                if not ok:
                    self.shutdown_after.set(False)
                    messagebox.showerror("Could not shut down", (message or "The system refused.") +
                                         ("\n\nOn macOS, allow BirdNET eBird to control System Events in System "
                                          "Settings → Privacy & Security → Automation." if platform_tools.IS_MAC else ""))
                return
            text.set(f"All jobs are done. The computer shuts down in {left[0]} s.")
            left[0] -= 1
            win.after(1000, tick)

        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
        ttk.Button(buttons, text="Shut down now", command=now).pack(side="right", padx=(0, 8))
        win.protocol("WM_DELETE_WINDOW", cancel)
        tick()

    def _start_next_job(self):
        if self.job_queue and not self.job_running and not self._closing:
            # งานของผู้ใช้ก่อนงานเบื้องหลังเสมอ
            index = next((i for i, queued in enumerate(self.job_queue) if not queued.get("background")), 0)
            self._launch_job(self.job_queue.pop(index))
        else:
            self._yield_to_user_jobs()

    def stop_job(self):
        job, proc = self.current_job, self.proc
        if job is None or proc is None or proc.poll() is not None:
            return
        if not messagebox.askokcancel("Stop this job?", f"Stop “{job['title']}”?\n"
                                      "Finished files are kept; run it again later to continue."):
            return
        job["stopping"] = True
        request_stop(proc)                # CLI เปลี่ยนเป็น KeyboardInterrupt แล้ว rollback เอง
        self.status.set(self._job_progress_text())

    def show_jobs(self):
        if self._jobs_window is not None and self._jobs_window.win.winfo_exists():
            self._jobs_window.win.lift()
            return
        self._jobs_window = JobsWindow(self)

    def run(self, confirmed_batch=False, button=None, fill_missing=False):
        audio = self.audio.get().strip().strip('"')
        if not audio:
            messagebox.showwarning("No audio selected", "Choose an audio file or folder first.")
            return
        if self.date.get().strip():
            try:
                parse_date_entry(self.date.get())
            except ValueError:
                messagebox.showwarning("Invalid date", f"Enter the date as {DATE_HINT}")
                return
        if self.start_time.get().strip():
            value = self.start_time.get().strip()
            try:
                parse_time_entry(value)
            except ValueError:
                messagebox.showwarning("Invalid time", f"Enter the recording start time as {TIME_HINT}")
                return
        same_date_for_all = False
        if self.date.get().strip() and Path(audio).is_dir() and not fill_missing:
            audio_files = collect_audio(audio)
            if len(audio_files) > 1:
                if not confirmed_batch and not messagebox.askokcancel(
                        "One date for every file?",
                        f"The entered date will be used for all {len(audio_files)} audio files. "
                        "Confirm they were recorded on the same date. Otherwise, process each date separately."):
                    return
                same_date_for_all = True
        if self.continuous.get() and not messagebox.askokcancel(
                "Continuous recording",
                "Keep one continuous span from the first to last detection of each species? "
                "Use this only when you have checked the recording is of the same individual."):
            return
        try:
            save_settings({OUTPUT_FOLDERS: remembered(self.out.get().strip() or DEF_OUT)})
        except OSError:
            pass
        cmd = self._launch() + [audio, "-o", self.out.get().strip(),
               "--model", self.model.get(),
               "--min-conf", str(self.min_conf.get()),
               "--overlap", str(self.overlap.get()),
               "--occurrence-gap", str(self.gap.get()),
               "--lead", str(self.lead.get()), "--tail", str(self.tail.get()),
               "--target-dbfs", str(self.dbfs.get()),
               "--mono" if self.mono.get() else "--no-mono",
               "--spectrogram" if self.spec.get() else "--no-spectrogram",
               "--alt-species" if self.alt.get() else "--no-alt-species",
               "--unknown" if self.unknown.get() else "--no-unknown",
               "--second-look" if self.second_look.get() else "--no-second-look"]
        if self.keep_out_of_range.get():
            cmd += ["--out-of-range-min-conf", str(self.out_of_range_min_conf.get())]
        if fill_missing:
            cmd += ["--fill-missing-metadata"]
        if self.force.get():
            if not messagebox.askokcancel("Redo analysis", "Redo will replace generated clips for this source "
                                            "and reset their review status to Pending. Continue?"):
                return
            cmd += ["--force"]
        if self.continuous.get():
            cmd += ["--keep-continuous"]
        if self.filetime.get():
            cmd += ["--use-filetime"]
        if str(self.coords.get()).strip():
            cmd += ["--coords", str(self.coords.get()).strip()]
        if self.date.get().strip():
            cmd += ["--date", self.date.get().strip()]
        if same_date_for_all:
            cmd += ["--same-date-for-all"]
        if self.start_time.get().strip():
            cmd += ["--start-time", self.start_time.get().strip()]
        if self.place.get().strip():
            cmd += ["--place", self.place.get().strip()]
        habitats = [key for key, var in self.habitat_vars.items() if var.get()]
        if habitats:
            cmd += ["--habitat", ",".join(habitats)]
        name = Path(audio).name + (" (folder)" if Path(audio).is_dir() else "")
        self._start(cmd, button or self.run_btn, f"Analyze {name}", is_analyze=True)

    def merge_run(self):
        files = list(self.merge_list.get(0, "end"))
        if len(files) < 2:
            messagebox.showwarning("Need clips", "Add at least 2 clips to merge")
            return
        folder = self._merge_target_folder()
        if folder is None:
            messagebox.showwarning("No folder", "Choose a folder for the merged file")
            return
        names = merge_clips.species_in(files)
        if len(names) > 1 and not messagebox.askokcancel(
                "Different species", f"These clips are from {len(names)} species: {', '.join(names)}.\n"
                "Only merge recordings of the same individual bird. Merge anyway?"):
            return
        out = merge_clips.free_path(folder, self.merge_name.get())

        def merged(code, _lines):
            if code == 0:
                self.merge_where.set(f"Saved: {out}")
                self.status.set(f"Merged: {out.name}")

        cmd = self._launch() + ["--group"] + files + ["-o", str(out)]
        self._start(cmd, self.merge_btn, f"Merge {len(files)} clips → {out.name}", is_analyze=False, on_done=merged)

    def _worker(self, cmd):
        try:
            self.proc = platform_tools.start_child(cmd)   # Windows: ไม่มี console เด้ง, หยุดผ่าน stdin
            for line in self.proc.stdout:
                self.q.put(line.rstrip())
            code = self.proc.wait()
            self.q.put(("__DONE__", code))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("__DONE__", f"error: {exc}"))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    self._finish(item[1])
                    return
                self._log(item)
                if isinstance(item, str) and self.current_job is not None:
                    job = self.current_job        # ความคืบหน้าจากข้อความของ CLI
                    if item.startswith(("MANUAL ", "ERROR ", "SLOW ")):
                        job["lines"].append(item)
                    total = re.match(r"Processing (\d+) file", item)
                    if total:
                        job["total"] = int(total.group(1))
                    current = re.match(r"\s*=== (.+?) ===", item)
                    if current:
                        job["index"] += 1
                        job["file"] = current.group(1)
                if isinstance(item, str):
                    match = re.search(r"\bDone: (\d+) clips\b", item)
                    if match:
                        self._clip_count = int(match.group(1))
                    if " -> " in item and "summary.xlsx" in item:
                        candidate = Path(item.split(" -> ", 1)[1].strip())
                        if candidate.is_file() and candidate not in self._generated_summaries:
                            self._generated_summaries.append(candidate)
                        if "already up to date" in item:
                            self._already_done = True
        except queue.Empty:
            pass
        if self.current_job is not None:
            self.status.set(self._job_progress_text())
        self.root.after(100, self._poll)

    def _finish(self, code):
        self.progress.stop()
        try:                                      # ปุ่มของหน้าต่างที่ปิดไปแล้วไม่มีให้คืนสถานะ
            getattr(self, "_active_btn", self.run_btn)["state"] = "normal"
        except (tk.TclError, TypeError):
            pass
        self.quick_file_btn.configure(state="normal")
        self.quick_folder_btn.configure(state="normal")
        self.quick_start_btn.configure(state="normal")
        self.proc = None
        self.job_running = False
        job, self.current_job = self.current_job, None
        if self._closing:
            return
        if job is not None and job.get("preempted"):      # หลีกทางให้งานของผู้ใช้: ทำต่อทีหลัง
            self.job_queue.append(dict(job, preempted=False, stopping=False, lines=[]))
            self.root.after(300, self._start_next_job)
            self._update_job_controls()
            return
        # งานถัดไปในคิว (เว้นช่วงให้เห็นผลของงานที่เพิ่งจบ)
        if self.job_queue:
            self.root.after(1500, self._start_next_job)
        self._update_job_controls()
        if job is not None and job.get("on_done"):
            self.status.set("Done ✓" if code == 0 else f"Failed: {job['title']}")
            job["on_done"](code, job["lines"])
            return
        if job is not None and job.get("stopping"):
            self.status.set(f"Stopped: {job['title']} — finished files are kept; run it again to continue")
            self._log("\n=== Stopped by you ===")
            return
        if code == 0:
            self.status.set("Done ✓")
            self._log("\n=== Done ===")
            if self._active_is_analyze:
                summaries = sorted(self._generated_summaries)    # โฟลเดอร์หลายไฟล์ = ผลชุดละไฟล์เสียง: เปิดชุดแรก
                if summaries:
                    self.review_summary.set(str(summaries[0]))
                    self.review_load()
                    self.refresh_results()                        # ผลชุดใหม่ทุกชุดขึ้นในรายการ Result เลย
                    if self.review_rows:
                        self.nb.select(self.review_tab)
                        more = (f" — {len(summaries)} results, choose the others in the Result list"
                                if len(summaries) > 1 else "")
                        self.status.set("Already analyzed — showing the existing results (tick “force redo” to analyze again)"
                                        if self._already_done and not self._clip_count
                                        else f"Done{more} — select a clip to listen and review ✓")
                elif self._clip_count == 0:
                    self.status.set("Done — no bird sounds met the current settings")
        else:
            self.status.set("Analysis failed — see the details below")
            self._log(f"\n!! finished with exit {code}")
            self._show_log()
            if code == 2:
                messagebox.showwarning("Recording details incomplete",
                                       "Enter the real recording date and location, then try again.")


class JobsWindow:
    """งานที่กำลังรันและงานที่รอคิว: หยุดงาน / เอางานออกจากคิว"""

    def __init__(self, app):
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.title("Jobs")
        win.geometry("640x280")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(body, columns=("state", "job", "time"), show="headings", height=8)
        for key, label, width in (("state", "State", 110), ("job", "Job", 380), ("time", "Time", 110)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "job"))
        self.tree.pack(fill="both", expand=True)
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="■ Stop running job", command=app.stop_job).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Remove from queue", command=self.remove).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Stop all", command=self.stop_all).pack(side="left")
        ttk.Button(actions, text="Close", command=win.destroy).pack(side="right")
        self.refresh()
        self._tick()

    def refresh(self):
        if not self.win.winfo_exists():
            return
        selected = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        job = self.app.current_job
        if job is not None:
            elapsed = int(time.monotonic() - job["started"])
            state = "Stopping" if job["stopping"] else "Running"
            detail = job["title"] + (f"  —  {job['file']} ({job['index']}/{job['total']})"
                                     if job["file"] and job["total"] and job["total"] > 1 else "")
            self.tree.insert("", "end", iid="running", values=(state, detail,
                                                               f"{elapsed // 60:02d}:{elapsed % 60:02d}"))
        for number, waiting in enumerate(self.app.job_queue, start=1):
            iid = f"queued{id(waiting)}"
            self.tree.insert("", "end", iid=iid, values=(f"Queued #{number}", waiting["title"],
                                                         "added " + waiting["added"].strftime("%H:%M")))
            if iid in selected:
                self.tree.selection_add(iid)
        if job is None and not self.app.job_queue:
            self.tree.insert("", "end", iid="idle", values=("—", "No jobs running or waiting", ""))

    def _tick(self):                       # เวลาที่ผ่านไปของงานที่รันอยู่
        if self.win.winfo_exists():
            self.refresh()
            self.win.after(1000, self._tick)

    def remove(self):
        chosen = set(self.tree.selection())
        self.app.job_queue[:] = [job for job in self.app.job_queue if f"queued{id(job)}" not in chosen]
        self.app._update_job_controls()

    def stop_all(self):
        if not self.app.job_queue and self.app.current_job is None:
            return
        self.app.job_queue.clear()
        self.app._update_job_controls()
        self.app.stop_job()


class ReferenceWindow:
    """เสียงตัวอย่างจาก xeno-canto ของชนิดหนึ่ง เล่นสลับกับคลิปที่กำลังตรวจได้"""
    TYPES = ("Any", "song", "call", "flight call", "alarm call")

    def __init__(self, app, scientific, common, lat, lon):
        self.app, self.scientific, self.common, self.lat, self.lon = app, scientific, common, lat, lon
        self.recordings = {}
        self.q = queue.Queue()
        self._search_id = 0
        win = self.win = tk.Toplevel(app.root)
        win.title(f"Reference sounds — {common} ({scientific})" if common else f"Reference sounds — {scientific}")
        win.geometry("940x430")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Sound type").pack(side="left")
        self.sound_type = tk.StringVar(value="Any")
        chooser = ttk.Combobox(top, textvariable=self.sound_type, values=self.TYPES, state="readonly", width=12)
        chooser.pack(side="left", padx=6)
        chooser.bind("<<ComboboxSelected>>", lambda _event: self.search())
        self.state = tk.StringVar()
        ttk.Label(top, textvariable=self.state, foreground="#808080").pack(side="left", padx=10)

        columns = ("id", "type", "q", "length", "distance", "where", "recordist", "license")
        self.tree = ttk.Treeview(body, columns=columns, show="headings", height=12)
        for key, label, width in (("id", "XC", 85), ("type", "Type", 140), ("q", "Q", 30),
                                  ("length", "Length", 60), ("distance", "Distance", 80),
                                  ("where", "Location", 230), ("recordist", "Recordist", 130),
                                  ("license", "License", 110)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "where"))
        self.tree.pack(fill="both", expand=True, pady=8)
        self.tree.bind("<Double-1>", lambda _event: self.play())

        actions = ttk.Frame(body)
        actions.pack(fill="x")
        ttk.Button(actions, text="▶ Play reference", command=self.play).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="▶ Play my clip", command=app.review_play).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="■ Stop", command=app.review_stop).pack(side="left", padx=(0, 8))
        playback.control(actions).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Open on xeno-canto ↗", command=self.open_page).pack(side="left")
        ttk.Label(body, text="Best quality and nearest first. Recordings © their recordists, "
                  "shared on xeno-canto.org under the license shown.", foreground="#888").pack(anchor="w", pady=(6, 0))
        self.search()
        self.win.after(100, self._poll)

    def search(self):
        self._search_id += 1
        search_id, sound_type = self._search_id, self.sound_type.get()
        self.state.set("Searching xeno-canto…")
        self.tree.delete(*self.tree.get_children())
        # งานเบื้องหลังไม่ถือหน้าต่าง (ปิดหน้าต่างระหว่างค้นได้): ส่งค่าที่ต้องใช้กับคิวไปแทน
        scientific, common, lat, lon, results = self.scientific, self.common, self.lat, self.lon, self.q
        key = self.app.xc_key.get().strip()

        def work():
            try:
                found, total = xeno_canto.search(scientific, key, "" if sound_type == "Any" else sound_type,
                                                 lat, lon, common=common)
                results.put(("results", search_id, found, total))
            except xeno_canto.XenoCantoError as exc:
                results.put(("error", search_id, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _selected(self):
        selected = self.tree.selection()
        return self.recordings.get(selected[0]) if selected else None

    def play(self):
        recording = self._selected()
        if recording is None:
            self.state.set("Select a recording first")
            return
        self.state.set(f"Downloading {recording['id']}…")
        results = self.q

        def work():
            try:
                results.put(("play", None, xeno_canto.download(recording)))
            except xeno_canto.XenoCantoError as exc:
                results.put(("error", None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def open_page(self):
        recording = self._selected()
        if recording:
            webbrowser.open(recording["page"])
        else:
            self.app.review_xeno_canto()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, search_id, *payload = self.q.get_nowait()
                if search_id is not None and search_id != self._search_id:
                    continue                          # ผลค้นเก่าหลังเปลี่ยนประเภทเสียง
                if kind == "results":
                    self._show(*payload)
                elif kind == "play":
                    self.app.review_stop()
                    try:
                        self.app.audio_player = play_audio(payload[0])
                    except (OSError, RuntimeError) as exc:
                        self.state.set(f"Cannot play: {exc}")
                        continue
                    self.state.set(f"Playing {payload[0].stem}")
                else:
                    self.state.set(payload[0])
        except queue.Empty:
            pass
        self.win.after(100, self._poll)

    def _show(self, found, total):
        self.recordings = {}
        for recording in found:
            self.recordings[recording["id"]] = recording
            distance = "" if recording["distance_km"] is None else f"{recording['distance_km']:,} km"
            where = ": ".join(part for part in (recording["country"], recording["location"]) if part)
            self.tree.insert("", "end", iid=recording["id"], values=(
                recording["id"], recording["type"], recording["quality"], recording["length"],
                distance, where, recording["recordist"], recording["license"]))
        self.state.set(f"Showing {len(found)} of {total:,} recordings" if found
                       else "No recordings found for this species and sound type")


def main():
    paths.migrate_old_locations()   # ข้อมูลจาก ~/Library (รุ่นก่อน) -> data/
    platform_tools.enable_high_dpi()
    root = tk.Tk()
    tk_threads.install(root)   # Tcl/Tk 9: ห้ามเรียก Tk จากเธรดอื่น แม้แต่ตอนเก็บขยะ
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
