"""หน้าต่าง Recorder flags (Find sounds): flag ทุกจุดที่กดตอนอัด พร้อมสิ่งที่อยู่ใกล้จุดนั้น
ตารางบน = flag แต่ละจุด (รวมไฟล์ที่ยังไม่ได้วิเคราะห์), ตารางล่าง = คลิปของ BirdNET และรูปถ่ายใกล้ flag ที่เลือก
กดฟังรอบจุดนั้น, เพิ่มคลิปเองเมื่อ BirdNET ไม่เจออะไร, เปิดคลิปในแท็บ Review หรือเปิดรูป

ไฟล์เสียงหาจากผลวิเคราะห์ทุกชุดในโฟลเดอร์ผลลัพธ์ + โฟลเดอร์ Sounds ใต้ Birding (จาก Field logs) + โฟลเดอร์ที่เลือกเอง
รูปถ่ายมาจาก Field log ของวันนั้น เวลาในรูปเป็นนาฬิกากล้อง ถ้าไม่ตรงกับเครื่องอัดให้ตั้ง Camera minus recorder
(บันทึกใน settings.json) การอยู่ใกล้กันไม่ได้แปลว่าเป็นนกตัวเดียวกัน
"""

import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import field_log
import manual_clips
import platform_tools
import playback
import recorder_flags
import species_names
from app_settings import load_settings, save_settings
from audio_io import audio_duration, collect_audio
from frequency_compare import TEXT_COLORS, FlowFrame, dark_mode
from manual_clips import format_offset
from result_roots import all_results
from sound_search import collect_sources

OFFSET_SETTING = "camera_minus_recorder_s"
CLIP_LEAD = 5.0                 # Add clip here: เปิดภาพก่อน flag เท่านี้วินาที


def sound_folders(root):
    """<Birding>/<ปี>/<เดือน>/<วัน>/Sounds (ไม่ต้องไล่ไฟล์รูปทั้งหมด) — ไม่ใช่โครงสร้างนี้ = ทั้งโฟลเดอร์"""
    found = sorted(Path(root).glob("*/*/*/Sounds"))
    return found or [Path(root)]


def collect(output, folders):
    """flag ทุกจุดเรียงตามเวลา -> [item] (item: path, sid, number, seconds, flags, start, clock, summary, clips, photos)
    clips = [(ระยะวินาที บวก = คลิปอยู่หลัง flag, แถวใน summary)] — ทำใน thread ได้ (ไม่แตะ Tk)"""
    from field_audio_to_ebird import source_id
    sources, _rows = collect_sources([info["path"] for info in all_results(output)])
    files = {}                                 # sid -> ไฟล์ที่มี flag ในโฟลเดอร์เสียง (ไฟล์ที่ย้ายมาก็เจอ)
    for folder in folders:
        for path in collect_audio(folder):
            if recorder_flags.read_flags(path):
                files[source_id(path)] = path
    items, photos = [], {}
    for sid in set(sources) | set(files):
        entry = sources.get(sid)
        path = files.get(sid) or entry["path"]
        flags = recorder_flags.flags_of(sid, path)
        start = ((entry.get("context") or {}).get("start") if entry else None) or recorder_flags.recording_start(path)
        for number, seconds in enumerate(flags, 1):
            clips = []
            for row in entry["rows"] if entry else ():
                first = manual_clips._number(row.get("Start offset (s)"))
                last = manual_clips._number(row.get("End offset (s)"))
                near = None if first is None else recorder_flags.distance([seconds], first,
                                                                          last if last is not None else first)
                if near:
                    clips.append((near[1], row))
            clock = start + timedelta(seconds=seconds) if start else None
            day = clock.date() if clock else None
            if day not in photos:
                photos[day] = field_log.photos_on(field_log.for_day(day))
            items.append({"path": Path(path), "listed": entry["path"] if entry else None, "sid": sid, "number": number,
                          "seconds": seconds, "flags": flags, "start": start, "clock": clock,
                          "summary": entry["summary"] if entry else None, "photos": photos[day],
                          "clips": sorted(clips, key=lambda item: abs(item[0]))})
    return sorted(items, key=lambda item: (item["clock"] or datetime.max, str(item["path"]), item["number"]))


def _camera_offset():
    try:
        return float(load_settings().get(OFFSET_SETTING) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _away(seconds):
    return "at the flag" if abs(seconds) < 0.5 else f"{seconds:+.0f} s"


class RecorderFlagsWindow:
    def __init__(self, app):
        self.app = app
        self.items, self.near, self.extra = {}, {}, []
        self.q = queue.Queue()
        self._busy = False
        win = self.win = tk.Toplevel(app.root)
        win.title("Recorder flags")
        win.geometry(f"1040x{min(700, win.winfo_screenheight() - 80)}")
        win.minsize(640, 460)
        win.update_idletasks()
        colors = TEXT_COLORS[dark_mode(win)]
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        bottom = ttk.Frame(body)                       # วางก่อน = ปุ่มไม่ตกขอบเมื่อจอเตี้ย
        bottom.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Close", command=self.close).pack(side="right", anchor="n")
        buttons = FlowFrame(bottom)
        buttons.pack(side="left", fill="x", expand=True)
        buttons.add(ttk.Button(buttons, text="▶ Listen around the flag", command=self.listen))
        buttons.add(playback.control(buttons))
        buttons.add(ttk.Button(buttons, text="Add clip here…", command=self.add_clip))
        buttons.add(ttk.Button(buttons, text="Analyze this recording", command=self.analyze))
        buttons.add(ttk.Button(buttons, text="Open selected clip or photo", command=self.open_near), space=10)
        top = FlowFrame(body)
        top.pack(fill="x")
        top.add(ttk.Button(top, text="Add recordings folder…", command=self.add_folder))
        top.add(ttk.Button(top, text="Refresh", command=self.refresh))
        top.add(ttk.Label(top, text="Camera minus recorder clock (s)"), space=14)
        self.offset = tk.StringVar(master=win, value=f"{_camera_offset():g}")
        spin = top.add(ttk.Spinbox(top, from_=-3600, to=3600, increment=1, width=7, textvariable=self.offset,
                                   command=self._offset_changed))
        spin.bind("<Return>", lambda _event: self._offset_changed())
        spin.bind("<FocusOut>", lambda _event: self._offset_changed())
        self.state = tk.StringVar(value="Reading the recordings …")
        ttk.Label(body, textvariable=self.state, foreground=colors["state"], wraplength=1000,
                  justify="left").pack(anchor="w", pady=(6, 4))
        columns = ("clock", "file", "at", "clips", "photos", "status")
        self.tree = ttk.Treeview(body, columns=columns, show="headings", height=9, selectmode="browse")
        for key, label, width in (("clock", "Flag time", 150), ("file", "Recording", 190), ("at", "In file", 62),
                                  ("clips", "BirdNET clips near (before − / after +)", 310),
                                  ("photos", "Photos near", 190), ("status", "Status", 110)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=key == "clips")
        self.tree.tag_configure("waiting", foreground=colors["waiting"])
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())
        self.tree.bind("<Double-1>", lambda _event: self.listen())
        ttk.Label(body, text="Near the selected flag (double-click opens the clip or photo)").pack(anchor="w", pady=(8, 2))
        columns = ("kind", "time", "away", "name", "detail")
        self.near_tree = ttk.Treeview(body, columns=columns, show="headings", height=6, selectmode="browse")
        for key, label, width in (("kind", "", 56), ("time", "Time", 76), ("away", "From flag", 80),
                                  ("name", "Species", 240), ("detail", "Detail", 400)):
            self.near_tree.heading(key, text=label)
            self.near_tree.column(key, width=width, stretch=key == "detail")
        self.near_tree.pack(fill="both", expand=True)
        self.near_tree.bind("<Double-1>", lambda _event: self.open_near())
        win.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_job = None
        self.refresh()

    # ---------- อ่านข้อมูล ----------
    def _folders(self):
        roots = field_log.stored()[1]
        return [folder for root in roots if Path(root).is_dir() for folder in sound_folders(root)] + self.extra

    def refresh(self):
        if self._busy:
            return
        self._busy = True
        self.state.set("Reading the recordings …")
        output, folders, results = self.app.out.get().strip() or str(Path.home()), self._folders(), self.q

        def work():   # ไม่ถือ self
            try:
                results.put((collect(output, folders), None))
            except Exception as exc:  # noqa: BLE001  ไฟล์เสีย: แจ้งในหน้าต่าง
                results.put(([], str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self._poll_job = self.win.after(100, self._poll)

    def _poll(self):
        self._poll_job = None
        try:
            items, error = self.q.get_nowait()
        except queue.Empty:
            self._poll_job = self.win.after(100, self._poll)
            return
        self._busy = False
        self._fill(items, error)

    def _fill(self, items, error=None):
        self.tree.delete(*self.tree.get_children())
        self.items = {str(n): item for n, item in enumerate(items)}
        offset = self._offset()
        for iid, item in self.items.items():
            clips = ", ".join(f"{species_names.row_names(row)[0]} {_away(away)}" for away, row in item["clips"][:3])
            clips += f" +{len(item['clips']) - 3} more" if len(item["clips"]) > 3 else ""
            if item["summary"] is None:
                clips = "not analyzed yet"
            elif not clips:
                clips = "BirdNET found nothing here"
            near = recorder_flags.photos_near(item["clock"], item["photos"], offset) if item["clock"] else []
            named = next((photo for _away_s, photo in near if photo["common"]), None)
            photos = f"{named['common']} +{len(near) - 1}" if named and len(near) > 1 else (
                named["common"] if named else f"{len(near)} photo(s)" if near else "")
            self.tree.insert("", "end", iid=iid, tags=() if item["summary"] else ("waiting",), values=(
                f"{item['clock']:%Y-%m-%d %H:%M:%S}" if item["clock"] else f"flag {item['number']}", item["path"].name,
                format_offset(item["seconds"], 0), clips, photos, "analyzed" if item["summary"] else "not analyzed"))
        recordings = len({str(item["path"]) for item in items})
        self.state.set(f"Could not read everything: {error}" if error else
                       f"{len(items)} flag(s) in {recordings} recording(s). A flag marks where to look: it does not "
                       "change BirdNET's scores or confirm a species. Sounds count as near from "
                       f"{recorder_flags.BEFORE:.0f} s before to {recorder_flags.AFTER:.0f} s after a flag, "
                       f"photos within {recorder_flags.PHOTO_WINDOW:.0f} s.")
        self._selected()

    def _offset(self):
        try:
            return float(self.offset.get())
        except ValueError:
            return 0.0

    def _offset_changed(self):
        value = self._offset()
        if value != _camera_offset():
            save_settings({OFFSET_SETTING: value})
        chosen = self.tree.selection()
        self._fill(list(self.items.values()))
        if chosen and self.tree.exists(chosen[0]):
            self.tree.selection_set(chosen[0])

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self.win, title="Folder of recordings (subfolders are searched too)")
        if folder and Path(folder) not in self.extra:
            self.extra.append(Path(folder))
            self.refresh()

    # ---------- flag ที่เลือก ----------
    def _item(self, quiet=False):
        chosen = self.tree.selection()
        item = self.items.get(chosen[0]) if chosen else None
        if item is None and not quiet:
            self.state.set("Select a flag first")
        return item

    def _selected(self):
        self.near_tree.delete(*self.near_tree.get_children())
        self.near = {}
        item = self._item(quiet=True)
        if item is None:
            return
        for number, (away, row) in enumerate(item["clips"]):
            status = {"Unsure": "Not sure"}.get(str(row.get("Review status") or "Pending"), row.get("Review status") or "Pending")
            score = "added by you" if manual_clips.is_manual(row) else f"{row.get('Max confidence')}"
            iid = f"c{number}"
            self.near[iid] = ("clip", row)
            self.near_tree.insert("", "end", iid=iid, values=("Clip", row.get("Clock time") or "", _away(away),
                                                              species_names.row_names(row)[0], f"{score} · {status}"))
        if item["clock"]:
            for number, (away, photo) in enumerate(recorder_flags.photos_near(item["clock"], item["photos"], self._offset())):
                iid = f"p{number}"
                self.near[iid] = ("photo", photo)
                self.near_tree.insert("", "end", iid=iid, values=(
                    "Photo", photo["time"][11:], _away(away), photo["common"] or "(not on the checklist)", photo["file"]))

    def _ready_file(self, item):
        if item["path"].is_file():
            return True
        self.state.set(f"Recording not found: {item['path']}")
        return False

    def listen(self):
        item = self._item()
        if item is None or not self._ready_file(item):
            return
        from listening_tools import ListeningToolsWindow
        first = max(0.0, item["seconds"] - recorder_flags.BEFORE)
        ListeningToolsWindow(self.win, item["path"], app=self.app, start=first,
                             end=min(item["seconds"] + recorder_flags.AFTER, audio_duration(item["path"])),
                             clock_start=item["start"] + timedelta(seconds=first) if item["start"] else None,
                             flags=item["flags"])

    def add_clip(self):
        item = self._item()
        if item is None or not self._ready_file(item):
            return
        if item["summary"] is None or not Path(item["summary"]).is_file():
            self.state.set("This recording has not been analyzed yet: press Analyze this recording first")
            return
        self.app.open_manual_at(item["summary"], item["listed"], max(0.0, item["seconds"] - CLIP_LEAD))

    def analyze(self):
        item = self._item()
        if item is None or not self._ready_file(item):
            return
        if item["summary"] is not None and not messagebox.askokcancel(
                "Analyze this recording", f"{item['path'].name} has been analyzed already. Run it again?", parent=self.win):
            return
        self.app.analyze_path(item["path"])

    def open_near(self):
        chosen = self.near_tree.selection()
        kind, value = self.near.get(chosen[0], (None, None)) if chosen else (None, None)
        item = self._item(quiet=True)
        if kind == "clip" and item is not None:
            self.app.show_clip(item["summary"], str(value.get("File")))
        elif kind == "photo":
            if Path(value["path"]).is_file():
                platform_tools.open_path(value["path"])
            else:
                self.state.set(f"Photo not found: {value['path']}")
        else:
            self.state.set("Select a clip or photo in the lower table first")

    def close(self):
        try:
            if self._poll_job:
                self.win.after_cancel(self._poll_job)
            self.win.destroy()
        except tk.TclError:
            pass
