"""หน้าต่าง Match recordings: แต่ละไฟล์เสียงของวันหนึ่งอยู่ checklist ไหน แถวละไฟล์ เลือกจาก drop down แล้วบันทึกเลย
เด้งเองหลังเพิ่มลิงก์หรือ Log ไฟล์ที่ยังไม่ได้เลือกใช้ checklist ที่เดาจากเวลาเริ่มอัด (field_log.for_recording) ขึ้น "by time"
แต่ละแถวบอกพิกัดที่ไฟล์นั้นจะใช้ตอนวิเคราะห์ (field_log.spot_for)
ปิดหน้าต่างแล้วแจ้ง <<RecordingsMatched>> ที่หน้าต่างหลัก ให้ย้ายคลิปใน Ready ตาม checklist ของวันที่เปิดดู (ready_folders.py)

ไฟล์เสียงหาจาก Sounds ของวันนั้นใต้ Birding (รวมไฟล์ที่ยังไม่ได้วิเคราะห์) + ผลวิเคราะห์ของวันนั้น
"""

import queue
import threading
from datetime import date, timedelta
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import field_log
import manual_clips
import recorder_flags
from audio_io import audio_duration, collect_audio
from frequency_compare import ScrollFrame
from result_roots import all_results
from review_store import load_reviews

NO_CHECKLIST = "No checklist"


def recordings_on(day, output=None):
    """ไฟล์เสียงของวันนี้ -> [{"start", "end", "file"}] เรียงตามเวลา — ทำใน thread ได้ (ไม่แตะ Tk)"""
    found = {}

    def note(start, name, seconds):
        if start.date() == day:
            found.setdefault(start, {"start": start, "end": start + timedelta(seconds=seconds), "file": name})

    for root in field_log.stored()[1]:
        sounds = field_log.day_folder(root, day) / "Sounds"
        folder = sounds / "Raw" if (sounds / "Raw").is_dir() else sounds
        for path in collect_audio(folder) if folder.is_dir() else ():
            start = recorder_flags.recording_start(path)
            if start:
                note(start, path.name, audio_duration(path))
    for info in all_results(output):
        if not info["folder"].startswith(f"{day:%Y.%m.%d}"):
            continue
        rows = load_reviews(info["path"])
        for source in manual_clips.sources_in(rows):
            start = (manual_clips.source_context(rows, source) or {}).get("start")
            if start and start not in found:
                ends = [manual_clips._number(row.get("End offset (s)")) for row in rows
                        if str(row.get("Source path") or "").strip() == str(source)]
                seconds = audio_duration(source) if source.is_file() else max(filter(None, ends), default=0)
                note(start, Path(source).name, seconds)
    return [found[start] for start in sorted(found)]


class RecordingChecklists:
    def __init__(self, master, output, day=None, changed=None):
        self.output, self.changed = output, changed     # changed() = เลือกแล้ว ให้หน้าที่เปิดอยู่แสดงใหม่
        self.q = queue.Queue()
        self.days = sorted({date.fromisoformat(c["date"][:10]) for c in field_log.stored()[0]} | ({day} if day else set()),
                           reverse=True)
        self._poll_job, self._reading, self._kept, self._seen = None, 0, [], set()
        win = self.win = tk.Toplevel(master)
        win.title("Match recordings to checklists")
        win.geometry(f"1020x{min(420, win.winfo_screenheight() - 80)}")
        win.minsize(640, 240)
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        bottom = ttk.Frame(body)                       # วางก่อน = ปุ่มไม่ตกขอบเมื่อจอเตี้ย
        bottom.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Close", command=win.destroy).pack(side="right")
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Day").pack(side="left")
        self.day = tk.StringVar(master=win, value=f"{(day or self.days[0]):%d %b %Y}" if self.days else "")
        pick_day = ttk.Combobox(top, textvariable=self.day, values=[f"{d:%d %b %Y}" for d in self.days],
                                state="readonly", width=12)
        pick_day.pack(side="left", padx=(6, 12))
        pick_day.bind("<<ComboboxSelected>>", lambda _event: self.read())
        self._no_wheel(pick_day)
        self.state = tk.StringVar(master=win)
        ttk.Label(top, textvariable=self.state, foreground="#888").pack(side="left", fill="x", expand=True)
        scroller = ScrollFrame(body)
        scroller.pack(fill="both", expand=True, pady=(8, 0))
        self.rows = scroller.inner
        self.rows.columnconfigure(2, weight=1)
        self.rows.grid_anchor("nw")                    # แถวเรียงจากบน ไม่ลอยกลางหน้าต่าง
        win.bind("<Destroy>", self._closed)
        self.read()

    @staticmethod
    def _no_wheel(box):
        """ปัดผ่าน drop down ต้องไม่เปลี่ยนค่า (บันทึกทันทีที่เลือก)"""
        for sequence in box.bind_class("TCombobox"):
            if "Wheel" in sequence or "Scroll" in sequence:
                box.bind(sequence, lambda _event: "break")

    def _chosen_day(self):
        return next((day for day in self.days if f"{day:%d %b %Y}" == self.day.get()), None)

    def read(self):
        day = self._chosen_day()
        for child in self.rows.winfo_children():
            child.destroy()
        self._kept = []
        if day is None:
            self.state.set("Add a field log first")
            return
        self.state.set("Reading the recordings …")
        self._seen.add(day)
        self._reading += 1
        reading, output, results = self._reading, self.output, self.q

        def work():   # ไม่ถือ self
            try:
                results.put((reading, day, recordings_on(day, output), None))
            except Exception as exc:  # noqa: BLE001  ไฟล์เสีย: แจ้งในหน้าต่าง
                results.put((reading, day, [], str(exc)))
        threading.Thread(target=work, daemon=True).start()
        if self._poll_job is None:
            self._poll_job = self.win.after(100, self._poll)

    def _poll(self):
        self._poll_job = None
        try:
            reading, day, recordings, error = self.q.get_nowait()
        except queue.Empty:
            self._poll_job = self.win.after(100, self._poll)
            return
        if reading == self._reading:                   # เปลี่ยนวันระหว่างอ่าน = ทิ้ง
            self.fill(day, recordings, error)
        else:
            self._poll_job = self.win.after(100, self._poll)

    @staticmethod
    def _where(start):
        spot = field_log.spot_for(start)
        return f"{spot['lat']:.6f}, {spot['lon']:.6f}" if spot else "no coordinates"

    def fill(self, day, recordings, error=None):
        checklists = field_log.for_day(day)
        labels = {field_log.describe_day([checklist]): checklist["id"] for checklist in checklists}
        labels[NO_CHECKLIST] = field_log.NONE
        names = {value: label for label, value in labels.items()}
        for index, recording in enumerate(recordings):
            start = recording["start"]
            current = field_log.for_recording(start)
            choice = tk.StringVar(master=self.win, value=names[current[0]["id"]] if current else NO_CHECKLIST)
            how = tk.StringVar(master=self.win, value="" if field_log.picked(start) else "by time")
            where = tk.StringVar(master=self.win, value=self._where(start))
            self._kept.append((choice, how, where))    # ตัวแปรหายเมื่อไม่มีใครถือ แล้ว drop down จะว่าง
            ttk.Label(self.rows, text=f"{start:%H:%M}–{recording['end']:%H:%M}").grid(row=index, column=0, sticky="w")
            ttk.Label(self.rows, text=recording["file"]).grid(row=index, column=1, sticky="w", padx=(10, 10))
            box = ttk.Combobox(self.rows, textvariable=choice, values=list(labels), state="readonly", width=44)
            box.grid(row=index, column=2, sticky="ew", pady=2)
            box.bind("<<ComboboxSelected>>",
                     lambda _event, r=recording, c=choice, h=how, w=where: self._pick(r, labels[c.get()], h, w))
            self._no_wheel(box)
            ttk.Label(self.rows, textvariable=where, width=21).grid(row=index, column=3, sticky="w", padx=(8, 0))
            ttk.Label(self.rows, textvariable=how, foreground="#888", width=7).grid(row=index, column=4, sticky="w",
                                                                                    padx=(4, 0))
        if error:
            self.state.set(f"Could not read the recordings: {error}")
        elif recordings:
            self.state.set(f"{len(recordings)} recording(s), {len(checklists)} checklist(s) — saved when you pick; "
                           "Ready folders follow when you close this window")
        else:
            self.state.set("No recordings found for this day yet")

    def _pick(self, recording, checklist, how, where):
        field_log.choose(recording, checklist)
        how.set("")
        where.set(self._where(recording["start"]))
        if self.changed:
            self.changed()

    def _closed(self, event):
        if event.widget is not self.win:
            return
        if self._poll_job is not None:
            self.win.after_cancel(self._poll_job)
            self._poll_job = None
        root = self.win._root()
        root._matched_days = getattr(root, "_matched_days", set()) | self._seen
        try:
            root.event_generate("<<RecordingsMatched>>", when="tail")
        except tk.TclError:                            # ปิดแอปอยู่
            pass
