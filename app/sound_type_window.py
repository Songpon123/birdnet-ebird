"""หน้าต่าง Sound types: ดูคำเดา Song/Call ของคลิปทุกผลชุด ฟัง/ดูภาพ แล้วเลือกประเภทที่ถูกด้วยคลิกเดียว (หรือปุ่มลัด)
ผลที่เลือกบันทึกลง summary.xlsx (คอลัมน์ Sound type) ทันทีโดยไม่เปลี่ยนสถานะการตรวจ แล้วฝึกตัวเดาใหม่เบื้องหลัง
(คลิปที่ยัง Pending ก็ใช้ฝึกได้; ไม่แสดงคลิปที่ Reject = ชนิดผิด หรือ Not sure = เสียงไม่ชัด)"""

import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import playback
import sound_type
import species_names
from frequency_compare import FlowFrame
from result_roots import all_results
from review_store import save_review
from spectrogram_view import SpectrogramView

GROUPS = ("Not sure — most uncertain first", "Guessed song (sure)", "Guessed call (sure)",
          "Different from your choice", "Your choices")
KEYS = {"s": "Song", "c": "Call", "f": "Flight call", "a": "Alarm call", "d": "Drumming", "o": "Other"}


def build_groups(records):
    """คลิป + คำเดาตอนนี้ -> {กลุ่ม: [(แถว, คำเดา)]} — กลุ่มที่แอปเดาเรียงจากก้ำกึ่งที่สุดก่อน (ตรวจแล้วได้ประโยชน์สุด)"""
    groups = {name: [] for name in GROUPS}
    for row in records:
        if str(row.get("Review status") or "Pending") in ("Rejected", "Unsure"):
            continue
        found = sound_type.guess(row)
        chosen = str(row.get("Sound type") or "").strip()
        if chosen:
            groups["Your choices"].append((row, found))
            kind = sound_type.reviewed_kind(chosen)
            if found and kind and found["type"] != kind:
                groups["Different from your choice"].append((row, found))
        elif found:
            name = GROUPS[0] if not found["sure"] else (GROUPS[1] if found["type"] == "Song" else GROUPS[2])
            groups[name].append((row, found))
    for name in GROUPS[:3]:
        groups[name].sort(key=lambda pair: pair[1]["p"])
    return groups


def guess_label(found):
    if not found:
        return "—"
    return f"{found['type'].lower()} {found['p']:.0%}" + ("" if found["sure"] else "?")


class SoundTypeWindow:
    def __init__(self, app, group=None):
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.title("Sound types — check the app's song / call guesses")
        win.geometry(f"1000x{min(780, win.winfo_screenheight() - 80)}")
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)

        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Show").pack(side="left")
        self.group = tk.StringVar(value=group or GROUPS[0])
        box = ttk.Combobox(top, textvariable=self.group, values=GROUPS, state="readonly", width=32)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda _event: self._fill())
        self.count = tk.StringVar(value="Loading …")
        ttk.Label(top, textvariable=self.count, foreground="#808080").pack(side="left", padx=8)
        ttk.Button(top, text="Open in Review clips", command=self.open_in_review, takefocus=0).pack(side="right")
        ttk.Button(top, text="Refresh guesses", command=self.refresh, takefocus=0).pack(side="right", padx=6)

        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True, pady=(6, 4))
        columns = ("species", "guess", "chosen", "status", "time", "result")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        for key, label, width in (("species", "Species (eBird name)", 230), ("guess", "App's guess", 120),
                                  ("chosen", "Your choice", 110), ("status", "Review", 90),
                                  ("time", "Time", 80), ("result", "Result", 220)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "result"), anchor="w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._selected)
        self.tree.bind("<Double-1>", lambda _event: self.open_in_review())

        self.view = SpectrogramView(body, width=900, height=150, overview_height=26, on_seek=self._seek,
                                    only_heard=getattr(app, "only_heard", None))
        self.view.pack(fill="x")
        actions = FlowFrame(body)
        actions.pack(fill="x", pady=(6, 0))
        actions.add(ttk.Button(actions, text="▶ Play", command=self.play, takefocus=0))
        actions.add(ttk.Button(actions, text="■ Stop", command=app.review_stop, takefocus=0))
        actions.add(playback.control(actions))
        actions.add(ttk.Label(actions, text="It is:"), space=8)
        for key, kind in KEYS.items():                # ปุ่มไม่รับโฟกัส: เคาะ space = เล่นเสียง ไม่ใช่กดปุ่ม
            actions.add(ttk.Button(actions, text=f"{kind} ({key})", command=lambda kind=kind: self.label(kind),
                                   takefocus=0))
        actions.add(ttk.Button(actions, text="Clear", command=lambda: self.label(""), takefocus=0), space=2)
        self.info = tk.StringVar()
        ttk.Label(body, textvariable=self.info, foreground="#808080", wraplength=960).pack(anchor="w", pady=(4, 0))
        ttk.Label(body, text="Keys: space = play, s / c / f / a / d / o = choose. Your choice is saved at once "
                  "(the review status does not change) and the guesser retrains. Rejected and Not sure clips "
                  "are not listed.", foreground="#808080",
                  wraplength=960).pack(anchor="w")

        for key, kind in KEYS.items():
            win.bind(f"<KeyPress-{key}>", lambda event, kind=kind: self._key(event, kind))
        win.bind("<space>", lambda _event: (self.play(), "break")[1])
        win.protocol("WM_DELETE_WINDOW", self.close)
        self.q = queue.Queue()
        self.groups, self.items = {}, {}
        self.refresh()
        self.win.after(150, self._poll)

    # ---------- โหลด ----------
    def refresh(self):
        self.count.set("Loading …")
        root = self.app.out.get().strip() or str(Path.home() / "BirdNET_eBird")
        extra = Path(self.app.review_summary.get().strip() or ".").expanduser()
        results = self.q

        def work():                                   # ไม่ถือ self: ส่งผลกลับทางคิวอย่างเดียว
            import review_progress
            summaries = {Path(info["path"]).resolve() for info in all_results(root)}
            if extra.is_file() and extra.name == "summary.xlsx":
                summaries.add(extra.resolve())
            results.put(build_groups(review_progress.collect(sorted(summaries))))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            self.groups = self.q.get_nowait()
            self._fill()
        except queue.Empty:
            pass
        self.win.after(200, self._poll)

    def _fill(self):
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        pairs = self.groups.get(self.group.get(), [])
        for index, (row, found) in enumerate(pairs):
            iid = str(index)
            self.items[iid] = (row, found)
            self.tree.insert("", "end", iid=iid, values=(
                species_names.row_names(row)[0], guess_label(found), row.get("Sound type") or "",
                {"Unsure": "Not sure"}.get(str(row.get("Review status") or "Pending"),
                                           str(row.get("Review status") or "Pending")),
                row.get("Clock time") or "", row["_summary"].parent.name))
        self.count.set(f"{len(pairs)} clips" + ("" if pairs else " — nothing here"))
        if pairs:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.tree.focus_set()
        else:
            self.view.message("Nothing to show in this group")

    # ---------- เลือก / ฟัง ----------
    def _current(self):
        chosen = self.tree.selection()
        return (chosen[0], *self.items[chosen[0]]) if chosen and chosen[0] in self.items else (None, None, None)

    def _path(self, row):
        return (row["_summary"].parent / str(row["File"])).resolve()

    def _selected(self, _event=None):
        _iid, row, found = self._current()
        if row is None:
            return
        from birdnet_gui import _clip_spans_and_clock
        spans, clock = _clip_spans_and_clock(row)
        self.view.load(self._path(row), clock_start=clock, spans=spans)
        score = row.get("Max confidence")
        self.info.set(f"{species_names.row_names(row)[0]}  ·  {row['_summary'].parent.name}  ·  "
                      f"{row.get('Clock time') or ''}"
                      + (f"  ·  BirdNET {float(score):.2f}" if isinstance(score, (int, float)) else "")
                      + (f"  ·  {sound_type.describe(found)}" if found else ""))

    def play(self):
        _iid, row, _found = self._current()
        if row is not None:
            self.app.play_from(self._path(row), 0.0, self.view)

    def _seek(self, seconds):
        _iid, row, _found = self._current()
        if row is not None:
            self.app.play_from(self._path(row), seconds, self.view)

    def open_in_review(self):
        _iid, row, _found = self._current()
        if row is not None:
            self.app.show_clip(row["_summary"], str(row["File"]), "Opened from Sound types")

    # ---------- เลือกประเภท ----------
    def _key(self, event, kind):
        if event.widget.winfo_class() not in ("TCombobox", "TEntry", "Entry"):   # พิมพ์ในช่องอื่นอยู่: ไม่ใช่ปุ่มลัด
            self.label(kind)

    def label(self, kind):
        """บันทึก Sound type ของคลิปที่เลือก (ไม่เปลี่ยนสถานะการตรวจ) แล้วไปคลิปถัดไป"""
        iid, row, found = self._current()
        if row is None:
            return
        status = str(row.get("Review status") or "Pending")
        rating = row.get("Quality rating")
        try:
            save_review(row["_summary"], int(row["_row"]), status=status,
                        common=str(row.get("Reviewed common") or ""), scientific=str(row.get("Reviewed scientific") or ""),
                        rating=int(rating) if isinstance(rating, (int, float)) and rating == rating else None,
                        notes=str(row.get("Review notes") or ""), expected_file=str(row["File"]), sound_type=kind,
                        sound_type_guess=None if row.get("Sound type guess") else (sound_type.guess_text(found) or None))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not save the sound type", str(exc), parent=self.win)
            return
        row["Sound type"] = kind or None
        if found and not row.get("Sound type guess"):
            row["Sound type guess"] = sound_type.guess_text(found)
        self.tree.set(iid, "chosen", kind)
        # แท็บ Review เปิดผลชุดเดียวกันอยู่: โหลดใหม่ ไม่งั้นช่อง Sound type ที่ค้างอยู่จะเขียนทับตอนกด Approve
        current = Path(self.app.review_summary.get()).expanduser()
        if current.is_file() and current.resolve() == row["_summary"].resolve():
            self.app.review_load(quiet=True)
        if sound_type.reviewed_kind(kind):
            self.app._retrain_sound_type()
        following = self.tree.next(iid)
        if following:
            self.tree.selection_set(following)
            self.tree.focus(following)
            self.tree.see(following)
        self.tree.focus_set()

    def close(self):
        self.app.review_stop()
        self.win.destroy()
