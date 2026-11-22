"""หน้าต่างค้นเสียง (ใช้ดัชนีเสียงใน sound_index.py):
  FindSimilarWindow   เสียงที่คล้ายคลิปหรือช่วงที่เลือก ในไฟล์นี้ ผลชุดนี้ หรือทุกผลชุด
  SoundGroupsWindow   จัดกลุ่มเสียงที่ BirdNET ยังไม่ได้ตัดเป็นคลิป แล้วตั้งชื่อทีละกลุ่ม
  RecognizersWindow   ตัวจำแนกต่อชนิดที่ฝึกจากผลตรวจของคุณ (recognizers.py)
ทุกหน้าต่างใช้ MatchList: ตารางเสียงที่เจอ + ตัวดู spectrogram (คลิก = เล่น) + เพิ่มเป็นคลิปทีละหลายอัน
งานหนัก (ทำดัชนี, จัดกลุ่ม, ฝึก, ตัดคลิป) เข้าคิวงานเดียวกับการวิเคราะห์; การค้นใช้ numpy ใน thread
"""

import json
import queue
import threading
import time
from datetime import timedelta
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import country_checklist
import manual_clips
import paths
import playback
import sound_index
from frequency_compare import FlowFrame
from manual_clips import format_offset
from review_store import SOUND_TYPES, load_reviews

PREVIEW_DIR = paths.CACHE / "search-preview"
PAD = 1.5                       # คลิปที่เพิ่มจากเสียงที่เจอ: เผื่อหน้า/หลัง (วินาที)
SECONDS_PER_AUDIO_HOUR = 130    # ทำดัชนี: วัดจริงบน M-series ~2 นาทีต่อเสียง 1 ชั่วโมง


def collect_sources(summaries):
    """ไฟล์ต้นฉบับของผลชุดเหล่านี้ -> ({sid: {"path", "name", "summary", "rows", "context"}}, แถวทุกแถว)
    ไฟล์เดียวกันอยู่หลายผลชุด (เช่น 3.0 กับ 2.4): ใช้ผล _v3preview ก่อน — ทำงานใน thread ได้ (ไม่แตะ Tk)"""
    found, every = {}, []
    for summary in summaries:
        try:
            rows = load_reviews(Path(summary))
        except Exception:  # noqa: BLE001  ไฟล์เสีย/กำลังเขียน: ข้าม
            continue
        every += rows
        for sid, entry in sound_index.sources(rows, summary).items():
            current = found.get(sid)
            if current is None or (Path(summary).parent.name.endswith("_v3preview")
                                   and not Path(current["summary"]).parent.name.endswith("_v3preview")):
                entry["context"] = manual_clips.source_context(entry["rows"], entry["path"]) or {}
                found[sid] = entry
    return found, every


def describe_clips(rows):
    """คลิปที่มีอยู่แล้วตรงเสียงนี้ -> 'Light-vented Bulbul (Approved); …' หรือ '—'"""
    if not rows:
        return "—"
    import species_names
    names = []
    for row in rows:
        common = row.get("Reviewed common") or species_names.row_names(row)[0]
        status = {"Unsure": "Not sure"}.get(str(row.get("Review status") or "Pending"), row.get("Review status") or "Pending")
        names.append(f"{common} ({status})")
    return "; ".join(dict.fromkeys(names))


def same_species(match, scientific):
    """มีคลิปของชนิดนี้ตรงเสียงนี้แล้วไหม (ไม่ระบุชนิด = คลิปชนิดไหนก็ได้)"""
    import species_names
    if not match.get("clips"):
        return False
    target = " ".join(str(scientific or "").split()[:2])
    if not target:
        return True
    return any(target in {" ".join(str(row.get("Reviewed scientific") or "").split()[:2]),
                          species_names.row_names(row)[1]} for row in match["clips"])


def enrich(matches, sources, spans):
    """เติมข้อมูลให้เสียงที่เจอ: ชื่อไฟล์, เวลาจริง, คำทายของ BirdNET, คลิปที่มีอยู่แล้ว (ใน thread ได้)"""
    for match in matches:
        entry = sources[match["sid"]]
        context = entry["context"]
        match["name"], match["source"], match["summary"] = entry["name"], entry["path"], entry["summary"]
        match["context"] = context
        match["clock"] = context["start"] + timedelta(seconds=match["start"]) if context.get("start") else None
        index = sound_index.load(match["sid"])
        guesses = sound_index.event_guesses(index, match, top=1) if index is not None else []
        match["guess"] = f"{guesses[0]['common']} {guesses[0]['confidence']:.2f}" if guesses else ""
        match["clips"] = sound_index.clips_at(spans, match["sid"], match["start"], match["end"])
    return matches


class MatchList(ttk.Frame):
    """ตารางเสียงที่เจอ (เลือกได้หลายแถว) + ตัวดู spectrogram + เพิ่มเป็นคลิป"""

    def __init__(self, master, app, species=("", ""), score_label="Similarity", on_status=None,
                 strong=sound_index.STRONG, strong_label=None):
        super().__init__(master)
        from spectrogram_view import SpectrogramView
        self.app = app
        self.on_status = on_status or (lambda _text: None)
        self.matches = {}
        self.score_label = score_label
        self.strong = strong                # คะแนนที่ทำตัวหนา + ปุ่ม Select strong
        self._preview = None              # (iid, path ของเสียงตัวอย่าง)
        self._number = 0
        columns = ("score", "recording", "at", "clock", "length", "birdnet", "clips")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=9, selectmode="extended")
        for key, label, width in (("score", score_label, 74), ("recording", "Recording", 120),
                                  ("at", "In file", 70), ("clock", "Clock time", 140), ("length", "Length", 58),
                                  ("birdnet", "BirdNET guess", 190), ("clips", "Clips already there", 250)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "clips"))
        self.tree.tag_configure("strong", font=("Helvetica", 12, "bold"))
        self.tree.tag_configure("added", foreground="#3a87c8")
        self.tree.tag_configure("clipped", foreground="#808080")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())
        self.tree.bind("<space>", lambda _event: self.play())
        self.view = SpectrogramView(self, width=860, height=110, overview_height=18, on_seek=self._seek, app=app)
        self.view.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.view.message("Select a sound to see and hear it")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)

        buttons = FlowFrame(self)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        buttons.add(ttk.Button(buttons, text="▶ Play", command=self.play))
        buttons.add(ttk.Button(buttons, text="■ Stop", command=app.review_stop))
        buttons.add(playback.control(buttons))
        buttons.add(ttk.Button(buttons, text="Open in Add clip window…", command=self.open_in_manual))
        buttons.add(ttk.Button(buttons, text=strong_label or f"Select strong (≥ {strong:.2f})", command=self.select_strong))
        buttons.add(ttk.Button(buttons, text="Select all", command=lambda: self.tree.selection_set(self.tree.get_children())))

        add = ttk.Frame(self)
        add.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        add.columnconfigure(1, weight=1)
        ttk.Label(add, text="Add selected as clips of").grid(row=0, column=0, sticky="w")
        self.common, self.scientific = tk.StringVar(value=species[0]), tk.StringVar(value=species[1])
        self.sound_type = tk.StringVar()
        common = ttk.Entry(add, textvariable=self.common)
        common.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(add, textvariable=self.scientific, width=24).grid(row=0, column=2, padx=(0, 6))
        ttk.Label(add, text="Sound type").grid(row=0, column=3)
        ttk.Combobox(add, textvariable=self.sound_type, values=SOUND_TYPES, state="readonly",
                     width=10).grid(row=0, column=4, padx=6)
        self.add_btn = ttk.Button(add, text="✚ Add selected", style="QuickAction.TButton", command=self.add_selected)
        self.add_btn.grid(row=0, column=5)
        from species_picker import SpeciesPicker
        self.picker = SpeciesPicker(common, self._country, lambda c, s: (self.common.set(c), self.scientific.set(s)),
                                    on_exact=self.scientific.set)
        ttk.Label(self, text=f"Clips use the original recording at full quality ({PAD:.1f} s padding for discovered "
                  "sounds; imported CSV detections use their exact bounds) and appear in Review as Pending. Sounds that already "
                  "have a clip of that species are skipped.", foreground="#808080", wraplength=900,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _country(self):
        match = next(iter(self.matches.values()), None)
        if match is None:
            return None
        context = match.get("context") or {}
        return country_checklist.country_at(context.get("lat"), context.get("lon"))

    # ---------- ตาราง ----------
    def set_matches(self, matches, keep_species=True):
        if self.view._play is not None:
            self.app.review_stop()
        self._preview = None
        self.view.message("Select a sound to see and hear it")
        self.tree.delete(*self.tree.get_children())
        self.matches = {}
        target = self.scientific.get()
        for number, match in enumerate(matches):
            iid = f"m{number}"
            self.matches[iid] = match
            clock = match["clock"].strftime("%Y-%m-%d %H:%M:%S") if match.get("clock") else ""
            tags = ("strong",) if self.score_label not in ("Excess dB", "Imported") and match["score"] >= self.strong else ()
            if same_species(match, target):                 # มีคลิปของชนิดนี้แล้ว (เพิ่มซ้ำจะถูกข้าม)
                tags += ("clipped",)
            self.tree.insert("", "end", iid=iid, tags=tags, values=(
                (f"{match['score']:.2f}" if self.score_label != "Imported" else f"row {match.get('csv_line', '')}"),
                match["name"], format_offset(match["start"], 1), clock,
                f"{match['end'] - match['start']:.1f} s", match.get("guess", ""), describe_clips(match.get("clips"))))
        if not matches:
            self.view.message("No candidate sounds found with these settings")

    def select_strong(self):
        self.tree.selection_set([iid for iid, match in self.matches.items() if match["score"] >= self.strong])

    def _selected(self):
        chosen = self.tree.selection()
        if len(chosen) != 1:
            return
        iid = chosen[0]
        match = self.matches[iid]
        from audio_io import read_channels, write_preview
        first = max(0.0, match["start"] - 2.0)
        last = match["end"] + 2.0
        try:
            samples, rate = read_channels(match["source"], first, last, match.get("channel", "stereo"))
            self._number += 1
            target = PREVIEW_DIR / f"match-{self._number}.wav"
            for old in PREVIEW_DIR.glob("match-*.wav"):     # Windows: ไฟล์ที่ยังเล่นอยู่ลบไม่ได้ — ข้าม
                try:
                    old.unlink()
                except OSError:
                    pass
            write_preview(samples, rate, target)
        except Exception as exc:  # noqa: BLE001
            self.view.message(f"Cannot read this sound: {exc}")
            return
        self._preview = (iid, target)
        clock = match["clock"] - timedelta(seconds=match["start"] - first) if match.get("clock") else None
        self.view.load(target, clock_start=clock, spans=[(match["start"] - first, match["end"] - first)])

    def _seek(self, seconds):
        if self._preview:
            self.app.play_from(self._preview[1], seconds, self.view)

    def play(self):
        if not self._preview or self._preview[0] not in self.tree.selection():
            self._selected()
        if self._preview:
            self.app.play_from(self._preview[1], 0.0, self.view)

    def open_in_manual(self):
        chosen = self.tree.selection()
        if not chosen:
            self.on_status("Select a sound first")
            return
        match = self.matches[chosen[0]]
        self.app.open_manual_at(match["summary"], match["source"], match["start"])

    # ---------- เพิ่มเป็นคลิป ----------
    def add_selected(self):
        chosen = [iid for iid in self.tree.selection() if iid in self.matches]
        common, scientific = self.common.get().strip(), " ".join(self.scientific.get().split())
        if not chosen:
            messagebox.showinfo("Add clips", "Select the sounds to add (Shift/Cmd-click selects several)",
                                parent=self.winfo_toplevel())
            return
        if not common or len(scientific.split()) < 2:
            messagebox.showinfo("Add clips", "Enter the species (type a few letters and pick it from the list)",
                                parent=self.winfo_toplevel())
            return
        items = [item for item in (batch_item(self.matches[iid], common, scientific, self.sound_type.get(), iid)
                                   for iid in chosen) if item]
        if not items:
            messagebox.showinfo("Add clips", "The recording date/time or location of these sounds is unknown",
                                parent=self.winfo_toplevel())
            return
        if len(items) > 25 and not messagebox.askokcancel(
                "Add many clips?", f"Add {len(items)} clips of {common}? Each one is a separate clip to review.",
                parent=self.winfo_toplevel()):
            return
        self.on_status(f"Adding {len(items)} clip(s) of {common} …"
                       + (" (waits for the running job)" if self.app.job_running else ""))
        start_batch(self.app, items, self.add_btn, common, self._added)

    def _added(self, code, lines, order, common):
        added = [json.loads(line[7:]) for line in lines if line.startswith("MANUAL ")]
        if self.winfo_exists():
            for info in added:
                iid = order[info["number"] - 1] if 0 < info.get("number", 0) <= len(order) else None
                if iid and self.tree.exists(iid):
                    self.tree.set(iid, "clips", f"added ✓ {info['common']} (Pending)")
                    self.tree.item(iid, tags=("added",))
        reload_review(self.app, added)
        skipped = len(order) - len(added)
        text = (f"Added {len(added)} clip(s) of {common} — they are in Review clips (Pending)"
                + (f"; {skipped} skipped (already a clip, or see Show details)" if skipped else ""))
        if code not in (0, None) and not added:
            text = f"Could not add the clips (exit {code}) — see Show details"
        self.on_status(text)
        self.app.status.set(text)


def batch_item(match, common, scientific, sound_type="", iid=None):
    """เสียงที่เจอ -> รายการของ --manual-clip add-batch (None = ไม่รู้วันเวลา/พิกัดของไฟล์)"""
    pad = match.get("padding", PAD)
    context = match.get("context") or {}
    if not context.get("start") or context.get("lat") is None:
        return None
    return {"summary": str(match["summary"]), "source": str(match["source"]),
            "start": max(0.0, match["start"] - pad), "end": match["end"] + pad,
            "recorded": context["start"].isoformat(), "lat": context["lat"], "lon": context["lon"],
            "place": context.get("place") or "", "date_source": context.get("date_source") or "",
            "common": common, "scientific": scientific, "sound_type": sound_type,
            "notes": match.get("note", ""), "channel": match.get("channel"), "_iid": iid}


def start_batch(app, items, button, what, on_added):
    """เข้าคิวงานตัดคลิปหลายอันทีเดียว -> on_added(code, lines, ลำดับ iid ของรายการ, what)"""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    batch = PREVIEW_DIR / f"add-{int(time.time() * 1000)}.json"
    batch.write_text(json.dumps([{k: v for k, v in item.items() if k != "_iid"} for item in items]),
                     encoding="utf-8")
    order = [item["_iid"] for item in items]

    def done(code, lines):
        batch.unlink(missing_ok=True)
        on_added(code, lines, order, what)
    app._start(app._launch() + ["--manual-clip", "add-batch", str(batch)], button,
               f"Add {len(items)} clip(s) of {what}", is_analyze=False, on_done=done)


def reload_review(app, added):
    """หลังเพิ่มคลิป: รายการผลชุด + แท็บ Review ที่เปิดผลชุดนั้นอยู่"""
    summaries = {info["summary"] for info in added}
    current = Path(app.review_summary.get()).expanduser()
    if summaries:
        app.refresh_results()
        if not current.is_file():
            current = Path(sorted(summaries)[0])
            app.review_summary.set(str(current))
    if current.is_file() and str(current.resolve()) in summaries:
        app.review_load()


def ensure_index(app, window, summaries, then, extra_paths=()):
    """ไฟล์ในผลชุดเหล่านี้ที่ยังไม่มีดัชนี -> เข้าคิวทำดัชนีก่อน (ถามเมื่อนาน) แล้วเรียก then(); มีครบแล้วเรียกทันที
    ตรวจใน thread (ผลหลายชุดอ่านนาน) แล้วกลับมาทำต่อใน thread หลัก"""
    found = queue.Queue()

    def work():
        try:
            from audio_io import audio_duration
            sources, _rows = collect_sources(summaries)
            todo = sound_index.missing(sources) + [(None, Path(p)) for p in extra_paths]
            found.put((todo, sum(audio_duration(path) for _sid, path in todo) / 3600, None))
        except Exception as exc:  # noqa: BLE001
            found.put(([], 0.0, str(exc)))

    def wait():
        if not window.winfo_exists():
            return
        try:
            todo, hours, error = found.get_nowait()
        except queue.Empty:
            window.after(150, wait)
            return
        if error:
            app.status.set(f"Cannot check the sound index: {error}")
            return
        if not todo:
            then()
            return
        minutes = max(1, round(hours * SECONDS_PER_AUDIO_HOUR / 60))
        if minutes >= 5 and not messagebox.askokcancel(
                "Make the sound index first?",
                f"{len(todo)} recording(s) ({hours:.1f} h of audio) need a sound index first — about {minutes} min "
                "on this Mac, done once per recording. It runs in the job queue; you can keep reviewing.",
                parent=window):
            then(cancelled=True)
            return
        app._start(app._launch() + ["--sound-index", "build"] + [str(s) for s in summaries]
                   + [str(p) for p in extra_paths], None, f"Sound index: {len(todo)} recording(s)", is_analyze=False,
                   on_done=lambda code, _lines: then() if code == 0 else then(failed=True))
    threading.Thread(target=work, daemon=True).start()
    window.after(150, wait)


class FindSimilarWindow:
    """เสียงที่คล้ายตัวอย่าง (คลิปในแท็บ Review หรือช่วงที่เลือกในหน้าต่างเพิ่มคลิป)
    query = {"sid", "source", "summary", "start", "end", "label", "common", "scientific"}"""
    SCOPES = (("this recording", "recording"), ("this result", "result"), ("all results", "all"))

    def __init__(self, app, query):
        self.app, self.query = app, query
        self.q = queue.Queue()
        self._search_id = 0
        win = self.win = tk.Toplevel(app.root)
        win.title(f"Find similar sounds — {query['label']}")
        win.geometry(f"1010x{min(820, win.winfo_screenheight() - 70)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Sounds like:", font=("Helvetica", 13, "bold")).pack(side="left")
        ttk.Label(top, text=query["label"]).pack(side="left", padx=6)
        ttk.Button(top, text="▶ Play example", command=self.play_example).pack(side="left", padx=6)
        playback.control(top).pack(side="left", padx=6)
        options = ttk.Frame(body)
        options.pack(fill="x", pady=(8, 0))
        ttk.Label(options, text="Search in").pack(side="left")
        self.scope = tk.StringVar(value="result")
        for text, value in self.SCOPES:
            ttk.Radiobutton(options, text=text, variable=self.scope, value=value).pack(side="left", padx=(6, 0))
        ttk.Label(options, text="Similarity ≥").pack(side="left", padx=(16, 4))
        self.threshold = tk.DoubleVar(value=sound_index.POSSIBLE)
        self.threshold_text = tk.StringVar(value=f"{sound_index.POSSIBLE:.2f}")
        ttk.Scale(options, from_=0.5, to=0.95, variable=self.threshold, length=140,
                  command=lambda value: self.threshold_text.set(f"{float(value):.2f}")).pack(side="left")
        ttk.Label(options, textvariable=self.threshold_text, width=5).pack(side="left", padx=(4, 0))
        self.search_btn = ttk.Button(options, text="Search", style="QuickAction.TButton", command=self.search)
        self.search_btn.pack(side="left", padx=(10, 0))
        self.state = tk.StringVar(value="Bold = very similar (≥ 0.80; the same sound type is usually 0.85–1.0). "
                                        "Listen before adding — similar is not the same species.")
        ttk.Label(body, textvariable=self.state, foreground="#3a87c8", wraplength=960,
                  justify="left").pack(anchor="w", pady=(6, 4))
        self.list = MatchList(body, app, (query.get("common", ""), query.get("scientific", "")),
                              on_status=self.state.set)
        self.list.pack(fill="both", expand=True)
        app._install_clipboard_fix()
        self.win.after(100, self._poll)
        self.win.after(200, self.search)

    def _summaries(self):
        scope = self.scope.get()
        if scope == "all":
            return sorted({Path(p) for p in self.app._results.values()} | {Path(self.query["summary"])})
        return [Path(self.query["summary"])]

    def play_example(self):
        from audio_io import read_mono, write_preview
        try:
            samples, rate = read_mono(self.query["source"], self.query["start"], self.query["end"])
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            target = PREVIEW_DIR / f"example-{int(time.time() * 1000)}.wav"
            write_preview(samples, rate, target)
            self.app.review_stop()
            import platform_tools
            self.app.audio_player = platform_tools.play_audio(target)
        except Exception as exc:  # noqa: BLE001
            self.state.set(f"Cannot play the example: {exc}")

    def search(self):
        summaries = self._summaries()
        self.search_btn.configure(state="disabled")
        self.state.set("Checking the sound index …")
        ensure_index(self.app, self.win, summaries, lambda **why: self._search(summaries, **why),
                     extra_paths=() if sound_index.has_index(self.query["sid"]) or self.query.get("summary")
                     else (self.query["source"],))

    def _search(self, summaries, cancelled=False, failed=False):
        if not self.win.winfo_exists():
            return
        if cancelled or failed:
            self.search_btn.configure(state="normal")
            self.state.set("Sound index failed — see Show details in the main window" if failed
                           else "Search cancelled — the sound index is needed first")
            return
        self._search_id += 1
        search_id, query, scope = self._search_id, dict(self.query), self.scope.get()
        threshold = round(float(self.threshold.get()), 2)
        self.state.set("Searching …")
        results = self.q

        def work():
            try:
                sources, rows = collect_sources(summaries)
                queries = sound_index.query_vectors(query["sid"], query["start"], query["end"])
                if not len(queries):
                    raise ValueError("this recording has no sound index yet")
                targets = [query["sid"]] if scope == "recording" else list(sources)
                found = sound_index.find_similar(queries, [sid for sid in targets if sid in sources], threshold,
                                                 exclude={query["sid"]: [(query["start"], query["end"])]})
                enrich(found, sources, sound_index.clip_spans(rows))
                for match in found:
                    match["note"] = f"Found by Find similar ({match['score']:.2f} like {query['label']})"
                results.put((search_id, found, None, len(targets)))
            except Exception as exc:  # noqa: BLE001
                results.put((search_id, [], str(exc), 0))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                search_id, found, error, searched = self.q.get_nowait()
                if search_id != self._search_id:
                    continue
                self.search_btn.configure(state="normal")
                if error:
                    self.state.set(f"Search failed: {error}")
                    continue
                self.list.set_matches(found)
                strong = sum(match["score"] >= sound_index.STRONG for match in found)
                fresh = sum(not same_species(match, self.list.scientific.get()) for match in found)
                self.state.set(f"{len(found)} sound(s) in {searched} recording(s) ≥ {self.threshold_text.get()} — "
                               f"{strong} very similar (bold), {fresh} without a clip of this species yet "
                               "(gray = already a clip). Listen, select, then add them as clips.")
        except queue.Empty:
            pass
        self.win.after(150, self._poll)


class SoundGroupsWindow:
    """กลุ่มเสียงในผลชุดนี้ (แนว Cluster Analysis): เสียงที่ BirdNET ว่าคล้ายนก (แม้คะแนนต่ำ) หรือดังมาก
    แต่ยังไม่มีคลิป จัดกลุ่มตามความคล้าย -> ฟังตัวอย่าง 2-3 เสียง แล้วเพิ่มทั้งกลุ่มเป็นคลิปของชนิดนั้น"""

    def __init__(self, app, summary):
        self.app, self.summary = app, Path(summary)
        self.groups = {}
        win = self.win = tk.Toplevel(app.root)
        win.title(f"Sound groups — {self.summary.parent.name}")
        win.geometry(f"1010x{min(880, win.winfo_screenheight() - 70)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        self.include = tk.BooleanVar(value=False)
        self.group_mode = tk.StringVar(value="Sound events")
        mode = ttk.Combobox(top, textvariable=self.group_mode, values=("Sound events", "3 s windows"),
                            width=15, state="readonly")
        mode.pack(side="left", padx=(0, 8))
        mode.bind("<<ComboboxSelected>>", lambda _event: self.show_cached())
        ttk.Checkbutton(top, text="Also sounds BirdNET already clipped", variable=self.include,
                        command=self.show_cached).pack(side="left")
        self.find_btn = ttk.Button(top, text="Find groups", style="QuickAction.TButton", command=self.find)
        self.find_btn.pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Find similar to this group…", command=self.similar).pack(side="left", padx=(10, 0))
        self.state = tk.StringVar()
        ttk.Label(body, textvariable=self.state, foreground="#3a87c8", wraplength=960,
                  justify="left").pack(anchor="w", pady=(6, 4))
        columns = ("group", "sounds", "seconds", "guess", "check", "example")
        self.tree = ttk.Treeview(body, columns=columns, show="headings", height=8)
        for key, label, width in (("group", "#", 40), ("sounds", "Sounds", 60), ("seconds", "Seconds", 64),
                                  ("guess", "BirdNET guess (hint only)", 330), ("check", "Here & now?", 90),
                                  ("example", "Example", 300)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "example"))
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._group_selected())
        self.list = MatchList(body, app, score_label="Like group", on_status=self.state.set)
        self.list.pack(fill="both", expand=True, pady=(8, 0))
        app._install_clipboard_fix()
        self.show_cached(first=True)

    def show_cached(self, first=False):
        if self.group_mode.get() == "Sound events":
            import event_groups
            try:
                found = json.loads(event_groups.groups_file(self.summary, self.include.get()).read_text())
            except (OSError, ValueError):
                found = None
        else:
            found = sound_index.load_groups(self.summary, self.include.get())
        if found is None:
            self.tree.delete(*self.tree.get_children())
            self.list.set_matches([])
            self.state.set("Sound events: finds and isolates sounds by frequency and duration, then groups their "
                           "embeddings. Includes quiet candidates and uses every detected event. "
                           "Choose 3 s windows for the previous grouping method. Click Find groups."
                           if self.group_mode.get() == "Sound events" else
                           "Groups sounds that BirdNET found bird-like (even with a low score) or that are very loud, "
                           "but that have no clip yet. Click Find groups — it needs the sound index (made once, "
                           "~2 min per hour of audio) and then takes a few seconds."
                           if not first or not self.include.get() else "")
            return
        self._show(found)

    def find(self):
        self.find_btn.configure(state="disabled")
        self.state.set("Finding groups … (in the job queue)")
        cmd = self.app._launch() + ["--sound-index", "groups", str(self.summary)]
        if self.group_mode.get() == "Sound events":
            cmd.append("--events")
        if self.include.get():
            cmd.append("--include-clipped")
        self.app._start(cmd, self.find_btn, f"Sound groups — {self.summary.parent.name}", is_analyze=False,
                        on_done=self._found)

    def _found(self, code, lines):
        if not self.win.winfo_exists():
            return
        self.find_btn.configure(state="normal")
        error = next((line[6:] for line in lines if line.startswith("ERROR ")), None)
        if code != 0:
            self.state.set(f"Could not find groups: {error or f'exit {code}'} — see Show details")
            return
        self.show_cached()

    def _show(self, found):
        self.tree.delete(*self.tree.get_children())
        self.groups = {}
        sources, rows = collect_sources([self.summary])
        spans = sound_index.clip_spans(rows)
        for group in found["groups"]:
            iid = f"g{group['id']}"
            events = [dict(event) for event in group["events"] if event["sid"] in sources]
            if not events:
                continue
            enrich(events, sources, spans)
            for event in events:
                event["note"] = f"Found in Sound groups (group {group['id']})"
            self.groups[iid] = dict(group, events=events)
            guess = group.get("guess") or []
            text = "; ".join(f"{item['common']} {item['confidence']:.2f}" for item in guess[:3])
            here = guess[0].get("expected") if guess else None
            first = events[0]
            example = f"{first['name']} {format_offset(first['start'], 0)}" + (
                f" ({first['clock']:%H:%M:%S})" if first.get("clock") else "")
            self.tree.insert("", "end", iid=iid, values=(
                group["id"], len(events), f"{group['seconds']:.0f}", text or "—",
                {True: "expected ✓", False: "not expected ✗", None: ""}[here], example))
        repeated = sum(len(group["events"]) > 1 for group in self.groups.values())
        unit = "isolated sound events" if found.get("mode") == "events" else "bird-like or loud 3-s windows"
        self.state.set(f"{repeated} groups of repeated sounds + {len(self.groups) - repeated} sounds heard once, "
                       f"from {found['windows']} {unit} (made {found['made'].replace('T', ' ')}). "
                       "Select a group → listen to its first sounds (most typical first) → name it → select sounds → "
                       "Add selected. BirdNET's guess is only a hint.")
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])

    def _group_selected(self):
        chosen = self.tree.selection()
        if not chosen:
            return
        group = self.groups[chosen[0]]
        self.list.set_matches(group["events"])
        guess = (group.get("guess") or [{}])[0]
        if guess.get("expected") and guess.get("common"):           # เติมชื่อให้ (แก้ได้) เฉพาะชนิดที่ควรพบที่นี่
            self.list.common.set(guess["common"])
            self.list.scientific.set(guess["scientific"])
        else:
            self.list.common.set("")
            self.list.scientific.set("")

    def similar(self):
        chosen = self.tree.selection()
        if not chosen:
            self.state.set("Select a group first")
            return
        group = self.groups[chosen[0]]
        event = group["events"][0]
        self.app.open_find_similar({
            "sid": event["sid"], "source": event["source"], "summary": event["summary"],
            "start": event["start"], "end": event["end"],
            "label": f"group {group['id']} ({event['name']} {format_offset(event['start'], 0)})",
            "common": self.list.common.get(), "scientific": self.list.scientific.get()})


class RecognizersWindow:
    """ตัวจำแนกของคุณเอง: ฝึกต่อชนิดจากคลิปที่ตรวจแล้ว แล้วสแกนหาเสียงชนิดนั้นในผลชุดนี้หรือทุกผลชุด"""

    def __init__(self, app):
        import recognizers
        self.app = app
        self.q = queue.Queue()
        self.species = {}
        win = self.win = tk.Toplevel(app.root)
        win.title("Your recognizers")
        win.geometry(f"1010x{min(880, win.winfo_screenheight() - 70)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"A recognizer learns one species from clips you approved (at least {recognizers.MIN_POSITIVE}), "
                  "clips where BirdNET wrongly said that species, and background sound — then finds that species "
                  "in your recordings, including sounds BirdNET missed. More reviewed clips = a better recognizer.",
                  wraplength=960, justify="left", foreground="#808080").pack(anchor="w")
        target = ttk.Frame(body)
        target.pack(fill="x", pady=5)
        ttk.Label(target, text="Learn this sound type").pack(side="left", padx=(0, 6))
        self.target_type = tk.StringVar(value="Any sound")
        kind_box = ttk.Combobox(target, textvariable=self.target_type,
                                values=("Any sound",) + tuple(t for t in SOUND_TYPES if t),
                                state="readonly", width=16)
        kind_box.pack(side="left")
        kind_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(target, text="Training examples / confusing sounds…", command=self.examples_window).pack(
            side="left", padx=10)
        columns = ("species", "approved", "wrong", "status")
        self.tree = ttk.Treeview(body, columns=columns, show="headings", height=8)
        for key, label, width in (("species", "Species", 220), ("approved", "Approved clips", 100),
                                  ("wrong", "Wrong BirdNET IDs", 120), ("status", "Recognizer", 500)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "status"))
        self.tree.pack(fill="x", pady=(8, 0))
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(8, 0))
        self.train_btn = ttk.Button(actions, text="Train selected", command=self.train)
        self.train_btn.pack(side="left")
        ttk.Label(actions, text="Scan").pack(side="left", padx=(16, 4))
        self.scope = tk.StringVar(value="result")
        ttk.Radiobutton(actions, text="this result", variable=self.scope, value="result").pack(side="left")
        ttk.Radiobutton(actions, text="all results", variable=self.scope, value="all").pack(side="left", padx=(6, 0))
        self.scan_btn = ttk.Button(actions, text="Scan with selected", style="QuickAction.TButton", command=self.scan)
        self.scan_btn.pack(side="left", padx=(10, 0))
        self.state = tk.StringVar(value="Counting your reviewed clips …")
        ttk.Label(body, textvariable=self.state, foreground="#3a87c8", wraplength=960,
                  justify="left").pack(anchor="w", pady=(6, 4))
        self.list = MatchList(body, app, score_label="Probability", on_status=self.state.set)
        self.list.pack(fill="both", expand=True)
        app._install_clipboard_fix()
        self.refresh()
        self.win.after(150, self._poll)

    def _summaries(self, scope="all"):
        current = Path(self.app.review_summary.get()).expanduser()
        if scope == "result":
            return [current] if current.is_file() else []
        return sorted({Path(p) for p in self.app._results.values()} | ({current} if current.is_file() else set()))

    def refresh(self):
        summaries, results = self._summaries(), self.q
        kind = "" if self.target_type.get() == "Any sound" else self.target_type.get()

        def work():
            import recognizers
            import review_progress
            try:
                records = review_progress.collect(summaries)
                results.put(("species", (recognizers.examples(records, kind), kind), recognizers.trained(kind)))
            except Exception as exc:  # noqa: BLE001
                results.put(("error", str(exc), None))
        threading.Thread(target=work, daemon=True).start()

    def _show_species(self, found, trained):
        import recognizers
        selected = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        self.species = {}
        order = sorted(found.items(), key=lambda pair: (-len(pair[1]["positive"]), pair[1]["common"]))
        for scientific, entry in order:
            if not entry["positive"]:
                continue
            iid = scientific
            self.species[iid] = dict(entry, scientific=scientific)
            meta = trained.get(scientific)
            missing = recognizers.MIN_POSITIVE - len(entry["positive"])
            status = recognizers.describe(meta) if meta else (
                f"needs {missing} more approved clip(s)" if missing > 0 else "ready to train")
            self.tree.insert("", "end", iid=iid, values=(entry["common"] or scientific, len(entry["positive"]),
                                                       len(entry["negative"]), status))
            if iid in selected:
                self.tree.selection_add(iid)
        ready = sum(len(entry["positive"]) >= recognizers.MIN_POSITIVE for entry in found.values())
        self.state.set(f"{ready} species have enough approved clips to train; {len(trained)} recognizer(s) trained. "
                       "Train → Scan → listen to the sounds it finds (highest probability first) → add them as clips.")

    def _selected(self):
        chosen = self.tree.selection()
        if chosen:
            entry = self.species[chosen[0]]
            self.list.common.set(entry["common"])
            self.list.scientific.set(entry["scientific"])
            self.list.sound_type.set("" if self.target_type.get() == "Any sound" else self.target_type.get())

    def examples_window(self):
        chosen = self.tree.selection()
        if not chosen:
            self.state.set("Select a species first")
            return
        entry = self.species[chosen[0]]
        win = tk.Toplevel(self.win)
        win.title(f"Training examples — {entry['common'] or chosen[0]}")
        win.geometry("920x450")
        ttk.Label(win, text="Positive = target sound. Negative = a different species or sound type, or a wrong ID. "
                  "Double-click an example to inspect its source. Validation keeps recordings from the same day together.",
                  wraplength=890, padding=10).pack(anchor="w")
        table = ttk.Treeview(win, columns=("class", "species", "kind", "source", "at"), show="headings")
        for key, label, width in (("class", "Training role", 100), ("species", "Reviewed species", 230),
                                  ("kind", "Sound type", 110), ("source", "Recording", 230), ("at", "In file", 90)):
            table.heading(key, text=label)
            table.column(key, width=width)
        table.pack(fill="both", expand=True, padx=10, pady=5)
        rows = {}
        for label, records in (("Positive", entry["positive"]), ("Negative", entry["negative"])):
            for row in records:
                iid = table.insert("", "end", values=(label, row.get("Reviewed common") or row.get("Species (common)"),
                    row.get("Sound type") or "unspecified", row.get("Source file"), row.get("Start offset (s)")))
                rows[iid] = row
        def inspect(_event):
            selected = table.selection()
            if selected:
                row = rows[selected[0]]
                if row.get("Source path") and row.get("_summary"):
                    self.app.open_manual_at(row["_summary"], row["Source path"], float(row.get("Start offset (s)") or 0))
        table.bind("<Double-1>", inspect)

    def train(self):
        import recognizers
        chosen = [iid for iid in self.tree.selection()
                  if len(self.species[iid]["positive"]) >= recognizers.MIN_POSITIVE]
        if not chosen:
            messagebox.showinfo("Train", f"Select species with at least {recognizers.MIN_POSITIVE} approved clips",
                                parent=self.win)
            return
        summaries = self._summaries()
        kind = "" if self.target_type.get() == "Any sound" else self.target_type.get()
        entries = dict(self.species)

        def train_next(remaining):
            if not remaining:
                self.refresh()
                return
            scientific = remaining[0]
            self.state.set(f"Training {entries[scientific]['common'] or scientific} …")
            self.app._start(self.app._launch() + ["--recognizer", "train", scientific] + [str(s) for s in summaries]
                            + (["--sound-type", kind] if kind else []),
                            self.train_btn, f"Train recognizer: {entries[scientific]['common'] or scientific}",
                            is_analyze=False, on_done=lambda code, lines: done(code, lines, remaining))

        def done(code, lines, remaining):
            if not self.win.winfo_exists():
                return
            error = next((line[6:] for line in lines if line.startswith("ERROR ")), None)
            if code != 0:
                self.state.set(f"Could not train: {error or f'exit {code}'}")
            train_next(remaining[1:])

        # คลิปที่ไฟล์ต้นฉบับยังไม่มีดัชนีใช้เวกเตอร์ของคลิปจากงาน Auto-checks แทน (ไฟล์ที่มีดัชนีมากขึ้น =
        # ตัวอย่างเสียงพื้นหลังมากขึ้น) จึงไม่ต้องรอทำดัชนีทุกผลชุดก่อนฝึก
        train_next(chosen)

    def scan(self):
        import recognizers
        chosen = self.tree.selection()
        if not chosen:
            messagebox.showinfo("Scan", "Select a species with a trained recognizer", parent=self.win)
            return
        kind = "" if self.target_type.get() == "Any sound" else self.target_type.get()
        model = recognizers.load(chosen[0], kind)
        if model is None:
            messagebox.showinfo("Scan", "Train this species first", parent=self.win)
            return
        summaries = self._summaries(self.scope.get())
        if not summaries:
            messagebox.showinfo("Scan", "Open a result in Review clips first", parent=self.win)
            return
        self.scan_btn.configure(state="disabled")
        self.state.set("Checking the sound index …")
        ensure_index(self.app, self.win, summaries, lambda **why: self._scan(model, summaries, **why))

    def _scan(self, model, summaries, cancelled=False, failed=False):
        if not self.win.winfo_exists():
            return
        if cancelled or failed:
            self.scan_btn.configure(state="normal")
            self.state.set("Scan needs the sound index first" if cancelled else "Sound index failed")
            return
        self.state.set("Scanning …")
        results, name = self.q, model["meta"]["common"] or model["meta"]["scientific"]

        def work():
            import recognizers
            try:
                sources, rows = collect_sources(summaries)
                found = enrich(recognizers.scan(model, list(sources))[:300], sources, sound_index.clip_spans(rows))
                for match in found:
                    match["note"] = f"Found by your {name} recognizer (p = {match['score']:.2f})"
                    match["sound_type"] = model["meta"].get("sound_type", "")
                results.put(("scan", found, len(sources)))
            except Exception as exc:  # noqa: BLE001
                results.put(("error", str(exc), None))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, payload, extra = self.q.get_nowait()
                if kind == "species":
                    pool, target_kind = payload
                    current_kind = "" if self.target_type.get() == "Any sound" else self.target_type.get()
                    if target_kind == current_kind:
                        self._show_species(pool, extra)
                elif kind == "scan":
                    self.scan_btn.configure(state="normal")
                    self.list.set_matches(payload)
                    if payload:
                        self.list.sound_type.set(payload[0].get("sound_type", ""))
                    fresh = sum(not same_species(match, self.list.scientific.get()) for match in payload)
                    self.state.set(f"{len(payload)} sound(s) in {extra} recording(s) above the recognizer's threshold — "
                                   f"{fresh} without a clip of this species yet (gray = already a clip). Listen before "
                                   "adding: a recognizer trained on few clips makes mistakes.")
                else:
                    self.scan_btn.configure(state="normal")
                    self.state.set(f"Failed: {payload}")
        except queue.Empty:
            pass
        self.win.after(150, self._poll)


def _clip_text(clips):
    """คลิปของชนิดนี้ในผลชุด -> '12 clips (5 ✓)' หรือ 'none'"""
    if not clips or not clips.get("clips"):
        return "none"
    return f"{clips['clips']} clip{'s' if clips['clips'] != 1 else ''}" + (
        f" ({clips['approved']} ✓)" if clips.get("approved") else "")


def _geo_text(probability):
    if probability is None:
        return "unknown"
    import near_misses
    return f"{probability:.0%}" if probability >= near_misses.GEO_EXPECTED else f"not expected ({probability:.1%})"


class NearMissWindow:
    """เสียงที่ BirdNET เกือบเจอในผลชุดนี้ (คะแนน 0.2 ขึ้นไปแต่ไม่ได้ตัดเป็นคลิปของชนิดนั้น) + ผลฟังซ้ำแบบวางหน้าต่าง
    ตรงเสียง (near_misses.py) -> เลือกชนิด ฟัง แล้วเพิ่มเป็นคลิป หรือเพิ่มทุกเสียงที่ฟังซ้ำผ่านทีเดียว"""

    def __init__(self, app, summary, species=None):
        self.app, self.summary = app, Path(summary).resolve()
        self.wanted = species
        self.found = None
        self.species = {}
        win = self.win = tk.Toplevel(app.root)
        win.title(f"Near misses — {self.summary.parent.name}")
        win.geometry(f"1080x{min(880, win.winfo_screenheight() - 70)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="BirdNET score ≥").pack(side="left")
        self.floor = tk.DoubleVar(value=0.3)
        self.floor_text = tk.StringVar(value="0.30")
        ttk.Scale(top, from_=0.2, to=0.5, variable=self.floor, length=120,
                  command=lambda value: (self.floor_text.set(f"{float(value):.2f}"), self._refilter())).pack(side="left")
        ttk.Label(top, textvariable=self.floor_text, width=5).pack(side="left", padx=(4, 10))
        self.unexpected = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Also species not expected here", variable=self.unexpected,
                        command=self._refilter).pack(side="left")
        self.find_btn = ttk.Button(top, text="Find again", command=self.find)
        self.find_btn.pack(side="left", padx=(10, 0))
        self.add_all_btn = ttk.Button(top, text="✚ Add all that passed the second look", style="QuickAction.TButton",
                                      command=self.add_passed)
        self.add_all_btn.pack(side="left", padx=(10, 0))
        self.state = tk.StringVar()
        ttk.Label(body, textvariable=self.state, foreground="#3a87c8", wraplength=1030,
                  justify="left").pack(anchor="w", pady=(6, 4))
        columns = ("species", "best", "look", "sounds", "here", "result", "checklist")
        self.tree = ttk.Treeview(body, columns=columns, show="headings", height=8)
        for key, label, width in (("species", "Species", 200), ("best", "BirdNET best", 86),
                                  ("look", "Second look", 110), ("sounds", "Sounds", 60),
                                  ("here", "Here & now?", 120), ("result", "Clips in this result", 130),
                                  ("checklist", "Country checklist", 260)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "checklist"))
        self.tree.tag_configure("new", font=("Helvetica", 12, "bold"))
        self.tree.tag_configure("unexpected", foreground="#808080")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._species_selected())
        self.list = MatchList(body, app, score_label="Score", on_status=self.state.set,
                              strong=0.5, strong_label="Select passed second look")
        self.list.pack(fill="both", expand=True, pady=(8, 0))
        app._install_clipboard_fix()
        self.show_cached(first=True)

    # ---------- ข้อมูล ----------
    def find(self):
        self.find_btn.configure(state="disabled")
        self.state.set("Finding near misses and listening again … (in the job queue; needs the sound index, "
                       "made once per recording, ~2 min per hour of audio)")
        self.app._start(self.app._launch() + ["--sound-index", "near", str(self.summary)], self.find_btn,
                        f"Near misses — {self.summary.parent.name}", is_analyze=False, on_done=self._found)

    def _found(self, code, lines):
        if not self.win.winfo_exists():
            return
        self.find_btn.configure(state="normal")
        error = next((line[6:] for line in lines if line.startswith("ERROR ")), None)
        if code != 0:
            self.state.set(f"Could not find near misses: {error or f'exit {code}'} — see Show details")
            return
        self.show_cached()

    def show_cached(self, first=False):
        import near_misses
        found = near_misses.load_near(self.summary)
        if found is None:
            if first:
                self.find()
            return
        self.found = found
        self.list.strong = found["min_conf"]                   # ตัวหนา/Select passed = ฟังซ้ำถึง min_conf
        self.sources, rows = collect_sources([self.summary])
        self.spans = sound_index.clip_spans(rows)
        self.clips = near_misses.species_clips(rows)           # คลิปตอนนี้ (อาจเพิ่งเพิ่มหลังหาเสียง)
        self._refilter()

    def _events(self, item):
        """เสียงของชนิดนี้ที่ยังไม่มีคลิปของชนิดนั้น: คะแนน >= ที่เลือก หรือฟังซ้ำผ่าน"""
        floor, passed = float(self.floor.get()), self.found["min_conf"]
        return [event for event in item["events"] if event["sid"] in self.sources
                and (event["score"] >= floor or (event.get("look") or 0.0) >= passed)]

    def _refilter(self):
        if self.found is None:
            return
        import near_misses
        selected = self.tree.selection()
        keep = selected[0] if selected else None
        self.tree.delete(*self.tree.get_children())
        self.species = {}
        min_conf = self.found["min_conf"]
        for item in self.found["species"]:
            expected = item["geo"] is not None and item["geo"] >= near_misses.GEO_EXPECTED
            if not expected and not self.unexpected.get():
                continue
            events = self._events(item)
            if not events:
                continue
            iid = item["scientific"]
            self.species[iid] = dict(item, shown=events)
            clips = self.clips.get(item["scientific"])
            look = item.get("look") or 0.0
            tags = () if expected else ("unexpected",)
            if not clips and expected:
                tags += ("new",)                            # ชนิดที่ผลชุดนี้ยังไม่มีคลิปเลย
            self.tree.insert("", "end", iid=iid, tags=tags, values=(
                item["common"], f"{item['best']:.2f}",
                (f"{look:.2f} ✓ ×{item['passed']}" if item["passed"] else f"{look:.2f}") if look else "—",
                len(events), _geo_text(item["geo"]), _clip_text(clips), item.get("checklist") or ""))
        passed = sum(item["passed"] for item in self.species.values())
        new = sum(1 for iid in self.species if not self.clips.get(iid)
                  and "unexpected" not in self.tree.item(iid, "tags"))
        self.state.set(f"{len(self.species)} species BirdNET heard (score ≥ {self.floor_text.get()}) at moments without "
                       f"a clip of that species — {new} have no clip in this result at all (bold). Second look: "
                       f"{passed} sound(s) reached {min_conf:g} when BirdNET listened again with the window centred on "
                       f"the sound (made {self.found['made'].replace('T', ' ')}). Listen before adding — a near miss is "
                       "only a hint.")
        wanted, self.wanted = self.wanted, None
        if wanted:
            self.select_species(wanted)
        elif not (keep and self.select_species(keep, quiet=True)) and self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])

    def select_species(self, name, quiet=False):
        """เลือกชนิดตามชื่อวิทยาศาสตร์หรือชื่อสามัญ (จากหน้าต่าง Expected species)"""
        key = str(name).casefold()
        for iid, item in self.species.items():
            if key in (iid.casefold(), item["common"].casefold()):
                self.tree.selection_set(iid)
                self.tree.see(iid)
                return True
        if not quiet:
            self.state.set(f"No near misses of {name} in this result (BirdNET score below {self.floor_text.get()}, "
                           "not expected here, or it already has a clip at those moments) — try a lower score or "
                           "tick Also species not expected here")
        return False

    def _matches(self, item):
        matches = enrich([dict(event) for event in item["shown"]], self.sources, self.spans)
        for match in matches:
            look, grid = match.get("look") or 0.0, match["score"]
            way = {"centred": "centred", "isolated": "on its own"}.get(match.get("look_how"), "")
            match["guess"] = f"{grid:.2f} → {look:.2f} ({way})" if look else f"{grid:.2f}"
            match["score"] = max(grid, look)
            match["note"] = (f"Near miss: BirdNET {grid:.2f} in its usual 3-s windows"
                             + (f"; second look {look:.2f} ({way})" if look else ""))
        matches.sort(key=lambda match: -match["score"])
        return matches

    def _species_selected(self):
        chosen = self.tree.selection()
        if not chosen or chosen[0] not in self.species:
            return
        item = self.species[chosen[0]]
        self.list.set_matches(self._matches(item))
        self.list.common.set(item["common"])
        self.list.scientific.set(item["scientific"])

    # ---------- เพิ่มทุกเสียงที่ฟังซ้ำผ่าน ----------
    def add_passed(self):
        if self.found is None:
            return
        min_conf = self.found["min_conf"]
        items, names = [], {}
        for iid, item in self.species.items():
            if "unexpected" in self.tree.item(iid, "tags"):
                continue
            for match in self._matches(item):
                if (match.get("look") or 0.0) >= min_conf and not same_species(match, item["scientific"]):
                    made = batch_item(match, item["common"], item["scientific"], iid=None)
                    if made:
                        items.append(made)
                        names[item["common"]] = names.get(item["common"], 0) + 1
        if not items:
            messagebox.showinfo("Add clips", f"No sound without a clip reached {min_conf:g} on the second look",
                                parent=self.win)
            return
        listing = ", ".join(f"{name} ×{count}" for name, count in sorted(names.items(), key=lambda kv: -kv[1]))
        if not messagebox.askokcancel("Add clips to review?",
                                      f"Add {len(items)} clip(s) as Pending clips to review: {listing}?\n\n"
                                      "BirdNET's second look is only a hint — listen to each clip and reject "
                                      "wrong IDs.", parent=self.win):
            return
        for number, item in enumerate(items):
            item["_iid"] = number
        self.state.set(f"Adding {len(items)} clip(s) …" + (" (waits for the running job)" if self.app.job_running else ""))
        start_batch(self.app, items, self.add_all_btn, f"{len(names)} species (second look)", self._added)

    def _added(self, code, lines, order, what):
        added = [json.loads(line[7:]) for line in lines if line.startswith("MANUAL ")]
        reload_review(self.app, added)
        if not self.win.winfo_exists():
            return
        self.show_cached()
        text = (f"Added {len(added)} clip(s) of {what} — they are in Review clips (Pending)"
                + (f"; {len(order) - len(added)} skipped (already a clip)" if len(order) > len(added) else ""))
        if code not in (0, None) and not added:
            text = f"Could not add the clips (exit {code}) — see Show details"
        self.state.set(text)
        self.app.status.set(text)


class ExpectedSpeciesWindow:
    """ชนิดที่โมเดลพื้นที่ของ BirdNET ว่าควรพบที่จุด/สัปดาห์ของผลชุดนี้
    เทียบกับคลิปที่มีในผลชุด และคะแนนสูงสุดที่ BirdNET ได้ยินในดัชนีเสียง — ไว้ดูว่าควรฟังหาชนิดไหนเพิ่ม"""
    SHOW = ("Expected this week", "Expected at any time of year", "Without a clip (this week)",
            "Heard but not clipped")

    def __init__(self, app, summary):
        self.app, self.summary = app, Path(summary).resolve()
        self.found = None
        win = self.win = tk.Toplevel(app.root)
        win.title(f"Expected species here — {self.summary.parent.name}")
        win.geometry(f"900x{min(760, win.winfo_screenheight() - 70)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Show").pack(side="left")
        self.show = tk.StringVar(value=self.SHOW[0])
        box = ttk.Combobox(top, textvariable=self.show, values=self.SHOW, state="readonly", width=28)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda _event: self._fill())
        self.refresh_btn = ttk.Button(top, text="Refresh", command=self.find)
        self.refresh_btn.pack(side="left", padx=(6, 0))
        self.state = tk.StringVar(value="Asking BirdNET's location/date model … (in the job queue)")
        ttk.Label(body, textvariable=self.state, foreground="#3a87c8", wraplength=860,
                  justify="left").pack(anchor="w", pady=(6, 4))
        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True)
        columns = ("species", "week", "year", "result", "heard", "checklist")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings")
        for key, label, width in (("species", "Species", 210), ("week", "This week", 76), ("year", "Peak of year", 90),
                                  ("result", "Clips in this result", 130), ("heard", "BirdNET heard (best)", 130),
                                  ("checklist", "Country checklist", 200)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=(key == "checklist"))
        self.tree.tag_configure("clipped", foreground="#808080")
        self.tree.tag_configure("heard", font=("Helvetica", 12, "bold"))
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="left", fill="y")
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())
        ttk.Label(body, text="Probabilities are from BirdNET's location/date model (eBird-based), not proof that a "
                  "species is present. Bold = BirdNET heard it (score ≥ 0.2) but this result has no clip of it — "
                  "double-click to open its near misses; gray = already has clips (double-click shows them in Review).",
                  foreground="#808080", wraplength=860, justify="left").pack(anchor="w", pady=(6, 0))
        import near_misses
        cached = near_misses.load_expected(self.summary)
        if cached is not None:
            self._show(cached)
        self.find()

    def find(self):
        self.refresh_btn.configure(state="disabled")
        self.app._start(self.app._launch() + ["--sound-index", "expected", str(self.summary)], self.refresh_btn,
                        f"Expected species — {self.summary.parent.name}", is_analyze=False, on_done=self._found)

    def _found(self, code, lines):
        if not self.win.winfo_exists():
            return
        self.refresh_btn.configure(state="normal")
        error = next((line[6:] for line in lines if line.startswith("ERROR ")), None)
        if code != 0:
            self.state.set(f"Could not ask the location/date model: {error or f'exit {code}'}")
            return
        import near_misses
        found = near_misses.load_expected(self.summary)
        if found is not None:
            self._show(found)

    def _show(self, found):
        import near_misses
        self.found = found
        rows = load_reviews(self.summary)
        self.clips = near_misses.species_clips(rows)
        self._fill()

    def _fill(self):
        import near_misses
        if self.found is None:
            return
        show = self.show.get()
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        expected = near_misses.GEO_EXPECTED
        for item in self.found["species"]:
            clips = self.clips.get(item["scientific"])
            heard = item["heard"] >= near_misses.NEAR_FLOOR
            if show == self.SHOW[1]:
                wanted = item["year"] >= expected
            elif show == self.SHOW[2]:
                wanted = item["week"] >= expected and not clips
            elif show == self.SHOW[3]:
                wanted = heard and not clips
            else:
                wanted = item["week"] >= expected
            if not wanted:
                continue
            iid = item["scientific"]
            self.items[iid] = item
            tags = ("clipped",) if clips else ("heard",) if heard else ()
            self.tree.insert("", "end", iid=iid, tags=tags, values=(
                item["common"], f"{item['week']:.0%}", f"{item['year']:.0%}", _clip_text(clips),
                f"{item['heard']:.2f}" if item["heard"] else "—", item.get("checklist") or ""))
        week = sum(item["week"] >= expected for item in self.found["species"])
        with_clips = sum(1 for item in self.found["species"] if item["week"] >= expected
                         and self.clips.get(item["scientific"]))
        heard_only = sum(1 for item in self.found["species"] if item["week"] >= expected
                         and item["heard"] >= near_misses.NEAR_FLOOR and not self.clips.get(item["scientific"]))
        start = self.found["start"].replace("T", " ")[:16]
        index_note = ("" if self.found.get("indexed") == self.found.get("recordings") else
                      f" (sound index made for {self.found.get('indexed', 0)} of {self.found.get('recordings', 0)} "
                      "recordings — open Near misses to make the rest)")
        self.state.set(f"{week} species expected at {self.found['lat']:.4f}, {self.found['lon']:.4f} in the week of "
                       f"{start}: {with_clips} have clips in this result, {heard_only} were heard by BirdNET without "
                       f"a clip (bold){index_note}. Showing {len(self.items)}.")

    def open_selected(self):
        chosen = self.tree.selection()
        if not chosen:
            return
        item = self.items[chosen[0]]
        if self.clips.get(item["scientific"]):
            current = Path(self.app.review_summary.get()).expanduser()
            if not (current.is_file() and current.resolve() == self.summary):
                self.app.review_summary.set(str(self.summary))
                self.app.review_load()
            self.app.review_species_filter.set(item["common"])
            self.app._fill_review_tree()
            self.app.nb.select(self.app.review_tab)
            self.app.status.set(f"Showing {item['common']} clips")
            return
        self.app.open_near_misses(item["scientific"])
