"""Candidate sounds from signal detection, including files with no BirdNET results."""

from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import manual_clips
import paths
import result_folders
import signal_events
from sound_search import MatchList, collect_sources, enrich


def source_entry(source, summary, rows=()):
    """Resolve recording context without inventing missing date or coordinates."""
    import field_audio_to_ebird as fa
    source = Path(source).resolve()
    context = manual_clips.source_context(rows, source)
    if context is None:
        meta = fa.recording_context_preview(source)
        context = dict(start=meta.get("datetime") if meta.get("has_time") else None,
                       lat=meta.get("lat"), lon=meta.get("lon"), place=meta.get("place") or "",
                       date_source=meta.get("datetime_source") or "")
    return fa.source_id(source), dict(path=source, name=source.name, summary=Path(summary),
                                     rows=list(rows), context=context)


class CandidateWindow:
    """Shared source/context editor and candidate list used by scans and CSV imports."""
    def __init__(self, app, title, score_label):
        self.app, self.sources = app, {}
        self.rows = []
        self.summary = Path(app.review_summary.get()).expanduser() if app.review_summary.get().strip() else None
        if self.summary is not None and self.summary.is_file():
            self.sources, self.rows = collect_sources([self.summary])
        else:
            self.summary = result_folders.folder_for(Path(app.out.get()).expanduser(), datetime.now(),
                                                     "_sound_events", seconds=True) / "summary.xlsx"
        win = self.win = tk.Toplevel(app.root)
        win.title(title)
        win.geometry(f"1080x{min(900, win.winfo_screenheight() - 70)}")
        self.body = ttk.Frame(win, padding=12)
        self.body.pack(fill="both", expand=True)
        self.top = ttk.Frame(self.body)
        self.top.pack(fill="x")
        self.state = tk.StringVar(value="Candidate sounds need listening and a species name before adding to Review.")
        self.context_bar = ttk.Frame(self.body)
        self.context_bar.pack(fill="x", pady=6)
        self.date, self.time, self.coords = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for label, var, width in (("Recording date YYYYMMDD", self.date, 11), ("Start HHMM", self.time, 8),
                                  ("Coordinates lat,lon", self.coords, 25)):
            ttk.Label(self.context_bar, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(self.context_bar, textvariable=var, width=width).pack(side="left", padx=(0, 10))
        ttk.Button(self.context_bar, text="Use details for selected source", command=self.apply_context).pack(side="left")
        ttk.Label(self.body, textvariable=self.state, foreground="#3a87c8", wraplength=1000,
                  justify="left").pack(anchor="w", pady=5)
        self.list = MatchList(self.body, app, score_label=score_label, on_status=self.state.set)
        self.list.pack(fill="both", expand=True)
        self.list.tree.bind("<<TreeviewSelect>>", self.show_context, add="+")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        app._install_clipboard_fix()

    def close(self):
        if self.list.view._play is not None:
            self.app.review_stop()
        self.win.destroy()

    def selected_sid(self):
        selected = self.list.tree.selection()
        if selected and selected[0] in self.list.matches:
            return self.list.matches[selected[0]]["sid"]
        return next(iter(self.sources), None)

    def show_context(self, _event=None):
        sid = self.selected_sid()
        if sid not in self.sources:
            return
        context = self.sources[sid]["context"]
        start = context.get("start")
        self.date.set(start.strftime("%Y%m%d") if start else "")
        self.time.set(start.strftime("%H%M%S") if start else "")
        self.coords.set(f"{context['lat']},{context['lon']}" if context.get("lat") is not None else "")

    def apply_context(self):
        from entry_formats import parse_date_entry, parse_time_entry
        from field_audio_to_ebird import parse_coords
        sid = self.selected_sid()
        if sid not in self.sources:
            return
        try:
            day = parse_date_entry(self.date.get())
            clock = parse_time_entry(self.time.get())
            coords = parse_coords(self.coords.get())
            if coords is None:
                raise ValueError("Enter valid recording coordinates, e.g. 13.8,100.5")
            lat, lon = coords
            start = datetime.combine(day.date(), clock)
        except (ValueError, TypeError) as exc:
            messagebox.showwarning("Recording details", str(exc), parent=self.win)
            return
        context = self.sources[sid]["context"]
        context.update(start=start, lat=lat, lon=lon, date_source="user recording details")
        for match in self.list.matches.values():
            if match["sid"] == sid:
                match["context"] = context
                match["clock"] = start + timedelta(seconds=match["start"])
        for iid, match in self.list.matches.items():
            if match["sid"] == sid:
                self.list.tree.set(iid, "clock", match["clock"].strftime("%Y-%m-%d %H:%M:%S"))
        self.state.set("Recording details set. Listen, choose the species, then add selected sounds to Review.")


class SignalEventsWindow(CandidateWindow):
    def __init__(self, app):
        super().__init__(app, "Find sound events — independent of BirdNET", "Excess dB")
        self.source = tk.StringVar()
        self.source_box = ttk.Combobox(self.top, textvariable=self.source, state="readonly", width=55)
        self.source_box.pack(side="left", padx=(0, 6))
        self.source_box.bind("<<ComboboxSelected>>", lambda _e: self.show_context())
        self._names = {}
        self.refresh_sources()
        ttk.Button(self.top, text="Browse audio…", command=self.browse).pack(side="left")
        self.scan_btn = ttk.Button(self.top, text="Find events", command=self.scan)
        self.scan_btn.pack(side="left", padx=8)
        options = ttk.Frame(self.body)
        options.pack(fill="x", before=self.context_bar, pady=6)
        self.params = {}
        for label, key, value, width in (("Low Hz", "low", 150, 7), ("High Hz", "high", 12000, 7),
                                         ("Min s", "minimum", .05, 6), ("Max s", "maximum", 15, 6),
                                         ("Gap s", "gap", .3, 6), ("Above background dB", "excess_db", 12, 6)):
            ttk.Label(options, text=label).pack(side="left", padx=(0, 3))
            var = self.params[key] = tk.StringVar(value=str(value))
            ttk.Entry(options, textvariable=var, width=width).pack(side="left", padx=(0, 8))
        self.channel = tk.StringVar(value="mix")
        ttk.Combobox(options, textvariable=self.channel, values=("mix", "left", "right"), state="readonly",
                     width=6).pack(side="left")
        self.force = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.body, text="Re-scan instead of using cached events", variable=self.force).pack(
            anchor="w", before=self.context_bar)
        self.state.set("Finds time/frequency events without a species model. Adjust parameters and listen; "
                       "wind, insects and other sounds may also be detected.")
        self.show_context()

    def selected_sid(self):
        return getattr(self, "_names", {}).get(self.source.get()) if hasattr(self, "source") else None

    def refresh_sources(self):
        self._names = {f"{entry['name']} · {sid}": sid for sid, entry in self.sources.items()}
        self.source_box.configure(values=list(self._names))
        if self.source.get() not in self._names:
            self.source.set(next(iter(self._names), ""))

    def browse(self):
        file = filedialog.askopenfilename(parent=self.win, title="Audio to scan", filetypes=[
            ("Audio", "*.wav *.WAV *.flac *.mp3 *.m4a *.ogg *.aiff"), ("All files", "*")])
        if not file:
            return
        try:
            sid, entry = source_entry(file, self.summary)
            self.sources[sid] = entry
            self.refresh_sources()
            self.source.set(next(name for name, key in self._names.items() if key == sid))
            self.show_context()
        except (OSError, ValueError, RuntimeError) as exc:
            self.state.set(str(exc))

    def scan(self):
        sid = self.selected_sid()
        if sid not in self.sources:
            self.state.set("Choose an audio recording first")
            return
        try:
            params = signal_events.DetectionParams(**{key: float(var.get()) for key, var in self.params.items()},
                                                   channel=self.channel.get()).validate()
        except ValueError as exc:
            self.state.set(str(exc))
            return
        paths.CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="event-job-", dir=paths.CACHE,
                                         encoding="utf-8", delete=False) as f:
            json.dump(dict(sources=[dict(sid=sid, path=str(self.sources[sid]["path"]))], params=asdict(params),
                           force=self.force.get()), f)
            job = Path(f.name)
        self.state.set("Scanning audio … (in the job queue; Stop stops the current job)")
        self.app._start(self.app._launch() + ["--signal-events", str(job)], self.scan_btn,
                        f"Sound events — {self.sources[sid]['name']}", is_analyze=False,
                        on_done=lambda code, lines: self.scanned(code, lines, job, sid, params))

    def scanned(self, code, lines, job, sid, params):
        job.unlink(missing_ok=True)
        if not self.win.winfo_exists():
            return
        self.scan_btn.configure(state="normal")
        if code:
            self.state.set(next((line[6:] for line in lines if line.startswith("ERROR ")), "Scan stopped or failed"))
            return
        try:
            result = json.loads(signal_events.cache_file(sid, params).read_text())
            self.source.set(next(name for name, key in self._names.items() if key == sid))
            self.show_context()
            import sound_index
            found = enrich(result["events"], self.sources, sound_index.clip_spans(self.rows))
            for event in found:
                event["note"] = (f"Found by signal detection: {event['low']}–{event['high']} Hz, "
                                 f"{event['score']:.1f} dB above local background; channel {params.channel}. "
                                 "Species selected by reviewer.")
            self.list.set_matches(found)
            self.state.set(f"{len(found)} candidate sounds. Listen and name selected sounds before adding; "
                           "new clips remain Pending. Missing date/location can be filled above.")
        except (OSError, ValueError) as exc:
            self.state.set(str(exc))
