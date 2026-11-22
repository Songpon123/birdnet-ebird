"""Preview detections from an external CSV and pass named selections to human Review."""

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

from event_window import CandidateWindow, source_entry
from detections_import import read_candidates
from sound_search import enrich
import sound_index


class DetectionsImportWindow(CandidateWindow):
    def __init__(self, app):
        super().__init__(app, "Import detections", "Imported")
        self.q = queue.Queue()
        self.csv, self.audio_root = tk.StringVar(), tk.StringVar()
        for label, var, command in (("CSV file", self.csv, self.choose_csv),
                                    ("Audio folder (optional remap)", self.audio_root, self.choose_folder)):
            row = ttk.Frame(self.body)
            row.pack(fill="x", before=self.context_bar, pady=3)
            ttk.Label(row, text=label, width=28).pack(side="left")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
            ttk.Button(row, text="Browse…", command=command).pack(side="left")
        self.load_btn = ttk.Button(self.top, text="Preview detections", command=self.load)
        self.load_btn.pack(side="left")
        ttk.Button(self.top, text="Skipped rows…", command=self.show_errors).pack(side="left", padx=8)
        self.errors = []
        self.state.set("Select cluster.csv or a compatible birdnet.csv. Preview resolves source audio and channels. "
                       "Listen, select a species, then Add selected. Imported labels remain unverified and clips Pending.")
        self.win.after(150, self.poll)

    def choose_csv(self):
        path = filedialog.askopenfilename(parent=self.win, filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if path:
            self.csv.set(path)

    def choose_folder(self):
        path = filedialog.askdirectory(parent=self.win, title="Folder containing original recordings")
        if path:
            self.audio_root.set(path)

    def load(self):
        source, audio_root = self.csv.get(), self.audio_root.get().strip() or None
        if not source:
            self.state.set("Choose a CSV file first")
            return
        self.load_btn.configure(state="disabled")
        self.state.set("Reading detections and validating source audio …")
        results, summary, known, rows = self.q, self.summary, dict(self.sources), list(self.rows)

        def work():
            try:
                imported = read_candidates(source, audio_root)
                matches, by_path = [], {}
                for candidate in imported["candidates"]:
                    path = candidate["source"]
                    if path not in by_path:
                        sid, entry = source_entry(path, summary, rows)
                        by_path[path] = sid
                        known.setdefault(sid, entry)
                    matches.append(dict(candidate, sid=by_path[path]))
                enrich(matches, known, sound_index.clip_spans(rows))
                for match in matches:
                    match["guess"] = "CSV label: " + match["label"] if match["label"] else "unlabeled"
                results.put((matches, known, imported, None))
            except Exception as exc:  # noqa: BLE001
                results.put(([], {}, {}, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                matches, sources, imported, error = self.q.get_nowait()
                self.load_btn.configure(state="normal")
                if error:
                    self.state.set(f"Cannot import: {error}")
                    continue
                self.sources, self.errors = sources, imported["errors"]
                self.list.set_matches(matches)
                self.state.set(f"{len(matches)} detections ready to inspect; {len(self.errors)} rows skipped; "
                               f"{imported['duplicates']} duplicates skipped. CSV labels are hints. "
                               "Select sounds of one species, enter its name, then add to Review.")
        except queue.Empty:
            pass
        self.win.after(150, self.poll)

    def show_errors(self):
        win = tk.Toplevel(self.win)
        win.title("Skipped CSV rows")
        text = tk.Text(win, width=110, height=24, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", "\n".join(self.errors) or "No skipped rows")
        text.configure(state="disabled")
