"""Named analysis presets: an explicit whitelist excludes credentials and destructive flags."""

import json
import math
import os
from pathlib import Path
import tempfile

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import paths

STORE = paths.SETTINGS_FILE.parent / "analysis-presets.json"
NUMBERS = {"min_conf": (0, 1), "overlap": (0, 2.99), "gap": (0, 3600), "lead": (0, 600),
           "tail": (0, 600), "dbfs": (-60, 0), "out_of_range_min_conf": (0, 1)}
BOOLS = ("mono", "spec", "alt", "unknown", "keep_out_of_range", "continuous", "second_look")
ALLOWED = set(NUMBERS) | set(BOOLS) | {"model", "habitats", "coords", "place"}


def validate(settings):
    from habitat import SITE_HABITATS
    if not isinstance(settings, dict) or set(settings) - ALLOWED:
        raise ValueError("Preset contains unsupported fields")
    result = {}
    for key, value in settings.items():
        if key in NUMBERS:
            low, high = NUMBERS[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}: expected {low}–{high}")
        elif key in BOOLS:
            if not isinstance(value, bool):
                raise ValueError(f"Invalid {key}: expected true/false")
        elif key == "model":
            if value not in ("3.0-preview", "3.0-pruned", "2.4"):    # 2.4: ชุดตั้งค่าเก่า
                raise ValueError("Unsupported BirdNET model")
        elif key == "habitats":
            if not isinstance(value, list) or any(x not in SITE_HABITATS for x in value):
                raise ValueError("Unsupported habitat")
        elif key in ("coords", "place"):
            if not isinstance(value, str) or len(value) > 2000:
                raise ValueError(f"Invalid {key}")
            if key == "coords" and value.strip():
                from field_audio_to_ebird import parse_coords
                if parse_coords(value) is None:
                    raise ValueError("Invalid recording coordinates")
        result[key] = value
    return result


def read_store(path=None):
    path = Path(path or STORE)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Preset store is not a JSON object")
    return {name: validate(settings) for name, settings in data.items()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     encoding="utf-8", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(f.name)
    try:
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_preset(name, settings, path=None):
    name = str(name).strip()
    if not name or len(name) > 80:
        raise ValueError("Preset name must be 1–80 characters")
    data = read_store(path)
    data[name] = validate(settings)
    write_json(path or STORE, data)


def snapshot(app, include_site=False):
    result = {key: getattr(app, key).get() for key in NUMBERS.keys() | set(BOOLS) | {"model"}}
    result["habitats"] = [key for key, var in app.habitat_vars.items() if var.get()]
    if include_site:
        result.update(coords=app.coords.get(), place=app.place.get())
    return validate(result)


def apply(app, settings):
    settings = validate(settings)  # validate the whole document before changing any widget
    for key, value in settings.items():
        if key == "habitats":
            for habitat, variable in app.habitat_vars.items():
                variable.set(habitat in value)
        else:
            getattr(app, key).set(value)
    if hasattr(app, "_update_quick_model_hint"):
        app._update_quick_model_hint()


class PresetsWindow:
    def __init__(self, app):
        self.app = app
        win = self.win = tk.Toplevel(app.root)
        win.title("Analysis presets")
        win.geometry("650x420")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Save and restore analysis settings. Applying a preset does not start analysis. "
                  "Files, dates, output folders, API keys and Force redo are excluded.", wraplength=610).pack(anchor="w")
        self.list = tk.Listbox(body, height=9)
        self.list.pack(fill="both", expand=True, pady=8)
        self.site = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Include coordinates and place when saving (site preset)", variable=self.site).pack(anchor="w")
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=8)
        for label, callback in (("Save current…", self.save), ("Apply selected", self.apply),
                                 ("Export…", self.export), ("Import…", self.import_file)):
            ttk.Button(controls, text=label, command=callback).pack(side="left", padx=(0, 6))
        self.status = tk.StringVar()
        ttk.Label(body, textvariable=self.status, foreground="#3a87c8", wraplength=610).pack(anchor="w")
        self.refresh()

    def refresh(self):
        try:
            self.presets = read_store()
            self.list.delete(0, "end")
            for name in sorted(self.presets):
                self.list.insert("end", name)
        except (OSError, ValueError) as exc:
            self.status.set(str(exc))

    def selected(self):
        selection = self.list.curselection()
        return self.list.get(selection[0]) if selection else None

    def save(self):
        name = simpledialog.askstring("Save preset", "Preset name", parent=self.win)
        if not name:
            return
        try:
            save_preset(name, snapshot(self.app, self.site.get()))
            self.refresh()
            self.status.set(f"Saved {name}")
        except (OSError, ValueError, tk.TclError) as exc:
            self.status.set(str(exc))

    def apply(self):
        name = self.selected()
        if name:
            apply(self.app, self.presets[name])
            self.status.set(f"Applied {name}. Check recording details before analysis.")

    def export(self):
        name = self.selected()
        if not name:
            self.status.set("Select a preset first")
            return
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension=".json", initialfile="analysis-preset.json")
        if path:
            try:
                write_json(path, dict(kind="birdnet-analysis-preset", version=1, name=name, settings=self.presets[name]))
                self.status.set("Exported selected preset")
            except (OSError, ValueError) as exc:
                self.status.set(str(exc))

    def import_file(self):
        path = filedialog.askopenfilename(parent=self.win, filetypes=[("Preset JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("kind") != "birdnet-analysis-preset" or data.get("version") != 1:
                raise ValueError("Unsupported preset format")
            save_preset(data["name"], data["settings"])
            self.refresh()
            self.status.set("Imported preset; select Apply to use it")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showwarning("Cannot import preset", str(exc), parent=self.win)
