"""Non-destructive listening: time/frequency selection, channel, tempo and looping.
Two displays: the Prism viewer (prism_view.py, default) or the Macaulay-style spectrogram."""

import queue
import tempfile
import threading
from datetime import timedelta
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from audio_io import audio_duration, audio_info, listening_preview
from manual_clips import format_offset, parse_offset
import playback
from platform_tools import play_audio


DISPLAYS = ("Prism", "Macaulay")
MAX_SPAN = 600.0


class ListeningToolsWindow:
    def __init__(self, master, source, *, app=None, start=0.0, end=None, clock_start=None, species=None, location=None,
                 boxes=None, flags=None):
        from prism_view import PrismView
        from spectrogram_view import SpectrogramView
        self.source, self.app = Path(source), app
        self.total = audio_duration(source)
        self.origin = float(start)
        self.end = min(float(end or self.total), self.origin + MAX_SPAN)
        # clock_start = เวลาจริงที่ตำแหน่ง start -> เก็บเป็นเวลาจริงที่ต้นไฟล์ (เลื่อนช่วงแล้วยังถูก)
        self.file_clock = clock_start - timedelta(seconds=self.origin) if clock_start else None
        self.q, self.player, self.generation = queue.Queue(), None, 0
        self._temp = tempfile.TemporaryDirectory(prefix="birdnet-listen-")
        self._closed = False
        self._cancel = threading.Event()
        self._loop_file = self._loop_start = None
        win = self.win = tk.Toplevel(master)
        win.title(f"{self.source.parent.name}/{self.source.name} — Listening tools")
        win.geometry(f"1100x{min(820, win.winfo_screenheight() - 80)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Display").pack(side="left")
        self.display = tk.StringVar(value=DISPLAYS[0])
        display = ttk.Combobox(top, textvariable=self.display, values=DISPLAYS, state="readonly", width=13)
        display.pack(side="left", padx=(4, 12))
        display.bind("<<ComboboxSelected>>", lambda _event: self.display_changed())
        self.hint = tk.StringVar()
        ttk.Label(top, textvariable=self.hint, wraplength=880, foreground="#808080").pack(side="left")
        self.views = ttk.Frame(body)
        self.views.pack(fill="both", expand=True, pady=8)
        self.channel = tk.StringVar(value="stereo")
        self.speed = tk.StringVar(value="1.0")
        self.macaulay = SpectrogramView(self.views, width=980, height=280, overview_height=28,
                                        on_seek=self.seek, on_select=self.selected, listening_tools=False)
        self.prism = PrismView(self.views, on_select=self.selected, on_play=self.play_view,
                                             on_stop=self.stop, speed_var=self.speed, channel_var=self.channel,
                                             on_channel=self.channel_changed, on_segment=self.segment)
        self.prism.species, self.prism.location = species, location   # ใช้ใน Compare frequencies
        self.prism.boxes = boxes          # Detection boxes ของคลิป (วินาทีนับจากต้นไฟล์) ให้เลือกในหน้าต่างเทียบ
        self.prism.flags = list(flags or ())                 # flag ที่กดตอนอัด (วินาทีนับจากต้นไฟล์นี้)
        self.macaulay.flags = [flag - self.origin for flag in flags or ()]
        self.view = self.prism
        fields = ttk.Frame(body)
        fields.pack(fill="x")
        self.start = tk.StringVar(value=format_offset(self.origin, 2))
        self.finish = tk.StringVar(value=format_offset(min(self.end, self.origin + 6), 2))
        try:
            self.nyquist = audio_info(source)["samplerate"] / 2
        except (OSError, RuntimeError, ValueError):
            self.nyquist = 24000.0
        self.low, self.high = tk.StringVar(value="0"), tk.StringVar(value=f"{self.nyquist:g}")
        for label, variable, width in (("Start in file", self.start, 12), ("End", self.finish, 12),
                                       ("Low Hz", self.low, 8), ("High Hz", self.high, 8)):
            ttk.Label(fields, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(fields, textvariable=variable, width=width).pack(side="left", padx=(0, 12))
        ttk.Button(fields, text="Full frequency range", command=self.reset_band).pack(side="left")
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=(10, 0))
        # ช่อง/ความเร็ว/เล่น ของหน้าเสียงแบบ Macaulay (แบบ Prism มีปุ่มเหล่านี้ในแถบของตัวเอง)
        self.macaulay_controls = ttk.Frame(controls)
        ttk.Label(self.macaulay_controls, text="Channel").pack(side="left")
        channel = ttk.Combobox(self.macaulay_controls, textvariable=self.channel,
                               values=("stereo", "mix", "left", "right"), state="readonly", width=8)
        channel.pack(side="left", padx=(4, 12))
        channel.bind("<<ComboboxSelected>>", self.channel_changed)
        ttk.Label(self.macaulay_controls, text="Speed").pack(side="left")
        speed_box = ttk.Combobox(self.macaulay_controls, textvariable=self.speed,
                                 values=("0.5", "0.75", "1.0", "1.25", "1.5", "2.0"), state="readonly", width=5)
        speed_box.pack(side="left", padx=4)
        speed_box.bind("<<ComboboxSelected>>", lambda _event: self.stop())
        self.play_btn = ttk.Button(self.macaulay_controls, text="▶ Play selection", command=self.play)
        self.play_btn.pack(side="left", padx=5)
        self.macaulay_controls.pack(side="left")
        self.loop = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Loop selection", variable=self.loop).pack(side="left", padx=(6, 4))
        ttk.Button(controls, text="■ Stop", command=self.stop).pack(side="left")
        playback.control(controls).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Close", command=self.close).pack(side="right")
        self.status = tk.StringVar(value=f"File {format_offset(self.origin)}–{format_offset(self.end)}; "
                                        "select up to 10 minutes. Stereo display combines channels.")
        ttk.Label(body, textvariable=self.status, wraplength=1000, foreground="#3a87c8").pack(anchor="w", pady=8)
        self.display_changed()
        win.protocol("WM_DELETE_WINDOW", self.close)
        win.bind("<Escape>", lambda _event: self.stop())
        win.after(80, self._poll)

    def display_changed(self):
        """สลับหน้าเสียง: Prism (ช่องซ้าย/ขวาทีละช่อง) หรือ Macaulay"""
        self.stop()
        prism = self.display.get() == DISPLAYS[0]
        self.view.pack_forget()
        self.view = self.prism if prism else self.macaulay
        self.view.pack(fill="both", expand=True)
        if prism:
            self.macaulay_controls.pack_forget()
            if self.channel.get() in ("stereo", "mix"):
                self.channel.set("left")              # Prism แสดง/เล่นทีละช่อง เริ่มที่ช่องซ้าย
            self.hint.set("Drag a box to select time and frequency, click once to "
                          "clear it, right-click = frequency reference line, ←/→ scroll, Space play, + − = zoom. "
                          "Listening only; speed keeps the pitch.")
        else:
            self.macaulay_controls.pack(side="left", before=self.macaulay_controls.master.winfo_children()[1])
            self.hint.set("Drag a box around a sound to select its time and frequency range. "
                          "Changes affect listening only; speed preserves pitch.")
        self._load_view()

    def _load_view(self):
        if self.view is self.prism:
            from manual_window import index_labels
            self.prism.load(self.source, clock_start=self.file_clock, start=self.origin, end=self.end,
                                   channel=self.channel.get(), labels=index_labels(self.source))
            return
        channel = "mix" if self.channel.get() == "stereo" else self.channel.get()
        clock = self.file_clock + timedelta(seconds=self.origin) if self.file_clock else None
        self.view.load(self.source, clock_start=clock, start=self.origin, end=self.end, channel=channel)

    def segment(self, direction):
        """ปุ่มช่วงก่อน/ถัดไป: ช่วงยาวเท่าเดิมที่อยู่ติดกันในไฟล์เดียวกัน"""
        length = self.end - self.origin
        start = min(max(0.0, self.origin + direction * length), max(0.0, self.total - length))
        if abs(start - self.origin) < 0.01:
            self.status.set("This is the " + ("start" if direction < 0 else "end") + " of the recording")
            return
        self.stop()
        self.origin, self.end = start, min(self.total, start + length)
        self.start.set(format_offset(self.origin, 2))
        self.finish.set(format_offset(min(self.end, self.origin + 6), 2))
        self.status.set(f"File {format_offset(self.origin)}–{format_offset(self.end)} "
                        f"(span {int(self.origin // length) + 1} of {max(1, round(self.total / length))})")
        self._load_view()

    def play_view(self):
        """ปุ่ม Play ของหน้าเสียง Prism: กรอบที่เลือก หรือจากขอบซ้ายของภาพถึงท้ายช่วง (ทุกความถี่)"""
        if self.prism.selection is None:
            self.start.set(format_offset(self.origin + self.prism.t0, 2))
            self.finish.set(format_offset(self.end, 2))
            self.reset_band()
        self.play()

    def channel_changed(self, _event=None):
        self.stop()
        self._load_view()

    def selected(self, a, b, low, high):
        self.stop()
        self.start.set(format_offset(self.origin + a, 2))
        self.finish.set(format_offset(self.origin + b, 2))
        self.low.set(f"{low:.0f}")
        self.high.set(f"{high:.0f}")

    def reset_band(self):
        self.stop()
        self.low.set("0")
        self.high.set(f"{self.nyquist:g}")
        self.view.clear_selection()

    def seek(self, seconds):
        self.start.set(format_offset(self.origin + seconds, 2))
        try:
            if parse_offset(self.finish.get()) <= self.origin + seconds:
                self.finish.set(format_offset(min(self.end, self.origin + seconds + 6), 2))
        except ValueError:
            self.finish.set(format_offset(self.end, 2))
        self.play()

    def play(self):
        try:
            a, b = parse_offset(self.start.get()), parse_offset(self.finish.get())
            low, high, speed = float(self.low.get()), float(self.high.get()), float(self.speed.get())
            if not self.origin <= a < b <= self.end + 0.01:
                raise ValueError("Start and end must be inside the displayed span")
        except (ValueError, TypeError) as exc:
            messagebox.showwarning("Listening selection", str(exc), parent=self.win)
            return
        self.stop()
        generation, results = self.generation, self.q
        target = Path(self._temp.name) / f"listen-{generation}.wav"
        source, channel, cancelled = self.source, self.channel.get(), self._cancel
        if channel == "right" and not self._stereo():
            channel = "mix"
        self.play_btn.configure(state="disabled")
        self.status.set("Preparing listening preview …")

        def work():
            try:
                if cancelled.is_set():
                    return
                listening_preview(source, a, b, target, channel, low, high, speed)
                if not cancelled.is_set():
                    results.put((generation, target, a, b, speed, None))
            except Exception as exc:  # noqa: BLE001  display decoding/filter errors in the window
                if not cancelled.is_set():
                    results.put((generation, target, a, b, speed, str(exc)))
            finally:
                if cancelled.is_set():
                    try:
                        target.unlink(missing_ok=True)
                        target.parent.rmdir()
                    except OSError:
                        pass
        threading.Thread(target=work, daemon=True).start()

    def _stereo(self):
        try:
            return audio_info(self.source)["channels"] > 1
        except (OSError, RuntimeError, ValueError):
            return False

    def _start_player(self, target, a, b, speed):
        if self.app is not None:
            self.app.review_stop()
        self.player = play_audio(target)
        if self.app is not None:
            self.app.audio_player = self.player
        self._loop_file, self._loop_start = target, (a, b, speed)
        self.view.start_playhead(a - self.origin, self.player, speed=speed)
        self.status.set(f"Listening {format_offset(a)}–{format_offset(b)} · {self.channel.get()} · {speed:g}×")

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                generation, target, a, b, speed, error = self.q.get_nowait()
                if generation != self.generation:
                    target.unlink(missing_ok=True)
                    continue
                self.play_btn.configure(state="normal")
                if error:
                    self.status.set(f"Cannot play: {error}")
                else:
                    try:
                        self._start_player(target, a, b, speed)
                    except (OSError, RuntimeError) as exc:
                        self.status.set(f"Cannot play: {exc}")
        except queue.Empty:
            pass
        if self.player is not None and self.player.poll() is not None:
            code = self.player.returncode
            self.player = None
            if code == 0 and self.loop.get() and self._loop_file is not None:
                try:
                    self._start_player(self._loop_file, *self._loop_start)
                except (OSError, RuntimeError) as exc:
                    self.status.set(str(exc))
                    self._loop_file = None
        self.win.after(80, self._poll)

    def stop(self):
        self.generation += 1  # a late decoding result must never start playing
        if self.player is not None and self.player.poll() is None:
            self.player.terminate()
        self.player = None
        self._loop_file = None
        self.view.stop_playhead()
        self.play_btn.configure(state="normal")

    def close(self):
        self.stop()
        self._cancel.set()
        self._closed = True
        self.win.destroy()
        self._temp.cleanup()
