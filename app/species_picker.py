"""ช่องชื่อชนิดแบบค้นได้ (แท็บ Review): คลิกช่อง = ตัวเลือกของคลิปนี้ก่อน (second opinion, ชนิดอื่นที่ BirdNET เดา)
พิมพ์ส่วนหนึ่งของชื่ออังกฤษ/วิทยาศาสตร์ = ค้นจากระบบชื่อ eBird (ไม่ต้องต่อเน็ต) ชนิดที่พบในประเทศที่บันทึกขึ้นก่อน
↓ เลื่อนลงไปเลือก, Enter/คลิก = ใช้, Esc = ปิด
multiple = ช่องหลายชื่อคั่นด้วย ; (Background species): ค้นเฉพาะชื่อที่กำลังพิมพ์ท้ายช่อง เลือกแล้วต่อท้าย"""

import re
import tkinter as tk

from manual_window import species_matches

ROWS = 8


class SpeciesPicker:
    """entry = ช่องชื่อสามัญ, country_of() -> รหัสประเทศหรือ None, on_pick(common, scientific)
    on_exact(scientific) เรียกเมื่อพิมพ์ชื่อสามัญตรงกับชนิดหนึ่งพอดี (เติมชื่อวิทยาศาสตร์ให้)"""

    def __init__(self, entry, country_of, on_pick, on_exact=None, candidates_of=None, multiple=False):
        """candidates_of() -> [(common, scientific, หมายเหตุ)] ตัวเลือกตอนคลิกช่อง (common = None: บรรทัดคำแนะนำ เลือกไม่ได้)"""
        self.entry, self.country_of, self.on_pick, self.on_exact = entry, country_of, on_pick, on_exact
        self.candidates_of = candidates_of
        self.multiple = multiple
        self.top = self.box = None
        self.names = []
        self._searching = False               # รายการที่แสดงอยู่เป็นผลค้น (ไม่ใช่ตัวเลือกของคลิป)
        entry.bind("<KeyRelease>", self._typed, add="+")
        entry.bind("<Button-1>", lambda _event: entry.after_idle(self.show_candidates), add="+")
        entry.bind("<Down>", self._into_list, add="+")
        entry.bind("<Escape>", lambda _event: self.hide(), add="+")
        entry.bind("<FocusOut>", lambda _event: entry.after(200, self._hide_if_left), add="+")

    def _typed(self, event=None):
        if event is not None and getattr(event, "keysym", "") in ("Down", "Up", "Return", "Escape", "Tab"):
            return
        text = self._query()
        if len(text) < 2:                         # ลบชื่อจนว่าง: กลับไปแสดงตัวเลือกของคลิปนี้
            self.show_candidates()
            return
        self.names = species_matches(text, self.country_of(), limit=30)
        exact = next((scientific for common, scientific in self.names if common.casefold() == text.casefold()), None)
        if exact and self.on_exact:
            self.on_exact(exact)
        if self.names and not exact:
            self._show([f"{common}  —  {scientific}" for common, scientific in self.names])
            self._searching = True
        else:
            self.hide()

    def _query(self):
        text = self.entry.get()
        return re.split(r"[;,]", text)[-1].strip() if self.multiple else text.strip()

    def show_candidates(self):
        items = self.candidates_of() if self.candidates_of else []
        self._searching = False
        if not items:
            self.names = []
            self.hide()
            return
        self.names = [(common, scientific) if common else None for common, scientific, _note in items]
        self._show([f"{common}  —  {scientific}   ·  {note}" if common else note for common, scientific, note in items])

    def _show(self, labels):
        if self.top is None:
            self.top = tk.Toplevel(self.entry)
            self.top.wm_overrideredirect(True)
            self.box = tk.Listbox(self.top, height=ROWS, activestyle="dotbox", exportselection=False)
            self.box.pack(fill="both", expand=True)
            self.box.bind("<ButtonRelease-1>", lambda _event: self.pick())
            self.box.bind("<Return>", lambda _event: self.pick())
            self.box.bind("<Escape>", lambda _event: (self.hide(), self.entry.focus_set()))
            self.box.bind("<FocusOut>", lambda _event: self.box.after(200, self._hide_if_left))
        self.box.delete(0, "end")
        for label in labels:
            self.box.insert("end", label)
        self.entry.update_idletasks()
        width = max(self.entry.winfo_width(), 560)
        x, y = self.entry.winfo_rootx(), self.entry.winfo_rooty() + self.entry.winfo_height()
        self.top.geometry(f"{width}x{min(ROWS, len(self.names)) * 19 + 6}+{x}+{y}")
        self.top.lift()

    def _into_list(self, _event=None):
        if self.top is None or not self.names:
            return None
        first = next((i for i, name in enumerate(self.names) if name is not None), 0)
        self.box.focus_set()
        self.box.selection_clear(0, "end")
        self.box.selection_set(first)
        self.box.activate(first)
        return "break"

    def pick(self, index=None):
        if self.box is None:
            return
        chosen = [index] if index is not None else self.box.curselection()
        if chosen and chosen[0] < len(self.names) and self.names[chosen[0]] is not None:
            common, scientific = self.names[chosen[0]]
            searching = self._searching
            self.hide()
            if self.multiple:
                self._add(common, replace_typed=searching)
            self.on_pick(common, scientific)
            self.entry.focus_set()
            self.entry.icursor("end")

    def _add(self, common, replace_typed):
        """ช่องหลายชื่อ: ผลค้น = แทนชื่อที่กำลังพิมพ์ท้ายช่อง, ตัวเลือกของคลิป = ต่อท้าย แล้วใส่ ; รอชื่อถัดไป"""
        parts = re.split(r"[;,]", self.entry.get())
        if replace_typed:
            parts = parts[:-1]
        names = [" ".join(part.split()) for part in parts if part.strip()]
        if common.casefold() not in {name.casefold() for name in names}:
            names.append(common)
        self.entry.delete(0, "end")
        self.entry.insert(0, "; ".join(names) + "; ")

    def _hide_if_left(self):
        try:
            focus = self.entry.focus_get()
        except (KeyError, tk.TclError):     # โฟกัสอยู่ในหน้าต่างอื่น/popup ของ Tk
            focus = None
        if focus not in (self.entry, self.box):
            self.hide()

    def hide(self):
        if self.top is not None:
            self.top.destroy()
        self.top = self.box = None
