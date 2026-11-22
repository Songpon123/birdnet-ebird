"""gc ในเธรดเบื้องหลังเก็บของ Tk: การลบใน Tcl ต้องรอเธรดหลัก (Tcl/Tk 9 + Python 3.12 ล่มถ้าเรียกจากเธรดอื่น)"""

import gc
from pathlib import Path
import sys
import threading
import tkinter as tk
import unittest

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import tk_threads  # noqa: E402


class WorkerGcTests(unittest.TestCase):
    def test_worker_gc_leaves_tcl_cleanup_to_the_main_thread(self):
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        enabled = gc.isenabled()
        gc.disable()
        try:
            tk_threads.install(root)
            gc.collect()
            tk_threads.drain()

            class ClosedWindow:
                pass

            def close_window():   # ของ Tk ในวงอ้างอิงกันเอง: เก็บได้ด้วย gc เท่านั้น
                window = ClosedWindow()
                window.var = tk.StringVar(root, "x")
                window.img = tk.PhotoImage(master=root, width=4, height=4)
                window.pil = ImageTk.PhotoImage(Image.new("RGB", (4, 4)), master=root)
                window.me = window
                return str(window.var), str(window.img), str(window.pil)

            var, img, pil = close_window()
            worker = threading.Thread(target=gc.collect)
            worker.start()
            worker.join()

            def alive():
                names = set(map(str, root.image_names()))
                return root.getboolean(root.tk.call("info", "exists", var)), img in names, pil in names
            self.assertEqual(alive(), (True, True, True))        # เธรดเบื้องหลังไม่ได้แตะ Tcl
            self.assertGreaterEqual(len(tk_threads._pending), 3)

            root.after(tk_threads.DRAIN_MS + 50, root.quit)        # เธรดหลักทำงานที่ฝากไว้เอง
            root.mainloop()
            gc.collect()
            self.assertEqual(alive(), (False, False, False))
            self.assertEqual(len(tk_threads._pending), 0)
        finally:
            if enabled:
                gc.enable()
            root.destroy()

    def test_main_thread_cleanup_is_unchanged(self):
        root = tk.Tk()
        root.withdraw()
        root.createcommand("::tk::mac::Quit", lambda: None)
        try:
            tk_threads.install(root)
            image = tk.PhotoImage(master=root, width=4, height=4)
            name = str(image)
            del image
            self.assertNotIn(name, set(map(str, root.image_names())))
            self.assertEqual(len(tk_threads._pending), 0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
