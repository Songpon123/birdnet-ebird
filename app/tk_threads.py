"""กันแอปล่มเมื่อ Python เก็บขยะของ Tk ในเธรดเบื้องหลัง

Python 3.12 ที่มากับแอปใช้คู่กับ Tcl/Tk 9 ซึ่งไม่ตั้งค่า tcl_platform(threaded) แล้ว _tkinter จึงเข้าใจว่า
Tcl ไม่รองรับเธรด และเรียก Tcl จากเธรดที่สั่งโดยตรง แทนที่จะส่งต่อไปให้เธรดหลัก → แอปล่ม (SIGSEGV ใน
EvalObjvCore) ทันทีที่มีการเรียก Tk จากเธรดเบื้องหลัง

โค้ดของแอปไม่เรียก Tk จากเธรดเบื้องหลังอยู่แล้ว (ส่งผลผ่าน queue) แต่ตัวเก็บขยะ (gc) ของ Python ทำงานในเธรด
ไหนก็ได้ เมื่อเขาเก็บ StringVar / PhotoImage / Font ของหน้าต่างที่ปิดไปแล้ว __del__ ของสิ่งเหล่านี้จะเรียก Tcl
จากเธรดนั้น โมดูลนี้ทำให้ __del__ ที่ถูกเรียกนอกเธรดหลักฝากงานไว้ แล้วให้เธรดหลักทำแทนทุก DRAIN_MS
"""

import collections
import threading

DRAIN_MS = 250
_pending = collections.deque()   # (__del__ เดิม, ออบเจ็กต์) ที่ฝากมาจากเธรดอื่น
_guarded = set()


def _on_main_thread():
    return threading.current_thread() is threading.main_thread()


def _guard(cls):
    original = cls.__dict__.get("__del__")
    if original is None or cls in _guarded:
        return

    def __del__(self):
        if _on_main_thread():
            original(self)
        else:
            _pending.append((original, self))   # เก็บออบเจ็กต์ไว้ก่อน (Python ไม่เรียก __del__ ซ้ำ)

    __del__.__doc__ = original.__doc__
    cls.__del__ = __del__
    _guarded.add(cls)


def guarded_classes():
    import tkinter
    import tkinter.font
    classes = [tkinter.Image, tkinter.Variable, tkinter.font.Font]
    try:
        from PIL import ImageTk
    except ImportError:
        pass
    else:
        classes += [ImageTk.PhotoImage, ImageTk.BitmapImage]
    return classes


def drain():
    """ทำงานลบที่ฝากไว้ (เรียกจากเธรดหลักเท่านั้น)"""
    while _pending:
        original, obj = _pending.popleft()
        try:
            original(obj)
        except Exception:  # noqa: BLE001  หน้าต่าง/โปรแกรม Tk ปิดไปแล้ว
            pass
        del original, obj


def install(root):
    """เรียกครั้งเดียวจากเธรดหลักหลังสร้าง Tk()"""
    for cls in guarded_classes():
        _guard(cls)

    def tick():
        drain()
        try:
            if root.winfo_exists():
                root.after(DRAIN_MS, tick)
        except Exception:  # noqa: BLE001  root ถูกปิดแล้ว
            pass
    root.after(DRAIN_MS, tick)
