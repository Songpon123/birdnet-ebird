"""บันทึกภาคสนามของคุณ: eBird checklist ที่ส่งแล้ว (พร้อมรูป) จากโฟลเดอร์ Log ของโปรแกรมจัดรูป
  <Birding>/<ปี>/<เดือน>/<YYYYMMDD>/Log/<รอบ>/checklist.json   (ชนิด จำนวน รูปใน Macaulay ของ checklist นั้น)
ใช้ประกอบการตรวจเสียง: ชนิดที่คุณลงไว้ (เห็น/ถ่ายรูปได้) วันนั้น หรือไม่ได้ลงไว้ — ไม่ใช่ตัวระบุชนิดของเสียง
ตัวอย่าง 25 ต.ค. 2026 (S200000001 มหาวิทยาลัยอู่ฮั่น 08:08): Swinhoe's White-eye 13 ตัว รูป 2 รูป, ไม่มี Chestnut-flanked

Log ที่เพิ่มแล้วจำไว้ใน data/field-logs.json พร้อมโฟลเดอร์ <Birding> — วันอื่นที่มีโฟลเดอร์ Log แบบเดียวกันหาให้เอง

ไฟล์เสียงหนึ่งไฟล์อยู่ checklist เดียว (ย้ายที่ = checklist ใหม่): เลือกในหน้าต่าง Match recordings เก็บเป็นช่วงเวลาของไฟล์
จึงใช้ได้กับคลิปทุกจุดในไฟล์นั้น ไฟล์ที่ยังไม่ได้เลือก = checklist ล่าสุดที่เริ่มก่อนเริ่มอัด (for_recording)

พิกัดมาจาก location_details ใน checklist.json: ไฟล์เสียงที่ไม่มี geotag ใช้พิกัดของ checklist ของไฟล์นั้นตอนวิเคราะห์ (spot_for)

วางลิงก์ checklist ได้เลย (add_links): ดึงชนิด จำนวน จำนวนรูปต่อชนิด วันเวลา ระยะเวลา สถานที่ พิกัด จาก eBird ล่าสุดเสมอ
ค่าจากลิงก์ชนะค่าใน Log (Log คือสำเนาตอนส่งรูป) แต่รายการรูปและเวลาถ่ายยังมาจาก Log
"""

import csv
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import paths

STORE = paths.SETTINGS_FILE.parent / "field-logs.json"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DATE_FORMATS = ("%d %b %Y %I:%M %p", "%d %b %Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d")
NONE = "none"                      # ไฟล์เสียงนี้ไม่อยู่ใน checklist ไหน
AHEAD = timedelta(minutes=15)      # เปิดเครื่องอัดก่อนเริ่ม checklist ได้เท่านี้
NO_CHECKLIST_FOLDER = "No checklist"
GUESS_WINDOW = timedelta(hours=6)  # ไฟล์ที่ยังไม่ได้เลือก: ใช้พิกัด checklist เมื่อเริ่มอัดหลังเริ่ม checklist ไม่เกินเท่านี้ (ไม่รู้ระยะเวลา)
CHECKLIST_ID = re.compile(r"\bS\d{6,}\b")
FROM_EBIRD = ("date", "location", "lat", "lon", "hours", "species", "fetched")   # ค่าที่ลิงก์ให้ใหม่กว่า Log


def _two(name):
    return " ".join(str(name or "").split()[:2]).casefold()


def _parse_date(text):
    for pattern in DATE_FORMATS:
        try:
            return datetime.strptime(str(text).strip(), pattern)
        except ValueError:
            continue
    return None


def _coordinate(value, limit):
    """ค่าพิกัดจาก location_details -> float หรือ None"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if abs(value) <= limit else None


def read_checklist(path):
    """checklist.json หนึ่งไฟล์ -> dict (None ถ้าอ่านไม่ได้หรือไม่ใช่ checklist)"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    when = _parse_date(data.get("date"))
    if not isinstance(data, dict) or not data.get("checklist_id") or when is None:
        return None
    species = [{"common": item.get("common_name") or "", "scientific": item.get("scientific_name") or "",
                "code": item.get("species_code") or "", "count": str(item.get("count") or ""),
                "photos": sum(1 for media in item.get("media") or () if media.get("media_type") == "photo"),
                "media": len(item.get("media") or ())}
               for item in data.get("species") or ()]
    place = data.get("location_details") or {}
    return {"id": data["checklist_id"], "url": data.get("url") or "", "date": when.isoformat(timespec="minutes"),
            "location": data.get("location") or "", "lat": _coordinate(place.get("latitude"), 90),
            "lon": _coordinate(place.get("longitude"), 180), "species": species, "source": str(Path(path).resolve()),
            "photos": read_photos(Path(path).with_name("files.csv"))}


def read_photos(path):
    """files.csv ข้าง checklist.json -> [{"time", "file", "path", "common", "scientific"}] (ไม่มีชื่อชนิด = รูปที่ยังไม่ได้ผูกกับ
    checklist) — time = เวลาในรูป (EXIF ตามนาฬิกากล้อง เขตเวลาท้องถิ่น) ไม่ใช่เวลาส่ง checklist"""
    photos = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                try:
                    taken = datetime.fromisoformat(str(row.get("capture_time") or "")).replace(tzinfo=None)
                except ValueError:
                    continue
                photos.append({"time": taken.isoformat(timespec="seconds"), "file": row.get("filename") or "",
                               "path": row.get("original_path") or "", "common": row.get("common_name") or "",
                               "scientific": row.get("scientific_name") or ""})
    except OSError:
        return []
    return sorted(photos, key=lambda photo: photo["time"])


def find_checklists(folder):
    """checklist.json ใน Log / รอบใน Log / checklist.json ที่เลือกตรง ๆ -> [checklist] (checklist เดียวกันหลายรอบ = รอบล่าสุด)"""
    folder = Path(folder).expanduser()
    files = [folder] if folder.is_file() else sorted(folder.glob("checklist.json")) + sorted(folder.glob("*/checklist.json"))
    found = {}
    for path in files:
        checklist = read_checklist(path)
        if checklist:
            found[checklist["id"]] = checklist
    return list(found.values())


def birding_root(folder):
    """<Birding>/<ปี>/<เดือน>/<YYYYMMDD>/Log… -> <Birding> (None ถ้าไม่ใช่โครงสร้างนี้)"""
    parts = Path(folder).expanduser().resolve().parts
    for index in range(len(parts) - 3):
        year, month, day = parts[index + 1:index + 4]
        if (year.isdigit() and len(year) == 4 and month.title() in MONTHS and day.isdigit() and len(day) == 8
                and day.startswith(year)):
            return str(Path(*parts[:index + 1]))
    return None


def _load(store=None):
    try:
        data = json.loads((store or STORE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data, store=None):
    store = store or STORE
    store.parent.mkdir(parents=True, exist_ok=True)
    temporary = store.with_name(f"{store.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(store)


def stored(store=None):
    """(checklist ที่เพิ่มไว้ เรียงตามวัน, โฟลเดอร์ Birding ที่หา Log ของวันอื่นให้เอง)"""
    data = _load(store)
    return sorted((data.get("checklists") or {}).values(), key=lambda c: c["date"]), list(data.get("roots") or ())


def _merged(log, known):
    """checklist จาก Log + ค่าที่เคยดึงจากลิงก์ (ถ้ามี)"""
    return dict(log, **{name: known[name] for name in FROM_EBIRD if name in known}) if known.get("fetched") else log


def checklist_ids(text):
    """รหัส checklist ทุกตัวในข้อความที่วาง (ลิงก์หรือรหัส S… หลายอันได้) ไม่ซ้ำ"""
    return list(dict.fromkeys(CHECKLIST_ID.findall(str(text))))


def add_links(text, key, store=None):
    """ดึง checklist จากลิงก์ที่วาง -> ([checklist ที่เพิ่ม], [ข้อความปัญหา]) ลิงก์เดิม = อัปเดตให้ตรง eBird"""
    import ebird
    added, problems = [], []
    for checklist_id in checklist_ids(text):
        try:
            checklist = ebird.checklist(checklist_id, key)
        except ebird.EBirdError as exc:
            problems.append(f"{checklist_id}: {exc}")
            continue
        data = _load(store)
        checklists = data.setdefault("checklists", {})
        known = checklists.get(checklist_id) or {}
        checklists[checklist_id] = {**known, **checklist, "photos": known.get("photos") or [],
                                    "fetched": datetime.now().isoformat(timespec="seconds")}
        _save(data, store)
        added.append(checklists[checklist_id])
    return added, problems


def add(folder, store=None):
    """เพิ่ม Log (โฟลเดอร์ Log, รอบ หรือ checklist.json) -> [checklist ที่เพิ่ม]"""
    found = find_checklists(folder)
    if not found:
        return []
    data = _load(store)
    checklists = data.setdefault("checklists", {})
    checklists.update({c["id"]: _merged(c, checklists.get(c["id"]) or {}) for c in found})
    root = birding_root(folder)
    if root and root not in data.setdefault("roots", []):
        data["roots"].append(root)
    _save(data, store)
    return found


def day_folder(root, day):
    """<Birding>/<ปี>/<เดือน>/<YYYYMMDD>"""
    return Path(root) / f"{day.year}" / MONTHS[day.month - 1] / day.strftime("%Y%m%d")


def for_day(day, store=None):
    """checklist ของวันนั้น: ที่เพิ่มไว้ + โฟลเดอร์ Log ของวันนั้นใต้ <Birding> ที่เคยเพิ่ม (หาเอง ไม่ต้องกดเพิ่มทุกวัน)
    Log อยู่ที่ <วัน>/Log หรือใต้ <วัน>/Pictures ก็ได้ เช่น Pictures/Done/Done/Log"""
    if day is None:
        return []
    day = day.date() if isinstance(day, datetime) else day
    data = _load(store)
    found = {cid: c for cid, c in (data.get("checklists") or {}).items() if c["date"][:10] == day.isoformat()}
    for root in data.get("roots") or ():
        folder = day_folder(root, day)
        for log in (folder / "Log", *sorted(folder.glob("Pictures/**/Log"))):
            if log.is_dir():
                found.update({c["id"]: _merged(c, found.get(c["id"]) or {}) for c in find_checklists(log)
                              if c["date"][:10] == day.isoformat()})
    return sorted(found.values(), key=lambda c: c["date"])


def by_time(checklists, when):
    """checklist ของเสียงที่เริ่มอัดเวลานี้: อันล่าสุดที่เริ่มก่อนนั้น (เผื่อ AHEAD) ก่อนทุกอัน = อันแรก"""
    reached = [c for c in checklists if datetime.fromisoformat(c["date"]) <= when + AHEAD]
    return reached[-1] if reached else checklists[0]


def picked(when, store=None):
    """checklist ที่เลือกให้ไฟล์เสียงซึ่งมีเวลานี้อยู่ในไฟล์ -> id, NONE หรือ None (ยังไม่ได้เลือก)"""
    stamp = when.isoformat(timespec="seconds")
    return next((item["checklist"] for start, item in (_load(store).get("recordings") or {}).items()
                 if start <= stamp <= item["end"]), None)


def choose(recording, checklist, store=None):
    """เลือก checklist ของไฟล์เสียง {"start", "end"} (NONE = ไม่อยู่ใน checklist ไหน)"""
    data = _load(store)
    data.setdefault("recordings", {})[recording["start"].isoformat(timespec="seconds")] = {
        "end": recording["end"].isoformat(timespec="seconds"), "checklist": checklist}
    _save(data, store)


def for_recording(when, store=None):
    """checklist ของไฟล์เสียงที่มีเวลานี้อยู่ (ทุกจุดในไฟล์ได้) -> [] หรือ [checklist]"""
    checklists = for_day(when, store)
    if not checklists:
        return []
    chosen = picked(when, store)
    if chosen == NONE:
        return []
    return [next((c for c in checklists if c["id"] == chosen), None) or by_time(checklists, when)]


def folder_name(checklist):
    """ชื่อโฟลเดอร์ใน Ready: '0636 教字垭镇 (Jiaoziya Town) S300000001' (เวลาเริ่มขึ้นก่อน เรียงตามลำดับใน Finder)"""
    place = str(checklist.get("location") or "").split(",")[0].strip()
    name = " ".join(part for part in (f"{datetime.fromisoformat(checklist['date']):%H%M}", place, checklist["id"]) if part)
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def ready_folder(when, store=None):
    """โฟลเดอร์ย่อยใน Ready ของไฟล์เสียงนี้ — '' = วันนั้นไม่มี checklist เลย ใช้ Ready/<ชนิด> แบบเดิม"""
    if when is None or not for_day(when, store):
        return ""
    checklists = for_recording(when, store)
    return folder_name(checklists[0]) if checklists else NO_CHECKLIST_FOLDER


def spot_for(when, store=None):
    """พิกัดของไฟล์เสียงที่เริ่มอัดเวลานี้จาก checklist -> {"lat", "lon", "checklist", "how"} หรือ None
    ยังไม่ได้เลือก checklist = ใช้เมื่อเริ่มอัดใกล้เวลา checklist เท่านั้น นอกช่วงนี้ห้ามเดาที่อัดเสียง"""
    checklists = for_recording(when, store)
    if not checklists or checklists[0].get("lat") is None or checklists[0].get("lon") is None:
        return None
    checklist = checklists[0]
    chosen = picked(when, store) == checklist["id"]
    began = datetime.fromisoformat(checklist["date"])
    hours = checklist.get("hours")
    if not chosen and not began - AHEAD <= when <= began + (timedelta(hours=hours) + AHEAD if hours else GUESS_WINDOW):
        return None
    return {"lat": checklist["lat"], "lon": checklist["lon"], "checklist": checklist["id"],
            "how": "your choice" if chosen else "by time"}


def describe_recording(when, store=None):
    """checklist ของไฟล์เสียงนี้ (แสดงในแท็บ Review) -> 'eBird checklist S… 02 Nov 08:00 ที่ (by time)' — '' = วันนั้นไม่มี"""
    if not for_day(when, store):
        return ""
    found = for_recording(when, store)
    if not found:
        return "eBird checklist: none for this recording (Settings › Match recordings…)"
    how = "your choice" if picked(when, store) == found[0]["id"] else "by time"
    return f"eBird checklist {describe_day(found)} ({how})"


def entry(checklists, scientific):
    """(checklist, ชนิดใน checklist) ของชนิดนี้ (เทียบชื่อวิทยาศาสตร์ 2 คำแรก) หรือ None"""
    key = _two(scientific)
    for checklist in checklists:
        for item in checklist["species"]:
            if key and _two(item["scientific"]) == key:
                return checklist, item
    return None


def describe_species(checklists, scientific, common=""):
    """ข้อความสั้นของชนิดหนึ่ง: 'on your checklist S… ×13, 2 photos' / 'not on your eBird checklist S…' / ''"""
    if not checklists:
        return ""
    found = entry(checklists, scientific)
    if found is None:
        return f"{common or scientific}: not on your eBird checklist {', '.join(c['id'] for c in checklists)}"
    checklist, item = found
    photos = f", {item['photos']} photo{'s' if item['photos'] != 1 else ''}" if item["photos"] else ""
    return f"{common or item['common']}: on your checklist {checklist['id']} ×{item['count'] or '?'}{photos}"


def photos_on(checklists):
    """รูปของ checklist เหล่านี้เรียงตามเวลา — ภาพเดียวกันหลายไฟล์ (.jpg .dng .CR3) นับใบเดียว ใช้ไฟล์ที่มีชื่อชนิด/jpg"""
    found = {}
    every = [photo for checklist in checklists for photo in checklist.get("photos") or ()]
    for photo in sorted(every, key=lambda p: (not p["common"], Path(p["file"]).suffix.lower() != ".jpg")):
        found.setdefault((photo["time"], re.sub(r"^\d{8}-", "", Path(photo["file"]).stem)), photo)
    return sorted(found.values(), key=lambda photo: photo["time"])


def describe_day(checklists):
    """'S200000001 25 Oct 08:08 武汉大学 (Wuhan University), Hubei, CN' ของ checklist เหล่านี้"""
    return "; ".join(f"{c['id']} {datetime.fromisoformat(c['date']):%d %b %H:%M} {c['location']}".strip()
                     for c in checklists)
