"""Offline country checklists for China, Hong Kong, Taiwan and Thailand: is this species native,
a rarity, introduced … in the country of the recording? Works without internet or API keys.

Checklists: Scythebill (Apache 2.0, https://bitbucket.org/scythebill/scythebill-birdlist),
names on the eBird/Clements 2025 taxonomy. Borders: Natural Earth 1:10m (public domain).
Rebuild assets/country_checklists.csv and country_borders.json with tools/build_country_checklists.py.
"""

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
CHECKLISTS = ASSETS / "country_checklists.csv"
BORDERS = ASSETS / "country_borders.json"
NAMES = {"CN": "China", "HK": "Hong Kong", "TW": "Taiwan", "TH": "Thailand"}
COAST_KM = 10     # จุดนอกแนวชายฝั่ง: หาดเลน คันนา เรือใกล้ฝั่ง
STATUS_TEXT = {
    "native": "native in {}",
    "endemic": "endemic to {}",
    "introduced": "introduced in {}",
    "rarity": "rarity in {}",
    "rarity-from-introduced": "rarity in {} (from introduced birds)",
    "escaped": "only escaped cage birds in {}",
    "extinct": "extinct in {}",
    "": "not on the {} checklist",
}
CHECK = {"rarity", "rarity-from-introduced", "escaped", "extinct", ""}


@lru_cache(maxsize=1)
def _species():
    """({ชื่อวิทยาศาสตร์: แถว}, {ชื่ออังกฤษตัวเล็ก: แถว}, {รหัส eBird: แถว})"""
    with CHECKLISTS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return ({row["scientific"]: row for row in rows}, {row["common"].casefold(): row for row in rows},
            {row["ebird_code"]: row for row in rows})


@lru_cache(maxsize=1)
def _rings():
    """(ประเทศ หรือ None = แผ่นดินประเทศอื่นใกล้ชายฝั่งเรา, xs, ys, ธงพรมแดนทางบกต่อเส้น, กรอบ)"""
    data = json.loads(BORDERS.read_text(encoding="utf-8"))
    parts = [(country, ring["lonlat"], ring["land"])
             for country, rings in data["countries"].items() for ring in rings]
    parts += [(None, lonlat, None) for lonlat in data["others"]]
    rings = []
    for country, lonlat, land in parts:
        xs, ys = lonlat[0::2], lonlat[1::2]
        rings.append((country, xs, ys, land, (min(xs), min(ys), max(xs), max(ys))))
    return rings


def _segment_km(lat, lon, ax, ay, bx, by):
    """ระยะ (กม.) จากจุดถึงเส้น A-B แบบระนาบรอบจุดนั้น (พอสำหรับระยะไม่กี่สิบกิโลเมตร)"""
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57
    ax, ay, bx, by = (ax - lon) * kx, (ay - lat) * ky, (bx - lon) * kx, (by - lat) * ky
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx, ay + t * dy)


@lru_cache(maxsize=4096)
def country_at(lat, lon):
    """CN / HK / TW / TH ของพิกัดนี้ หรือ None (ประเทศอื่น ทะเลไกลฝั่ง หรือไม่มีพิกัด)
    จุดใกล้ชายแดนทางบกยึดเส้นเขตแดนตามจริง (เส้น 1:10m คลาดได้ราว 1 กม.)"""
    if lat is None or lon is None:
        return None
    margin = COAST_KM / 50                  # องศา (เผื่อกว้างกว่า 10 กม. ทุกละติจูดที่ใช้)
    nearby = [ring for ring in _rings()
              if ring[4][0] - margin <= lon <= ring[4][2] + margin
              and ring[4][1] - margin <= lat <= ring[4][3] + margin]
    inside = {}
    for country, xs, ys, _land, _box in nearby:     # even-odd: ring ที่เป็นรูใน polygon สลับกลับเอง
        for i in range(len(xs) - 1):
            ay, by = ys[i], ys[i + 1]
            if (ay > lat) != (by > lat) and lon < xs[i] + (lat - ay) * (xs[i + 1] - xs[i]) / (by - ay):
                inside[country] = not inside.get(country, False)
    if any(inside.values()):
        return next(country for country, is_inside in inside.items() if is_inside)  # None = ประเทศอื่น
    # ในทะเล: ใช้ฝั่งที่ใกล้ที่สุด ถ้าเป็นชายฝั่งของเรา (ไม่ใช่พรมแดนทางบก ไม่ใช่ฝั่งประเทศอื่น) ≤ COAST_KM
    best = None
    for country, xs, ys, land, _box in nearby:
        for i in range(len(xs) - 1):
            km = _segment_km(lat, lon, xs[i], ys[i], xs[i + 1], ys[i + 1])
            if best is None or km < best[0]:
                best = (km, country, country is not None and land[i] == "0")
    if best and best[2] and best[0] <= COAST_KM:
        return best[1]
    return None


def check(scientific, lat, lon, common=""):
    """{'country', 'name', 'status', 'split'} — status '' = not on that country's checklist;
    split = [(common, scientific)] ชนิดที่แยกออกจากกลุ่มเดียวกันและพบตามปกติในประเทศนั้น
    (BirdNET ใช้ชื่อเดียวกับทั้งกลุ่ม: Little Heron ในจีนขึ้นว่า Striated Heron)
    None = no checklist for this place, or the name is not a bird species in eBird/Clements 2025.
    ชื่อวิทยาศาสตร์ของ BirdNET ต่างจาก eBird ~240 ชื่อ (เช่น Charadrius dubius = Thinornis dubius)
    จึงลองชื่ออังกฤษถ้าชื่อวิทยาศาสตร์ไม่ตรง"""
    country = country_at(lat, lon)
    by_scientific, by_common, by_code = _species()
    row = (by_scientific.get(" ".join(str(scientific or "").split()[:2]))
           or by_common.get(str(common or "").strip().casefold()))
    if country is None or row is None:
        return None
    split = []
    if row[country] in CHECK:
        split = [(other["common"], other["scientific"]) for other in map(by_code.get, row["split_with"].split())
                 if other and other[country] and other[country] not in CHECK]
    return {"country": country, "name": NAMES[country], "status": row[country], "split": split}


def needs_check(result):
    return bool(result) and result["status"] in CHECK


def describe(result):
    """'rarity in Thailand — verify by ear' / 'native in China' / '' (ไม่มีข้อมูล)"""
    if not result:
        return ""
    text = STATUS_TEXT[result["status"]].format(result["name"])
    if result["split"]:
        names = ", ".join(f"{common} ({scientific})" for common, scientific in result["split"])
        verb = "is" if len(result["split"]) == 1 else "are"
        return f"{text}, but split species {names} {verb} — check the name"
    return text + " — verify by ear" if needs_check(result) else text
