"""eBird API 2.0, read-only. The taxonomy needs no key; nearby sightings, regional species
lists and hotspots use the user's own key (Settings tab). Results are cached on this Mac."""

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import paths

API = "https://api.ebird.org/v2"
USER_AGENT = "BirdNET-eBird-mac (bird recording review app)"
CACHE = paths.CACHE / "ebird"
NEARBY_KM = 25                      # รายงานล่าสุดใกล้จุดบันทึก (API ให้สูงสุด 50 กม.)
RECENT_DAYS = 30                    # API ย้อนหลังได้สูงสุด 30 วัน
HOTSPOT_KM = 50
DAY = 24 * 3600
SEASON_YEARS = 3                    # ช่วงเดียวกันของ 3 ปีก่อน + ปีที่บันทึก
SEASON_HALF_WINDOW = 3              # ±3 วันรอบวันที่บันทึก
SEASON_MIN_SPECIES = 40             # มณฑลที่รายงานน้อยกว่านี้ในช่วงนั้น: "ไม่มีรายงาน" แทบไม่มีความหมาย
SEASON_WORKERS = 6                  # คำขอพร้อมกัน (แต่ละวันใช้ 2-9 วิ)


class EBirdError(RuntimeError):
    pass


def _get(path, params=None, key=None, timeout=30):
    url = f"{API}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["X-eBirdApiToken"] = key
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise EBirdError("eBird did not accept the API key; check it in the Settings tab") from exc
        raise EBirdError(f"eBird returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise EBirdError(f"Cannot reach eBird: {exc}") from exc


def _cached(name, max_age_s, fetch):
    path = CACHE / f"{name}.json"
    try:
        if time.time() - path.stat().st_mtime < max_age_s:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data = fetch()
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)
    return data


_taxonomy_memo = {}


def taxonomy():
    """{'by_sci': {sciName: {code, common}}, 'by_common': {comName lower: ...}} (no key needed)
    เก็บไว้ในหน่วยความจำ 1 ชม. (เรียกบ่อยตอนพิมพ์ชื่อชนิด)"""
    if _taxonomy_memo and time.time() - _taxonomy_memo["at"] < 3600:
        return _taxonomy_memo["data"]
    rows = _cached("taxonomy", 60 * DAY,
                   lambda: _get("ref/taxonomy/ebird", {"fmt": "json", "cat": "species", "locale": "en"}))
    by_sci, by_common = {}, {}
    for row in rows:
        entry = {"code": row["speciesCode"], "common": row["comName"], "scientific": row["sciName"]}
        by_sci[row["sciName"]] = entry
        by_common.setdefault(row["comName"].casefold(), entry)
    _taxonomy_memo.update(at=time.time(), data={"by_sci": by_sci, "by_common": by_common})
    return _taxonomy_memo["data"]


def species(scientific, common=""):
    tax = taxonomy()
    return tax["by_sci"].get(" ".join(str(scientific).split()[:2])) or tax["by_common"].get(str(common).casefold())


def _spot(lat, lon):
    return f"{lat:.2f}_{lon:.2f}"


def nearest_hotspots(lat, lon, key):
    rows = _cached(f"hotspots_{_spot(lat, lon)}", 7 * DAY,
                   lambda: _get("ref/hotspot/geo", {"lat": f"{lat:.4f}", "lng": f"{lon:.4f}",
                                                    "dist": HOTSPOT_KM, "fmt": "json"}, key))
    for row in rows:
        row["km"] = round(_distance_km(lat, lon, float(row["lat"]), float(row["lng"])), 1)
    return sorted(rows, key=lambda row: row["km"])


def region_species(region, key):
    return set(_cached(f"spplist_{region}", 7 * DAY, lambda: _get(f"product/spplist/{region}", key=key)))


def region_name(region, key):
    try:
        info = _cached(f"region_{region}", 60 * DAY, lambda: _get(f"ref/region/info/{region}", key=key))
        return str(info.get("result") or region).split(",")[0]
    except EBirdError:
        return region


def recent_nearby(lat, lon, key):
    """ล่าสุดของแต่ละชนิดภายใน NEARBY_KM ในช่วง RECENT_DAYS วันก่อนวันนี้"""
    return _cached(f"recent_{_spot(lat, lon)}_{date.today().isoformat()}", 6 * 3600,
                   lambda: _get("data/obs/geo/recent", {"lat": f"{lat:.4f}", "lng": f"{lon:.4f}",
                                                        "dist": NEARBY_KM, "back": RECENT_DAYS,
                                                        "includeProvisional": "true"}, key))


def region_of(lat, lon, key):
    """รหัสมณฑล/รัฐ (เช่น CN-42) ของจุดบันทึก จาก hotspot ที่ใกล้ที่สุด -> str หรือ None"""
    hotspots = nearest_hotspots(lat, lon, key)
    return (hotspots[0].get("subnational1Code") or hotspots[0].get("countryCode")) if hotspots else None


def season_dates(recorded):
    """วันที่จะถาม: ±SEASON_HALF_WINDOW วันรอบวันเดียวกันของ SEASON_YEARS ปีก่อน และปีที่บันทึก
    (ไม่รวมวันที่บันทึกเอง กันรายการ eBird ของคุณเองยืนยันตัวเอง และไม่รวมวันในอนาคต)"""
    when = recorded.date() if hasattr(recorded, "date") else recorded
    days = []
    for year in range(when.year - SEASON_YEARS, when.year + 1):
        try:
            center = when.replace(year=year)
        except ValueError:                           # 29 ก.พ. ในปีที่ไม่ใช่ปีอธิกสุรทิน
            center = when.replace(year=year, day=28)
        for offset in range(-SEASON_HALF_WINDOW, SEASON_HALF_WINDOW + 1):
            day = center + timedelta(days=offset)
            if day != when and day < date.today():
                days.append(day)
    return days


def historic_day(region, day, key):
    """รหัสชนิดที่มีรายงานในมณฑลนี้ในวันนั้น (eBird API: historic observations on a date)"""
    recent = (date.today() - day).days <= 60          # วันใกล้ ๆ ยังมีรายการส่งเข้ามาเพิ่ม
    rows = _cached(f"historic_{region}_{day.isoformat()}", (2 if recent else 365) * DAY,
                   lambda: _get(f"data/obs/{region}/historic/{day.year}/{day.month}/{day.day}",
                                {"rank": "mrec", "detail": "simple", "maxResults": 10000}, key))
    return sorted({row["speciesCode"] for row in rows if row.get("speciesCode")})


def season_reports(region, recorded, key):
    """ชนิดที่มีรายงานในมณฑลนี้ช่วงเดียวกันของปีก่อน ๆ -> dict (เก็บ cache ทีละวัน ครั้งแรก ~10 วิ)"""
    days = season_dates(recorded)
    with ThreadPoolExecutor(SEASON_WORKERS) as pool:
        found = dict(zip(days, pool.map(lambda day: historic_day(region, day, key), days)))
    reported = {}
    for day, codes in sorted(found.items()):
        for code in codes:
            reported.setdefault(code, []).append(day.isoformat())
    return {"region": region, "center": (recorded.date() if hasattr(recorded, "date") else recorded).isoformat(),
            "days": len(days), "days_with_reports": sum(bool(codes) for codes in found.values()),
            "species_total": len(reported), "first_year": min(day.year for day in days) if days else None,
            "last_year": max(day.year for day in days) if days else None, "reported": reported}


def season_status(season, code):
    """'reported' / 'not reported' / 'few reports' (มณฑลนี้มีข้อมูลน้อยเกินจะสรุป)"""
    if season["reported"].get(code):
        return "reported"
    return "few reports" if season["species_total"] < SEASON_MIN_SPECIES else "not reported"


def check(scientific, lat, lon, recorded, key, common=""):
    """ชนิดนี้สอดคล้องกับข้อมูล eBird ใกล้จุดบันทึกไหม -> dict (recorded = datetime/date หรือ None)"""
    entry = species(scientific, common)
    if entry is None:
        return {"known": False, "scientific": scientific}
    result = {"known": True, **entry, "region": None, "region_name": None, "in_region": None,
              "recent_checked": False, "recent": None, "hotspot": None}
    if lat is None or lon is None or not key:
        return result
    hotspots = nearest_hotspots(lat, lon, key)
    if hotspots:
        nearest = hotspots[0]
        result["hotspot"] = {"name": nearest["locName"], "km": nearest["km"], "id": nearest["locId"]}
        result["region"] = nearest.get("subnational1Code") or nearest.get("countryCode")
    if result["region"]:
        result["region_name"] = region_name(result["region"], key)
        result["in_region"] = entry["code"] in region_species(result["region"], key)
    when = getattr(recorded, "date", lambda: recorded)() if recorded else None
    if when is not None and 0 <= (date.today() - when).days <= RECENT_DAYS:
        result["recent_checked"] = True
        sightings = [row for row in recent_nearby(lat, lon, key) if row.get("speciesCode") == entry["code"]]
        if sightings:
            latest = max(sightings, key=lambda row: row.get("obsDt", ""))
            result["recent"] = {"date": latest.get("obsDt", ""), "where": latest.get("locName", "")}
    return result


def _place(view, key):
    """จุดของ checklist -> (ชื่อเต็ม, lat, lon) feed รายวันของมณฑลให้จุดส่วนตัวด้วย แต่ได้แค่ 200 รายการ
    ref/hotspot/info ให้เฉพาะ hotspot (จุดส่วนตัวตอบว่าง) — ไม่เจอ = ('', None, None)"""
    location, region, day = view.get("locId"), view.get("subnational1Code"), str(view.get("obsDt") or "")[:10].split("-")
    if region and len(day) == 3:
        path = f"product/lists/{region}/{int(day[0])}/{int(day[1])}/{int(day[2])}"
        for item in _get(path, {"maxResults": 200}, key):
            place = item.get("loc") or {}
            if item.get("locId") == location and place.get("lat") is not None:
                return place.get("hierarchicalName") or place.get("name") or "", place["lat"], place.get("lng")
    try:
        place = _get(f"ref/hotspot/info/{location}", key=key)
    except EBirdError:
        return "", None, None
    return place.get("hierarchicalName") or place.get("name") or "", place.get("latitude"), place.get("longitude")


def checklist(checklist_id, key):
    """checklist จาก eBird -> dict แบบเดียวกับ field_log.read_checklist
    ได้จำนวนรูปต่อชนิด แต่ไม่ได้ไฟล์รูปและเวลาถ่าย (มีเฉพาะใน Log ของโปรแกรมจัดรูป)"""
    view = _get(f"product/checklist/view/{checklist_id}", key=key)
    observed = view.get("obs") or []
    codes = list(dict.fromkeys(item["speciesCode"] for item in observed))
    rows = _get("ref/taxonomy/ebird", {"fmt": "json", "locale": "en", "species": ",".join(codes)}, key) if codes else []
    names = {row["speciesCode"]: row for row in rows}
    species = []
    for item in observed:
        name = names.get(item["speciesCode"], {})
        counts = item.get("mediaCounts") or {}
        species.append({"common": name.get("comName") or item["speciesCode"], "scientific": name.get("sciName") or "",
                        "code": item["speciesCode"], "count": str(item.get("howManyStr") or ""),
                        "photos": int(counts.get("P") or 0), "media": sum(int(n) for n in counts.values())})
    location, lat, lon = _place(view, key)
    return {"id": checklist_id, "url": f"https://ebird.org/checklist/{checklist_id}",
            "date": str(view.get("obsDt") or "")[:16].replace(" ", "T"), "location": location, "lat": lat, "lon": lon,
            "hours": view.get("durationHrs"), "species": species, "photos": []}


def _distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))
