"""สร้าง app/assets/country_checklists.csv และ country_borders.json ใหม่ (จีน ฮ่องกง ไต้หวัน ไทย)

ข้อมูลต้นทาง (โหลดครั้งแรกเก็บใน build/country-checklists-cache/ ตรวจ SHA-256 ทุกครั้ง):
  Scythebill (Apache 2.0, https://bitbucket.org/scythebill/scythebill-birdlist)
    clements.transpose   สถานภาพของแต่ละชนิดในแต่ละประเทศ (native, rarity, introduced, …)
    ebird2025.xml.gz     รหัสชนิดของ Scythebill -> รหัส eBird + ชื่อวิทยาศาสตร์ (eBird/Clements 2025)
    2015to2025.csv       ชนิดที่ถูกแยก (split) ตั้งแต่ Clements 2015 เช่น Striated Heron -> Little Heron
                         (BirdNET ยังใช้ชื่อเดียวกับทั้งกลุ่ม: ได้ยิน Little Heron ในจีนแต่ขึ้นว่า Striated Heron)
  Natural Earth 1:10m admin-0 countries v5.1.2 (public domain)   เส้นเขตแดน

ใช้: ./python/bin/python3.12 -I tools/build_country_checklists.py
อัปเดตข้อมูล: เปลี่ยน SCYTHEBILL_COMMIT แล้วใส่ sha256 ใหม่ (สคริปต์บอกค่าที่ได้ถ้าไม่ตรง)
"""

import csv
import gzip
import hashlib
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROGRAM = Path(__file__).resolve().parent.parent
CACHE = PROGRAM / "build" / "country-checklists-cache"
ASSETS = PROGRAM / "app" / "assets"

SCYTHEBILL_COMMIT = "7157112b44d7a9e76a23db3c081c2ee60a3911ae"   # 2026-10-13
SCYTHEBILL = f"https://bitbucket.org/scythebill/scythebill-birdlist/raw/{SCYTHEBILL_COMMIT}/"
SOURCES = {   # ชื่อในแคช -> (url, sha256)
    "clements.transpose": (
        SCYTHEBILL + "model/src/main/resources/com/scythebill/birdlist/model/checklist/clements.transpose",
        "f9894464304b9a318be9ffc5aefdb0912316cc03c4be358d368c5360f5d1ab25"),
    "ebird2025.xml.gz": (
        SCYTHEBILL + "taxonomy/ebird/ebird2025.xml.gz",
        "8e32e957eb7691a64089cb669369fd895461a8667eb59e66d7eed06d16911fad"),
    "2015to2025.csv": (
        SCYTHEBILL + "app/src/main/resources/2015to2025.csv",
        "bf1139aac67e7e1b5e99c51d47cd62f4acd62c395bc8f82218744e72530b045b"),
    "ne_10m_admin_0_countries.geojson": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
        "ne_10m_admin_0_countries.geojson",
        "239eec57ac17f100a11e2536cffc56752c318b50ae765b0918ff7aab4ce8f255"),
}
COUNTRIES = ("CN", "HK", "TW", "TH")       # Natural Earth ISO_A2_EH (ไต้หวัน = TW) = รหัสใน Scythebill
NATIVE = "native"                          # Scythebill ไม่เขียนสถานะนี้ (บรรทัดที่ไม่มี /สถานะ)
# ถ้าชนิดเดียวกันมีหลายสถานะในประเทศเดียว ใช้ตัวที่มาก่อน (ลำดับเดียวกับ Checklist.Status ของ Scythebill)
STATUS_ORDER = (NATIVE, "endemic", "extinct", "introduced", "rarity", "rarity-from-introduced", "escaped")


def fetch(name):
    url, expected = SOURCES[name]
    path = CACHE / name
    if not path.exists():
        print(f"Downloading {name} ...")
        CACHE.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(name + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            out.write(response.read())
        partial.replace(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        sys.exit(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return path


def ebird_species(path):
    """id ของ Scythebill -> (รหัส eBird, ชื่อวิทยาศาสตร์, ชื่ออังกฤษ) เฉพาะระดับชนิด"""
    found = {}
    with gzip.open(path) as stream:
        genus = None
        for event, element in ET.iterparse(stream, events=("start", "end")):
            tag = element.tag.rpartition("}")[2]
            if event == "start" and tag == "genus":
                genus = element.get("name")
            elif event == "end" and tag == "species":
                if element.get("accountId"):
                    common = element.findtext("{*}common-name") or ""
                    found[element.get("id")] = (element.get("accountId"), f"{genus} {element.get('name')}", common)
                element.clear()
    return found


def statuses(path):
    """id ของ Scythebill -> {ประเทศ: สถานะ} เฉพาะ COUNTRIES"""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, codes = line.partition("=")
        species_id, _, status = key.partition("/")
        status = status or NATIVE
        if status not in STATUS_ORDER:
            sys.exit(f"Unknown Scythebill status {status!r}; add it to STATUS_ORDER")
        for country in set(codes.split(",")) & set(COUNTRIES):
            current = result.setdefault(species_id, {}).get(country)
            if current is None or STATUS_ORDER.index(status) < STATUS_ORDER.index(current):
                result[species_id][country] = status
    return result


def split_groups(path, species):
    """id ของ Scythebill -> id อื่นในกลุ่มที่เคยเป็นชนิดเดียวกัน (แถว MULT: ชนิดเดิม -> หลายชนิดในปี 2025)
    กลุ่มที่ซ้อนกัน (แยกหลายรอบ) รวมเป็นกลุ่มเดียว"""
    group_of = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 3 or row[0] != "MULT" or not row[1].startswith("sp"):
                continue
            ids = {i for i in row[2].split(",") if i in species}
            if len(ids) < 2:
                continue
            merged = set(ids)
            for i in ids:
                merged |= group_of.get(i, set())
            for i in merged:
                group_of[i] = merged
    groups = {id(g): g for g in group_of.values()}.values()
    print(f"  {len(groups)} split groups, largest {max(len(g) for g in groups)} species")
    return {i: group - {i} for i, group in group_of.items()}


def write_checklists(species, status_by_id, siblings):
    unmatched = sorted(set(status_by_id) - set(species))
    if unmatched:
        print(f"  {len(unmatched)} checklist ids have no eBird species (skipped), e.g. {unmatched[:5]}")
    by_code = {code: status_by_id.get(species_id, {}) for species_id, (code, _, _) in species.items()}
    out = ASSETS / "country_checklists.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ebird_code", "scientific", "common", *COUNTRIES, "split_with"))
        for species_id, (code, scientific, common) in sorted(species.items(), key=lambda item: item[1][1]):
            split_with = " ".join(sorted(species[i][0] for i in siblings.get(species_id, ())))
            writer.writerow((code, scientific, common, *(by_code[code].get(c, "") for c in COUNTRIES),
                             split_with))
    for country in COUNTRIES:
        counts = {}
        for status in by_code.values():
            if country in status:
                counts[status[country]] = counts.get(status[country], 0) + 1
        print(f"  {country}: " + ", ".join(f"{name} {counts[name]}" for name in STATUS_ORDER if name in counts))
    print(f"Wrote {out.relative_to(PROGRAM)} ({len(species)} species)")


def _polygons(geometry):
    return geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]


def _flat(ring):
    return [round(value, 4) for point in ring for value in point]


def write_borders(path):
    """countries: ขอบเขตของ COUNTRIES (ทุก ring ของทุก polygon) + ธงต่อเส้น: 1 = พรมแดนทางบกกับประเทศอื่น
    (Natural Earth ใช้จุดยอดร่วมกันตามแนวพรมแดน) 0 = ชายฝั่ง หรือแนวที่ติดกันเองใน COUNTRIES
    others: ring ของประเทศอื่นที่อยู่ใกล้ชายฝั่งของ COUNTRIES (เช่นเกาะสอง พม่า ตรงข้ามระนอง) ใช้บอกว่า
    จุดนอกชายฝั่งอยู่บนแผ่นดินประเทศอื่น ไม่ใช่ในทะเลใกล้ฝั่งเรา"""
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    targets = {f["properties"]["ISO_A2_EH"]: f for f in features if f["properties"]["ISO_A2_EH"] in COUNTRIES}
    missing = set(COUNTRIES) - set(targets)
    if missing:
        sys.exit(f"Natural Earth has no {sorted(missing)}")
    other_rings = [ring for f in features if f["properties"]["ISO_A2_EH"] not in COUNTRIES
                   for polygon in _polygons(f["geometry"]) for ring in polygon]
    shared_points = {(round(x, 6), round(y, 6)) for ring in other_rings for x, y in ring}
    countries, coast_cells = {}, set()
    for country in COUNTRIES:
        rings = []
        for polygon in _polygons(targets[country]["geometry"]):
            for ring in polygon:
                shared = [(round(x, 6), round(y, 6)) in shared_points for x, y in ring]
                land = "".join("1" if shared[i] and shared[i + 1] else "0" for i in range(len(ring) - 1))
                rings.append({"lonlat": _flat(ring), "land": land})
                for i, flag in enumerate(land):
                    if flag == "0":
                        coast_cells.update(_cells(ring[i]) | _cells(ring[i + 1]))
        countries[country] = rings
        segments = sum(len(r["land"]) for r in rings)
        land = sum(r["land"].count("1") for r in rings)
        print(f"  {country}: {len(rings)} rings, {segments} segments ({land} land border)")
    others = [_flat(ring) for ring in other_rings
              if any(_cell(point) in coast_cells for point in ring)]
    print(f"  others near the coast: {len(others)} rings, {sum(len(r) // 2 for r in others)} points")
    out = ASSETS / "country_borders.json"
    out.write_text(json.dumps({"source": "Natural Earth 1:10m admin-0 countries v5.1.2 (public domain)",
                               "countries": countries, "others": others}, separators=(",", ":")),
                   encoding="utf-8")
    print(f"Wrote {out.relative_to(PROGRAM)} ({out.stat().st_size // 1024} KB)")


CELL = 0.2   # องศา (~17-22 กม. ในแถบนี้) ช่องรอบจุดชายฝั่ง 3x3 ช่อง ครอบรัศมี COAST_KM (10 กม.) ของแอป


def _cell(point):
    return int(point[0] // CELL), int(point[1] // CELL)


def _cells(point):
    x, y = _cell(point)
    return {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def main():
    species = ebird_species(fetch("ebird2025.xml.gz"))
    write_checklists(species, statuses(fetch("clements.transpose")),
                     split_groups(fetch("2015to2025.csv"), species))
    write_borders(fetch("ne_10m_admin_0_countries.geojson"))


if __name__ == "__main__":
    main()
