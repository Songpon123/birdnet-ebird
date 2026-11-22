"""ชื่อนกตาม eBird (eBird/Clements 2025) — ชื่อที่ต้องใช้ตอนอัปโหลดขึ้น eBird / Macaulay Library

ป้ายชื่อของ BirdNET ใช้รายชื่อชุดเก่า: จาก 9,834 ชนิดนกของ BirdNET 3.0 มี 1,564 ชนิดที่ชื่ออังกฤษต่างจาก eBird
(Black-throated Bushtit = Black-throated Tit, Grey Heron = Gray Heron) และ 58 ชนิดที่ชื่อวิทยาศาสตร์ต่างกัน
(Charadrius dubius = Thinornis dubius) ใช้ข้อมูลในเครื่อง (assets/country_checklists.csv) ไม่ต้องต่อเน็ต
"""

import csv
import re
from functools import lru_cache

import country_checklist
import paths


def _key(common):
    """ชื่ออังกฤษที่ตัดความต่างของการสะกดออก: grey/gray, ขีด, Bushtit/Tit"""
    text = str(common).casefold().replace("grey", "gray").replace("-", " ").replace("’", "'")
    return " ".join(re.sub(r"\bbushtit\b", "tit", text).split())


@lru_cache(maxsize=1)
def _by_key():
    index = {}
    for row in country_checklist._species()[0].values():
        index.setdefault(_key(row["common"]), row)
    return index


def to_ebird(scientific, common=""):
    """ชื่อจาก BirdNET -> {'common', 'scientific', 'code'} ตาม eBird หรือ None (ไม่ใช่ชนิดนกใน eBird)
    ลำดับ: ชื่อวิทยาศาสตร์ > ชื่ออังกฤษ > ชื่ออังกฤษที่ไม่สนการสะกด"""
    by_scientific, by_common, _codes = country_checklist._species()
    row = by_scientific.get(" ".join(str(scientific or "").split()[:2]))
    if row is None and str(common or "").strip():
        row = by_common.get(str(common).strip().casefold()) or _by_key().get(_key(common))
    if row is None:
        return None
    return {"common": row["common"], "scientific": row["scientific"], "code": row["ebird_code"]}


@lru_cache(maxsize=1)
def _label_groups():
    """ชื่อวิทยาศาสตร์ในป้ายของ BirdNET 3.0 -> กลุ่ม (Aves, Insecta, Amphibia, Mammalia …)"""
    groups = {}
    for path in sorted(paths.BIRDNET_MODELS.glob("acoustic-models/v3.0/labels_raw-*.csv")):
        try:
            with path.open(encoding="utf-8") as stream:
                for row in csv.reader(stream, delimiter=";"):
                    if len(row) > 4:
                        groups[row[2]] = row[4]
        except OSError:
            continue
    return groups


def is_bird_label(scientific, common=""):
    """ป้ายนี้ของ BirdNET เป็นนกไหม (แมลง กบ สัตว์อื่น = False) ใช้กลุ่มในไฟล์ป้ายของโมเดล ถ้าไม่มีดูจากระบบชื่อ eBird"""
    group = _label_groups().get(str(scientific))
    return group == "Aves" if group else to_ebird(scientific, common) is not None


def row_names(row):
    """(ชื่อสามัญ, ชื่อวิทยาศาสตร์) ตาม eBird ของคลิปใน summary (ผลรุ่นเก่ายังเป็นชื่อของ BirdNET)"""
    common, scientific = row.get("Species (common)") or "", row.get("Species (scientific)") or ""
    names = to_ebird(scientific, common)
    return (names["common"], names["scientific"]) if names else (str(common), str(scientific))


def birdnet_name(row):
    """ชื่อที่ BirdNET ใช้ ถ้าต่างจากชื่อ eBird ('' ถ้าเหมือนกัน)"""
    original = str(row.get("BirdNET name") or row.get("Species (common)") or "")
    return original if original and original != row_names(row)[0] else ""
