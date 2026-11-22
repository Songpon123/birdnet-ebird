"""คลิปที่ผู้ตรวจเพิ่มเอง (กึ่ง manual): เลือกช่วงในไฟล์ต้นฉบับ ใส่ชนิดเอง -> ตัดแบบเดียวกับคลิปของ BirdNET
(คุณภาพเต็ม คง bit depth, normalize, spectrogram) แล้วขึ้นในแท็บ Review เป็น Pending

แถวเหล่านี้มี Origin = Manual: การวิเคราะห์ใหม่ (รวม --force) ไม่แทนที่และไม่ลบไฟล์ของมัน
GUI เรียกผ่าน `field_audio_to_ebird.py --manual-clip add|suggest ...` (process แยก เหมือนงานอื่น)
"""

import argparse
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from review_store import SOUND_TYPES  # noqa: F401  ชุดเดียวกับช่อง Sound type ในแท็บ Review

ORIGIN = "Origin"
MANUAL = "Manual"
MIN_SECONDS = 0.5
MAX_SECONDS = 600.0          # คลิปยาวกว่านี้ให้ตัดในโปรแกรมตัดต่อเสียง
SUGGEST_FLOOR = 0.02         # คะแนนต่ำสุดที่ยังแสดงเป็นตัวเลือก (เดาเท่านั้น ไม่ยืนยัน)
BIRDNET_WINDOW = 3.0


# ----------------------------- ใช้ได้ทั้ง GUI และ CLI (ไม่ import ของหนัก) -----------------------------
def is_manual(row) -> bool:
    """row = dict จาก load_reviews หรือแถว pandas"""
    return row is not None and str(row.get(ORIGIN) or "").strip() == MANUAL


def manual_mask(df):
    """DataFrame -> Series[bool] แถวที่ผู้ตรวจเพิ่มเอง"""
    import pandas as pd
    if ORIGIN not in df.columns:
        return pd.Series(False, index=df.index)
    return df[ORIGIN].astype(str).str.strip().eq(MANUAL)


def parse_offset(text) -> float:
    """ตำแหน่งในไฟล์: '904', '904.5', '15:04', '15:04.5', '1:15:04' -> วินาที"""
    value = str(text).strip()
    match = re.fullmatch(r"(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d*)?)", value)
    if not match or not value:
        raise ValueError(f"Use minutes:seconds, e.g. 15:04 or 15:04.5 (got {text!r})")
    first, second, seconds = match.groups()
    hours, minutes = (int(first), int(second)) if second is not None else (0, int(first or 0))
    if second is not None or first is not None:
        if float(seconds) >= 60 or (second is not None and minutes >= 60):
            raise ValueError(f"Seconds and minutes must be below 60 (got {text!r})")
    return hours * 3600 + minutes * 60 + float(seconds)


def parse_clock(text, recording_start: datetime) -> float:
    """เวลาจริง '14:54:49', '14:54', '1454', '145449' -> วินาทีนับจากเริ่มบันทึก (ข้ามเที่ยงคืนได้)"""
    value = str(text).strip()
    match = (re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d*)?))?", value)
             or re.fullmatch(r"(\d{2})(\d{2})(\d{2}(?:\.\d*)?)?", value))
    if not match:
        raise ValueError(f"Use clock time HH:MM:SS, e.g. 14:54:49 (got {text!r})")
    hours, minutes, seconds = int(match[1]), int(match[2]), float(match[3] or 0)
    if hours > 23 or minutes > 59 or seconds >= 60:
        raise ValueError(f"Not a clock time: {text!r}")
    clock = recording_start.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    clock += timedelta(seconds=seconds)
    if clock < recording_start - timedelta(hours=1):          # หลังเที่ยงคืนของการบันทึกข้ามวัน
        clock += timedelta(days=1)
    return (clock - recording_start).total_seconds()


def format_offset(seconds, decimals=1) -> str:
    """904.5 -> '15:04.5'; 3725 -> '1:02:05.0'"""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{secs:0{3 + decimals if decimals else 2}.{decimals}f}"
    return f"{int(hours)}:{int(minutes):02d}:{text}" if hours >= 1 else f"{int(minutes)}:{text}"


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def recording_start_of(row):
    """เวลาเริ่มบันทึกของไฟล์ต้นฉบับจากแถวใน summary -> datetime หรือ None
    Clock time ถูกปัดทิ้งเศษวินาที และเวลาเริ่มบันทึกเป็นวินาทีเต็มเสมอ จึง
    เวลาเริ่ม = วันที่ใน Analysis settings + Clock time - floor(Start offset) (คลิปหลังเที่ยงคืนเลื่อนวันให้)"""
    try:
        settings = json.loads(row.get("Analysis settings") or "{}")
    except (TypeError, ValueError):
        return None
    offset, clock, day = _number(row.get("Start offset (s)")), row.get("Clock time"), settings.get("date")
    if offset is None or not clock or not day:
        return None
    try:
        first_day = datetime.fromisoformat(str(day)).date()
        start = datetime.fromisoformat(f"{first_day.isoformat()}T{str(clock).strip()}")
    except ValueError:
        return None
    start = start.replace(microsecond=0) - timedelta(seconds=math.floor(offset))
    while start.date() < first_day:                        # คลิปหลังเที่ยงคืน
        start += timedelta(days=1)
    return start


def heard_span(row):
    """ช่วงเวลาจริงที่ได้ยินเสียงในคลิปนี้ (BirdNET เจอ / ผู้ตรวจเลือก) -> (datetime, datetime) หรือ None"""
    start = recording_start_of(row)
    first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
    if start is None or first is None:
        return None
    return start + timedelta(seconds=first), start + timedelta(seconds=last if last is not None else first)


def sources_in(rows):
    """ไฟล์ต้นฉบับใน summary (ตามลำดับที่พบ) -> [Path]"""
    seen = {}
    for row in rows:
        path = str(row.get("Source path") or "").strip()
        if path and path not in seen:
            seen[path] = Path(path)
    return list(seen.values())


def source_context(rows, source):
    """เวลาเริ่มบันทึก พิกัด สถานที่ ของไฟล์ต้นฉบับ จากแถวใน summary (ไม่ต้องอ่าน metadata ใหม่)"""
    source = str(source)
    mine = [row for row in rows if str(row.get("Source path") or "") == source]
    if not mine:
        return None
    context = {"start": None, "lat": None, "lon": None, "place": "", "date_source": "",
               "habitat": [], "source_id": ""}
    for row in mine:
        try:
            settings = json.loads(row.get("Analysis settings") or "{}")
        except (TypeError, ValueError):
            settings = {}
        context["lat"] = context["lat"] if context["lat"] is not None else _number(row.get("Latitude"))
        context["lon"] = context["lon"] if context["lon"] is not None else _number(row.get("Longitude"))
        context["place"] = context["place"] or str(row.get("Place") or "").strip()
        context["date_source"] = context["date_source"] or str(row.get("Date/time source") or "")
        context["source_id"] = context["source_id"] or str(row.get("Source ID") or "")
        if not context["habitat"] and isinstance(settings.get("habitat"), list):
            context["habitat"] = settings["habitat"]
        if context["start"] is None:
            context["start"] = recording_start_of(row)
    return context


def aggregate_suggestions(detections, top=8):
    """detection ของ BirdNET ในช่วงที่เลือก -> ชนิดเรียงตามคะแนนสูงสุด (หนึ่งแถวต่อชนิด)"""
    best = {}
    for det in detections:
        key = det["scientific_name"]
        if key not in best or det["confidence"] > best[key]["confidence"]:
            best[key] = {"common": det["common_name"], "scientific": key,
                         "confidence": round(float(det["confidence"]), 3),
                         "expected": (det.get("is_predicted_for_location_and_date")
                                      if det.get("location_filter_available", True) else None)}
    return sorted(best.values(), key=lambda item: -item["confidence"])[:top]


# ----------------------------- CLI (process แยก): ตัดคลิป / ถาม BirdNET -----------------------------
def _settings_of(existing, source_id):
    """การตั้งค่าตัดคลิป (format/mono/target dB/habitat) ของไฟล์นี้จากแถวของ BirdNET ถ้ามี"""
    import field_audio_to_ebird as fa
    settings = {"format": fa.EXPORT_FORMAT, "mono": fa.MAKE_MONO, "target_dbfs": fa.TARGET_DBFS}
    if existing.empty or "Analysis settings" not in existing.columns or "Source ID" not in existing.columns:
        return settings
    auto = existing[~manual_mask(existing) & existing["Source ID"].astype(str).eq(source_id)]
    for value in auto["Analysis settings"]:
        try:
            found = json.loads(value)
        except (TypeError, ValueError):
            continue
        settings.update({key: found[key] for key in ("format", "mono", "target_dbfs", "habitat")
                         if key in found})
        break
    return settings


def _read_span(source: Path, start_s, end_s):
    """ช่วง [start_s, end_s] ของไฟล์ต้นฉบับเต็มคุณภาพ -> (AudioSegment, subtype สำหรับ export)"""
    import field_audio_to_ebird as fa
    mode, src = fa._open_source(source)
    if mode == "sf":
        return fa._read_segment(source, src, start_s, end_s), fa._depth(src.subtype)[2]
    return src[int(start_s * 1000):int(end_s * 1000)], None


_GEO_MODEL = []                 # โหลดโมเดลพื้นที่ครั้งเดียวต่อ process (เพิ่มหลายคลิปทีเดียว)


def location_expected(scientific, common, lat, lon, recorded):
    """โมเดลพื้นที่ BirdNET 3.0: True/False = ควร/ไม่ควรพบที่นี่ในสัปดาห์นั้น, None = ตรวจไม่ได้"""
    try:
        from v3_model import location_check
        if not _GEO_MODEL:
            import birdnet
            _GEO_MODEL.append(birdnet.load("geo", "3.0", "onnx", precision="fp16"))
        return location_check(scientific, common, lat, lon, recorded, geo_model=_GEO_MODEL[0])
    except Exception:  # noqa: BLE001  ไม่มีโมเดล/ชื่อไม่ตรง: ให้คนตรวจเอง
        return None


def add_clip(summary, source, start_s, end_s, common, scientific, *, recorded, lat, lon,
             place="", sound_type="", notes="", date_source="", spectrogram=True,
             check_location=True, log=print, channel=None):
    """ตัดช่วงที่เลือกเป็นคลิปใหม่ + เพิ่มแถว Origin=Manual ใน summary -> dict ของแถว"""
    import pandas as pd

    import country_checklist
    import field_audio_to_ebird as fa
    from habitat import fits_site, species_habitat

    summary, source = Path(summary), Path(source)
    common, scientific = " ".join(str(common).split()), " ".join(str(scientific).split())
    if not common or len(scientific.split()) < 2:
        raise ValueError("Enter the common name and the scientific name (genus species)")
    if sound_type not in SOUND_TYPES:
        raise ValueError(f"Sound type must be one of: {', '.join(t for t in SOUND_TYPES if t)}")
    if channel not in (None, "mix", "stereo", "left", "right"):
        raise ValueError("Invalid source channel")
    if not source.is_file():
        raise FileNotFoundError(f"Recording not found: {source}")
    start_s, end_s = float(start_s), float(end_s)
    duration = fa._audio_duration(source)
    if duration:
        end_s = min(end_s, duration)
    if start_s < 0 or end_s - start_s < MIN_SECONDS:
        raise ValueError(f"Select at least {MIN_SECONDS} s of sound")
    if end_s - start_s > MAX_SECONDS:
        raise ValueError(f"Select at most {MAX_SECONDS / 60:.0f} minutes; cut longer spans in an audio editor")
    if lat is None or lon is None:
        raise ValueError("The recording location is missing")

    date_folder = summary.parent
    existing = pd.DataFrame()
    if summary.exists():
        existing = pd.read_excel(summary, dtype={"Source ID": str})
    sid = fa.source_id(source)
    if not existing.empty and {"Source path", "Source ID"}.issubset(existing.columns):
        recorded_ids = set(existing.loc[existing["Source path"].astype(str) == str(source.resolve()),
                                        "Source ID"].dropna().astype(str))
        if recorded_ids and sid not in recorded_ids:
            raise ValueError(f"{source.name} changed since it was analyzed; analyze it again first")
    settings = _settings_of(existing, sid)

    clip, subtype = _read_span(source, start_s, end_s)
    if clip is None or len(clip) == 0:
        raise ValueError("The selected span has no audio")
    if channel in ("left", "right"):
        if channel == "right" and clip.channels < 2:
            raise ValueError("This recording has no right channel")
        clip = clip.split_to_mono()[0 if channel == "left" else 1]
    if settings["mono"] and clip.channels > 1:
        clip = clip.set_channels(1)
    length = len(clip) / 1000.0
    peak_dbfs = clip.max_dBFS
    stats_clip = clip[:15000]
    snr = fa.estimate_snr(stats_clip)
    peak_hz, flo, fhi = fa.freq_stats(stats_clip)
    boxes = {"windows": [], "sounds": fa._sound_events(clip, 0.0, length, max_boxes=12 + int(length)),
             "manual": True}
    clip = fa.normalize_to(clip, float(settings["target_dbfs"]))

    clock = recorded + timedelta(seconds=start_s)
    date_tag = recorded.strftime("%Y.%m.%d_%H%M")
    genus_sp = fa.sanitize(scientific).replace(" ", ".")
    species_dir = date_folder / fa.sanitize(common)
    taken = set(existing["File"].astype(str)) if "File" in existing.columns else set()
    # ลำดับต่อจากคลิปที่เพิ่มเองของชนิดนี้ในไฟล์นี้ (#1, #2 … เหมือน occurrence ของ BirdNET)
    number = 1
    if {"Source ID", "Species (scientific)"}.issubset(existing.columns):
        number += int((manual_mask(existing) & existing["Source ID"].astype(str).eq(sid)
                       & existing["Species (scientific)"].astype(str).eq(scientific)).sum())
    while True:
        out_path = species_dir / f"{date_tag}_{clock:%H%M}_{genus_sp}_{sid}_M{number:03d}_{fa.DEFAULT_RATING}.{settings['format']}"
        if str(out_path.relative_to(date_folder)) not in taken and not out_path.exists():
            break
        number += 1
    species_dir.mkdir(parents=True, exist_ok=True)
    fa._export_clip(clip, out_path, settings["format"], subtype)
    try:
        site_habitats = tuple(settings.get("habitat") or ())
        habitat = species_habitat(scientific)
        expected = location_expected(scientific, common, lat, lon, clock) if check_location else None
        row_settings = {"manual": True, "date": recorded.date().isoformat(), "lat": lat, "lon": lon,
                        "format": settings["format"], "mono": settings["mono"],
                        "target_dbfs": settings["target_dbfs"]}
        if site_habitats:
            row_settings["habitat"] = list(site_habitats)
        row = {
            "Species (common)": common,
            "Species (scientific)": scientific,
            "Occurrence #": number,
            "Start offset (s)": round(start_s, 1),
            "Clock time": clock.strftime("%H:%M:%S"),
            "End offset (s)": round(end_s, 1),
            "Clock end": (recorded + timedelta(seconds=end_s)).strftime("%H:%M:%S"),
            "Duration (s)": round(length, 1),
            "Max confidence": None,
            "Expected by location/date": ("Not checked — verify by ear" if expected is None
                                          else "Yes" if expected else "No — verify by ear"),
            "Species habitat": habitat,
            "Fits site habitat": {True: "Yes", False: "No — verify by ear", None: ""}[
                fits_site(habitat, site_habitats)],
            "Country checklist": country_checklist.describe(
                country_checklist.check(scientific, lat, lon, common)),
            "Peak dBFS": round(peak_dbfs, 1),
            "Clipping": "YES" if peak_dbfs >= -0.1 else "",
            "SNR (dB approx, first 15s)": snr if snr is not None else "",
            "Peak freq (Hz)": peak_hz if peak_hz is not None else "",
            "Freq low (Hz)": flo if flo is not None else "",
            "Freq high (Hz)": fhi if fhi is not None else "",
            "Detection boxes": json.dumps(boxes),
            "Place": place or "",
            "File": str(out_path.relative_to(date_folder)),
            "Generated sha256": fa.file_sha256(out_path),
            "Source file": source.name,
            "Source path": str(source.resolve()),
            "Source ID": sid,
            "Source channel": channel if channel in ("left", "right") else "original",
            "Analysis settings": json.dumps(row_settings, sort_keys=True, separators=(",", ":")),
            "Date/time source": date_source or "",
            "Latitude": lat,
            "Longitude": lon,
            "Review status": "Pending",
            "Review notes": notes.strip(),
            ORIGIN: MANUAL,
            "Sound type": sound_type,
        }
        fa.append_summary_rows([row], summary)
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    log(f"  added {common} {clock:%H:%M:%S} ({length:.1f} s) -> {row['File']}")
    if spectrogram:
        try:
            fa.gen_spectrograms([date_folder])
        except Exception as exc:  # noqa: BLE001  คลิปเพิ่มแล้ว; ภาพสร้างใหม่ได้ตอนวิเคราะห์ครั้งหน้า
            log(f"  warning: could not draw the spectrogram: {exc}")
    return row


def _already_clipped(summary, source, start_s, end_s, scientific, channel=None):
    """มีคลิปของชนิดนี้ (ชื่อที่ตรวจแล้วหรือที่ BirdNET ทาย) ทับช่วงนี้อย่างน้อย 1 วินาทีแล้วไหม"""
    from review_store import load_reviews
    if not Path(summary).is_file():
        return False
    for row in load_reviews(Path(summary)):
        if channel in ("left", "right") and row.get("Source channel") != channel:
            continue
        if str(row.get("Source path") or "") != str(Path(source).resolve()):
            continue
        names = {" ".join(str(row.get(key) or "").split()[:2]) for key in ("Reviewed scientific", "Species (scientific)")}
        first, last = _number(row.get("Start offset (s)")), _number(row.get("End offset (s)"))
        if scientific in names and first is not None and \
                min(last if last is not None else first, end_s) - max(first, start_s) >= min(1.0, (end_s - start_s) / 2):
            return True
    return False


def add_batch(items, log=print):
    """เพิ่มหลายคลิปทีเดียว (จาก Find similar / Sound groups / recognizers) -> [แถวที่เพิ่ม]
    items = [{"summary", "source", "start", "end", "recorded", "lat", "lon", "common", "scientific",
              "sound_type"?, "notes"?, "place"?, "date_source"?}] — ช่วงที่มีคลิปของชนิดนั้นแล้วจะข้าม
    spectrogram สร้างทีเดียวตอนจบ"""
    import field_audio_to_ebird as fa
    added, folders = [], set()
    for number, item in enumerate(items, start=1):
        label = f"{item['common']} {float(item['start']):.1f}-{float(item['end']):.1f} s"
        if _already_clipped(item["summary"], item["source"], float(item["start"]), float(item["end"]),
                            " ".join(str(item["scientific"]).split()[:2]), channel=item.get("channel")):
            log(f"SKIPPED {label}: already a clip of this species")
            continue
        try:
            row = add_clip(item["summary"], item["source"], item["start"], item["end"], item["common"],
                           item["scientific"], recorded=datetime.fromisoformat(item["recorded"]),
                           lat=item["lat"], lon=item["lon"], place=item.get("place", ""),
                           sound_type=item.get("sound_type", ""), notes=item.get("notes", ""),
                           date_source=item.get("date_source", ""), spectrogram=False, log=log,
                           channel=item.get("channel"))
        except (OSError, ValueError, RuntimeError) as exc:
            log(f"SKIPPED {label}: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
            continue
        added.append(row)
        folders.add(Path(item["summary"]).parent)
        log("MANUAL " + json.dumps({"summary": str(Path(item["summary"]).resolve()), "file": row["File"],
                                    "common": row["Species (common)"], "clock": row["Clock time"],
                                    "number": number}))
    for folder in sorted(folders):
        try:
            fa.gen_spectrograms([folder])
        except Exception as exc:  # noqa: BLE001  คลิปเพิ่มแล้ว; ภาพสร้างใหม่ได้ภายหลัง
            log(f"  warning: could not draw spectrograms: {exc}")
    log(f"BATCH added {len(added)} of {len(items)}")
    return added


def suggest(source, start_s, end_s, lat, lon, recorded, analyzer=None, top=8):
    """BirdNET 3.0 ฟังเฉพาะช่วงที่เลือก (เกณฑ์ต่ำมาก) -> ชนิดที่น่าจะเป็น พร้อมผลโมเดลพื้นที่/checklist"""
    import soundfile as sf

    import country_checklist

    source = Path(source)
    start_s, end_s = float(start_s), float(end_s)
    if end_s - start_s < BIRDNET_WINDOW:                     # BirdNET ฟังทีละ 3 วินาที: ขยายรอบกลางช่วง
        middle = (start_s + end_s) / 2
        start_s, end_s = max(0.0, middle - BIRDNET_WINDOW / 2), middle + BIRDNET_WINDOW / 2
    clip, _subtype = _read_span(source, start_s, min(end_s, start_s + MAX_SECONDS))
    if clip is None or len(clip) == 0:
        raise ValueError("The selected span has no audio")
    import numpy as np
    samples = np.array(clip.set_channels(1).get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * clip.sample_width - 1))
    if analyzer is None:
        from v3_model import V3Analyzer
        analyzer = V3Analyzer()
    with tempfile.TemporaryDirectory() as temp:
        wav = Path(temp) / "selection.wav"
        sf.write(str(wav), samples, clip.frame_rate, subtype="PCM_16")
        detections = analyzer.detections(wav, 1.5, SUGGEST_FLOOR, lat, lon,
                                         recorded + timedelta(seconds=start_s))
    from field_audio_to_ebird import _ebird_names
    ranking = aggregate_suggestions([_ebird_names(d) for d in detections], top)
    for item in ranking:
        item["checklist"] = country_checklist.describe(
            country_checklist.check(item["scientific"], lat, lon, item["common"]))
    return ranking


def _parser():
    parser = argparse.ArgumentParser(prog="--manual-clip")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add-batch").add_argument("items", help="JSON file: list of clips to add")
    for name in ("add", "suggest"):
        command = commands.add_parser(name)
        command.add_argument("--source", required=True)
        command.add_argument("--start", type=float, required=True, help="seconds into the recording")
        command.add_argument("--end", type=float, required=True)
        command.add_argument("--recorded", required=True, help="recording start, ISO date-time")
        command.add_argument("--lat", type=float, required=True)
        command.add_argument("--lon", type=float, required=True)
        if name == "add":
            command.add_argument("--summary", required=True)
            command.add_argument("--common", required=True)
            command.add_argument("--scientific", required=True)
            command.add_argument("--place", default="")
            command.add_argument("--sound-type", default="")
            command.add_argument("--notes", default="")
            command.add_argument("--date-source", default="")
    return parser


def main(argv):
    """GUI อ่านบรรทัด MANUAL {...} / SUGGEST [...] / ERROR ..."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "add-batch":
            items = json.loads(Path(args.items).read_text(encoding="utf-8"))
            add_batch(items, log=lambda text: print(text, flush=True))
            return 0
        recorded = datetime.fromisoformat(args.recorded)
        if args.command == "add":
            row = add_clip(args.summary, args.source, args.start, args.end, args.common, args.scientific,
                           recorded=recorded, lat=args.lat, lon=args.lon, place=args.place,
                           sound_type=args.sound_type, notes=args.notes, date_source=args.date_source,
                           log=lambda text: print(text, flush=True))
            print("MANUAL " + json.dumps({"summary": str(Path(args.summary).resolve()), "file": row["File"],
                                          "common": row["Species (common)"], "clock": row["Clock time"]}),
                  flush=True)
        else:
            print("SUGGEST " + json.dumps(suggest(args.source, args.start, args.end, args.lat, args.lon,
                                                  recorded)), flush=True)
    except KeyboardInterrupt:
        print("ERROR stopped", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
