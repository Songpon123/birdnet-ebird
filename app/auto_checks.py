"""ตรวจคลิปของ BirdNET อัตโนมัติเบื้องหลัง (งาน "Auto-checks" หลังเปิดผลชุดในแท็บ Review):

  1. เสียงที่ไม่ใช่นกดังแข่ง: BirdNET 3.0 ฟังคลิปอีกรอบ ถ้าในช่วงที่เจอนก มีแมลง กบ หรือสัตว์อื่นได้คะแนน
     เท่ากับหรือสูงกว่านก -> เตือน (สิ่งที่ BirdNET ว่าเป็นนกอาจเป็นแมลง/กบ)
  2. Second opinion (เทียบกับเสียง xeno-canto) ให้คลิปคะแนน BirdNET ต่ำกว่า SECOND_OPINION_BELOW
     (ทุก ID ผิดที่ตรวจไว้ได้คะแนน 0.53-0.60) ต้องมี xeno-canto key
  3. ประเภทเสียง (Song / Call) ดู sound_type.py: เวกเตอร์ของทุกคลิป (รวมที่เพิ่มเอง) + เสียงอ้างอิงที่ระบุ
     ว่าเป็น song/call ของแต่ละชนิด (ต้องมี xeno-canto key) แล้วฝึกตัวเดาใหม่

ผลเก็บใน data/cache/auto-checks/<sha256 ของคลิป>.json — ไม่เขียนลง summary.xlsx เพราะผู้ตรวจอาจบันทึกผลตรวจ
พร้อมกัน; หยุดกลางคันได้ รันใหม่ทำต่อจากคลิปที่ยังไม่ได้ตรวจ
GUI เรียกผ่าน `field_audio_to_ebird.py --auto-checks SUMMARY.xlsx`
"""

import json
import sys
from pathlib import Path

import numpy as np

import manual_clips
import paths
import result_folders
import second_opinion
import sound_type
import species_names
from result_roots import all_results
from review_store import load_reviews

CACHE = paths.CACHE / "auto-checks"
VERSION = 1
SLOW_BELOW = 50 * 1024          # ไบต์/วิ: โหลดเสียงอ้างอิงช้ากว่านี้ 20 วิ = หยุดโหลดในรอบนี้ (รอบหน้าลองใหม่)
SECOND_OPINION_BELOW = 0.65
NONBIRD_MIN = 0.25             # คะแนนขั้นต่ำของเสียงที่ไม่ใช่นกที่จะเตือน


def cached(clip_sha):
    try:
        data = json.loads((CACHE / f"{clip_sha}.json").read_text(encoding="utf-8"))
        return data if data.get("version") == VERSION else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _save(clip_sha, data):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{clip_sha}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(data, version=VERSION), indent=1), encoding="utf-8")
    temporary.replace(path)


def _score(row):
    try:
        value = float(row.get("Max confidence"))
    except (TypeError, ValueError):
        return None
    return None if value != value else value


def wants_second_opinion(row):
    score = _score(row)
    return score is not None and score < SECOND_OPINION_BELOW


def needs_check(row, has_xc_key):
    """คลิปนี้ยังต้องตรวจอัตโนมัติไหม (ยังไม่เคยฟังซ้ำ หรือยังไม่มี second opinion ที่ควรมี)"""
    sha = str(row.get("Generated sha256") or "")
    if not sha or sha == "nan":
        return False
    if sound_type.needs_work(row, has_xc_key):
        return True
    if manual_clips.is_manual(row):
        return False
    data = cached(sha)
    if data is None:
        return True
    return (has_xc_key and wants_second_opinion(row) and "second_error" not in data
            and second_opinion.cached_result(sha) is None)


def flags_for(row):
    """{'nonbird': True/False/None, 'second': True/False/None} — None = ยังไม่ได้ตรวจ/บอกไม่ได้"""
    sha = str(row.get("Generated sha256") or "")
    data = cached(sha) if sha else None
    second = second_opinion.disagrees(second_opinion.cached_result(sha)) if sha else None
    return {"nonbird": None if data is None else bool(data.get("nonbird")),
            "second": second if wants_second_opinion(row) else None}


def describe_nonbird(row):
    """'Field Cricket 0.83 vs bird 0.61' หรือ ''"""
    sha = str(row.get("Generated sha256") or "")
    found = (cached(sha) or {}).get("nonbird") if sha else None
    return f"{found['common']} {found['score']:.2f} vs bird {found['bird']:.2f}" if found else ""


def nonbird_competitor(detections, bird_scientific, spans=None):
    """เสียงที่ไม่ใช่นกที่ดังแข่งในช่วงที่เจอนก -> dict หรือ None
    detections = [(start, end, label 'Scientific_Common', score)], bird_scientific = ชื่อตาม eBird ของนกในคลิป"""
    def inside(start, end):
        return spans is None or any(start < b and end > a for a, b in spans)

    bird, other = 0.0, None
    for start, end, label, score in detections:
        if not inside(start, end):
            continue
        scientific, _, common = str(label).partition("_")
        if species_names.is_bird_label(scientific, common):
            names = species_names.to_ebird(scientific, common)
            if names and names["scientific"] == bird_scientific:
                bird = max(bird, score)
        elif other is None or score > other["score"]:
            other = {"scientific": scientific, "common": common or scientific, "score": round(float(score), 3)}
    if other and other["score"] >= max(bird, NONBIRD_MIN):
        return dict(other, bird=round(bird, 3))
    return None


def _spans(row):
    try:
        windows = json.loads(row.get("Detection boxes") or "")["windows"]
        return [(float(a), float(b)) for a, b, *_rest in windows] or None
    except (TypeError, ValueError, KeyError):
        return None


BATCH = 40                     # BirdNET ฟังทีละหลายคลิปเร็วกว่าทีละคลิป ~5 เท่า


def run(summary, xc_key="", analyzer=None, log=print):
    """ตรวจทุกคลิปของ BirdNET ใน summary ที่ยังไม่ได้ตรวจ -> จำนวนคลิปที่ตรวจ"""
    summary = Path(summary)
    rows = [row for row in load_reviews(summary) if needs_check(row, bool(xc_key))]
    if not rows:
        log("auto-checks: nothing to do")
        return 0
    if analyzer is None:
        log("Loading BirdNET 3.0 …")
        from v3_model import V3Analyzer
        analyzer = V3Analyzer()
    import xeno_canto
    xeno_canto.MIN_SPEED, xeno_canto.downloads_paused = SLOW_BELOW, False
    waiting = 0                     # second opinion / เสียงอ้างอิงที่เลื่อนไปรอบหน้าเพราะเน็ตช้า

    heard, encoded = {}, {}        # รอบนี้ BirdNET ฟังแต่ละคลิปครั้งเดียว second opinion ใช้ผลนี้ต่อ
    # 1) เสียงที่ไม่ใช่นกดังแข่ง: ฟังคลิปซ้ำทีละชุด บันทึกผลทุกชุด (หยุดกลางคันได้)
    listen = []
    for row in rows:
        clip = (summary.parent / str(row["File"])).resolve()
        if (not manual_clips.is_manual(row) and "nonbird" not in (cached(str(row["Generated sha256"])) or {})
                and clip.is_file()):
            listen.append((row, clip))
    for first in range(0, len(listen), BATCH):
        batch = listen[first:first + BATCH]
        log(f"auto-check: listening to clips {first + 1}-{first + len(batch)} of {len(listen)} …")
        by_clip = second_opinion.hear(analyzer.model, [clip for _row, clip in batch], top_k=25,
                                      speed=second_opinion.MANY_CLIPS)
        heard.update(by_clip)
        for row, clip in batch:
            sha = str(row["Generated sha256"])
            found = nonbird_competitor(by_clip[str(clip)], species_names.row_names(row)[1], _spans(row))
            if found:
                log(f"  {species_names.row_names(row)[0]} {row.get('Clock time') or ''}: non-bird as strong — "
                    f"{found['common']} {found['score']:.2f}")
            _save(sha, dict(cached(sha) or {}, nonbird=found))

    # 3a) ประเภทเสียง: เวกเตอร์ของคลิป (เร็ว ทำก่อนงานที่ต้องโหลดไฟล์)
    encoded.update(_clip_vectors(summary, rows, analyzer, log))

    # 2) second opinion ของคลิปคะแนนต่ำ (ชนิดใหม่ต้องโหลดเสียงอ้างอิงจาก xeno-canto ครั้งแรก)
    asks = [row for row in rows if xc_key and wants_second_opinion(row) and not manual_clips.is_manual(row)
            and "second_error" not in (cached(str(row["Generated sha256"])) or {})
            and second_opinion.cached_result(str(row["Generated sha256"])) is None]
    for first in range(0, len(asks), BATCH):
        batch = asks[first:first + BATCH]
        _listen_for_second_opinion(summary, batch, analyzer.model, heard, encoded, log)
        for number, row in enumerate(batch, start=first + 1):
            sha = str(row["Generated sha256"])
            clip = str((summary.parent / str(row["File"])).resolve())
            if not xeno_canto.downloads_paused:
                log(f"auto-check: second opinion {number}/{len(asks)} — {species_names.row_names(row)[0]} "
                    f"{row.get('Clock time') or ''}")
            try:
                second_opinion.run(summary, row["_row"], xc_key, log=lambda text: log("  " + text), analyzer=analyzer,
                                   row=row, heard=heard.get(clip), encoded=encoded.get(clip))
            except (ValueError, FileNotFoundError) as exc:      # ถาวร (เช่น ไม่มีชนิดให้เทียบ): จำไว้ ไม่ลองซ้ำ
                error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                log(f"  second opinion: {error}")
                _save(sha, dict(cached(sha) or {}, second_error=error))
            except xeno_canto.XenoCantoSlow as exc:          # ต้องโหลดไฟล์อ้างอิงแต่เน็ตช้า: รอบหน้า
                if waiting == 0:
                    log(f"  {exc} — clips that need new references wait for next time")
                waiting += 1
            except Exception as exc:  # noqa: BLE001  เน็ต/ไฟล์อ้างอิงชั่วคราว: ข้ามคลิปนี้ไปก่อน รอบหน้าลองใหม่
                log(f"  second opinion skipped for now ({type(exc).__name__}: "
                    f"{str(exc).splitlines()[0] if str(exc) else ''})")

    # 3b) เสียงอ้างอิงแยก song/call ของแต่ละชนิด แล้วฝึกตัวเดาใหม่ (รวมคลิปที่ผู้ตรวจเลือก Sound type ไว้ทุกผลชุด)
    if xc_key:
        waiting += _typed_references(rows, xc_key, analyzer, log)
    removed = second_opinion.tidy_prepared() + xeno_canto.tidy_partial()
    if removed:
        log(f"auto-check: removed {removed} temporary reference file(s)")
    if xeno_canto.downloads_paused:     # GUI อ่านบรรทัด SLOW แล้วบอกในแถบสถานะ
        log(f"SLOW xeno-canto was slower than {SLOW_BELOW // 1024} KB/s: {waiting} second opinion(s)/reference "
            "set(s) need downloads and will run next time; everything else is done")
    summaries = {Path(info["path"]).resolve() for info in all_results(result_folders.root_of(summary))} | {summary.resolve()}
    sound_type.train(sorted(summaries), log=log)
    return len(rows)


def _listen_for_second_opinion(summary, batch, model, heard, encoded, log):
    """ฟังคลิปชุดนี้ที่รอบนี้ยังไม่ได้ฟัง (เช่น หยุดแล้วทำต่อ) รวดเดียว — ฟังไม่ได้ second_opinion.run ลองทีละคลิปเอง"""
    clips = [(summary.parent / str(row["File"])).resolve() for row in batch]
    clips = [clip for clip in clips if clip.is_file()]
    try:
        unheard = [clip for clip in clips if str(clip) not in heard]
        if unheard:
            heard.update(second_opinion.hear(model, unheard, speed=second_opinion.MANY_CLIPS))
        unencoded = [clip for clip in clips if str(clip) not in encoded]
        if unencoded:
            encoded.update(second_opinion.vectors(model, unencoded, speed=second_opinion.MANY_CLIPS))
    except RuntimeError as exc:
        log(f"  could not listen to these clips together: {exc}")


def _clip_vectors(summary, rows, analyzer, log):
    wanted = []
    for row in rows:
        if not sound_type.needs_vectors(row):
            continue
        clip = (summary.parent / str(row["File"])).resolve()
        if clip.is_file():
            wanted.append((row, clip))
        else:
            sound_type.save_clip_vectors(str(row["Generated sha256"]), np.zeros((0, 1)))   # ไม่ต้องลองซ้ำ
    every = {}
    for first in range(0, len(wanted), BATCH):
        batch = wanted[first:first + BATCH]
        log(f"auto-check: sound type of clips {first + 1}-{first + len(batch)} of {len(wanted)} …")
        encoded = second_opinion.vectors(analyzer.model, [clip for _row, clip in batch],
                                         speed=second_opinion.MANY_CLIPS)
        every.update(encoded)
        for row, clip in batch:
            windows = None
            try:
                windows = json.loads(row.get("Detection boxes") or "")["windows"]
            except (TypeError, ValueError, KeyError):
                pass
            sound_type.save_clip_vectors(str(row["Generated sha256"]), sound_type.pick_segments(encoded[str(clip)], windows))
    return every


def _typed_references(rows, xc_key, analyzer, log):
    model = analyzer.model
    labels = {str(label).partition("_")[0]: str(label) for label in model.species_list}
    places = {}
    for row in rows:
        places.setdefault(sound_type.species_of(row), (row.get("Latitude"), row.get("Longitude")))
    todo = [name for name in sorted(places) if sound_type.needs_references(name)]
    waiting = 0
    for number, name in enumerate(todo, start=1):
        scientific = second_opinion.birdnet_label(name, labels)
        if scientific is None:                    # BirdNET ไม่รู้จักชนิดนี้ (เช่น ชื่อที่ผู้ตรวจพิมพ์เอง)
            sound_type._mark_checked(name)
            continue
        import xeno_canto
        if not xeno_canto.downloads_paused:
            log(f"auto-check: song/call references {number}/{len(todo)} — {name}")
        lat, lon = (second_opinion._number(value) for value in places[name])
        try:
            sound_type.ensure_references(model, scientific, labels[scientific], xc_key, lat, lon,
                                         log=lambda text: log("  " + text))
            sound_type._mark_checked(name)
        except xeno_canto.XenoCantoSlow:           # เน็ตช้า: ชนิดนี้รอรอบหน้า (ไม่จำว่าเช็กแล้ว)
            waiting += 1
        except Exception as exc:  # noqa: BLE001  เน็ต/xeno-canto ล่มชั่วคราว: รอบหน้าลองใหม่
            log(f"  skipped for now ({type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''})")
    return waiting


def main(argv):
    from app_settings import load_settings
    if len(argv) != 1:
        print("usage: --auto-checks SUMMARY.xlsx")
        return 2
    try:
        done = run(argv[0], load_settings().get("xeno_canto_key", ""), log=lambda text: print(text, flush=True))
    except KeyboardInterrupt:
        print("auto-checks stopped; finished clips are kept", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR {exc}", flush=True)
        return 1
    print(f"AUTO done {done}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
