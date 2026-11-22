"""ชื่อและที่เก็บไฟล์ของแท็บ Merge clips — ตัวต่อเสียงจริงคือ group_clips ใน field_audio_to_ebird.py"""

import re
from pathlib import Path

CLIP_NAME = re.compile(r"_[A-Z][a-z]+\.[a-z-]+_")      # …_Parus.cinereus_… = คลิปที่แอปตัด


def species_of(clip):
    """คลิปของแอปอยู่ในโฟลเดอร์ชื่อชนิด (<ผล>/Asian Tit/…, Ready/Asian Tit/…) -> 'Asian Tit' หรือ None"""
    path = Path(clip)
    return path.parent.name if CLIP_NAME.search(path.name) else None


def species_in(clips):
    """ชนิดในรายการ ตามลำดับที่พบ ไม่ซ้ำ"""
    return list(dict.fromkeys(name for name in map(species_of, clips) if name))


def suggested_name(clips):
    """'Asian Tit merged' — หลายชนิด = สองชื่อแรก, ไฟล์ที่ไม่ได้มาจากแอป = ชื่อไฟล์แรก"""
    if not clips:
        return ""
    return f"{' and '.join(species_in(clips)[:2]) or Path(clips[0]).stem} merged"


def free_path(folder, name):
    """<โฟลเดอร์>/<ชื่อ>.wav ที่ยังไม่มีไฟล์ ชื่อซ้ำ = ต่อท้าย 2, 3, … ไม่ทับไฟล์เดิม"""
    stem = re.sub(r"\.wav$", "", str(name).strip(), flags=re.IGNORECASE)
    stem = re.sub(r'[\\/:*?"<>|]', "_", stem).strip(" .") or "Merged"
    path, number = Path(folder) / f"{stem}.wav", 2
    while path.exists():
        path, number = Path(folder) / f"{stem} {number}.wav", number + 1
    return path
