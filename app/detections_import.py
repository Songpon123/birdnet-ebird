"""Read detections from an external CSV as review candidates; labels never imply approval."""

import csv
import io
import math
from pathlib import Path, PureWindowsPath
import re

from audio_io import audio_info


def _key(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _relative(value):
    return Path(str(value or "").replace("\\", "/"))


def resolve_source(row, csv_path, audio_root=None):
    name = row.get("INFILE") or row.get("FILENAME") or row.get("FILE") or row.get("SOURCEFILE")
    if not name:
        raise ValueError("Missing IN FILE / filename")
    name_path = _relative(name)
    folder = _relative(row.get("FOLDER") or "")
    candidates = []
    if audio_root:
        root = Path(audio_root).expanduser().resolve()
        for relative in (folder / name_path, folder / name_path.name, Path(name_path.name)):
            target = (root / relative).resolve()
            if target.is_relative_to(root):
                candidates.append(target)
    else:
        if name_path.is_absolute():
            candidates.append(name_path)
        indir = row.get("INDIR") or ""
        if indir and not PureWindowsPath(indir).drive:
            base = _relative(indir)
            if not base.is_absolute():
                base = Path(csv_path).parent / base
            candidates.append(base / folder / name_path)
        candidates += [Path(csv_path).parent / folder / name_path, Path(csv_path).parent / name_path.name]
    for path in dict.fromkeys(p.resolve() for p in candidates):
        if path.is_file():
            return path
    raise ValueError(f"Audio not found: {name}; choose the folder containing the recordings")


def read_candidates(csv_path, audio_root=None):
    csv_path = Path(csv_path)
    text = csv_path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:16384], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = {_key(h) for h in (reader.fieldnames or [])}
    if not headers.intersection({"INFILE", "FILE", "FILENAME", "SOURCEFILE"}) or not headers.intersection(
            {"OFFSET", "OFFSETSECONDS", "START", "STARTTIME", "STARTSECONDS"}):
        raise ValueError("Expected IN FILE, OFFSET and DURATION columns (or start/end equivalents)")
    candidates, errors, seen, durations = [], [], {}, {}
    duplicates = 0
    for line, raw in enumerate(reader, 2):
        row = {_key(key): value.strip() if isinstance(value, str) else value for key, value in raw.items() if key}
        try:
            source = resolve_source(row, csv_path, audio_root)
            start = float(next(row[k] for k in ("OFFSET", "OFFSETSECONDS", "START", "STARTTIME", "STARTSECONDS")
                               if row.get(k)))
            duration = next((row[k] for k in ("DURATION", "DURATIONSECONDS", "LENGTH") if row.get(k)), None)
            end = start + float(duration) if duration is not None else float(next(
                row[k] for k in ("END", "ENDTIME", "ENDSECONDS") if row.get(k)))
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Invalid offset or duration")
            if source not in durations:
                durations[source] = audio_info(source)
            length = durations[source]["duration"]
            if not length or start >= length or end > length + 0.05:
                raise ValueError("Detection is outside the source audio")
            end = min(end, length)
            raw_channel = str(row.get("CHANNEL") or "").lower()
            if raw_channel not in ("", "0", "1", "0.0", "1.0", "left", "right", "mono", "mix"):
                raise ValueError(f"Unsupported channel: {raw_channel}")
            channel = {"0": "left", "0.0": "left", "1": "right", "1.0": "right"}.get(raw_channel,
                      raw_channel if raw_channel in ("left", "right") else "stereo")
            if channel == "right" and durations[source]["channels"] < 2:
                raise ValueError("Channel 1/right does not exist in this mono recording")
            identity = (source, round(start, 5), round(end, 5), channel)
            manual = row.get("MANUALID") or ""
            suggested = manual or row.get("TOP1MATCH") or row.get("TOP1NAME") or ""
            if identity in seen:
                duplicates += 1
                previous = seen[identity]
                if suggested and suggested not in previous["label"].split("; "):
                    previous["label"] = "; ".join(filter(None, (previous["label"], suggested)))
                    previous["note"] += f" Additional unverified label from CSV row {line}: {suggested}."
                continue
            note = f"Imported from {csv_path.name}, row {line}; channel {raw_channel or 'original'}."
            if suggested:
                note += f" Original label (unverified): {suggested}."
            if row.get("TOP1DIST"):
                note += f" Cluster distance: {row['TOP1DIST']}."
            candidates.append(dict(source=source, start=start, end=end, score=0.0, channel=channel,
                                   best=start, note=note, label=suggested, csv_line=line,
                                   padding=0.0))
            seen[identity] = candidates[-1]
        except (ValueError, OSError, StopIteration, TypeError) as exc:
            errors.append(f"Row {line}: {exc or 'missing duration/end'}")
    return dict(candidates=candidates, errors=errors, duplicates=duplicates)
