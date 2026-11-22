"""Stream audio into time/frequency events, independently of species predictions."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

import paths
from audio_io import audio_duration, read_channels

CACHE = paths.CACHE / "signal-events"
VERSION = 1


@dataclass(frozen=True)
class DetectionParams:
    low: float = 150.0
    high: float = 12000.0
    minimum: float = 0.05
    maximum: float = 15.0
    gap: float = 0.3
    excess_db: float = 12.0
    channel: str = "mix"

    def validate(self):
        values = (self.low, self.high, self.minimum, self.maximum, self.gap, self.excess_db)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Detection parameters must be finite numbers")
        if not 0 <= self.low < self.high <= 192000:
            raise ValueError("Choose a valid frequency range in Hz")
        if not 0.02 <= self.minimum <= self.maximum <= 60:
            raise ValueError("Detection duration must be between 0.02 and 60 seconds")
        if not 0 <= self.gap <= 5 or not 3 <= self.excess_db <= 60:
            raise ValueError("Gap must be 0–5 seconds and threshold 3–60 dB above background")
        if self.channel not in ("mix", "left", "right"):
            raise ValueError("Choose mix, left, or right")
        return self


def _same_band(a, b):
    return max(a["low"], b["low"]) <= min(a["high"], b["high"]) + 150


def merge_events(events, gap):
    """Join neighbouring syllables in the same frequency band, including chunk duplicates."""
    merged, complete = [], []
    for event in sorted(events, key=lambda e: e["start"]):
        event = dict(event)
        complete.extend(old for old in merged if old["end"] + gap < event["start"])
        merged = [old for old in merged if old["end"] + gap >= event["start"]]
        matching = [i for i, old in enumerate(merged)
                    if old["end"] + gap >= event["start"] and _same_band(old, event)]
        for i in reversed(matching):
            old = merged.pop(i)
            event.update(start=min(old["start"], event["start"]), end=max(old["end"], event["end"]),
                         low=min(old["low"], event["low"]), high=max(old["high"], event["high"]),
                         score=max(old["score"], event["score"]))
        merged.append(event)
    return sorted(complete + merged, key=lambda e: e["start"])


def detect_samples(samples, sr, params=None):
    """Return candidate sounds, not species IDs. Thresholds are local spectral excess in dB."""
    from scipy import ndimage, signal
    p = (params or DetectionParams()).validate()
    samples = np.asarray(samples, np.float32)
    n_fft = max(256, 2 ** int(round(math.log2(sr * 0.032))))
    if len(samples) < n_fft * 2 or p.low >= sr / 2:
        return []
    freq, times, z = signal.stft(samples, fs=sr, nperseg=n_fft, noverlap=3 * n_fft // 4,
                                  boundary=None, padded=False)
    power = 10 * np.log10(ndimage.uniform_filter(np.abs(z) ** 2, size=(3, 3)) + 1e-15)
    floor = np.median(power, axis=1, keepdims=True)
    excess = power - floor
    band = (freq >= p.low) & (freq <= min(p.high, sr / 2))
    mask = (excess >= p.excess_db) & band[:, None]
    mask = ndimage.binary_closing(mask, structure=np.ones((2, 3)))
    labels, _count = ndimage.label(mask)
    events = []
    hop = n_fft / (4 * sr)
    for number, region in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = region
        mine = labels[region] == number
        if mine.sum() < 8:
            continue
        a, b = max(0, float(times[cols.start]) - hop / 2), min(len(samples) / sr, float(times[cols.stop - 1]) + hop / 2)
        if b - a < min(p.minimum, 0.03):
            continue
        events.append(dict(start=a, end=b, low=float(freq[rows.start]), high=float(freq[rows.stop - 1]),
                           score=float(excess[region][mine].max())))
    # Filter after joining, so a phrase can comprise several short syllables.
    return [e for e in merge_events(events, p.gap) if p.minimum <= e["end"] - e["start"] <= p.maximum]


def cache_file(sid, params=None):
    p = (params or DetectionParams()).validate()
    key = hashlib.sha256(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()[:16]
    safe = hashlib.sha256(str(sid).encode()).hexdigest()[:24]
    return CACHE / f"{safe}-{key}.json"


def scan_source(source, sid=None, params=None, *, force=False, chunk_seconds=60.0, log=print):
    from field_audio_to_ebird import source_id
    p = (params or DetectionParams()).validate()
    source = Path(source)
    current_sid = source_id(source)
    if sid and sid != current_sid:
        raise ValueError(f"{source.name} changed since analysis; analyze it again")
    sid = current_sid
    target = cache_file(sid, p)
    if not force:
        try:
            cached = json.loads(target.read_text())
            if cached.get("version") == VERSION and cached.get("sid") == sid:
                return cached
        except (OSError, ValueError):
            pass
    if chunk_seconds <= 0:
        raise ValueError("Chunk duration must be positive")
    duration = audio_duration(source)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Cannot determine audio duration: {source.name}")
    events, begin = [], 0.0
    margin = p.maximum + p.gap + 1.0
    while begin < duration:
        end = min(duration, begin + chunk_seconds)
        first, last = max(0, begin - margin), min(duration, end + margin)
        samples, sr = read_channels(source, first, last, p.channel)
        for event in detect_samples(samples, sr, p):
            event = dict(event, start=event["start"] + first, end=event["end"] + first)
            center = (event["start"] + event["end"]) / 2
            if begin <= center < end or (end == duration and center == end):
                events.append(event)
        log(f"{source.name}: {end:.0f}/{duration:.0f} s · {len(events)} candidate sounds")
        begin = end
    events = [e for e in merge_events(events, 0.02) if p.minimum <= e["end"] - e["start"] <= p.maximum]
    for event in events:
        event.update(start=round(event["start"], 4), end=round(event["end"], 4),
                     low=round(event["low"]), high=round(event["high"]), score=round(event["score"], 2),
                     best=round((event["start"] + event["end"]) / 2, 4), sid=sid, channel=p.channel)
    if source_id(source) != sid:
        raise ValueError("The source changed while scanning; results were not saved")
    result = dict(version=VERSION, sid=sid, source=str(source.resolve()), duration=duration,
                  params=asdict(p), events=events)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv):
    """Job input is a JSON document of sources and parameters, never executable text."""
    if len(argv) != 1:
        print("usage: --signal-events JOB.json")
        return 2
    try:
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        params = DetectionParams(**job.get("params", {})).validate()
        for item in job["sources"]:
            result = scan_source(item["path"], item.get("sid"), params, force=job.get("force", False),
                                 log=lambda text: print(text, flush=True))
            print("EVENTS " + str(cache_file(result["sid"], params)), flush=True)
        return 0
    except KeyboardInterrupt:
        print("ERROR stopped; completed recordings are kept", flush=True)
        return 130
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR {exc}", flush=True)
        return 1
