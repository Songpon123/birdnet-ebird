"""Group isolated sound events, keeping quiet events and all candidates at scale."""

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from audio_io import read_channels, write_preview
import signal_events
import sound_index


def groups_file(summary, include_clipped=False):
    digest = hashlib.sha256(str(Path(summary).resolve()).encode()).hexdigest()[:12]
    return sound_index.CACHE / f"groups-{digest}-events{'-all' if include_clipped else ''}.json"


def cluster_vectors(vectors):
    """No loudest-only truncation. Large datasets use bounded-memory mini-batch clustering."""
    vectors = np.asarray(vectors, np.float32)
    if len(vectors) <= sound_index.MAX_GROUP_WINDOWS:
        return sound_index.group_labels(vectors), "average linkage"
    from sklearn.cluster import MiniBatchKMeans
    clusters = min(sound_index.MAX_GROUPS, max(2, int(math.sqrt(len(vectors)) * 2)))
    labels = MiniBatchKMeans(n_clusters=clusters, batch_size=512, n_init=3, random_state=0).fit_predict(vectors)
    return labels, "mini-batch k-means (all events)"


def event_vectors(source, result, model, log=print):
    """Encode each isolated event; pad short events with silence rather than unrelated calls."""
    events = result["events"]
    signature = hashlib.sha256(json.dumps(dict(version=1, sid=result["sid"], events=events),
                                          sort_keys=True).encode()).hexdigest()[:24]
    target = signal_events.CACHE / f"features-{signature}.npz"
    try:
        with np.load(target) as saved:
            if int(saved["version"]) == 1 and len(saved["vectors"]) == len(events):
                return saved["vectors"].astype(np.float32)
    except (OSError, ValueError, KeyError):
        pass
    vectors = []
    with tempfile.TemporaryDirectory(prefix="birdnet-event-encode-") as temp:
        for first in range(0, len(events), 64):
            files = []
            for number, event in enumerate(events[first:first + 64], first):
                samples, sr = read_channels(source, event["start"], event["end"], event.get("channel", "mix"))
                if len(samples) < sr * 3:
                    remaining = sr * 3 - len(samples)
                    samples = np.pad(samples, (remaining // 2, remaining - remaining // 2))
                wav = Path(temp) / f"event-{number:07d}.wav"
                write_preview(samples, sr, wav)
                files.append(wav)
            encoded = model.encode([str(wav) for wav in files], n_workers=1, overlap_duration_s=1.5,
                                   show_stats=None).to_structured_array()
            by_file = {wav.name: [] for wav in files}
            for row in encoded:
                # BirdNET's encoding result carries the input path for each window.
                name = Path(str(row["input"])).name
                if name in by_file:
                    by_file[name].append(row["embedding"])
            for wav in files:
                if not by_file[wav.name]:
                    raise RuntimeError(f"No embedding for {wav.name}")
                vector = np.mean(by_file[wav.name], axis=0).astype(np.float32)
                if not np.isfinite(vector).all():
                    raise RuntimeError(f"Invalid embedding for {wav.name}")
                vector /= max(float(np.linalg.norm(vector)), 1e-9)
                vectors.append(vector)
                wav.unlink()
            log(f"{Path(source).name}: encoded {len(vectors)}/{len(events)} isolated events")
    if result.get("source"):
        from field_audio_to_ebird import source_id
        if source_id(Path(source)) != result["sid"]:
            raise ValueError("Source changed while encoding sound events")
    vectors = np.asarray(vectors, np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as f:
            np.savez_compressed(f, version=1, vectors=vectors.astype(np.float16))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return vectors


def find_groups(summary, include_clipped=False, log=print, model=None):
    from review_store import load_reviews
    rows = load_reviews(Path(summary))
    sources = sound_index.sources(rows, summary)
    spans = sound_index.clip_spans(rows)
    events, vectors = [], []
    for sid, entry in sources.items():
        result = signal_events.scan_source(entry["path"], sid, log=log)
        if not result["events"]:
            continue
        model = model or sound_index._load_birdnet()
        encoded = event_vectors(entry["path"], result, model, log)
        for event, vector in zip(result["events"], encoded):
            if not include_clipped and sound_index.clips_at(spans, sid, event["start"], event["end"]):
                continue
            events.append(dict(event, name=entry["name"], loud=event["score"]))
            vectors.append(vector)
    groups, method = [], "average linkage"
    if events:
        vectors = np.asarray(vectors, np.float32)
        labels, method = cluster_vectors(vectors)
        for label in sorted(set(labels.tolist())):
            members = np.where(labels == label)[0]
            center = vectors[members].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-9)
            mine = [dict(events[i], score=round(float(vectors[i] @ center), 3)) for i in members]
            mine.sort(key=lambda e: -e["score"])
            groups.append(dict(events=mine, windows=len(mine), loud=max(e["loud"] for e in mine),
                               seconds=round(sum(e["end"] - e["start"] for e in mine), 2)))
    groups.sort(key=lambda g: (-len(g["events"]), -g["loud"]))
    for number, group in enumerate(groups, 1):
        group["id"] = number
    sound_index._guess_groups(groups[:sound_index.MAX_GROUPS], sources, None, log)
    result = dict(summary=str(Path(summary).resolve()), made=datetime.now().isoformat(timespec="seconds"),
                  mode="events", method=method, include_clipped=include_clipped, windows=len(events), groups=groups,
                  sources={sid: dict(path=str(e["path"]), name=e["name"]) for sid, e in sources.items()})
    target = groups_file(summary, include_clipped)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(target)
    log(f"Grouped all {len(events)} events into {len(groups)} groups using {method}")
    return result
