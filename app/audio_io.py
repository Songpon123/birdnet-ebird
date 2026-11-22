"""อ่านเสียงจากไฟล์ต้นฉบับทีละช่วง (ไฟล์ยาวหลายชั่วโมงก็ไม่ต้องโหลดทั้งก้อน) + เขียนเสียงตัวอย่าง
ใช้ได้ทั้งหน้าจอ (หน้าต่างเพิ่มคลิป/ค้นเสียง/ตัวดู spectrogram) และงานเบื้องหลัง (ดัชนีเสียง) — ไม่ import Tk"""

import subprocess
import json
import math
import io
import wave
from pathlib import Path

import numpy as np

import platform_tools

PREVIEW_PEAK = 0.7          # เสียงตัวอย่างดังขึ้นให้ได้ยินเสียงเบา ๆ (คลิปจริง normalize -3 dBFS เหมือนกัน)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff"}


def collect_audio(folder):
    """ไฟล์เสียงในโฟลเดอร์รวมโฟลเดอร์ย่อย (Zoom เก็บแต่ละไฟล์ในโฟลเดอร์ของตัวเอง เช่น Raw/261030_062349/…)
    ข้ามไฟล์ซ่อน และผลวิเคราะห์ของแอป (โฟลเดอร์ที่มี summary.xlsx, Ready และ Done) ไม่ให้คลิปถูกวิเคราะห์ซ้ำ"""
    folder = Path(folder)
    results = {path.parent for path in folder.rglob("summary.xlsx")}
    found = []
    for path in folder.rglob("*"):
        parts = path.relative_to(folder).parts
        if (path.suffix.lower() in AUDIO_EXTS and path.is_file() and not any(part.startswith(".") for part in parts)
                and not {"Ready", "Done"}.intersection(parts[:-1]) and not results.intersection(path.parents)):
            found.append(path)
    return sorted(found)


def audio_duration(source: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(source)).duration)
    except Exception:  # noqa: BLE001  m4a/aac: ถาม ffprobe
        pass
    probe = platform_tools.bundled_tool("ffprobe")
    if not probe:
        return 0.0
    try:
        result = subprocess.run([probe, "-v", "quiet", "-show_entries", "format=duration",
                                 "-of", "default=nw=1:nk=1", str(source)], capture_output=True,
                                text=True, timeout=30, creationflags=platform_tools.NO_WINDOW)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def read_mono(source: Path, t0: float, t1: float):
    """ช่วง [t0, t1] วินาทีของไฟล์ -> (float32 mono -1..1, sample rate)"""
    try:
        import soundfile as sf
        info = sf.info(str(source))
        start, stop = max(0, int(t0 * info.samplerate)), min(info.frames, int(t1 * info.samplerate))
        data, sr = sf.read(str(source), start=start, stop=max(start, stop), dtype="float32", always_2d=True)
        return data.mean(axis=1), sr
    except Exception:  # noqa: BLE001  ไฟล์ที่ libsndfile อ่านไม่ได้ (m4a/aac): ให้ ffmpeg ถอดเป็น WAV
        pass
    ffmpeg = platform_tools.bundled_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    result = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{max(0.0, t1 - t0):.3f}",
                             "-i", str(source), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "-"],
                            capture_output=True, timeout=120, creationflags=platform_tools.NO_WINDOW)
    data = result.stdout
    at = data.find(b"data")
    if result.returncode != 0 or len(data) < 44 or at < 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or "cannot decode audio")
    sr = int.from_bytes(data[24:28], "little")
    samples = np.frombuffer(data[at + 8:len(data) - (len(data) - at - 8) % 2], dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sr


def audio_info(source):
    """Audio properties for formats handled by either soundfile or the bundled ffprobe."""
    import soundfile as sf
    try:
        info = sf.info(str(source))
        return dict(samplerate=info.samplerate, channels=info.channels, duration=info.duration)
    except (RuntimeError, OSError):
        probe = platform_tools.bundled_tool("ffprobe")
        if not probe:
            raise RuntimeError("ffprobe is required for this audio format")
        result = subprocess.run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                 "stream=sample_rate,channels:format=duration", "-of", "json", str(source)],
                                capture_output=True, check=True, timeout=30, creationflags=platform_tools.NO_WINDOW)
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        return dict(samplerate=int(stream["sample_rate"]), channels=int(stream["channels"]),
                    duration=float(metadata["format"]["duration"]))


def read_channels(source, t0, t1, channel="mix"):
    """Read a bounded span, preserving stereo or selecting a channel without touching the source."""
    if channel not in ("mix", "stereo", "left", "right"):
        raise ValueError("Choose mix, stereo, left, or right")
    if not all(math.isfinite(float(x)) for x in (t0, t1)) or t0 < 0 or t1 <= t0:
        raise ValueError("Choose a valid start and end time")
    import soundfile as sf
    try:
        info = sf.info(str(source))
        start, stop = int(t0 * info.samplerate), min(info.frames, int(t1 * info.samplerate))
        data, sr = sf.read(str(source), start=start, stop=max(start, stop), dtype="float32", always_2d=True)
    except (RuntimeError, OSError):
        ffmpeg, probe = platform_tools.bundled_tool("ffmpeg"), platform_tools.bundled_tool("ffprobe")
        if not ffmpeg or not probe:
            raise RuntimeError("ffmpeg and ffprobe are required for this audio format")
        result = subprocess.run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                 "stream=sample_rate,channels", "-of", "json", str(source)],
                                capture_output=True, check=True, timeout=30, creationflags=platform_tools.NO_WINDOW)
        stream = json.loads(result.stdout)["streams"][0]
        sr = int(stream["sample_rate"])
        result = subprocess.run([ffmpeg, "-v", "error", "-ss", str(t0), "-i", str(source),
                                 "-t", str(t1 - t0), "-map", "0:a:0", "-f", "wav", "-acodec", "pcm_f32le", "-"],
                                capture_output=True, check=True, timeout=120, creationflags=platform_tools.NO_WINDOW)
        data, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
    if channel == "right" and data.shape[1] < 2:
        raise ValueError("This recording has no right channel; choose Left or Mix")
    if channel == "mix":
        return data.mean(axis=1), sr
    if channel in ("left", "right"):
        return data[:, 0 if channel == "left" else 1], sr
    return data, sr


def bandpass(samples, sr, low=0.0, high=None):
    """Preview-only filter. Cutoffs are in Hz; filtering retains the channel axis."""
    from scipy.signal import butter, sosfilt, sosfiltfilt
    high = sr / 2 if high is None else float(high)
    low = float(low)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= sr / 2):
        raise ValueError(f"Frequency range must be between 0 and {sr / 2:g} Hz")
    data = np.asarray(samples, dtype=np.float32)
    if low == 0 and high >= sr / 2:
        return data.copy()
    if low == 0:
        sos = butter(5, high, btype="lowpass", fs=sr, output="sos")
    elif high >= sr / 2:
        sos = butter(5, low, btype="highpass", fs=sr, output="sos")
    else:
        sos = butter(5, (low, high), btype="bandpass", fs=sr, output="sos")
    try:
        return sosfiltfilt(sos, data, axis=0).astype(np.float32)
    except ValueError:  # very short selection
        return sosfilt(sos, data, axis=0).astype(np.float32)


def listening_preview(source, start, end, target, channel="stereo", low=0.0, high=None, speed=1.0):
    """Make a disposable filtered preview; tempo changes preserve pitch and source timestamps."""
    speed = float(speed)
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("Playback speed must be between 0.5 and 2")
    if end - start > 600:
        raise ValueError("Select at most 10 minutes for listening tools")
    samples, sr = read_channels(source, start, end, channel)
    if not len(samples):
        raise ValueError("The selection has no audio")
    samples = bandpass(samples, sr, low, high)
    if speed != 1:
        import librosa
        # librosa is bundled on both platforms; the small bundled ffmpeg has no atempo filter.
        samples = librosa.effects.time_stretch(samples.T, rate=speed).T
    write_preview(samples, sr, Path(target))
    return sr


def write_preview(samples, sr, path: Path):
    """เสียงตัวอย่าง 16-bit mono ดังขึ้นให้ peak = PREVIEW_PEAK"""
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    gain = PREVIEW_PEAK / peak if peak > 0 else 1.0
    data = (np.clip(samples * gain, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(data.shape[1] if data.ndim == 2 else 1)
        out.setsampwidth(2)
        out.setframerate(int(sr))
        out.writeframes(data.tobytes())
