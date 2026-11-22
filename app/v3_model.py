"""BirdNET 3.0 preview acoustic and location/date model adapter."""

import hashlib
import urllib.request

import paths  # noqa: F401  ตั้ง BIRDNET_APP_DATA -> data/birdnet ก่อน import birdnet


GEO_MIN_CONFIDENCE = 0.03
# รุ่นย่อ (pruned) — ชุดเดียวกับโมเดลเต็มบน Zenodo (CC BY-SA 4.0), 71.5 MB
# ทดลอง 2026-10-24: เร็วกว่า ~16%, เวกเตอร์เสียงเหมือนโมเดลเต็ม แต่คะแนนต่าง — คลิปที่ตรวจแล้ว ID ถูกตกต่ำกว่า 0.5 ไป 14%
PRUNED_FILE = "BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx"
PRUNED_URL = f"https://zenodo.org/records/20703646/files/{PRUNED_FILE}?download=1"
PRUNED_MD5 = "8278be6e7d1a9fc1717cf953e269b2e3"
PRUNED_DIR = paths.BIRDNET_MODELS / "acoustic-models" / "v3.0-pruned"
EBIRD_COMMON_NAMES = {
    "Eophona migratoria": "Yellow-billed Grosbeak",
}


def birdnet_week(recorded_at):
    """BirdNET geo models use four weeks per month, numbered 1 through 48."""
    return (recorded_at.month - 1) * 4 + min(4, (recorded_at.day - 1) // 7 + 1)


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pruned_model_path(log=print):
    """ไฟล์โมเดล pruned ใน data/birdnet (ไม่มี = โหลดจาก Zenodo แล้วตรวจ md5 ก่อนใช้)"""
    target = PRUNED_DIR / PRUNED_FILE
    if target.is_file():
        return target
    PRUNED_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    log(f"Downloading the pruned BirdNET 3.0 model (71.5 MB) from Zenodo …")
    try:
        with urllib.request.urlopen(PRUNED_URL, timeout=60) as response, open(partial, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if _md5(partial) != PRUNED_MD5:
            raise RuntimeError("the downloaded pruned model is damaged (md5 mismatch); try again")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def load_pruned(log=print):
    """โมเดล pruned ผ่าน birdnet.load_custom — API เดียวกับโมเดลเต็ม (predict, predict_arrays, encode)
    ป้ายชื่อใช้รายการของโมเดลเต็ม (ลำดับ output เดียวกัน 11,560 ชนิด) ชื่อในผลจึงตรงกัน"""
    import birdnet
    model = pruned_model_path(log)
    species = PRUNED_DIR / "species_list.txt"
    if not species.is_file():
        labels = birdnet.load("acoustic", "3.0", "onnx", precision="fp16").species_list
        temporary = species.with_name(species.name + ".tmp")
        temporary.write_text("\n".join(map(str, labels)) + "\n", encoding="utf-8")
        temporary.replace(species)
    return birdnet.load_custom("acoustic", "3.0", "onnx", model, species, precision="fp16")


class V3Analyzer:
    model_name = "BirdNET 3.0 preview ONNX fp16 + geo 3.0.4 (0.03)"
    direct_prediction = True
    folder_suffix = "_v3preview"          # ผลแต่ละโมเดลแยกโฟลเดอร์ (เทียบกันได้)

    def __init__(self):
        try:
            import birdnet
        except ImportError as exc:
            raise RuntimeError(
                "BirdNET 3.0 preview is unavailable. Install birdnet and onnxruntime "
                "in the app's Python environment."
            ) from exc
        self.model = self._acoustic()
        self.geo_model = birdnet.load("geo", "3.0", "onnx", precision="fp16")
        self._geo_cache = {}

    def _acoustic(self):
        import birdnet
        return birdnet.load("acoustic", "3.0", "onnx", precision="fp16")

    def _geo_probabilities(self, latitude, longitude, recorded_at):
        """(by scientific name, by lower-case common name) for this place and week."""
        key = (latitude, longitude, birdnet_week(recorded_at))
        if key not in self._geo_cache:
            prediction = self.geo_model.predict(
                latitude, longitude, week=key[2], min_confidence=0.0
            )
            by_scientific, by_common = {}, {}
            for label, probability in zip(prediction.species_list, prediction.species_probs):
                scientific, _, common = str(label).partition("_")
                by_scientific[scientific] = float(probability)
                if common:
                    by_common.setdefault(common.casefold(), (float(probability), scientific))
            self._geo_cache[key] = (by_scientific, by_common)
        return self._geo_cache[key]

    def detections(self, audio_path, overlap, confidence_floor,
                   latitude, longitude, recorded_at):
        by_scientific, by_common = self._geo_probabilities(latitude, longitude, recorded_at)
        result = self.model.predict(
            audio_path,
            top_k=10,
            n_workers=1,
            overlap_duration_s=overlap,
            default_confidence_threshold=confidence_floor,
            show_stats=None,
        )
        detections = []
        for row in result.to_structured_array():
            species = str(row["species_name"])
            scientific, separator, common = species.partition("_")
            if not separator:
                common = scientific
            # The two models use different taxonomies for ~900 labels (e.g. Charadrius vs
            # Thinornis dubius), so fall back to the common name before giving up.
            geo_probability, geo_scientific = by_scientific.get(scientific), scientific
            if geo_probability is None:
                geo_probability, geo_scientific = by_common.get(common.casefold(), (None, None))
            common = EBIRD_COMMON_NAMES.get(scientific, common)
            detections.append({
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "common_name": common,
                "scientific_name": scientific,
                "confidence": float(row["confidence"]),
                "is_predicted_for_location_and_date": (
                    geo_probability is None or geo_probability >= GEO_MIN_CONFIDENCE
                ),
                "location_filter_available": geo_probability is not None,
                "geo_scientific_name": geo_scientific,   # other taxonomy's name, for AVONET lookup
                "birdnet_label": species,                # ป้ายของโมเดลเสียง (second look ถามชนิดเดิม)
            })
        return detections


class PrunedV3Analyzer(V3Analyzer):
    """โมเดลรุ่นย่อ (เลือกแทน 2.4 ไว้เปรียบเทียบ) — โมเดลพื้นที่และตัวกรองเหมือนกันทุกอย่าง"""
    model_name = "BirdNET 3.0 preview pruned ONNX fp16 + geo 3.0.4 (0.03)"
    folder_suffix = "_v3pruned"

    def _acoustic(self):
        return load_pruned()


def location_check(scientific, common, latitude, longitude, recorded_at, geo_model=None):
    """เฉพาะโมเดลพื้นที่ (ไม่โหลดโมเดลเสียง) สำหรับชนิดที่ผู้ตรวจใส่เอง:
    True/False = ควร/ไม่ควรพบที่นี่ในสัปดาห์นั้น, None = โมเดลไม่รู้จักชื่อนี้"""
    analyzer = V3Analyzer.__new__(V3Analyzer)
    if geo_model is None:
        import birdnet
        geo_model = birdnet.load("geo", "3.0", "onnx", precision="fp16")
    analyzer.geo_model, analyzer._geo_cache = geo_model, {}
    by_scientific, by_common = analyzer._geo_probabilities(latitude, longitude, recorded_at)
    probability = by_scientific.get(" ".join(str(scientific).split()[:2]))
    if probability is None:
        probability = by_common.get(str(common).strip().casefold(), (None, None))[0]
    return None if probability is None else probability >= GEO_MIN_CONFIDENCE
