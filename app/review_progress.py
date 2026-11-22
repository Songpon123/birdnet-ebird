"""หน้าต่าง Review progress: ตรวจไปแล้วกี่คลิป ชนิดไหน ข้ามผลทุกชุด และตัวกรองไหนจับ ID ผิดของ BirdNET ได้

ผลตรวจของคน = คำตอบจริงสำหรับวัดตัวกรอง (ผู้ตรวจ Reject เฉพาะเมื่อชนิดผิด; ชนิดถูกแต่เสียงไม่ดี = Approve คะแนน 1):
  ID ถูก = Approve ด้วยชนิดเดียวกับ BirdNET
  ID ผิด = Reject หรือ Approve หลังแก้เป็นชนิดอื่น
  Not sure (Unsure) = เสียงไม่ชัดพอจะระบุ: ไม่นับทั้งถูกและผิด (ไม่ใช้วัดตัวกรอง/ฝึกโมเดล)
คลิปที่ผู้ตรวจเพิ่มเองไม่นับ (BirdNET ไม่ได้ทาย)
"""

import json
import queue
import threading
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import auto_checks
import country_checklist
import manual_clips
import second_opinion
import sound_type
import species_names
from result_roots import all_results
from review_store import load_reviews

GOAL_LOW, GOAL_HIGH = 100, 200
LOW_SCORE = 0.6
MIN_FOR_VERDICT = 10            # คลิปที่ตรวจแล้วในแต่ละฝั่งน้อยกว่านี้: ยังสรุปไม่ได้
FILTERS = (
    ("location", "BirdNET location/date model"),
    ("habitat", "Site habitat (AVONET)"),
    ("checklist", "Country checklist"),
    ("season", "eBird: same dates in past years"),
    ("low_score", f"BirdNET score below {LOW_SCORE}"),
    ("nonbird", "Insect/frog/other sound as strong"),
    ("second", f"Second opinion disagrees (score < {auto_checks.SECOND_OPINION_BELOW})"),
    ("second_look", "Found only by the second look (near miss)"),
)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def outcome(row):
    """'correct' / 'wrong' / 'unsure' / 'pending' ของ ID ที่ BirdNET ทาย ตามผลตรวจของคน"""
    status = str(row.get("Review status") or "Pending")
    if status == "Approved":
        reviewed = " ".join(str(row.get("Reviewed scientific") or "").split()[:2])
        birdnet = " ".join(species_names.row_names(row)[1].split()[:2])
        return "correct" if not reviewed or reviewed == birdnet else "wrong"
    if status == "Rejected":
        return "wrong"
    if status == "Unsure":
        return "unsure"
    return "pending"


def decided(row):
    """ผลตรวจบอกได้ว่า ID ถูกหรือผิด (ใช้วัดตัวกรอง/ฝึกโมเดล)"""
    return row["_outcome"] in ("correct", "wrong")


def flags(row, season_flagged=None):
    """ตัวกรองไหนเตือนคลิปนี้: True/False หรือ None = ตัวกรองนี้ยังไม่ได้ตรวจคลิปนี้
    (ใช้ชนิดที่ BirdNET ทาย ไม่ใช่ชื่อที่ผู้ตรวจแก้)"""
    common, scientific = species_names.row_names(row)
    checklist = country_checklist.check(scientific, _number(row.get("Latitude")), _number(row.get("Longitude")), common)
    score = _number(row.get("Max confidence"))
    return {
        "location": str(row.get("Expected by location/date") or "").startswith("No"),
        "habitat": str(row.get("Fits site habitat") or "").startswith("No"),
        "checklist": country_checklist.needs_check(checklist),
        "season": None if season_flagged is None else row.get("_season_key") in season_flagged,
        "low_score": score is not None and score < LOW_SCORE,
        **auto_checks.flags_for(row),
        # คลิปจากการฟังซ้ำ (near_misses.py): วัดว่าเสียงที่เกือบเจอแล้วฟังซ้ำผ่าน ID ถูกบ่อยแค่ไหน
        "second_look": str(row.get("Found by") or "").startswith("Second look"),
    }


def collect(summaries):
    """คลิปจากทุกผลชุด -> list ของ dict (แถวใน summary + _summary, _species, _outcome)"""
    records = []
    for summary in summaries:
        try:
            rows = load_reviews(Path(summary))
        except Exception:  # noqa: BLE001  ไฟล์เสีย/กำลังเขียน
            continue
        for row in rows:
            row["_summary"] = Path(summary)
            row["_species"], row["_scientific"] = species_names.row_names(row)
            row["_manual"] = manual_clips.is_manual(row)
            row["_outcome"] = outcome(row)
            records.append(row)
    return records


def species_table(records, season_flagged=None):
    """ต่อชนิด (ชื่อ eBird): clips, reviewed, approved, wrong, pending, flagged, results"""
    table = defaultdict(lambda: {"clips": 0, "reviewed": 0, "approved": 0, "wrong": 0, "unsure": 0, "pending": 0,
                                 "flagged": 0, "results": set()})
    for row in records:
        item = table[row["_species"]]
        item["clips"] += 1
        item["results"].add(row["_summary"])
        item["pending"] += row["_outcome"] == "pending"
        item["reviewed"] += decided(row)
        item["unsure"] += row["_outcome"] == "unsure"
        item["approved"] += str(row.get("Review status")) == "Approved"
        item["wrong"] += row["_outcome"] == "wrong" and not row["_manual"]
        item["flagged"] += not row["_manual"] and any(flags(row, season_flagged).values())
    return dict(table)


def filter_report(records, season_flagged=None):
    """ต่อตัวกรอง: คลิปของ BirdNET ที่ตรวจแล้ว แยกฝั่งที่ถูกเตือน/ไม่ถูกเตือน -> (reviewed, wrong)"""
    report = {key: {"flagged": [0, 0], "clear": [0, 0]} for key, _label in FILTERS}
    for row in records:
        if row["_manual"] or not decided(row):
            continue
        wrong = row["_outcome"] == "wrong"
        for key, flagged in flags(row, season_flagged).items():
            if flagged is None:                        # ตัวกรองนี้ยังไม่ได้ตรวจคลิปนี้: ไม่นับ
                continue
            side = report[key]["flagged" if flagged else "clear"]
            side[0] += 1
            side[1] += wrong
    return report


# ---------------- ML: เรียนจากผลตรวจของผู้ใช้ว่า ID ของ BirdNET แบบไหนมักผิด ----------------
MODEL_MIN_REVIEWED, MODEL_MIN_WRONG, MODEL_MIN_CORRECT = 150, 30, 30
MODEL_FOLDS = 5
MODEL_BETTER_BY = 0.03          # AUC ต้องดีกว่าคะแนนอย่างเดียวอย่างน้อยเท่านี้ ถึงจะนับว่าโมเดลช่วย
FEATURES = ("BirdNET score", "gap to 2nd species", "detections in clip", "clip length", "SNR",
            "peak level", "location model warns", "habitat warns", "checklist warns", "eBird past years warns",
            "non-bird as strong", "second opinion margin", "second opinion available")


def _second_margin(row):
    """ความคล้ายของชนิดปัจจุบัน ลบ ชนิดอื่นที่คล้ายที่สุด (บวก = เห็นด้วย) หรือ None ถ้ายังไม่มี second opinion"""
    result = second_opinion.cached_result(str(row.get("Generated sha256") or ""))
    ranking = [r for r in (result or {}).get("ranking", []) if r.get("similarity") is not None]
    current = next((r["similarity"] for r in ranking if r["scientific"] == (result or {}).get("current")), None)
    others = [r["similarity"] for r in ranking if r["scientific"] != (result or {}).get("current")]
    return None if current is None or not others else current - max(others)


def model_features(row, season_flagged=None):
    """ตัวเลขของคลิปหนึ่งสำหรับโมเดล (ค่าที่ไม่รู้: 0.5 สำหรับคำเตือน, 0 สำหรับตัวเลข)"""
    def number(key, default=0.0):
        value = _number(row.get(key))
        return default if value is None else value

    def flag(value):
        return 0.5 if value is None else float(value)
    try:
        windows = json.loads(row.get("Detection boxes") or "{}").get("windows", [])
    except (TypeError, ValueError, AttributeError):
        windows = []
    score = number("Max confidence")
    found = flags(row, season_flagged)
    margin = _second_margin(row)
    return [score, score - number("Alt1 conf"), float(len(windows)), number("Duration (s)"),
            number("SNR (dB approx, first 15s)"), number("Peak dBFS", -40.0),
            flag(found["location"]), flag(found["habitat"]), flag(found["checklist"]), flag(found["season"]),
            flag(found["nonbird"]), margin or 0.0, float(margin is not None)]


def model_check(records, season_flagged=None):
    """ฝึกโมเดลจากผลตรวจ แล้ววัดแบบ cross-validation แบ่งตามไฟล์ต้นฉบับ เทียบกับคะแนน BirdNET อย่างเดียว"""
    rows = [row for row in records if not row["_manual"] and decided(row)]
    wrong = sum(row["_outcome"] == "wrong" for row in rows)
    need = {"reviews": max(0, MODEL_MIN_REVIEWED - len(rows)), "wrong": max(0, MODEL_MIN_WRONG - wrong),
            "correct": max(0, MODEL_MIN_CORRECT - (len(rows) - wrong))}
    result = {"ready": not any(need.values()), "need": need, "reviewed": len(rows), "wrong": wrong}
    if not result["ready"]:
        return result
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

    X, y = _training(rows, season_flagged)
    groups = [str(row.get("Source ID") or row["_summary"]) for row in rows]
    models = _models()
    probabilities = {}
    for name, model in models.items():
        try:          # คลิปจากไฟล์เดียวกันอยู่ฝั่งเดียวกัน (นกตัวเดิมไม่ช่วยโมเดลโกงข้อสอบ)
            cv = StratifiedGroupKFold(n_splits=MODEL_FOLDS, shuffle=True, random_state=0)
            probabilities[name] = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
            result["by_recording"] = True
        except ValueError:    # ไฟล์ต้นฉบับน้อยเกินจะแบ่ง: แบ่งตามคลิปแทน
            cv = StratifiedKFold(n_splits=MODEL_FOLDS, shuffle=True, random_state=0)
            probabilities[name] = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            result["by_recording"] = False
    score = X[:, 0]
    result["auc"] = {"BirdNET score alone": float(roc_auc_score(y, -score)),
                     **{name: float(roc_auc_score(y, p)) for name, p in probabilities.items()}}
    best = max(probabilities, key=lambda name: result["auc"][name])
    result["best"] = best
    result["better"] = result["auc"][best] >= result["auc"]["BirdNET score alone"] + MODEL_BETTER_BY
    # เตือนเท่ากับกฎ 'คะแนน < LOW_SCORE': กฎจับ ID ผิดได้กี่คลิป เทียบกับโมเดล
    warnings = int((score < LOW_SCORE).sum())
    order = np.argsort(-probabilities[best])[:warnings]
    result["same_warnings"] = {"count": warnings, "score_rule": int(y[score < LOW_SCORE].sum()),
                               "model": int(y[order].sum())}
    # เกณฑ์เตือนในตาราง: ความเสี่ยงของคลิปที่ warnings ในการทดสอบ (เตือนเท่ากฎคะแนน แต่จับ ID ผิดได้มากกว่า)
    result["threshold"] = float(np.sort(probabilities[best])[::-1][warnings - 1]) if warnings else 1.0
    forest = models["Random Forest"].fit(X, y)
    ranked = np.argsort(-forest.feature_importances_)[:3]
    result["top_signals"] = [FEATURES[i] for i in ranked]
    return result


def _training(rows, season_flagged=None):
    import numpy as np
    X = np.array([model_features(row, season_flagged) for row in rows])
    y = np.array([row["_outcome"] == "wrong" for row in rows], dtype=int)
    return X, y


def _models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=0),
            "Logistic regression": make_pipeline(StandardScaler(),
                                                 LogisticRegression(max_iter=2000, class_weight="balanced"))}


def season_flags_all(records, ebird_key):
    """ตัวกรอง eBird ช่วงเดียวกันของปีก่อน ๆ ของทุกคลิป (ใช้ cache ของ eBird) -> set หรือ None (ไม่มี key/ถามไม่ได้)"""
    from birdnet_gui import _season_group, season_flags
    for row in records:
        group = _season_group(row)
        row["_season_key"] = (group[0], str(row.get("Species (scientific)"))) if group else None
    if not ebird_key:
        return None
    import ebird
    from datetime import date
    seasons = {}
    try:
        for group in {g for g in map(_season_group, records) if g}:
            region = ebird.region_of(group[1], group[2], ebird_key)
            if region:
                seasons[group] = ebird.season_reports(region, date.fromisoformat(group[0]), ebird_key)
    except (ebird.EBirdError, ValueError):
        return None
    # ใช้ชนิดที่ BirdNET ทาย ไม่ใช่ชื่อที่ผู้ตรวจแก้
    plain = [{k: v for k, v in row.items() if k not in ("Reviewed scientific", "Reviewed common")} for row in records]
    return season_flags(plain, seasons)


def warning_model(records, season_flagged=None):
    """โมเดลที่ใช้เตือนในตารางของแท็บ Review: ฝึกกับผลตรวจทั้งหมด — ใช้เฉพาะเมื่อทดสอบแล้วดีกว่าคะแนนอย่างเดียว
    -> (dict หรือ None, ผลของ model_check)"""
    result = model_check(records, season_flagged)
    if not result.get("better"):
        return None, result
    rows = [row for row in records if not row["_manual"] and decided(row)]
    model = _models()[result["best"]].fit(*_training(rows, season_flagged))
    return {"model": model, "threshold": result["threshold"], "season": season_flagged, "result": result}, result


def warning_risk(warner, row):
    """โอกาสที่ ID ของ BirdNET ในคลิปนี้ผิด ตามโมเดล (0-1)"""
    return float(warner["model"].predict_proba([model_features(row, warner["season"])])[0, 1])


def describe_model(result):
    """ข้อความหนึ่งย่อหน้าสำหรับหน้าต่าง"""
    if not result["ready"]:
        need = result["need"]
        parts = [f"{need['reviews']} more reviewed clips" if need["reviews"] else "",
                 f"{need['wrong']} more wrong IDs" if need["wrong"] else "",
                 f"{need['correct']} more right IDs" if need["correct"] else ""]
        return (f"Needs {' and '.join(p for p in parts if p)} (now {result['reviewed']} reviewed, "
                f"{result['wrong']} wrong). Then it learns from your reviews which BirdNET IDs tend to be wrong "
                "and is tested against the BirdNET score alone on clips it has not seen.")
    auc = result["auc"]
    same = result["same_warnings"]
    text = ("Ranking quality (1.0 = perfect, 0.5 = guessing), tested on recordings the model did not learn from"
            if result.get("by_recording") else "Ranking quality (1.0 = perfect, 0.5 = guessing)")
    text += ": " + " · ".join(f"{name} {value:.2f}" for name, value in auc.items())
    text += (f". With the same {same['count']} warnings as “score < {LOW_SCORE}”, the {result['best']} catches "
             f"{same['model']} wrong IDs vs {same['score_rule']}. Strongest signals: {', '.join(result['top_signals'])}. ")
    text += ("→ The model is better than the score alone, so it marks risky clips “Check” in Review clips "
             f"(risk {result['threshold']:.0%} or more)." if result["better"]
             else "→ Not better than the BirdNET score alone yet; keep using the score (no model warnings).")
    return text


def verdict(item):
    """ข้อความสั้นของตัวกรองหนึ่งตัว"""
    flagged, clear = item["flagged"], item["clear"]
    total_wrong = flagged[1] + clear[1]
    if flagged[0] < MIN_FOR_VERDICT or clear[0] < MIN_FOR_VERDICT:
        return "need more reviews"
    caught = flagged[1] / total_wrong if total_wrong else 0
    false_alarms = (flagged[0] - flagged[1]) / flagged[0]
    return f"catches {caught:.0%} of wrong IDs; {false_alarms:.0%} of its warnings were right IDs"


def next_clip(records, season_flagged=None):
    """คลิปที่ควรตรวจต่อ: ชนิดที่ตรวจน้อยที่สุดก่อน ในชนิดเดียวกันเอาคลิปที่ถูกตัวกรองเตือนก่อน
    (ทุกชนิดและทุกตัวกรองจะได้ถูกทดสอบ) -> row หรือ None"""
    reviewed = defaultdict(int)
    for row in records:
        reviewed[row["_species"]] += row["_outcome"] != "pending"
    pending = [row for row in records if row["_outcome"] == "pending" and not row["_manual"]]
    if not pending:
        return None
    return min(pending, key=lambda row: (reviewed[row["_species"]], not any(flags(row, season_flagged).values()),
                                         -(_number(row.get("Max confidence")) or 0)))


class ReviewProgressWindow:
    def __init__(self, app):
        self.app = app
        self.records = []
        self.season_flagged = None          # None = ยังไม่ได้ถาม eBird
        self._season_state = "not checked"
        self.q = queue.Queue()
        self._loading = False
        win = self.win = tk.Toplevel(app.root)
        win.title("Review progress")
        win.geometry(f"980x{min(760, win.winfo_screenheight() - 80)}")
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Wrong ID = rejected, or approved under another species (right species with poor "
                  "sound: approve with quality 1). Clips you added yourself are not counted. To also test "
                  "BirdNET's location filter, analyse some recordings with “also keep out-of-area” ticked — "
                  "otherwise those detections are dropped before clips are made.",
                  foreground="#888", wraplength=950).pack(side="bottom", anchor="w", pady=(6, 0))

        top = ttk.Frame(body)
        top.pack(fill="x")
        self.summary = tk.StringVar(value="Loading …")
        ttk.Label(top, textvariable=self.summary, font=("Helvetica", 13, "bold")).pack(anchor="w")
        self.bar = ttk.Progressbar(top, maximum=GOAL_HIGH, length=600)
        self.bar.pack(anchor="w", pady=(4, 2))
        ttk.Label(top, text=f"Goal: {GOAL_LOW}–{GOAL_HIGH} reviewed BirdNET clips across many species, "
                  "then the filters below can be compared.", foreground="#808080").pack(anchor="w")
        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="▶ Review next suggested clip", style="QuickAction.TButton",
                   command=self.review_next).pack(side="left")
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=8)
        self.next_info = tk.StringVar(value="Picks species with the fewest reviews, warned clips first, "
                                            "so every species and filter gets tested")
        ttk.Label(actions, textvariable=self.next_info, foreground="#808080", wraplength=560).pack(side="left", padx=8)

        ttk.Label(body, text="Filters — reviewed BirdNET clips", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(12, 2))
        columns = ("filter", "flagged", "flagged_wrong", "clear", "clear_wrong", "verdict")
        self.filter_tree = ttk.Treeview(body, columns=columns, show="headings", height=len(FILTERS))
        for key, label, width in (("filter", "Filter", 230), ("flagged", "Warned: reviewed", 115),
                                  ("flagged_wrong", "of them wrong ID", 110), ("clear", "Not warned: reviewed", 135),
                                  ("clear_wrong", "of them wrong ID", 110), ("verdict", "So far", 240)):
            self.filter_tree.heading(key, text=label)
            self.filter_tree.column(key, width=width, stretch=(key == "verdict"),
                                    anchor="w" if key in ("filter", "verdict") else "center")
        self.filter_tree.pack(fill="x")
        ttk.Label(body, text="ML model — learns from your reviews", font=("Helvetica", 12, "bold")).pack(
            anchor="w", pady=(10, 2))
        self.model_info = tk.StringVar(value="…")
        ttk.Label(body, textvariable=self.model_info, wraplength=950).pack(anchor="w")
        actions = ttk.Frame(body)
        actions.pack(anchor="w", pady=(4, 0))
        # คลิปที่โมเดลว่าเสี่ยงที่สุดก่อน (ยังไม่ได้ตรวจ ทุกผลชุด)
        self.risky_button = ttk.Button(actions, text="▶ Review riskiest clip", command=self.review_risky,
                                       state="disabled")
        self.risky_button.pack(side="left")
        self.risky_info = tk.StringVar()
        ttk.Label(actions, textvariable=self.risky_info, foreground="#808080").pack(side="left", padx=8)
        self._risky = []
        self._model_run = 0
        ttk.Label(body, text="Sound type guess — song / call", font=("Helvetica", 12, "bold")).pack(
            anchor="w", pady=(10, 2))
        self.sound_info = tk.StringVar(value="…")
        ttk.Label(body, textvariable=self.sound_info, wraplength=950).pack(anchor="w")
        actions = ttk.Frame(body)
        actions.pack(anchor="w", pady=(4, 0))
        # คลิปที่ตัวเดาลังเลที่สุด: เลือก Sound type ให้คลิปพวกนี้ ตัวเดาเรียนได้มากที่สุด
        self.unsure_button = ttk.Button(actions, text="▶ Label an unsure clip", command=self.review_unsure,
                                        state="disabled")
        self.unsure_button.pack(side="left")
        ttk.Button(actions, text="Sound types…", command=self.app.open_sound_types).pack(side="left", padx=(6, 0))
        self.unsure_info = tk.StringVar()
        ttk.Label(actions, textvariable=self.unsure_info, foreground="#808080").pack(side="left", padx=8)
        self._unsure = []

        species = ttk.Frame(body)
        species.pack(fill="x", pady=(12, 2))
        ttk.Label(species, text="Species", font=("Helvetica", 12, "bold")).pack(side="left")
        self.show = tk.StringVar(value="All")
        box = ttk.Combobox(species, textvariable=self.show, state="readonly", width=24,
                           values=("All", "Not reviewed yet", "Warned by a filter", "Has wrong IDs"))
        box.pack(side="left", padx=10)
        box.bind("<<ComboboxSelected>>", lambda _event: self._fill_species())
        ttk.Label(species, text="Double-click a species: Review clips shows only that species, at its next clip",
                  foreground="#808080").pack(side="left")
        columns = ("species", "clips", "reviewed", "approved", "wrong", "unsure", "pending", "flagged", "results")
        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True)
        self.species_tree = ttk.Treeview(frame, columns=columns, show="headings")
        for key, label, width in (("species", "Species (eBird name)", 250), ("clips", "Clips", 60),
                                  ("reviewed", "Reviewed", 75), ("approved", "✓ Approved", 85),
                                  ("wrong", "✗ Wrong ID", 85), ("unsure", "? Not sure", 75), ("pending", "Pending", 70),
                                  ("flagged", "Warned", 65), ("results", "Results", 65)):
            self.species_tree.heading(key, text=label, command=lambda key=key: self._sort(key))
            self.species_tree.column(key, width=width, stretch=(key == "species"),
                                     anchor="w" if key == "species" else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.species_tree.yview)
        self.species_tree.configure(yscrollcommand=bar.set)
        self.species_tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.species_tree.bind("<Double-1>", lambda _event: self.review_species())
        self._sort_key, self._sort_reverse = "clips", True
        self.refresh()
        self.win.after(150, self._poll)

    # ---------- โหลดข้อมูล ----------
    def refresh(self):
        if self._loading or not self.win.winfo_exists():
            return
        self._loading = True
        root = self.app.out.get().strip() or str(Path.home() / "BirdNET_eBird")
        extra = self.app.review_summary.get().strip()

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            summaries = [info["path"] for info in all_results(root)]
            if extra and Path(extra).expanduser().is_file() and \
                    Path(extra).expanduser().resolve() not in {Path(p).resolve() for p in summaries}:
                summaries.append(Path(extra).expanduser())
            results.put(("records", collect(summaries)))
        threading.Thread(target=work, daemon=True).start()

    def _start_season(self):
        """ตัวกรอง eBird ช่วงเดียวกันของปีก่อน ๆ: ต้องถาม eBird (มี cache) ทำเบื้องหลัง"""
        key = self.app.ebird_key.get().strip()
        if not key:
            self._season_state = "needs your eBird API key (Settings tab)"
            return
        self._season_state = "checking eBird …"
        records = [row for row in self.records if not row["_manual"]]

        results = self.q

        def work():  # ไม่ถือ self (ปิดหน้าต่างระหว่างทำได้)
            import ebird
            from datetime import date
            from birdnet_gui import _season_group, season_flags
            seasons = {}
            try:
                for group in {g for g in map(_season_group, records) if g}:
                    region = ebird.region_of(group[1], group[2], key)
                    if region:
                        seasons[group] = ebird.season_reports(region, date.fromisoformat(group[0]), key)
                # ใช้ชนิดที่ BirdNET ทาย ไม่ใช่ชื่อที่ผู้ตรวจแก้
                plain = [{k: v for k, v in row.items() if k not in ("Reviewed scientific", "Reviewed common")}
                         for row in records]
                results.put(("season", season_flags(plain, seasons), None))
            except (ebird.EBirdError, ValueError) as exc:
                results.put(("season", set(), str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "records":
                    self._loading = False
                    self.records = payload[0]
                    from birdnet_gui import _season_group
                    for row in self.records:
                        group = _season_group(row)
                        row["_season_key"] = (group[0], str(row.get("Species (scientific)"))) if group else None
                    if self._season_state == "not checked":   # ก่อน _fill: โมเดลรอผล eBird (สัญญาณสำคัญ)
                        self._start_season()
                    self._fill()
                elif kind in ("model", "sound") and payload[0] != self._model_run:
                    continue                                # ผลของรอบเก่า (ข้อมูลเปลี่ยนไปแล้ว)
                elif kind == "model":
                    text, self._risky, pending, threshold = payload[1:]
                    self.model_info.set(text)
                    new = sum(not any(value for key, value in flags(row, self.season_flagged).items()
                                      if key != "low_score") for _risk, row in self._risky)
                    self.risky_info.set(f"{len(self._risky)} of {pending} unreviewed clips at risk {threshold:.0%}+ "
                                        f"({new} not flagged by any other check)" if threshold is not None else "")
                    self.risky_button.configure(state="normal" if self._risky else "disabled")
                elif kind == "sound":
                    text, counts, self._unsure = payload[1:]
                    self.sound_info.set(text)
                    self.unsure_info.set("Unreviewed clips: " + ", ".join(f"{n} {label}" for label, n in counts.most_common())
                                         if counts else "")
                    self.unsure_button.configure(state="normal" if self._unsure else "disabled")
                elif kind == "season":
                    flagged, error = payload
                    self.season_flagged = None if error else flagged
                    self._season_state = f"eBird check failed: {error}" if error else "done"
                    self._fill()
        except queue.Empty:
            pass
        self.win.after(200, self._poll)

    # ---------- แสดงผล ----------
    def _fill(self):
        birdnet = [row for row in self.records if not row["_manual"]]
        reviewed = [row for row in birdnet if decided(row)]
        wrong = sum(row["_outcome"] == "wrong" for row in reviewed)
        approved = sum(str(row.get("Review status")) == "Approved" for row in reviewed)
        unsure = sum(row["_outcome"] == "unsure" for row in birdnet)
        species = {row["_species"] for row in reviewed}
        results = {row["_summary"] for row in self.records}
        self.summary.set(f"Reviewed {len(reviewed)} of {len(birdnet)} BirdNET clips "
                         f"({approved} approved · {wrong} wrong ID"
                         + (f" · {unsure} not sure, not counted" if unsure else "")
                         + f") — {len(species)} species, {len(results)} results")
        self.bar.configure(value=min(len(reviewed), GOAL_HIGH))
        self._fill_filters()
        self._fill_species()
        self._start_model()

    def _start_model(self):
        """ฝึก/วัดโมเดลเบื้องหลัง (scikit-learn มีในโปรแกรมแล้ว) — รอผล eBird ก่อนถ้ากำลังถามอยู่"""
        if self._season_state == "checking eBird …":
            self.model_info.set("Waiting for the eBird check …")
            return
        records, season, results = list(self.records), self.season_flagged, self.q
        self._model_run += 1
        run = self._model_run
        self.model_info.set("Testing …")
        self.sound_info.set("…")

        def work():                               # ไม่ถือ self: ส่งผลกลับทางคิวอย่างเดียว
            try:
                warner, result = warning_model(records, season)
                pending = [row for row in records if row["_outcome"] == "pending" and not row["_manual"]]
                risky = []
                if warner is not None:
                    scored = sorted(((warning_risk(warner, row), row) for row in pending), key=lambda pair: -pair[0])
                    risky = [(risk, row) for risk, row in scored if risk >= warner["threshold"]]
                results.put(("model", run, describe_model(result), risky, len(pending),
                             warner["threshold"] if warner else None))
            except Exception as exc:  # noqa: BLE001  โมเดลพังไม่ควรทำให้หน้าต่างใช้ไม่ได้
                results.put(("model", run, f"Could not test the model: {exc}", [], 0, None))
            try:
                counts, unsure = sound_type.unreviewed_guesses(records)
                results.put(("sound", run, sound_type.describe_accuracy(records), counts, unsure))
            except Exception as exc:  # noqa: BLE001
                results.put(("sound", run, f"Could not read the sound type guesses: {exc}", None, []))
        threading.Thread(target=work, daemon=True).start()

    def review_risky(self):
        while self._risky:
            risk, row = self._risky.pop(0)
            if row["_outcome"] == "pending":
                self._open(row)
                self.next_info.set(self.next_info.get() + f" — ML risk {risk:.0%}")
                return
        self.next_info.set("No risky unreviewed clips left")

    def review_unsure(self):
        while self._unsure:
            row = self._unsure.pop(0)
            if row["_outcome"] == "pending":
                self._open(row)
                self.next_info.set(self.next_info.get() + " — choose its Sound type when you approve")
                return
        self.next_info.set("No unsure clips left")

    def _fill_filters(self):
        self.filter_tree.delete(*self.filter_tree.get_children())
        report = filter_report(self.records, self.season_flagged)
        for key, label in FILTERS:
            item = report[key]
            text = verdict(item)
            if key == "season" and self._season_state != "done":
                text = self._season_state
            self.filter_tree.insert("", "end", values=(label, item["flagged"][0], item["flagged"][1],
                                                       item["clear"][0], item["clear"][1], text))

    def _fill_species(self):
        table = species_table(self.records, self.season_flagged)
        show = self.show.get()
        rows = [(name, item) for name, item in table.items()
                if show == "All" or (show == "Not reviewed yet" and not item["reviewed"])
                or (show == "Warned by a filter" and item["flagged"]) or (show == "Has wrong IDs" and item["wrong"])]
        key = self._sort_key
        rows.sort(key=lambda pair: (pair[0].casefold() if key == "species" else
                                    len(pair[1]["results"]) if key == "results" else pair[1][key]),
                  reverse=self._sort_reverse)
        self.species_tree.delete(*self.species_tree.get_children())
        for name, item in rows:
            self.species_tree.insert("", "end", iid=name, values=(
                name, item["clips"], item["reviewed"], item["approved"], item["wrong"], item["unsure"], item["pending"],
                item["flagged"], len(item["results"])))

    def _sort(self, key):
        self._sort_reverse = not self._sort_reverse if key == self._sort_key else key != "species"
        self._sort_key = key
        self._fill_species()

    # ---------- ไปตรวจ ----------
    def review_next(self):
        row = next_clip(self.records, self.season_flagged)
        if row is None:
            self.next_info.set("Every BirdNET clip has been reviewed ✓")
            return
        self._open(row)

    def review_species(self):
        chosen = self.species_tree.selection()
        if not chosen:
            return
        mine = [row for row in self.records if row["_species"] == chosen[0]]
        row = next_clip(mine, self.season_flagged) or next((r for r in mine if r["_outcome"] == "pending"), None)
        self.app.review_species_filter.set(chosen[0])          # แท็บ Review แสดงเฉพาะชนิดนี้
        if row is None:                                        # ตรวจครบแล้ว: เปิดผลชุดล่าสุดที่มีชนิดนี้ให้ดู
            latest = max(mine, key=lambda r: r["_summary"].parent.name, default=None)
            if latest is not None:
                self.app.show_clip(latest["_summary"], str(latest["File"]), f"All {chosen[0]} clips are reviewed")
            self.next_info.set(f"All {chosen[0]} clips are reviewed")
            return
        self._open(row)

    def _open(self, row):
        warned = [label for key, label in FILTERS if flags(row, self.season_flagged)[key]]
        self.next_info.set(f"Opened {row['_species']} {row.get('Clock time') or ''} in "
                           f"{row['_summary'].parent.name}" + (f" — warned by: {', '.join(warned)}" if warned else ""))
        self.app.show_clip(row["_summary"], str(row["File"]), f"Review this {row['_species']} clip")
