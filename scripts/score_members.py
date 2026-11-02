"""Score each ensemble member on its own through the deployment code path, the same way BSW-YOLO11s is scored
as the reference in tune_ensemble.py: per-class thresholds = best F1 on India + Japan VAL, then one pass on TEST.

Uses only the cached member outputs (runs/ensemble/views_val.pkl, views_test.pkl); no model is run.
Writes runs/ensemble/members_test.json.

    python scripts/score_members.py
"""
import json
import pickle
from pathlib import Path

from tune_ensemble import OUT, best_thresholds, evaluate, fused, strip

LABELS = {"india_japan": "BSW-YOLO11s", "gia_cdsa": "GGC-YOLO (YOLO11s + GDGF + GIA + CDSA)",
          "plain_s": "YOLO11s + GDGF"}


def main() -> None:
    val = pickle.loads((OUT / "views_val.pkl").read_bytes())
    test = pickle.loads((OUT / "views_test.pkl").read_bytes())
    assert val["members"] == test["members"], "val and test caches come from different member lists"
    report = {"members": test["members"], "note": "deployment path; thresholds = best F1 per class on val"}
    for m, name in enumerate(test["members"]):
        th = best_thresholds(evaluate(val, fused(val, None, member=m)))
        preds = fused(test, None, member=m)
        report[name] = {"label": LABELS.get(name, name), "thresholds": th,
                        **{label: strip(evaluate(test, preds, th, subset))
                           for subset, label in ((None, "combined"), ("India", "india"), ("Japan", "japan"))}}
        c = report[name]["combined"]; a, il = c["at_thresholds"], c["image_level"]
        print(f"{LABELS.get(name, name):40s} mAP50 {c['mAP50']:.4f} mAP50-95 {c['mAP50-95']:.4f} P {a['precision']:.3f} "
              f"R {a['recall']:.3f} F1 {a['f1']:.3f} | photos right {il['photo_accuracy']:.1%}, damaged flagged "
              f"{il['damaged_flagged']:.1%}, clean false alarm {il['clean_false_alarm']:.1%}, "
              f"right type {il['right_type_when_flagged']:.1%}")
    (OUT / "members_test.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"Saved: {OUT / 'members_test.json'}")


if __name__ == "__main__":
    main()
