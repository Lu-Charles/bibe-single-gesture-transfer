"""Recompute revision follow-up means from portable saved trial predictions."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
old = json.loads((DATA / "trial_predictions.json").read_text())
rows = json.loads((DATA / "followup_results.json").read_text())
expected = json.loads((RESULTS / "followup_summary.json").read_text())
G = [8, 9, 12, 13, 16, 21, 23, 24]
assert len(rows) == 288
for variant in expected["summary"]:
    for method, metrics in expected["summary"][variant].items():
        part = []
        for person in [2, 3, 4, 5]:
            case = []
            for gesture in G:
                match = next(
                    r
                    for r in old
                    if (
                        r["participant"],
                        r["source_session"],
                        r["target_session"],
                        r["gesture"],
                        r["method"],
                    )
                    == (person, 1, 2, gesture, method)
                )
                y = np.array(match["truth"])
                label = G.index(gesture)
                if variant == "original":
                    pred = np.array(match["prediction"])
                else:
                    row = next(
                        r
                        for r in rows
                        if (r["participant"], r["variant"], r["method"], r["gesture"])
                        == (person, variant, method, gesture)
                    )
                    pred = np.array(row["metrics"]["trial_predictions"])
                case.append(
                    [
                        np.mean(y == pred),
                        np.mean(pred[y == label] == label),
                        np.mean(y[y != label] == pred[y != label]),
                    ]
                )
            part.append(np.mean(case, 0))
        np.testing.assert_allclose(
            np.mean(part, 0),
            [metrics[k] for k in ["overall", "calibrated", "other"]],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            part, expected["participant_means"][variant][method], rtol=0, atol=1e-12
        )
print(
    json.dumps(
        {
            "passed": True,
            "followup_records": len(rows),
            "scope": "Summary reproduction from saved trial predictions; no raw-data or model replay",
        }
    )
)
