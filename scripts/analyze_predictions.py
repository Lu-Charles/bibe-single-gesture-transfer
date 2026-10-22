"""Reproduce BIBE error partitions, conditional confusion matrices and exact tests.

Uses saved trial predictions, not raw signals. Requires NumPy only.
"""

from pathlib import Path
import itertools, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
G = [8, 9, 12, 13, 16, 21, 23, 24]


def main():
    rows = json.loads((DATA / "trial_predictions.json").read_text())
    assert len(rows) == 3840
    methods = [
        "frozen",
        "gain_only",
        "spatial_only",
        "spatial_gain",
        "classifier_fine",
        "classifier_fine_gain",
    ]
    participants = {}
    summary = {}
    confusions = {}
    saturation = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        assert len(selected) == 640
        pp = []
        for person in [2, 3, 4, 5]:
            vals = []
            for r in selected:
                if r["participant"] != person:
                    continue
                y = np.array(r["truth"])
                p = np.array(r["prediction"])
                c = G.index(r["gesture"])
                other = y != c
                assert len(y) == len(p) == len(r["trial_ids"]) and len(
                    set(r["trial_ids"])
                ) == len(y)
                vals.append(
                    [
                        np.mean(y == p),
                        np.mean(p[~other] == c),
                        np.mean(p[other] == y[other]),
                        np.mean(p[other] == c),
                        np.mean((p[other] != y[other]) & (p[other] != c)),
                        float(np.all(p == c)),
                    ]
                )
            assert len(vals) == 160
            pp.append(np.mean(vals, 0).tolist())
        participants[method] = pp
        summary[method] = dict(
            zip(
                [
                    "overall",
                    "calibrated",
                    "other",
                    "other_to_calibrated",
                    "other_to_wrong_other",
                    "collapse_fraction",
                ],
                np.mean(pp, 0).tolist(),
            )
        )
        assert (
            abs(
                sum(
                    summary[method][k]
                    for k in ["other", "other_to_calibrated", "other_to_wrong_other"]
                )
                - 1
            )
            < 1e-12
        )
        confusions[method] = {}
        for gesture in G:
            pm = []
            for person in [2, 3, 4, 5]:
                matrices = []
                for r in selected:
                    if (r["participant"], r["gesture"]) != (person, gesture):
                        continue
                    cm = np.zeros((8, 8), int)
                    np.add.at(cm, (r["truth"], r["prediction"]), 1)
                    matrices.append(cm / cm.sum(1, keepdims=True))
                assert len(matrices) == 20
                pm.append(np.mean(matrices, 0))
            confusions[method][str(gesture)] = np.mean(pm, 0).tolist()
        if method.startswith("spatial"):
            saturation[method] = {
                "zero_minimum_cases": sum(r["search"]["loss"] == 0 for r in selected),
                "cases_with_zero_candidates": sum(
                    r["search"]["zero_loss_candidates"] > 0 for r in selected
                ),
                "cases": len(selected),
            }
    contrasts = {}
    definitions = {
        "gain_vs_frozen": {"gain_only": 1, "frozen": -1},
        "spatial_vs_frozen": {"spatial_only": 1, "frozen": -1},
        "spatial_gain_vs_frozen": {"spatial_gain": 1, "frozen": -1},
        "spatial_added_to_gain": {"spatial_gain": 1, "gain_only": -1},
        "interaction": {
            "spatial_gain": 1,
            "spatial_only": -1,
            "gain_only": -1,
            "frozen": 1,
        },
    }
    for name, terms in definitions.items():
        effects = sum(w * np.array(participants[m])[:, 2] for m, w in terms.items())
        observed = abs(effects.mean())
        null = [
            abs(np.mean(effects * np.array(s)))
            for s in itertools.product([-1, 1], repeat=4)
        ]
        contrasts[name] = {
            "participant_effects": effects.tolist(),
            "mean": float(effects.mean()),
            "negative_participants": int(sum(effects < 0)),
            "two_sided_sign_flip_p": sum(v >= observed - 1e-14 for v in null) / 16,
        }
    expected = json.loads((RESULTS / "analysis.json").read_text())
    for method in methods:
        for key, old in [
            ("overall", "overall"),
            ("calibrated", "seen"),
            ("other", "unseen"),
        ]:
            np.testing.assert_allclose(
                summary[method][key],
                expected["methods"][method][old]["mean"],
                rtol=0,
                atol=1e-12,
            )
    result = {
        "passed": True,
        "scope": "Post-hoc analysis of saved trial predictions; n=4 independent participant means",
        "summary": summary,
        "participants": participants,
        "contrasts": contrasts,
        "loss_saturation": saturation,
        "confusion_normalization": "Normalize truth rows within each case, average 20 pairs within person, then average four people; condition separately on calibration gesture",
        "confusions_by_calibration_gesture": confusions,
    }
    (RESULTS / "revision_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["participants", "confusions_by_calibration_gesture"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
