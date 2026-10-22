"""Recompute reported means and contrasts from included per-case records.
This audits the analysis only. It does not retrain models or replay raw signals.
"""

import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def read(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows]))


def check(value, expected):
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)


def boot(a):
    a = np.array(a)
    rng = np.random.default_rng(2026)
    return np.quantile(a[rng.integers(0, 4, (10000, 4))].mean(1), [0.025, 0.975])


def main():
    rows = read("primary_cases.csv")
    assert len(rows) == 3840
    a = json.loads((RESULTS / "analysis.json").read_text())
    c = json.loads((RESULTS / "controls_analysis.json").read_text())
    participant = {}
    for method, expected in a["methods"].items():
        selected = [r for r in rows if r["method"] == method]
        assert len(selected) == 640
        participant[method] = {}
        for field in ["overall", "unseen", "seen", "calibration_fit", "unseen_delta"]:
            part = [
                mean([r for r in selected if r["participant"] == str(p)], field)
                for p in [2, 3, 4, 5]
            ]
            participant[method][field] = part
            check(np.mean(part), expected[field]["mean"])
            check(boot(part), expected[field]["participant_bootstrap_95"])
        high = [r for r in selected if float(r["calibration_fit"]) >= 0.99]
        assert len(high) == expected["high_fit_cases"]
        assert (
            sum(r["negative_transfer"] == "True" for r in high)
            == expected["high_fit_negative"]
        )
    for name, terms in {
        "gain_vs_frozen": {"gain_only": 1, "frozen": -1},
        "spatial_added_to_gain": {"spatial_gain": 1, "gain_only": -1},
        "spatial_vs_frozen": {"spatial_only": 1, "frozen": -1},
        "interaction": {
            "spatial_gain": 1,
            "spatial_only": -1,
            "gain_only": -1,
            "frozen": 1,
        },
    }.items():
        part = sum(
            weight * np.array(participant[m]["unseen"]) for m, weight in terms.items()
        )
        check(part, a["contrasts"][name]["participant_effects"])
        check(boot(part), a["contrasts"][name]["participant_bootstrap_95"])
    frozen = [
        r
        for r in rows
        if r["method"] == "frozen"
        and r["source_session"] == "1"
        and r["target_session"] == "2"
    ]
    check(
        [
            mean([r for r in frozen if r["participant"] == str(p)], "unseen")
            for p in [2, 3, 4, 5]
        ],
        c["search_pair_frozen"]["participant_unseen"],
    )
    controls = read("control_cases.csv")
    for dataset, label in [
        ("order", "resample_then_gain"),
        ("identity", "identity_tie"),
    ]:
        for method in ["spatial_only", "spatial_gain"]:
            selected = [
                r for r in controls if r["dataset"] == dataset and r["method"] == method
            ]
            assert len(selected) == 640
            for field in ["overall", "unseen"]:
                part = [
                    mean([r for r in selected if r["participant"] == str(p)], field)
                    for p in [2, 3, 4, 5]
                ]
                check(np.mean(part), c["controls"][label][method][field]["mean"])
                check(
                    boot(part),
                    c["controls"][label][method][field]["participant_bootstrap_95"],
                )
    for seed in ["42", "43", "44"]:
        pool = (
            rows
            if seed == "42"
            else [
                r
                for r in controls
                if r["dataset"] == "search_seeds" and r["seed"] == seed
            ]
        )
        for method in ["spatial_only", "spatial_gain"]:
            selected = [
                r
                for r in pool
                if r["method"] == method
                and r["source_session"] == "1"
                and r["target_session"] == "2"
            ]
            assert len(selected) == 32
            part = [
                mean([r for r in selected if r["participant"] == str(p)], "unseen")
                for p in [2, 3, 4, 5]
            ]
            check(
                part, c["search_seed_sensitivity"][method][seed]["participant_unseen"]
            )
    report = {
        "passed": True,
        "primary_records": len(rows),
        "control_records": len(controls),
        "independent_participants": 4,
        "scope": "Analysis reproduction from included case records; no raw-data or checkpoint replay.",
    }
    (RESULTS / "analysis_reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
