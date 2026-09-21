"""Verify feature parity, refit fixed LR configuration, and stage V2 artifacts.
Run from the project root; requires the saved local research checkpoints.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from features_v2 import new_state, update_state, FEATURE_VERSION
from predictor_v2 import MatchPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing artifacts will not be overwritten.")
    research = joblib.load(args.research / "research_state.joblib")
    frozen = joblib.load(args.frozen / "model_bundle.joblib")
    if frozen["model_name"] != "LR raw":
        raise ValueError("This deployment recipe is specifically for selected LR raw.")
    history = research["matches"].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    history["scoring_level"] = history["home_rolling_goals"] + history["away_rolling_goals"]
    columns = list(frozen["features"])
    assert history["date"].notna().all()
    assert history["tournament"].ne("Friendly").all()
    assert set(history["target"].unique()) == {0, 1, 2}
    expected_target = np.where(history["home_score"] > history["away_score"], 2,
                               np.where(history["home_score"] < history["away_score"], 0, 1))
    np.testing.assert_array_equal(history["target"], expected_target)

    state = new_state()
    checked = 0
    print("Replaying historical features...", flush=True)
    for date, daily in history.groupby("date", sort=False):
        generated = update_state(state, daily, columns)
        np.testing.assert_allclose(
            generated.to_numpy(), daily[columns].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-8, equal_nan=True,
            err_msg=f"Feature mismatch on {date}")
        checked += len(daily)
    assert checked == len(history)
    print(f"Feature parity passed for {checked} matches.", flush=True)

    # Clone removes fitted parameters but preserves the selected configuration.
    # This refits preprocessing as well as LR on all eligible historical results.
    deployment_model = clone(frozen["model"])
    deployment_model.fit(history[columns].astype(float), history["target"])
    np.testing.assert_array_equal(deployment_model.classes_, [0, 1, 2])
    bundle = {
        "model": deployment_model,
        "features": columns,
        "model_name": "LR raw",
        "version": "v2-deployment",
        "feature_version": FEATURE_VERSION,
        "class_names": frozen["class_names"],
        "target_definition": frozen["target_definition"],
        "training_cutoff": history["date"].max().isoformat(),
        "training_matches": len(history),
        "source_checkpoint": args.frozen.name,
        "evaluation_note": "Historical scores belong to the frozen evaluation model, not this deployment refit.",
    }
    state["checkpoint_id"] = args.output.name
    args.output.mkdir(parents=True)
    joblib.dump(bundle, args.output / "model_bundle.joblib")
    joblib.dump(state, args.output / "latest_team_state.joblib")

    predictor = MatchPredictor(args.output)
    before = deepcopy(predictor.state)
    for venue in ["neutral", "home"]:
        prediction = predictor.predict("Japan", "Australia", venue=venue)
        assert prediction == predictor.predict("Japan", "Australia", venue=venue)
        probabilities = [prediction[k] for k in ["home_win", "draw", "away_win"]]
        assert np.isfinite(probabilities).all()
        assert all(0 <= p <= 1 for p in probabilities)
        np.testing.assert_allclose(sum(probabilities), 1.0)
        tenths = sum(int(round(float(prediction[k]) * 10))
                     for k in ["home_percent", "draw_percent", "away_percent"])
        assert tenths == 1000
    assert predictor.state == before
    for home, away, venue in [("Japan", "Japan", "neutral"),
                              ("Unknown team", "Japan", "neutral"),
                              ("Japan", "Australia", "invalid")]:
        try:
            predictor.predict(home, away, venue)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    reloaded = MatchPredictor(args.output)
    assert reloaded.predict("Japan", "Australia") == predictor.predict("Japan", "Australia")
    report = (
        f"Historical feature parity: {checked} matches passed.\n"
        "Prediction repeatability, state immutability, reload, probabilities, "
        "display totals, and invalid-input checks passed.\n"
        f"Data cutoff: {state['data_cutoff']}\n"
        "Existing application artifacts were not changed.\n"
    )
    (args.output / "verification.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(predictor.predict("Japan", "Australia"), flush=True)


if __name__ == "__main__":
    main()
