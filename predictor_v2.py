from pathlib import Path

import joblib
import numpy as np

from features_v2 import build_match_features

def display_percentages(probabilities):
    """Round to tenths of a percent while preserving a 100% total."""
    probabilities = np.asarray(probabilities, dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError("Invalid probabilities.")

    # Normalize only for negligible floating-point differences.
    scaled = probabilities / probabilities.sum() * 1000
    units = np.floor(scaled).astype(int)

    remaining = 1000 - int(units.sum())
    order = np.argsort(-(scaled - units), kind="stable")
    units[order[:remaining]] += 1

    return [f"{value / 10:.1f}" for value in units]

# Replacing the notebook's global variable based prediction
# function with an object that owns its model and state
class MatchPredictor:
    def __init__(self, artifact_dir):
        artifact_dir = Path(artifact_dir)

        self.bundle = joblib.load(
            artifact_dir / "model_bundle.joblib"
        )
        self.state = joblib.load(
            artifact_dir / "latest_team_state.joblib"
        )

        if (
            self.bundle["feature_version"]
            != self.state["feature_version"]
        ):
            raise ValueError("Model and state feature versions differ.")

        if set(self.bundle["model"].classes_) != {0, 1, 2}:
            raise ValueError("Unexpected outcome classes.")

        self.teams = sorted(self.state["teams"])

    def predict(self, home_team, away_team, venue="neutral"):
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            raise ValueError("Provide valid team names.")

        home_team = home_team.strip()
        away_team = away_team.strip()

        if home_team not in self.state["teams"]:
            raise ValueError("Select a supported home team.")

        if away_team not in self.state["teams"]:
            raise ValueError("Select a supported away team.")

        # Keep this initial interface simple:
        # neutral venue or first-listed team hosting.
        if venue not in {"neutral", "home"}:
            raise ValueError("Select a valid venue.")

        features = build_match_features(
            self.state,
            home_team,
            away_team,
            self.bundle["features"],
            is_neutral=(venue == "neutral"),
            is_home_team_host=(venue == "home"),
            allow_unseen=False,
        )

        model = self.bundle["model"]
        probabilities = model.predict_proba(features)[0]

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise RuntimeError("Model returned invalid probabilities.")

        by_class = dict(zip(model.classes_, probabilities))

        home_pct, draw_pct, away_pct = display_percentages([
            by_class[2],
            by_class[1],
            by_class[0],
        ])

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win": float(by_class[2]),
            "draw": float(by_class[1]),
            "away_win": float(by_class[0]),
            "model_version": self.bundle["version"],
            "data_cutoff": self.state["data_cutoff"],
            "target_definition": self.bundle["target_definition"],
            "home_percent": home_pct,
            "draw_percent": draw_pct,
            "away_percent": away_pct,
            "home_last_match": self.state["teams"][home_team]["last_match_date"],
            "away_last_match": self.state["teams"][away_team]["last_match_date"]
        }


if __name__ == "__main__":
    artifact_dir = Path(__file__).resolve().parent / "artifacts_v2"
    predictor = MatchPredictor(artifact_dir)
    print(predictor.predict("Japan", "Australia"))