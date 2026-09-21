# small automated check for probability adding to 1.0
from copy import deepcopy
from pathlib import Path

import numpy as np

from predictor_v2 import MatchPredictor

artifact_dir = Path(__file__).resolve().parent / "artifacts_v2"
predictor = MatchPredictor(artifact_dir)

state_before = deepcopy(predictor.state)

first = predictor.predict("Japan", "Australia", venue="neutral")
second = predictor.predict("Japan", "Australia", venue="neutral")

probability_keys = ["home_win", "draw", "away_win"]
probabilities = np.array([first[key] for key in probability_keys])

assert np.isfinite(probabilities).all()
assert ((probabilities >= 0) & (probabilities <= 1)).all()
assert np.isclose(probabilities.sum(), 1.0)

assert first == second
assert predictor.state == state_before

for home, away, venue in [
    ("Japan", "Japan", "neutral"),
    ("Unknown Team", "Australia", "neutral"),
    ("Japan", "Australia", "invalid"),
]:
    try:
        predictor.predict(home, away, venue=venue)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"Expected invalid input rejection: {home}, {away}, {venue}"
        )

# Confirm a freshly loaded model/state pair reproduces predictions.
reloaded = MatchPredictor(artifact_dir)
assert reloaded.predict(
    "Japan", "Australia", venue="neutral"
) == first

print("Prediction checks passed.")