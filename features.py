import numpy as np
import pandas as pd

def new_team_state():
    return {
        "elo": 1500.0,
        "recent_goals": [],
        "last_match_date": None,
    }


# Build the model's inputs: predict_match_latest()
# converts two team's saved information into input columns
def build_match_features(
    state,
    home_team,
    away_team,
    feature_names,
    *,
    is_neutral,
    is_home_team_host,
    allow_unseen=False,
):
    if home_team == away_team:
        raise ValueError("Select two different teams.")

    for value in [is_neutral, is_home_team_host]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("Venue flags must be booleans.")

    if is_neutral and is_home_team_host:
        raise ValueError("A neutral match cannot have home-host=True.")

    expected_features = {
        "elo_difference",
        "is_home_team_host",
        "neutral",
        "form_difference",
        "abs_elo_difference",
    }

    if (
        set(feature_names) != expected_features
        or len(feature_names) != len(expected_features)
    ):
        raise ValueError("Unexpected version 1 feature schema.")

    def get_team(team):
        if team in state["teams"]:
            return state["teams"][team]

        # Needed to reproduce training features for first appearances.
        if allow_unseen:
            return new_team_state()

        raise ValueError(f"No recorded competitive history for {team!r}.")

    home = get_team(home_team)
    away = get_team(away_team)

    def mean_goals(team):
        goals = team["recent_goals"]
        return float(np.mean(goals)) if goals else 0.0

    elo_difference = home["elo"] - away["elo"]

    values = {
        "elo_difference": elo_difference,
        "is_home_team_host": float(is_home_team_host),
        "neutral": float(is_neutral),
        "form_difference": mean_goals(home) - mean_goals(away),
        "abs_elo_difference": abs(elo_difference),
    }

    return pd.DataFrame(
        [values],
        columns=list(feature_names),
    ).astype(float)