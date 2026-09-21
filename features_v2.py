import math
import numpy as np
import pandas as pd

def new_state():
    return {
        "teams": {},
        "data_cutoff": None,
        "feature_version": "v2_equal_defense_context_scoring",
    }


def new_team_state():
    return {
        "elo": 1500.0,
        "recent_goals": [],
        "recent_defense": [],
        "last_match_date": None,
    }


# updating elo, recent goals, last-played dates after a match
# -> preparing  data or adding actual results
def _update_base_state(state, match):
    """Update state AFTER a completed competitive match."""

    if match["tournament"] == "Friendly":
        return  # Match the training notebook's filtering.

    if pd.isna(match["home_score"]) or pd.isna(match["away_score"]):
        raise ValueError("Cannot update state with an unplayed match.")

    home = match["home_team"]
    away = match["away_team"]
    date = pd.Timestamp(match["date"])

    if home == away:
        raise ValueError("A team cannot play itself.")

    if (
        state["data_cutoff"] is not None
        and date < pd.Timestamp(state["data_cutoff"])
    ):
        raise ValueError("Process matches in chronological order.")

    home_score = float(match["home_score"])
    away_score = float(match["away_score"])

    if (
        not np.isfinite([home_score, away_score]).all()
        or min(home_score, away_score) < 0
    ):
        raise ValueError("Scores must be finite and nonnegative.")

    for team in [home, away]:
        if team not in state["teams"]:
            state["teams"][team] = new_team_state()

    home_state = state["teams"][home]
    away_state = state["teams"][away]

    home_elo = home_state["elo"]
    away_elo = away_state["elo"]

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    # Preserve the notebook Elo rules for feature parity.
    tournament = str(match["tournament"])

    if tournament == "FIFA World Cup":
        k = 60
    elif (
        "World Cup qualification" in tournament
        or "UEFA Euro" in tournament
        or "Copa America" in tournament
    ):
        k = 50
    else:
        k = 40

    margin = abs(home_score - away_score)
    multiplier = math.sqrt(margin) if margin > 0 else 1.0

    expected_home = 1 / (
        1 + 10 ** ((away_elo - home_elo) / 400)
    )

    change = k * multiplier * (actual_home - expected_home)

    home_state["elo"] = home_elo + change
    away_state["elo"] = away_elo - change

    for team_state, goals in [
        (home_state, home_score),
        (away_state, away_score),
    ]:
        team_state["recent_goals"] = (
            team_state["recent_goals"] + [goals]
        )[-5:]

        team_state["last_match_date"] = date.isoformat()

    state["data_cutoff"] = date.isoformat()

FEATURE_VERSION = "v2_equal_defense_context_scoring"
FEATURE_NAMES = [
    "elo_difference", "is_home_team_host", "neutral",
    "form_difference", "abs_elo_difference",
    "home_goals_conceded_5", "away_goals_conceded_5",
    "home_opponent_elo_5", "away_opponent_elo_5", "scoring_level",
]


def build_match_features(state, home_team, away_team, feature_names, *,
                         is_neutral, is_home_team_host, allow_unseen=False):
    """Read current state without changing it; latest predictions follow cutoff."""
    if state["feature_version"] != FEATURE_VERSION:
        raise ValueError("Expected a matching V2 state, not a V1 state.")
    if home_team == away_team:
        raise ValueError("Select two different teams.")
    if any(not isinstance(v, (bool, np.bool_))
           for v in [is_neutral, is_home_team_host]):
        raise ValueError("Venue flags must be booleans.")
    if is_neutral and is_home_team_host:
        raise ValueError("A neutral match cannot have home-host=True.")
    if set(feature_names) != set(FEATURE_NAMES) or len(feature_names) != len(FEATURE_NAMES):
        raise ValueError("Unexpected V2 feature schema.")

    def get_team(name):
        if name in state["teams"]:
            return state["teams"][name]
        if allow_unseen:
            return new_team_state()
        raise ValueError(f"No recorded competitive history for {name!r}.")

    home, away = get_team(home_team), get_team(away_team)
    def form(team):
        return float(np.mean(team["recent_goals"])) if team["recent_goals"] else 0.0
    difference = home["elo"] - away["elo"]
    values = {
        "elo_difference": difference,
        "abs_elo_difference": abs(difference),
        "neutral": float(is_neutral),
        "is_home_team_host": float(is_home_team_host),
        "form_difference": form(home) - form(away),
        "scoring_level": form(home) + form(away),
    }
    for side, team in [("home", home), ("away", away)]:
        history = team["recent_defense"]
        values[f"{side}_goals_conceded_5"] = (
            float(np.mean([r[0] for r in history])) if history else np.nan
        )
        values[f"{side}_opponent_elo_5"] = (
            float(np.mean([r[1] for r in history])) if history else np.nan
        )
    return pd.DataFrame([values], columns=list(feature_names)).astype(float)


def update_state(state, daily_matches, feature_names=FEATURE_NAMES):
    """Process ONE complete date in original stable order.

    Returns each match's pre-match features. Elo/scoring update sequentially,
    matching the baseline notebook. Defense updates only after the full date.
    Rebuild from history when adding corrections or more results to a saved date.
    """
    records = daily_matches.to_dict("records")
    if not records:
        raise ValueError("Provide a complete nonempty match date.")
    dates = pd.to_datetime(daily_matches["date"])
    if dates.isna().any() or dates.nunique() != 1:
        raise ValueError("Pass exactly one date at a time.")
    date = dates.iloc[0]
    if state["data_cutoff"] is not None and date <= pd.Timestamp(state["data_cutoff"]):
        raise ValueError("Each complete date must be later than the saved cutoff.")
    # Validate the batch before mutating state.
    for match in records:
        if match["tournament"] == "Friendly":
            raise ValueError("Use the notebook's competitive-only history.")
        scores = np.array([match["home_score"], match["away_score"]], dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Require completed matches with valid scores.")
        build_match_features(state, match["home_team"], match["away_team"],
                             feature_names, is_neutral=match["neutral"],
                             is_home_team_host=match["is_home_team_host"], allow_unseen=True)
    pending, frames = [], []
    for match in records:
        home, away = match["home_team"], match["away_team"]
        frames.append(build_match_features(
            state, home, away, feature_names, is_neutral=match["neutral"],
            is_home_team_host=match["is_home_team_host"], allow_unseen=True))
        home_elo = state["teams"].get(home, new_team_state())["elo"]
        away_elo = state["teams"].get(away, new_team_state())["elo"]
        pending.extend([
            (home, float(match["away_score"]), away_elo),
            (away, float(match["home_score"]), home_elo),
        ])
        _update_base_state(state, match)
    for team, conceded, opponent_elo in pending:
        item = state["teams"][team]
        item["recent_defense"] = (item["recent_defense"] + [(conceded, opponent_elo)])[-5:]
    state["matches_processed"] = state.get("matches_processed", 0) + len(records)
    return pd.concat(frames, ignore_index=True)
