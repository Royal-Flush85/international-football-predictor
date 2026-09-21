import random
import numpy as np


def estimate_tournament_probabilities(predictor, teams, n_simulations=10000, seed=None):
    """Estimate advancement frequencies in a fixed eight-team bracket.

    Compute 28 possible matchup probabilities once per call. Simulate rounds
    in arrays; simulated outcomes never update the predictor or saved state.
    Left/right ordering follows the selected bracket, as in simulate_tournament.
    """
    if not isinstance(teams, (list, tuple)) or len(teams) != 8:
        raise ValueError("Select exactly eight teams.")
    if any(not isinstance(team, str) for team in teams):
        raise ValueError("Team names must be strings.")
    teams = [team.strip() for team in teams]
    if len(set(teams)) != 8:
        raise ValueError("Select eight different teams.")
    if any(team not in predictor.teams for team in teams):
        raise ValueError("Every selected team must have recorded history.")
    if type(n_simulations) is not int or not 1 <= n_simulations <= 10000:
        raise ValueError("Simulation count must be an integer from 1 to 10000.")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
        raise ValueError("Seed must be a whole number from 0 to 4294967295.")

    advancement = np.full((8, 8), np.nan)
    for first in range(8):
        for second in range(first + 1, 8):
            prediction = predictor.predict(teams[first], teams[second], venue="neutral")
            p = np.array([prediction["home_win"], prediction["draw"], prediction["away_win"]], dtype=float)
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1.0):
                raise ValueError("The predictor returned invalid probabilities.")
            # Equivalent to sampling win/draw/loss, then a fair draw tiebreak.
            advancement[first, second] = p[0] + 0.5 * p[1]

    rng = np.random.default_rng(seed)
    entrants = np.tile(np.arange(8), (n_simulations, 1))
    stage_counts = {}
    for stage in ["semifinal", "final", "title"]:
        first, second = entrants[:, ::2], entrants[:, 1::2]
        # Winners stay within their original bracket halves; first < second.
        probability = advancement[first, second]
        entrants = np.where(rng.random(probability.shape) < probability, first, second)
        stage_counts[stage] = np.bincount(entrants.ravel(), minlength=8)

    rows = []
    for index, team in enumerate(teams):
        row = {"team": team}
        for stage, counts in stage_counts.items():
            row[f"{stage}_count"] = int(counts[index])
            row[f"{stage}_probability"] = float(counts[index] / n_simulations)
        rows.append(row)
    rows.sort(key=lambda row: -row["title_count"])
    return {
        "teams": teams,
        "n_simulations": n_simulations,
        "seed": seed,
        "rows": rows,
        "matchups_evaluated": 28,
        "model_version": predictor.bundle["version"],
        "data_cutoff": predictor.state["data_cutoff"],
    }


def simulate_tournament(predictor, teams, seed=None):
    """
    Simulate an eight-team knockout tournament.

    Assumptions:
    - All matches use a neutral venue.
    - Team features remain fixed.
    - A sampled draw is resolved by a 50/50 random tiebreak.
    - Team order defines the bracket:
      1 vs 2, 3 vs 4, 5 vs 6, 7 vs 8.
    """
    if not isinstance(teams, (list, tuple)) or len(teams) != 8:
        raise ValueError("Select exactly eight teams.")

    if any(not isinstance(team, str) for team in teams):
        raise ValueError("Team names must be strings.")

    teams = [team.strip() for team in teams]

    if len(set(teams)) != 8:
        raise ValueError("Select eight different teams.")

    if any(team not in predictor.teams for team in teams):
        raise ValueError("Every selected team must have recorded history.")

    # Local random generator: does not change global random state.
    # after choosing 8 teams, create random seed
    rng = random.Random(seed)

    # These belong to this simulation, not to the saved predictor state.
    matches = []
    probability_cache = {}

    def play_match(first_team, second_team, round_name, match_id):
        key = (first_team, second_team)

        if key not in probability_cache:
            probability_cache[key] = predictor.predict(
                first_team,
                second_team,
                venue="neutral",
            )

        prediction = probability_cache[key]

        # "Home" means first-listed here; the venue is neutral.
        first_win = float(prediction["home_win"])
        draw = float(prediction["draw"])
        second_win = float(prediction["away_win"])

        sample = rng.random()

        if sample < first_win:
            sampled_outcome = "first_win"
            winner = first_team
            tiebreak_used = False

        elif sample < first_win + draw:
            sampled_outcome = "draw"
            winner = (
                first_team if rng.random() < 0.5 else second_team
            )
            tiebreak_used = True

        else:
            sampled_outcome = "second_win"
            winner = second_team
            tiebreak_used = False

        # saving match results
        matches.append({
            "match_id": match_id,
            "round": round_name,
            "first_team": first_team,
            "second_team": second_team,
            "probabilities": {
                "first_win": first_win,
                "draw": draw,
                "second_win": second_win,
            },
            "first_team_advancement_probability": (
                first_win + 0.5 * draw
            ),
            "sampled_outcome": sampled_outcome,
            "tiebreak_used": tiebreak_used,
            "winner": winner,
        })

        return winner

    def resolve_bracket(bracket_teams, start_slot=0):
        # Base case: one team needs no further match in this subtree.
        if len(bracket_teams) == 1:
            return bracket_teams[0]

        midpoint = len(bracket_teams) // 2

        first_finalist = resolve_bracket(
            bracket_teams[:midpoint],
            start_slot,
        )
        second_finalist = resolve_bracket(
            bracket_teams[midpoint:],
            start_slot + midpoint,
        )

        round_name, prefix = {
            2: ("Quarterfinal", "QF"),
            4: ("Semifinal", "SF"),
            8: ("Final", "F"),
        }[len(bracket_teams)]

        match_number = start_slot // len(bracket_teams) + 1
        match_id = f"{prefix}{match_number}"

        # playing matches, part of recursive process of tournament bracket
        return play_match(
            first_finalist,
            second_finalist,
            round_name,
            match_id,
        )

    # running the tournament simulation
    champion = resolve_bracket(teams)

    # Recursion finishes each branch before visiting the next.
    # Sort the returned list into display order.
    round_order = {
        "Quarterfinal": 0,
        "Semifinal": 1,
        "Final": 2,
    }

    matches.sort(
        key=lambda match: (
            round_order[match["round"]],
            match["match_id"],
        )
    )

    return {
        "teams": teams,
        "seed": seed,
        "matches": matches,
        "champion": champion,
        "model_version": predictor.bundle["version"],
        "data_cutoff": predictor.state["data_cutoff"],
        "assumptions": [
            "All venues are neutral.",
            "Team features remain fixed throughout the tournament.",
            "Sampled draws use a 50/50 random tiebreaker.",
            "Outcomes include extra time where present in the training data.",
        ],
    }
