# testing the tournament simulation
from collections import Counter
from copy import deepcopy
from pathlib import Path

from predictor_v2 import MatchPredictor
from tournament import simulate_tournament


project_root = Path(__file__).resolve().parent
predictor = MatchPredictor(project_root / "artifacts_v2")

teams = [
    "Japan", "Australia",
    "Argentina", "Brazil",
    "France", "Germany",
    "Spain", "England",
]

state_before = deepcopy(predictor.state)

result = simulate_tournament(predictor, teams, seed=42)

assert len(result["matches"]) == 7
assert len({match["match_id"] for match in result["matches"]}) == 7
assert result["champion"] in teams

assert Counter(match["round"] for match in result["matches"]) == {
    "Quarterfinal": 4,
    "Semifinal": 2,
    "Final": 1,
}

by_id = {
    match["match_id"]: match
    for match in result["matches"]
}

# Check that every later-round participant won the preceding match.
for next_match, preceding_matches in {
    "SF1": ("QF1", "QF2"),
    "SF2": ("QF3", "QF4"),
    "F1": ("SF1", "SF2"),
}.items():
    assert [
        by_id[next_match]["first_team"],
        by_id[next_match]["second_team"],
    ] == [
        by_id[previous]["winner"]
        for previous in preceding_matches
    ]

assert by_id["F1"]["winner"] == result["champion"]

# Fixed seed produces the same bracket.
assert result == simulate_tournament(predictor, teams, seed=42)

for invalid_teams in [
    teams[:-1],
    teams[:-1] + [teams[0]],
    teams[:-1] + ["Unknown Team"],
]:
    try:
        simulate_tournament(predictor, invalid_teams)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid team selection was accepted.")

assert predictor.state == state_before

for match in result["matches"]:
    note = " (random tiebreak)" if match["tiebreak_used"] else ""
    print(
        f"{match['match_id']}: "
        f"{match['first_team']} vs {match['second_team']} "
        f"-> {match['winner']}{note}"
    )

print("\nChampion:", result["champion"])
print("Tournament checks passed.")