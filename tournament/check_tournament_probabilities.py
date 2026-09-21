"""Run with python check_tournament_probabilities.py from the project root."""
from copy import deepcopy
from time import perf_counter
import re
from tournament import estimate_tournament_probabilities
from app_v2 import app, predictor


class FakePredictor:
    teams = list("ABCDEFGH")
    bundle = {"version": "test"}
    state = {"data_cutoff": "test"}

    def __init__(self, first_wins=False):
        self.first_wins = first_wins
        self.calls = 0

    def predict(self, first, second, venue):
        assert venue == "neutral" and first != second
        self.calls += 1
        return {"home_win": float(self.first_wins), "draw": float(not self.first_wins), "away_win": 0.0}


def verify_counts(result):
    rows, n = result["rows"], result["n_simulations"]
    assert len(rows) == 8
    assert sum(r["semifinal_count"] for r in rows) == 4 * n
    assert sum(r["final_count"] for r in rows) == 2 * n
    assert sum(r["title_count"] for r in rows) == n
    for row in rows:
        assert 0 <= row["title_count"] <= row["final_count"] <= row["semifinal_count"] <= n


fair = FakePredictor()
result = estimate_tournament_probabilities(fair, fair.teams, seed=42)
assert fair.calls == 28
verify_counts(result)
for row in result["rows"]:
    assert abs(row["semifinal_probability"] - .5) < .03
    assert abs(row["final_probability"] - .25) < .03
    assert abs(row["title_probability"] - .125) < .03
assert result == estimate_tournament_probabilities(FakePredictor(), fair.teams, seed=42)

fixed = estimate_tournament_probabilities(FakePredictor(first_wins=True), fair.teams, seed=42)
verify_counts(fixed)
assert fixed["rows"][0]["team"] == "A"
assert fixed["rows"][0]["title_count"] == 10000
assert {r["team"] for r in fixed["rows"] if r["semifinal_count"]} == set("ACEG")
assert {r["team"] for r in fixed["rows"] if r["final_count"]} == set("AE")

for bad_teams in [fair.teams[:-1], fair.teams + ["I"], list("ABCDEFGG"), list("ABCDEFGZ")]:
    try:
        estimate_tournament_probabilities(fair, bad_teams, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid teams accepted")
for n in [0, 10001, True, 1.5]:
    try:
        estimate_tournament_probabilities(fair, fair.teams, n_simulations=n)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid simulation count accepted")

teams = ["Japan", "Australia", "Argentina", "Brazil", "France", "Germany", "Spain", "England"]
before = deepcopy(predictor.state)
start = perf_counter()
real = estimate_tournament_probabilities(predictor, teams, seed=42)
verify_counts(real)
print(f"28 matchup predictions + 10,000 simulations: {perf_counter() - start:.2f} seconds locally.")
app.config["TESTING"] = True
with app.test_client() as client:
    response = client.post("/tournament", data={"teams": teams, "seed": "42", "action": "probabilities"})
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert html.count('data-odds-team=') == 8
    assert html.count('data-match-id=') == 7
    assert "10,000 runs" in html
    labels = re.findall(r'<td><strong>([0-9.]+)%</strong></td>', html)
    assert len(labels) == 8
    assert sum(round(float(x) * 10) for x in labels) == 1000
    assert client.post("/tournament", data={"teams": teams, "seed": "42", "action": "probabilities"}).data == response.data
    single = client.post("/tournament", data={"teams": teams, "seed": "42", "action": "bracket"})
    assert single.status_code == 200 and b'data-odds-team=' not in single.data
    assert client.post("/tournament", data={"teams": teams, "action": "unknown"}).status_code == 400
    invalid = client.post("/tournament", data={"teams": teams[:-1], "action": "probabilities"})
    assert invalid.status_code == 400 and b'data-odds-team=' not in invalid.data
assert predictor.state == before
print("Probability-table checks passed: counts, bracket logic, cache, seed, validation, display totals and immutable state.")
