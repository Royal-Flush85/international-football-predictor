"""Run from the project root: python check_tournament_page.py"""
from copy import deepcopy
import re
from app_v2 import app, predictor

teams = ["Japan", "Australia", "Argentina", "Brazil", "France", "Germany", "Spain", "England"]
before = deepcopy(predictor.state)
app.config["TESTING"] = True
with app.test_client() as client:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Tournament playground" in response.data
    response = client.post("/", data={"home_team": "Japan", "away_team": "Australia", "venue": "neutral"})
    assert response.status_code == 200 and b"v2-deployment" in response.data
    response = client.get("/tournament")
    assert response.status_code == 200
    assert response.data.count(b'name="teams"') == 8
    assert b"50/50 random tiebreaker" in response.data
    assert client.get("/static/tournament.css").status_code == 200
    result = client.post("/tournament", data={"teams": teams, "seed": "42"})
    assert result.status_code == 200
    html = result.get_data(as_text=True)
    ids = re.findall(r'data-match-id="([A-Z0-9]+)"', html)
    assert ids == ["QF1", "QF2", "QF3", "QF4", "SF1", "SF2", "F1"]
    assert "Simulated champion" in html and "Simulation seed:" in html
    assert "2026-06-27" in html
    again = client.post("/tournament", data={"teams": teams, "seed": "42"})
    assert again.data == result.data
    assert client.post("/tournament", data={"teams": teams, "seed": ""}).status_code == 200
    assert client.post("/tournament", data={"teams": teams, "seed": "0"}).status_code == 200
    for invalid in [teams[:-1], teams + ["Italy"], teams[:-1] + [teams[0]], teams[:-1] + ["Unknown"]]:
        response = client.post("/tournament", data={"teams": invalid, "seed": "42"})
        assert response.status_code == 400
        assert b'role="alert"' in response.data
        assert b'data-match-id=' not in response.data
    for bad_seed in ["-1", "1.5", "abc", "4294967296", "9" * 100]:
        response = client.post("/tournament", data={"teams": teams, "seed": bad_seed})
        assert response.status_code == 400 and b"Seed must be" in response.data
    assert client.get("/tournament").data.count(b'data-match-id=') == 0
assert predictor.state == before
print("Tournament page, bracket, seed, validation, immutable state, and predictor checks passed.")
