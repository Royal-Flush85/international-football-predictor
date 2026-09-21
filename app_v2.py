# Connecting Flask to the predictor
from pathlib import Path
import secrets

from flask import Flask, render_template, request

from predictor_v2 import MatchPredictor, display_percentages
from tournament import simulate_tournament, estimate_tournament_probabilities


app = Flask(__name__)

artifact_dir = Path(__file__).resolve().parent / "artifacts_v2"
predictor = MatchPredictor(artifact_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    home_team = request.form.get("home_team", "")
    away_team = request.form.get("away_team", "")
    venue = request.form.get("venue", "neutral")

    if request.method == "POST":
        try:
            result = predictor.predict(
                home_team,
                away_team,
                venue=venue,
            )
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        teams=predictor.teams,
        tournament_enabled=True,
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        result=result,
        error=error,
    )

@app.route("/tournament", methods=["GET", "POST"])
def tournament_page():
    defaults = ["Japan", "Australia", "Argentina", "Brazil",
                "France", "Germany", "Spain", "England"]
    selected = [team if team in predictor.teams else "" for team in defaults]
    seed_text = ""
    result = None
    odds_result = None
    error = None
    status = 200
    if request.method == "POST":
        selected = request.form.getlist("teams")
        seed_text = request.form.get("seed", "").strip()
        try:
            action = request.form.get("action", "bracket")
            if action not in {"bracket", "probabilities"}:
                raise ValueError("Choose a supported simulation action.")
            if seed_text:
                if (not seed_text.isascii() or not seed_text.isdecimal()
                        or len(seed_text) > 10):
                    raise ValueError("Seed must be a whole number from 0 to 4294967295.")
                seed = int(seed_text)
                if seed > 4294967295:
                    raise ValueError("Seed must be a whole number from 0 to 4294967295.")
            else:
                seed = secrets.randbits(32)
            result = simulate_tournament(predictor, selected, seed=seed)
            if action == "probabilities":
                odds_result = estimate_tournament_probabilities(
                    predictor, selected, n_simulations=10000, seed=seed
                )
                # Largest-remainder display rounding keeps title percentages at 100.0%.
                title_labels = display_percentages([
                    row["title_probability"] for row in odds_result["rows"]
                ])
                for row, label in zip(odds_result["rows"], title_labels):
                    row["title_percent"] = label
            selected = result["teams"]
            for match in result["matches"]:
                first = round(match["first_team_advancement_probability"] * 100, 1)
                match["first_advance_percent"] = f"{first:.1f}"
                match["second_advance_percent"] = f"{100 - first:.1f}"
        except ValueError as exc:
            result = None
            odds_result = None
            error = str(exc)
            status = 400
    # Keep the form at exactly eight slots even for malformed requests.
    selected = (selected + [""] * 8)[:8]
    rounds = [
        ("Quarterfinals", ["QF1", "QF2", "QF3", "QF4"]),
        ("Semifinals", ["SF1", "SF2"]),
        ("Final", ["F1"]),
    ]
    match_map = {m["match_id"]: m for m in result["matches"]} if result else {}
    return render_template(
        "tournament.html", teams=predictor.teams, selected=selected,
        seed_text=seed_text, result=result, odds_result=odds_result, error=error,
        rounds=rounds, match_map=match_map,
        model_version=predictor.bundle["version"],
        data_cutoff=predictor.state["data_cutoff"],
        team_dates={team: predictor.state["teams"][team]["last_match_date"]
                    for team in predictor.teams},
    ), status
