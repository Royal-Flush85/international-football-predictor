# Connecting Flask to the predictor
from pathlib import Path

from flask import Flask, render_template, request

from predictor import MatchPredictor


app = Flask(__name__)

artifact_dir = Path(__file__).resolve().parent / "artifacts"
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
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        result=result,
        error=error,
    )