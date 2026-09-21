# International Football Match Predictor

A Flask application estimating home-win, draw, and away-win probabilities
from historical men's international football results.

[Open the live demo](https://international-football-predictor.onrender.com/).
The running model version and results cutoff are shown with each prediction.
The live service may still show v1.1 until the v2 release is deployed.

## What the prediction means

The target is the recorded match outcome, including extra time where played
and excluding penalty shootouts. It is not consistently a 90-minute result
or a probability of advancing.

The app uses a saved data snapshot, not live results, lineups or injury feeds.
Its current deployment artifacts include results through **June 27, 2026**.

## Version 2 model

The v2 application uses **uncalibrated logistic regression (C=0.01)** inside
a fitted preprocessing pipeline. Raw and sigmoid-calibrated versions of LR
and XGBoost were compared on the selection period; LR raw was selected by
log loss, with only a small margin over the closest candidate.

Features:

- Elo difference and absolute Elo difference.
- Recent scoring-form difference and scoring level (the sum of both averages).
- Neutral-venue and home-host indicators.
- Each team's average goals conceded over its last five competitive matches.
- Each team's average opponent pre-match Elo over those same five matches.

V2 uses equal weights for these histories. Opponent Elo is a separate context
feature, not a direct opponent-adjusted transformation of goals conceded.
Experiments compared LR, XGBoost and LightGBM and tested 180/365-day recency
weighting. Greater model complexity did not automatically improve results.

## Evaluation

Chronological partitions:

- Development: before 2018, with four expanding validation folds covering
  2010-2011, 2012-2013, 2014-2015 and 2016-2017.
- Calibration: 2018-2019, used only to fit candidate calibration mappings.
- Selection: 2020-2021, used to compare raw and calibrated candidates.
- Historical holdout: 2022 through June 27, 2026.

Both frozen evaluation models were compared on the same 3,367 holdout matches.

| Frozen evaluation model | Log loss | Accuracy |
|---|---:|---:|
| V1.1: LR with sigmoid calibration | 0.886572 | 60.26% |
| V2: LR raw with defense, opponent context and scoring level | 0.870196 | 60.35% |

Lower log loss is better. V2 reduced log loss by **0.016376**, approximately
**1.85% relative to v1.1**, while accuracy changed little. This measures better
probabilistic predictions overall; it does not establish perfect calibration.
The constant development-class-frequency reference scored **1.056874**.

| Holdout segment | Matches | V1.1 log loss | V2 log loss |
|---|---:|---:|---:|
| Neutral | 1,205 | 0.954683 | 0.915620 |
| Non-neutral | 2,162 | 0.848610 | 0.844879 |
| World Cup finals | 136 | 1.048881 | 1.025665 |

V2 improved each reported year, including the partial 2026 period. These
periods were previously inspected during development, so this is a
**retrospective comparison, not a fresh independent test or a claim of
statistical significance**.

### Evaluation versus deployment

The frozen v2 evaluation model was fitted on development data. Its separate
**v2 deployment refit** uses the same selected configuration and all 31,089
eligible historical matches through the cutoff. No sigmoid calibration is
applied to this refit. The reported holdout score belongs to the frozen
evaluation model, not to the deployment refit.

Application-generated features were checked against all 31,089 historical
rows. Prediction checks cover valid probabilities, rounding to a displayed
100.0% total, repeatability, artifact reloads, invalid inputs and unchanged
team state during prediction.

## Run locally

Developed using Python 3.13. Use an environment compatible with the saved
artifacts. Requirements include both application and notebook dependencies.

```bash
python -m pip install -r requirements.txt
python check_app_v2.py
python -m flask --app app_v2 run
```

Open the local address printed by Flask. No notebook execution is required:
the application loads `artifacts_v2/model_bundle.joblib` and
`artifacts_v2/latest_team_state.joblib`.

The v1 application and `artifacts/` remain available for rollback.

## Deployment

For the existing Python web service:

- Build command: `pip install -r requirements.txt`
- V2 start command:

```bash
gunicorn app_v2:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

Push the v2 code and both artifacts together before changing the start
command. Verify the live page reports `v2-deployment`. To roll back to the
retained v1 application, restore the start command's module to `app:app`.

## Research records and rebuilding

- `code.ipynb`: v1 development and historical evaluation.
- `goals_conceded_experiment.ipynb`: v2 feature experiments, tuning,
  calibration comparisons and diagnostics.
- `archive.ipynb`: superseded experiments, not the active workflow.
- `prepare_v2_artifacts.py`: historical feature verification and deployment refit.

The notebooks are **research records**. Some recovery, evaluation and
artifact-preparation sections depend on local `checkpoints/` files, which
are excluded from Git. They are not self-contained, top-to-bottom
reproduction workflows from a fresh clone. The preparation script likewise
requires a compatible local research checkpoint and frozen evaluation bundle:

```bash
python prepare_v2_artifacts.py --research checkpoints/YOUR_RESEARCH_CHECKPOINT --frozen checkpoints/YOUR_FROZEN_CHECKPOINT --output checkpoints/NEW_DEPLOYMENT_CHECKPOINT
```

The state updater processes complete dates in stable historical order. Defense
and opponent-context histories exclude same-day results. Baseline Elo and
scoring form retain the notebook's sequential ordering within each date;
this preserves the evaluated feature definitions but does not establish
actual intraday kickoff order. Rebuild state when correcting earlier results.

## Notebook publication hygiene

Notebook files contain saved outputs as well as code. Before committing:

```bash
python check_notebook_privacy.py
```

If outputs contain local paths or saved exceptions, close or reload the
notebook around this cleanup so a stale editor copy does not overwrite it:

```bash
python check_notebook_privacy.py --clean
python check_notebook_privacy.py
```

This removes flagged text/error outputs and preserves clean tables and
figures. It reports source-code paths for manual correction. It does not
inspect text inside images or constitute a complete secret scan; review
figures and the Git diff as well. Local VS Code preferences, environments,
checkpoints and `.env` files are excluded from Git.

## Limitations

- No current lineup, injury or squad-strength information.
- Recorded outcomes mix regulation-time and extra-time results.
- World Cup finals remain a weaker, relatively small evaluation segment.
- Reliability checks show overconfidence in moderate home-win probabilities:
  the 0.6-0.7 bin averaged 0.651 predicted versus 0.566 observed (394 matches).
- Historical teams may remain selectable; check each team's last recorded date.
- Historical evaluation periods have already influenced project development.
- The app does not automatically download new results or update its state.

## Data

International Football Results, maintained by martj42:

- [Kaggle dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [Upstream repository](https://github.com/martj42/international_results)
- [Dataset license: CC0 1.0 Universal](https://github.com/martj42/international_results/blob/master/LICENSE)

Source file: `results.csv`. This project excludes matches labeled `Friendly`
and rows with missing scores, leaving 31,089 competitive matches. Scores
include extra time where played and exclude shootouts. June 27, 2026 is the
prepared results cutoff, not the download date or a guarantee of coverage.

## Possible extension

A separately labeled simplified tournament guessing feature could sample
outcomes from the model using explicit tiebreaker assumptions. It is not
implemented and would not constitute a validated official-format simulator.
