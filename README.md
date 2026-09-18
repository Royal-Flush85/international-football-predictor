# International Football Match Predictor

A Flask application that estimates home-win, draw, and away-win
probabilities from historical international football results.

## Set up
Developed using Python 3.13. The requirements include dependencies for both the Flask application and research notebooks.

## What the prediction means

The target is the recorded match outcome, including extra time where
played and excluding penalty shootouts. It is not consistently a
90-minute result or a probability of advancing.

## Model

The deployed system uses logistic regression with sigmoid calibration.

Features:
- Elo difference
- Absolute Elo difference
- Recent scoring-form difference
- Neutral-venue indicator
- Home-host indicator

## Evaluation

The corrected historical benchmark uses logistic regression with
sigmoid calibration and chronological data partitions:

- Development: before 2018.
- Calibration: 2018–2019.
- Model selection: 2020–2021.
- Historical holdout: 2022 through June 27, 2026.

| Partition | Matches | Model log loss | Reference log loss | Accuracy |
|---|---:|---:|---:|---:|
| Selection | 1,143 | 0.834752 | 1.062585 | 62.47% |
| Historical holdout | 3,367 | 0.886572 | 1.056874 | 60.26% |

Lower log loss is better. The reference predicts the same outcome
probabilities for every match, using class frequencies estimated
from development data.

The model reduced historical holdout log loss by approximately
16.1% relative to this reference.

The selection period informed model choice, and the historical
holdout was inspected during project development. These results
should therefore be interpreted as retrospective evaluation.

The application uses a separate deployment refit trained and
calibrated on newer data. 

Historical benchmark scores are not an independent evaluation of
that deployment artifact.

## Limitations

- No current lineup or injury information.
- Weaker historical results on World Cup finals.
- Predictions use results available through the saved data cutoff.
- Historical periods have already been inspected during development.

## Research notebook

`code.ipynb` documents model development and includes saved evaluation
tables and figures.

Some historical recovery, diagnostic, and artifact-preparation sections
depend on local files in `checkpoints/`, which are excluded from this
repository. These sections are labeled **Local checkpoint required**.
The notebook is therefore a research record rather than a fully
self-contained, top-to-bottom reproduction workflow.

The Flask app runs independently using the deployment model and team
state included in `artifacts/`. No notebook execution is required.

`archive.ipynb` preserves superseded experiments and debugging history.
It is not the active training or application workflow.

## Run locally

Use the Python environment compatible with the saved model artifacts.

    python -m pip install -r requirements.txt
    python check_app.py
    python -m flask --app app run

Open http://127.0.0.1:5000.

## Data

This project uses the International Football Results dataset
maintained by martj42.

- Download source: [Kaggle dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- Upstream repository: [martj42/international_results](https://github.com/martj42/international_results)
- Dataset license: [CC0 1.0 Universal](https://github.com/martj42/international_results/blob/master/LICENSE)
- Source file used: `results.csv`
- Latest completed match included in this project's prepared data:
  June 27, 2026.

The source contains men's international football results. This
version of the project excludes matches labeled `Friendly` and
rows with missing scores, leaving 31,089 competitive matches.

The score columns include extra time where played and exclude
penalty shootouts. Consequently, the model predicts the recorded
match outcome, not consistently the 90-minute outcome or which
team advances.

The application uses a saved data snapshot and does not automatically
download new results. The date above is the match-data cutoff,
not the download date or a guarantee of complete coverage.

## Next experiments

- Recency-weighted training.
- Defensive and opponent-adjusted form.
- A separately evaluated World Cup simulation approach.