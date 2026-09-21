# Add the 10,000-run table

Copy these replacements into your project, preserving the folder structure:

- tournament.py
- app_v2.py
- templates/tournament.html
- static/tournament.css
- README.md (review against any newer edits before replacing)

Add check_tournament_probabilities.py. The included check_tournament_page.py
is copied from your existing project for convenience. Keep all model artifacts
and other application files unchanged. No new dependencies are needed.

The existing recursive function still generates one bracket. The new
estimate_tournament_probabilities() function calculates 28 matchup predictions
once, then uses NumPy arrays to simulate the three rounds across 10,000 runs.
It returns counts and probabilities for each team at each stage. This is
equivalent in distribution to sampling match outcomes and fair drawn-match
tiebreaks. It does not produce a match log for all 70,000 simulated matches.

Run:

```powershell
python check_tournament.py
python check_tournament_page.py
python check_tournament_probabilities.py
python check_app_v2.py
python -m flask --app app_v2 run
```

Open http://127.0.0.1:5000/tournament and click Estimate chances. The table
shows chances to reach the semifinals, reach the final and win the tournament.
It also shows title counts out of 10,000. Title percentages round to 100.0%.
The single bracket below is a separate illustrative draw. Inspect desktop
and mobile appearance yourself before publishing.

The seed makes runs repeatable with the same selections, model, data and
software environment. Blank generates a new seed. More runs reduce sampling
noise, not model uncertainty. Do not describe this as an official World Cup
forecast. Read the Tournament playground README section for the assumptions.

Review and publish using your usual Git workflow. Explicitly stage intended
files so an accidentally extracted package directory is not included.
Render's current app_v2:app start command does not change.
