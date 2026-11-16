# MLP player rating forecast

This CSC 671 project uses a multi-layer perceptron (MLP) to forecast a soccer player's rating in their next rated Premier League appearance. Using the 2025–26 season, we will learn from each player's previous five appearances and compare how different model settings affect prediction accuracy.
Predictions use only information available before kickoff and are evaluated on appearances with positive recorded minutes and a match rating.

## Project goals

- [x] Prepare the dataset.
- [ ] Split the data into training, validation, and test sets by date.
- [ ] Build and train an MLP to predict player ratings.
- [ ] Tune the model's hyperparameters.
- [ ] Test the model and compare results with simple baselines.

## Setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The curated CSVs are committed under `data/processed/`, so teammates do not need to download or rebuild the data to start modeling. Use `forecasts.csv` and the feature list in `summary.json`.

To regenerate the curated dataset after changing its preparation:

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python -m unittest discover -s tests -v
```

## Data

The source is [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights), a community collection of real match statistics and derived metrics used for fantasy-football(soccer) analysis. This project uses the completed 2025–26 Premier League season. The exact source commit and scope are in `data/source.json`.

We prepare our own joined dataset from existing source CSVs.

- `playermatchstats.csv`: player-match statistics and authoritative `minutes_played`.
- `player_match_enrichment.csv`: the recorded match rating.
- `matches.csv`: match IDs and kickoff dates.
- `players.csv`: names, IDs, and positions.

Only the `By Tournament/Premier League` match files are downloaded.
Originals stay in `data/raw/<commit>/` unchanged and are ignored by Git.
Commit the three curated files in `data/processed/` together with the preparation scripts, source manifest, and `uv.lock`. When the dataset changes, update the preparation code, regenerate the files, run the checks, and commit the code and data together rather than manually editing generated CSVs.

The preparation script writes:

- `data/processed/appearances.csv`: joined source records, including same-match outcomes. This is an intermediate table, not a safe model feature matrix.
- `data/processed/forecasts.csv`: target-match identifiers, descriptive metadata, historical features, and `target_rating`.
- `data/processed/summary.json`: row counts, source commit, and the explicit feature list.

The initial features are averages over the previous five positive-minute appearances. Unrated appearances still contribute their observed statistics. Missing historical ratings are omitted from the rating average and counted explicitly.

A target needs five prior appearances and nonmissing feature values. Model inputs must use the explicit `features` list in the summary, not every numeric column. Names and IDs are metadata.

## Modeling rules

Train one shared MLP across players. Split by target kickoff date: earlier matches for training, later matches for validation, and the latest matches for testing. Fit scaling and any other learned preprocessing on training data only. Compare against each player's recent average rating using MAE and RMSE.

Explicitly sort by `kickoff_time` before splitting; the forecast CSV is grouped by player, not globally chronological. Keep every appearance from the same match in the same split. Agree on fixed date cutoffs before comparing team members' models.

Prior match outcomes may become history for later predictions once that match has finished, including during a rolling evaluation. Do not use target-match goals, assists, minutes, ratings, final scores, or final season totals as forecast inputs. Avoid cumulative `playerstats.csv` for the initial pipeline.

Past minutes are valid features.

A separate conditional experiment could be involving actual target-match minutes, aka minutes played by a player, but this would involve using information about the match that is unavailable before kickoff.

## Source limitations

The public repository does not claim complete accuracy with upstream lineage from Sofascore. From testing 9/10 ratings matched Sofascore articles exactly and one differed slightly by 0.1. Treat ratings as a 'good-enough' approximation rather than absolute truth.

Missing target ratings are excluded, and never filled with zero. FPL assists can differ from conventional assists. 

Minutes exclude stoppage time; use `minutes_played`, not `finish_min - start_min`. 

Some technical fields are missing or unavailable. See the upstream [integration review](https://github.com/olbauday/FPL-Core-Insights/blob/e6b2df626bc62c0fedc97c59a8a6e34c09212f85/DATA_INTEGRATION_REVIEW.md) and [FPL minutes explanation](https://fantasy.premierleague.com/help/faqs).

Credit the upstream project when presenting results. This repository does not grant new rights to upstream data.
