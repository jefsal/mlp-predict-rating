"""Join appearances and build features strictly from earlier appearances."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["player_id", "match_id"]
WINDOW = 5
HISTORY_STATS = [
    "minutes_played", "goals", "assists", "total_shots", "xg", "xa",
    "accurate_passes", "tackles_won", "interceptions", "saves", "rating",
]
FEATURES = [f"past_{WINDOW}_{column}_mean" for column in HISTORY_STATS]


def build_forecasts(appearances):
    """One target per rated positive-minute appearance after five prior appearances.

    Unrated appearances still contribute their recorded statistics to history.
    Prior ratings may be missing; retain their count so this is explicit.
    Actual target-match minutes and outcomes are never exported as features.
    """
    data = appearances.loc[appearances["minutes_played"] > 0].copy()
    data = data.sort_values(["player_id", "kickoff_time", "match_id"])
    grouped = data.groupby("player_id", sort=False)
    data["prior_appearances"] = grouped.cumcount()
    for column in HISTORY_STATS:
        data[f"past_{WINDOW}_{column}_mean"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(WINDOW, min_periods=1).mean()
        )
    data[f"past_{WINDOW}_rating_count"] = grouped["rating"].transform(
        lambda values: values.shift(1).rolling(WINDOW, min_periods=1).count()
    )
    eligible = (
        data["rating"].notna()
        & (data["prior_appearances"] >= WINDOW)
        & data[FEATURES].notna().all(axis=1)
    )
    columns = [
        *KEY, "kickoff_time", "first_name", "second_name", "position",
        "prior_appearances", *FEATURES, f"past_{WINDOW}_rating_count", "rating",
    ]
    return data.loc[eligible, columns].rename(columns={"rating": "target_rating"})


def main():
    source = json.loads((ROOT / "data/source.json").read_text())
    raw = ROOT / "data/raw" / source["commit"] / "data" / source["season"]
    league = raw / "By Tournament" / source["competition"]

    def read_gameweeks(name):
        return pd.concat(
            [pd.read_csv(league / f"GW{gw}" / f"{name}.csv")
             for gw in range(1, source["gameweeks"] + 1)],
            ignore_index=True,
        )

    stats = read_gameweeks("playermatchstats")
    ratings = read_gameweeks("player_match_enrichment")
    matches = read_gameweeks("matches")
    players = pd.read_csv(raw / "players.csv")
    appearances = stats.merge(
        ratings[[*KEY, "rating"]], on=KEY, how="left", validate="one_to_one"
    ).merge(
        matches[["match_id", "kickoff_time"]], on="match_id", how="left",
        validate="many_to_one",
    ).merge(
        players[["player_id", "first_name", "second_name", "position"]],
        on="player_id", how="left", validate="many_to_one",
    )
    appearances["kickoff_time"] = pd.to_datetime(appearances["kickoff_time"], utc=True)
    if appearances[["kickoff_time", "first_name", "position"]].isna().any().any():
        raise ValueError("A player-match row has missing match or player metadata.")
    for column in HISTORY_STATS:
        appearances[column] = pd.to_numeric(appearances[column], errors="raise")
    appearances = appearances.sort_values(["kickoff_time", "match_id", "player_id"])
    forecasts = build_forecasts(appearances)
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    appearances.to_csv(output / "appearances.csv", index=False)
    forecasts.to_csv(output / "forecasts.csv", index=False)
    summary = {
        "source_commit": source["commit"],
        "matches": int(matches["match_id"].nunique()),
        "appearance_rows_including_nonappearances": len(appearances),
        "positive_minute_rows": int((appearances["minutes_played"] > 0).sum()),
        "rated_positive_minute_rows": int(
            ((appearances["minutes_played"] > 0) & appearances["rating"].notna()).sum()
        ),
        "forecast_rows": len(forecasts),
        "history_window": WINDOW,
        "features": [*FEATURES, f"past_{WINDOW}_rating_count"],
        "target": "target_rating",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
