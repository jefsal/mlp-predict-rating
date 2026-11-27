"Data split, Train, Validation, test"

from pathlib import Path
import pandas as pd

TRAIN_END = pd.Timestamp("2026-03-03", tz="UTC")
VAL_END = pd.Timestamp("2026-04-24", tz="UTC")

FORECASTS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "forecasts.csv"

def load_forecasts(path=FORECASTS_PATH):

    df = pd.read_csv(path)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)

    return df.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)


def split_by_date(df):

    kickoff_day = df["kickoff_time"].dt.normalize()
    train = df[kickoff_day <= TRAIN_END]
    validation = df[(kickoff_day > TRAIN_END) & (kickoff_day <= VAL_END)]
    test = df[kickoff_day > VAL_END]
    return train, validation, test


if __name__ == "__main__":
    data = load_forecasts()
    print(data.shape)  # expect (8872, 20)
    print(data[["match_id", "kickoff_time", "target_rating"]].head())

    train, validation, test = split_by_date(data)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        print(f"{name:10s} {len(part):5d} rows | "
              f"{part['kickoff_time'].min().date()} -> {part['kickoff_time'].max().date()}")