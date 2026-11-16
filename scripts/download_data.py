"""Download the pinned source CSVs without changing their contents."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / "data/source.json").read_text())
    repo = source["repository"].removeprefix("https://github.com/")
    base = f"https://raw.githubusercontent.com/{repo}/{source['commit']}/"
    season = source["season"]
    prefix = f"data/{season}/By Tournament/{source['competition']}"
    paths = [f"data/{season}/players.csv"]
    paths += [
        f"{prefix}/GW{gw}/{name}.csv"
        for gw in range(1, source["gameweeks"] + 1)
        for name in ("matches", "playermatchstats", "player_match_enrichment")
    ]

    def download(path):
        destination = ROOT / "data/raw" / source["commit"] / path
        if destination.exists():
            return
        with urlopen(base + quote(path), timeout=60) as response:
            content = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, paths))
    print(f"Ready: {len(paths)} source files at data/raw/{source['commit']}/")


if __name__ == "__main__":
    main()
