"""Build GW12 'Wonderkids' inputs: every player's birth date from FPL's own API (bootstrap-static
carries birth_date). FPL element ids equal the master player_id this season.

    python build_ages.py       -> inputs/fpl_challenge_birthdates.csv (player_id, web_name, birth_date)
"""
import json
import os
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "inputs", "fpl_challenge_birthdates.csv")
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    rows = [{"player_id": e["id"], "web_name": e["web_name"], "birth_date": e.get("birth_date")}
            for e in data["elements"]]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} players, {out['birth_date'].notna().sum()} with a birth date")


if __name__ == "__main__":
    sys.exit(main())
