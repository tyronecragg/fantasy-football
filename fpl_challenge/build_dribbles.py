"""Rebuild GW9 'Trick or Treat' dribble inputs from FotMob pastes.

FotMob successful-dribbles pages:
  this season (2026/27, id 36781):  https://www.fotmob.com/leagues/47/stats/season/36781/players/won_contest
  last season (2025/26, id 27110):  https://www.fotmob.com/leagues/47/stats/season/27110/players/won_contest

Save each leaderboard as one player per line, Name;success_pct;per90 (e.g. "Sávio;52.9;4.9"):
  raw/fotmob_dribbles.txt       -> inputs/fpl_challenge_dribbles.csv        (this season)
  raw/fotmob_dribbles_last.txt  -> inputs/fpl_challenge_dribbles_last.csv   (last season, blended
                                   into GW9's rates at half weight)
then:

    python build_dribbles.py

Names are matched to the FPL master (accent/case-insensitive, then unique surname), with an alias
table for FotMob spellings that don't resolve. Unmatched this-season names are printed loudly;
unmatched last-season names are expected (players who have left the league) and just listed.
"""
import csv
import os
import sys
import unicodedata

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(HERE, "raw", "fotmob_dribbles.txt")
OUT = os.path.join(ROOT, "inputs", "fpl_challenge_dribbles.csv")
RAW_LAST = os.path.join(HERE, "raw", "fotmob_dribbles_last.txt")
OUT_LAST = os.path.join(ROOT, "inputs", "fpl_challenge_dribbles_last.csv")
MASTER = os.path.join(ROOT, "outputs", "players_master.csv")

# FotMob name -> FPL master name, for ones the matcher can't resolve on its own
ALIASES = {
    "Jaden Philogene-Bidace": "Jaden Philogene",
    "Joshua King": "Josh King",
    "Jair Cunha": "Jair Paula da Cunha Filho",
    "Gabriel": "Gabriel Magalhaes",
    "Chema Andrés": "José María Andrés Baixauli",
    "António Silva": "António João Pereira de Albuquerque Tavares da Silva",
    "Pape Sarr": "Pape Matar Sarr",
    "Joseph Gomez": "Joe Gomez",
    "Pascal Groß": "Pascal Gross",
}


def norm(s):
    s = str(s).replace("ı", "i").replace("ø", "o").replace("Ø", "O")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def build(raw_path, out_path, last_season=False):
    rows = [ln.strip().split(";") for ln in open(raw_path, encoding="utf-8") if ln.strip()]
    rows = [r for r in rows if len(r) == 3 and r[0] != "TOTAL"]
    m = pd.read_csv(MASTER)
    m.columns = m.columns.str.strip()
    m["_k"] = m["Player Name"].map(norm)
    by = dict(zip(m["_k"], m["Player Name"]))
    names = set(m["Player Name"])

    out, missed, fuzzy = {}, [], []
    for nm, pct, p90 in rows:
        canon = ALIASES.get(nm)
        if canon is None:
            canon = by.get(norm(nm))
        if canon is None:
            last = norm(nm).split()[-1]
            hit = m[m["_k"].str.split().apply(lambda ws: last in ws)]
            if len(hit) == 1:
                canon = hit.iloc[0]["Player Name"]
                fuzzy.append(f"{nm} -> {canon}")
        if canon is None or canon not in names:
            missed.append(nm)
            continue
        out.setdefault(canon, (float(p90), float(pct), nm))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Player", "dribbles_per90", "success_pct", "source_name"])
        for canon, (p90, pct, src) in out.items():
            w.writerow([canon, p90, pct, src])
    print(f"{len(rows)} FotMob rows -> {len(out)} matched, written to {out_path}")
    if fuzzy:
        print("surname matches (check):", "; ".join(fuzzy))
    if missed and last_season:
        print(f"not in this season's FPL (left the league - expected): {len(missed)}: {', '.join(missed)}")
    elif missed:
        bar = "!" * 74
        print(f"\n{bar}\n!!  UNMATCHED ({len(missed)}): {', '.join(missed)}\n"
              f"!!  add them to ALIASES in build_dribbles.py\n{bar}")


def main():
    build(RAW, OUT)
    if os.path.exists(RAW_LAST):
        print()
        build(RAW_LAST, OUT_LAST, last_season=True)


if __name__ == "__main__":
    sys.exit(main())
