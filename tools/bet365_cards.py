"""Ingest hand-transcribed bet365 'To Be Booked' odds into sportsbet_booking_odds.csv.

Unlike tools/ladbrokes_cards.py (which parses saved HTML and game-max-fills each squad), this takes
a small hand-typed CSV of what is visible on the bet365 card page and does a PLAIN PER-PLAYER UPSERT:
players present get their real 'Yes' price, everyone else KEEPS their existing (synthetic) odds. Use
this when only the top-bookable names are visible (collapsed 'Show more' lists), where a squad fill
off a partial capture would be wrong. Booking provenance stays 'synthetic' for now regardless.

Input CSV columns: match, player_name, odds   (match is a label only; the pipeline never reads it).
bet365 lists names 'Given Surname', so resolution is easier than Ladbrokes; still routed through the
shared card name-map (inputs/bet365_name_mappings.csv) + accent/case match. Unresolved names are
reported and skipped. Odds are stored as decimals; the pipeline applies config.MARGIN_CARD downstream.

    python tools/bet365_cards.py --input inputs/bet365_cards_gw2.csv [--dry-run]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names, provenance  # noqa: E402
from tools.ladbrokes_cards import candidates  # reuse the word-order candidate generator

BOOKING_FILE = "sportsbet_booking_odds.csv"
BOOKING_CSV = os.path.join(config.SPORTSBET_DIR, BOOKING_FILE)
ROSTER_CSV = os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv")


def make_resolver(roster_names):
    norm_map = {}
    for n in roster_names:
        norm_map.setdefault(names._norm(n), []).append(n)

    def resolve(raw):
        for cand in candidates(str(raw).strip()):
            mapped = names.apply_bet365_names(pd.Series([cand])).iloc[0]
            hits = norm_map.get(names._norm(mapped))
            if hits and len(hits) == 1:
                return hits[0]
        return None
    return resolve


def main():
    ap = argparse.ArgumentParser(description="Ingest bet365 booking odds (per-player upsert).")
    ap.add_argument("--input", required=True, help="CSV: match, player_name, odds")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    src = pd.read_csv(args.input)
    ros = pd.read_csv(ROSTER_CSV)
    resolve = make_resolver(list(ros["name"]))
    team = ros.set_index("name")["team"]

    rows, unmatched = {}, []
    for _, r in src.iterrows():
        hit = resolve(r["player_name"])
        if hit:
            rows[hit] = float(r["odds"])            # last price wins if a name repeats
        else:
            unmatched.append(str(r["player_name"]).strip())

    resolved = pd.DataFrame({"player_name": list(rows), "odds_decimal": list(rows.values())})
    resolved["team"] = resolved["player_name"].map(team)
    print(f"resolved {len(resolved)}/{len(src)} rows across "
          f"{resolved['team'].nunique()} teams:")
    for t, g in resolved.groupby("team"):
        print(f"  {t:<16} {len(g):>2} players  ({g['odds_decimal'].min():g}-{g['odds_decimal'].max():g})")
    if unmatched:
        print(f"\n  UNMATCHED ({len(sorted(set(unmatched)))}) - non-FPL depth or a spelling needing a map:")
        for u in sorted(set(unmatched)):
            print(f"    {u}")
        print("  Add real FPL players to inputs/bet365_name_mappings.csv and re-run.")

    new = pd.DataFrame({"match_name": "bet365 cards", "date": "",
                        "player_name": resolved["player_name"], "odds_decimal": resolved["odds_decimal"]})
    if args.dry_run:
        print(f"\n[dry-run] would upsert {len(new)} players into {os.path.basename(BOOKING_CSV)}.")
        return
    existing = pd.read_csv(BOOKING_CSV) if os.path.exists(BOOKING_CSV) else \
        pd.DataFrame(columns=["match_name", "date", "player_name", "odds_decimal"])
    kept = existing[~existing["player_name"].isin(new["player_name"])]
    out = pd.concat([kept, new], ignore_index=True)
    out.to_csv(BOOKING_CSV, index=False)
    # Kept SYNTHETIC by choice: coverage is partial (top-bookable names only), so bookings are not
    # yet a trusted full market — the manifest records the bet365 source but not "real" state.
    provenance.mark(BOOKING_FILE, "synthetic", "bet365 (partial)",
                    f"{len(new)} players priced over synthetic base")
    print(f"\nWrote {len(new)} players ({len(existing) - len(kept)} replaced); {len(out)} rows total. "
          f"Uncovered players keep their synthetic booking odds (market still flagged synthetic).")


if __name__ == "__main__":
    main()
