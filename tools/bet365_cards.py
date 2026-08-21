"""Ingest hand-collected Bet365 "player to be booked" odds into sportsbet_booking_odds.csv.

    python tools/bet365_cards.py --match "Arsenal v Coventry" --input odds_raw/cards.txt
    python tools/bet365_cards.py --match "Arsenal v Coventry"        # then paste, Ctrl-Z (Win) / Ctrl-D
    python tools/bet365_cards.py --match "Arsenal v Coventry" --date "2026-08-21T21:00" --dry-run

Paste either the Bet365 HTML block (the market card copied from the page) or plain
'Player Name <tab/spaces> 2.75' lines. Names are resolved to their canonical roster spelling —
exact match, then inputs/bet365_name_mappings.csv, then an accent/case-insensitive match — and
only players that land on a real FPL roster name are written. Anything unresolved is printed for
you to eyeball: if one SHOULD be an FPL player, add a row to inputs/bet365_name_mappings.csv and
re-run.

The booking market is keyed by PLAYER (the pipeline never looks at match_name/date), so this
upserts by player: any existing row for a collected player is replaced with the real Bet365
price, and every other player's row — real or the season-start synthetic placeholder — is left
untouched. Collect a fixture at a time and the synthetic cards give way to real ones one match
at a time.
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names  # noqa: E402

BOOKING_CSV = os.path.join(config.SPORTSBET_DIR, "sportsbet_booking_odds.csv")
ROSTER_CSV = os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv")

# A span of visible text immediately followed by a span holding a decimal price — the Bet365
# row shape. Class-agnostic on purpose: Bet365 rotates its obfuscated class names, but the
# name-span-then-odds-span structure is stable.
HTML_PAIR = re.compile(r"<span[^>]*>([^<>]+?)</span>\s*<span[^>]*>([0-9]+(?:\.[0-9]+)?)</span>")
# Plain-text fallback: 'Player Name 2.75' — the price is the final whitespace-separated token,
# so a single space works and internal spaces in the name are kept (greedy name capture).
TEXT_PAIR = re.compile(r"^(.*\S)\s+([0-9]+(?:\.[0-9]+)?)\s*$")


def extract_pairs(text):
    """[(name, odds_float)] from a Bet365 HTML block or plain 'name  odds' lines."""
    pairs = [(n.strip(), float(o)) for n, o in HTML_PAIR.findall(text)]
    if not pairs:
        for line in text.splitlines():
            m = TEXT_PAIR.match(line)
            if m:
                pairs.append((m.group(1).strip(), float(m.group(2))))
    # a name must contain a letter; a price is plausible odds; drop exact duplicates
    seen, out = set(), []
    for name, odds in pairs:
        if re.search(r"[A-Za-z]", name) and 1.01 <= odds <= 1001 and name not in seen:
            seen.add(name)
            out.append((name, odds))
    return out


def resolve(pairs, roster_names):
    """Map each raw name to its canonical roster spelling; split into matched / unmatched."""
    raw = pd.Series([n for n, _ in pairs])
    canon = names.resolve_to_roster(names.apply_bet365_names(raw), roster_names)
    roster_set = set(roster_names)
    matched, unmatched = [], []
    for (name, odds), fixed in zip(pairs, canon):
        (matched if fixed in roster_set else unmatched).append((name, fixed, odds))
    return matched, unmatched


def main():
    ap = argparse.ArgumentParser(description="Ingest Bet365 booking odds into the pipeline CSV.")
    ap.add_argument("--match", required=True, help='e.g. "Arsenal v Coventry" (cosmetic; lookup is by player)')
    ap.add_argument("--date", default="", help="kickoff label, stored as-is (the pipeline ignores it)")
    ap.add_argument("--input", help="file to read; omit to read the paste from stdin")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    text = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    pairs = extract_pairs(text)
    if not pairs:
        sys.exit("No 'name odds' pairs found — paste the Bet365 market block or 'Name  2.75' lines.")

    roster_names = list(pd.read_csv(ROSTER_CSV)["name"])
    matched, unmatched = resolve(pairs, roster_names)

    print(f"{args.match}: {len(pairs)} selections read, {len(matched)} matched to roster.")
    if unmatched:
        print(f"\n  UNMATCHED ({len(unmatched)}) - non-FPL depth, or a spelling that needs a mapping:")
        for raw, fixed, odds in unmatched:
            note = "" if raw == fixed else f'  (tried -> "{fixed}")'
            print(f"    {raw:<28} {odds}{note}")
        print("  If any SHOULD be an FPL player, add a row to inputs/bet365_name_mappings.csv and re-run.")

    new_rows = pd.DataFrame({
        "match_name": args.match, "date": args.date,
        "player_name": [fixed for _, fixed, _ in matched],
        "odds_decimal": [odds for _, _, odds in matched],
    })
    if args.dry_run:
        print(f"\n[dry-run] would upsert {len(new_rows)} players into {os.path.basename(BOOKING_CSV)}.")
        return

    existing = pd.read_csv(BOOKING_CSV) if os.path.exists(BOOKING_CSV) else \
        pd.DataFrame(columns=["match_name", "date", "player_name", "odds_decimal"])
    kept = existing[~existing["player_name"].isin(new_rows["player_name"])]
    replaced = len(existing) - len(kept)
    out = pd.concat([kept, new_rows], ignore_index=True)
    out.to_csv(BOOKING_CSV, index=False)
    print(f"\nWrote {len(new_rows)} real Bet365 rows ({replaced} placeholder/older rows replaced); "
          f"{len(out)} rows total in {os.path.basename(BOOKING_CSV)}.")


if __name__ == "__main__":
    main()
