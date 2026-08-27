"""Build an external DefCon prior from raw defensive-action component CSVs.

Some players carry no Premier League history — promoted-club players (Championship) and
foreign signings (Ligue 1, Bundesliga, ...). Their prior-season DefCon is therefore absent
and they fall to the population average until they bank current PL minutes. This tool turns a
season's raw defensive components (from a stats provider) into a DefCon-per-90 PROXY, matched
to the FPL roster by name, and writes inputs/external_dc_prior.csv. ingest.load_defensive_
contributions merges that file as a prior for players with no FPL prior, so a good new player
starts from real evidence instead of the average.

DefCon proxy (FPL's own definition, taken at FACE VALUE — no league/level adjustment):
    DEF      : Clearances + Blocked Shots + Interceptions + Tackles         (CBIT)
    MID / FWD: CBIT + Ball Recoveries                                        (CBIRT)
    dc90     = actions / Games        (per-GAME; provider gives no minutes, so a rotation
                                       player reads a little low — regulars are fine)

Recoveries: sources vary. The Championship export HAS Ball Recoveries; the Ligue 1 one does
NOT. When recoveries are absent, a MID/FWD's proxy would be badly understated (recoveries are
~half a defensive-mid's count), so we fill the recoveries-per-game from the position median of
the sources that DO carry it (written to inputs/_dc_prior_recovery_rates.csv), and flag it.

Usage:  python -m tools.build_dc_prior <components.csv> --source championship_2025_26
        python -m tools.build_dc_prior <ligue1.csv>      --source ligue1_2025_26 --teams Lens,...
Each run UPSERTS its --source rows into inputs/external_dc_prior.csv (idempotent per source).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, ingest  # noqa: E402

PRIOR_CSV = os.path.join(config.INPUTS_DIR, "external_dc_prior.csv")
RECOVERY_RATES_CSV = os.path.join(config.INPUTS_DIR, "_dc_prior_recovery_rates.csv")

# provider column name -> canonical. Two accepted input shapes:
#   RAW components  : Tackles Made / Interceptions / Clearances / Blocked Shots / Ball Recoveries + Games
#   PRE-AGGREGATED  : a per-90 DefCon value (dc90 / Average) + nineties   (already averaged elsewhere)
COLMAP = {
    "Interceptions": "Int", "Clearances": "Clr", "Blocked Shots": "Blocks",
    "Tackles Made": "Tkl", "Ball Recoveries": "Rec", "Games": "Games",
    "dc90": "dc90", "Average": "dc90", "nineties": "nineties", "90s": "nineties",
    "Player": "Player", "Team": "Team",
}


def _ascii(s):
    return (s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
            .str.lower().str.strip())


def _match_to_roster(df, roster):
    """Match provider names to FPL roster names, WITHIN club where the team is known."""
    roster = roster.copy()
    roster["_a"] = _ascii(roster["name"])
    roster["_surname"] = roster["_a"].str.split().str[-1]
    df = df.copy()
    df["_a"] = _ascii(df["Player"])
    df["_surname"] = df["_a"].str.split().str[-1]

    out_name, out_pos = [], []
    for _, r in df.iterrows():
        pool = roster[roster["team"] == r["Team"]] if r["Team"] in set(roster["team"]) else roster
        hit = pool[pool["_a"] == r["_a"]]                                   # exact (ascii)
        if hit.empty:
            hit = pool[pool["_surname"] == r["_surname"]]                   # surname within club
            if len(hit) > 1:                                               # disambiguate by first initial
                hit = hit[hit["_a"].str[0] == r["_a"][0]]
        if len(hit) == 1:
            out_name.append(hit.iloc[0]["name"]); out_pos.append(hit.iloc[0]["position"])
        else:
            out_name.append(None); out_pos.append(None)
    df["fpl_name"] = out_name
    df["position"] = out_pos
    return df


def _recovery_rates():
    if os.path.exists(RECOVERY_RATES_CSV):
        return pd.read_csv(RECOVERY_RATES_CSV).set_index("position")["rec_per_game"].to_dict()
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--source", required=True, help="tag stored in the source column (upsert key)")
    ap.add_argument("--teams", default=None, help="optional comma-list to restrict rows")
    args = ap.parse_args()

    try:
        raw = pd.read_csv(args.csv, encoding="utf-8-sig")   # hand-made / UTF-8 exports
    except UnicodeDecodeError:
        raw = pd.read_csv(args.csv, encoding="latin-1")      # provider exports (accented Latin-1)
    raw = raw[raw["Player"] != "Player"]
    df = raw.rename(columns={k: v for k, v in COLMAP.items() if k in raw.columns})
    if args.teams:
        df = df[df["Team"].isin([t.strip() for t in args.teams.split(",")])]
    preagg = "dc90" in df.columns and "nineties" in df.columns
    numeric = ("dc90", "nineties") if preagg else ("Int", "Clr", "Blocks", "Tkl", "Games")
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _match_to_roster(df, ingest.load_fpl_players())
    matched = df[df["fpl_name"].notna()].copy()

    rec_est = False
    if preagg:                                                              # dc90 supplied directly
        matched["dc90"] = matched["dc90"].round(2)
        matched["minutes"] = (matched["nineties"] * 90).round().astype(int)
    else:                                                                   # compute from raw components
        cbit = matched["Clr"] + matched["Blocks"] + matched["Int"] + matched["Tkl"]
        if "Rec" in matched.columns:
            rec = pd.to_numeric(matched["Rec"], errors="coerce").fillna(0.0)
            rates = matched.assign(_r=rec / matched["Games"]).groupby("position")["_r"].median()
            rates.rename("rec_per_game").to_csv(RECOVERY_RATES_CSV, header=True)  # learn for rec-less sources
        else:
            rec = matched["position"].map(_recovery_rates()).fillna(0.0) * matched["Games"]  # estimated
            rec_est = True
        rec = rec.where(matched["position"].isin(["MID", "FWD"]), 0.0)      # recoveries: MID/FWD only
        matched["dc90"] = ((cbit + rec) / matched["Games"]).round(2)
        matched["minutes"] = (matched["Games"] * 90).astype(int)            # face-value per-game->per-90
    matched["source"] = args.source
    matched["rec_estimated"] = rec_est
    out = matched[["fpl_name", "Team", "position", "dc90", "minutes", "source", "rec_estimated"]]
    out = out.rename(columns={"fpl_name": "name", "Team": "team"})

    if os.path.exists(PRIOR_CSV):
        prev = pd.read_csv(PRIOR_CSV)
        prev = prev[prev["source"] != args.source]                          # upsert this source
        out = pd.concat([prev, out], ignore_index=True)
    out = out.sort_values(["source", "team", "dc90"], ascending=[True, True, False])
    out.to_csv(PRIOR_CSV, index=False)

    unmatched = df[df["fpl_name"].isna()]
    print(f"[{args.source}] matched {len(matched)}/{len(df)} rows"
          + (f"  (recoveries ESTIMATED from {RECOVERY_RATES_CSV})" if rec_est else ""))
    print(f"  wrote {len(out)} total prior rows -> {PRIOR_CSV}")
    print(f"  unmatched (not in FPL roster / name miss): {list(unmatched['Player'].head(20))}")


if __name__ == "__main__":
    main()
