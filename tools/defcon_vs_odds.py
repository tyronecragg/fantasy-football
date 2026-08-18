"""Does a player's defensive contribution depend on how hard the fixture is?

    python tools/defcon_vs_odds.py                    # 2025-26, per-match data
    python tools/defcon_vs_odds.py --min-matches 15

`dc90` is currently a FLAT per-90 rate: the same value whether a defender faces Liverpool
away or a promoted side at home. But defensive contributions are *defending* actions
(tackles, interceptions, blocks, clearances, recoveries), so a side camped in its own half
might rack up more of them. If so, the pipeline under-rates defenders and midfielders in
hard fixtures and over-rates them in easy ones — exactly backwards for squad selection.

DATA: one row per player per MATCH, from
`By Gameweek/GW*/playermatchstats.csv` (`defensive_contributions`, `minutes_played`) joined
to `matches.csv` for the fixture and `lineups.csv` for the player's team.

An earlier version of this tool differenced the cumulative season-to-date `playerstats.csv`
instead, which was wrong in two ways: a double gameweek collapses two matches into one row
that then gets a single fixture's win probability, and a postponed match lands in whichever
gameweek window it was played in rather than the one it belongs to. Both blur the fixture
signal toward zero, which is precisely the false null this test is trying to avoid. Keyed on
`match_id`, neither can happen.

METHOD: everything is measured WITHIN a player, because between-player variation is enormous
(a holding midfielder out-tackles a winger regardless of fixture) and would swamp the effect.

    ratio = (this match's DC per 90) / (that player's own season DC per 90)

1.0 is his normal game, 1.3 is 30% above his own norm. That ratio is then regressed on the
team's de-margined win probability (Pinnacle closing prices, football-data.co.uk) and — as a
direct test of the mechanism — on the team's possession share, which `matches.csv` records.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names  # noqa: E402

MIN_MINUTES = 60          # a per-90 from a short cameo is mostly noise


def season_dir(season):
    """Sibling of the configured season directory (config points at the CURRENT season)."""
    return os.path.join(os.path.dirname(config.FPL_DATA_DIR.rstrip("\\/")), season)


def per_match(season):
    """One row per player per match: DC, minutes, team, opponent, venue, possession."""
    root = os.path.join(season_dir(season), "By Gameweek")
    teams = pd.read_csv(os.path.join(season_dir(season), "teams.csv"))
    code_name = dict(zip(teams["code"], teams["name"]))
    players = pd.read_csv(os.path.join(season_dir(season), "players.csv"))
    pos_of = dict(zip(players["player_id"], players["position"]))

    frames = []
    for gw_dir in sorted(glob.glob(os.path.join(root, "GW*")),
                         key=lambda p: int(re.search(r"GW(\d+)", p).group(1))):
        try:
            stats = pd.read_csv(os.path.join(gw_dir, "playermatchstats.csv"))
            matches = pd.read_csv(os.path.join(gw_dir, "matches.csv"))
            lineups = pd.read_csv(os.path.join(gw_dir, "lineups.csv"))
        except FileNotFoundError:
            continue
        stats = stats.merge(lineups[["match_id", "player_id", "team_code", "team_side"]],
                            on=["match_id", "player_id"], how="left")
        keep = ["match_id", "gameweek", "home_team", "away_team", "kickoff_time"]
        if "home_possession" in matches:
            keep.append("home_possession")
        frames.append(stats.merge(matches[keep], on="match_id", how="left"))

    df = pd.concat(frames, ignore_index=True)
    df["team_name"] = df["team_code"].map(code_name)
    df["position"] = df["player_id"].map(pos_of).map(config.POSITION_MAP)
    df["home"] = (df["team_side"] == "home").astype(int)
    # matches.csv stores home_team/away_team as numeric TEAM CODES, not names — running
    # them through the name mapper directly just returns the numbers.
    for side in ("home_team", "away_team"):
        df[side] = names.apply_team_names(
            pd.to_numeric(df[side], errors="coerce").map(code_name).fillna(""))
    df["team_name"] = names.apply_team_names(df["team_name"].fillna(""))
    df["opponent"] = np.where(df["home"] == 1, df["away_team"], df["home_team"])
    if "home_possession" in df:
        poss = pd.to_numeric(df["home_possession"], errors="coerce")
        df["possession"] = np.where(df["home"] == 1, poss, 100 - poss)
    df["date"] = pd.to_datetime(df["kickoff_time"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return df[df["minutes_played"] >= MIN_MINUTES].copy()


def match_odds(season):
    """De-margined 1X2 per LEAGUE FIXTURE, keyed by the (home, away) pair.

    The fixture pair is the right join key, not the date. In a double round-robin each
    ordered pair occurs EXACTLY ONCE per season, so (home_team, away_team) is unique —
    while dates drift between football-data's match date and the feed's kickoff timestamp,
    which silently dropped 170 of 380 matches when we joined on them.

    It also filters cup ties for free: a League Cup tie between two league sides is the one
    case where a pair could collide, and that is resolved by date proximity at the join.
    """
    tag = season[2:4] + season[-2:]
    e = pd.read_csv(os.path.join(config.ROOT, "fpl_data", "football_data", f"E0_{tag}.csv"))
    h, d, a = [("PSC" + s if f"PSC{s}" in e else "B365" + s) for s in ("H", "D", "A")]
    inv = 1 / e[[h, d, a]].astype(float)
    over = inv.sum(axis=1)
    e["p_home"], e["p_away"] = inv[h] / over, inv[a] / over
    e["odds_date"] = pd.to_datetime(e["Date"], dayfirst=True, errors="coerce")
    for side in ("HomeTeam", "AwayTeam"):
        e[side] = names.apply_team_names(e[side])
    return e.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})[
        ["home_team", "away_team", "odds_date", "p_home", "p_away"]]


def join_odds(df, odds):
    """Attach each player-match's own win probability via the fixture pair."""
    merged = df.merge(odds, on=["home_team", "away_team"], how="inner")
    # A cup tie can repeat a league pairing; keep whichever match sits nearest the
    # bookmaker's date for that fixture.
    if merged["match_id"].duplicated().any():
        merged["gap"] = (merged["date"] - merged["odds_date"]).abs()
        merged = (merged.sort_values("gap")
                        .drop_duplicates(subset=["match_id", "player_id"], keep="first"))
    merged["win_prob"] = np.where(merged["home"] == 1, merged["p_home"], merged["p_away"])
    return merged


def fit(x, y):
    """Closed-form OLS slope with a t-stat (the SVD path trips over this data under MKL)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 50 or x.std() == 0:
        return None
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    intercept = y.mean() - slope * x.mean()
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    return slope, intercept, slope / se, np.corrcoef(x, y)[0, 1], len(x)


def analyse(season="2025-2026", min_matches=10):
    raw = per_match(season)
    df = join_odds(raw, match_odds(season))
    print(f"joined {df['match_id'].nunique()} of {raw['match_id'].nunique()} feed matches "
          f"({raw['match_id'].nunique() - df['match_id'].nunique()} dropped: cup ties and "
          f"anything without league odds)")

    df["dc_per90"] = df["defensive_contributions"] / df["minutes_played"] * 90
    df = df[np.isfinite(df["dc_per90"])]
    own = df.groupby("player_id")["dc_per90"].transform("mean")
    n_obs = df.groupby("player_id")["dc_per90"].transform("size")
    df = df[(n_obs >= min_matches) & (own > 0)].copy()
    df["ratio"] = df["dc_per90"] / df.groupby("player_id")["dc_per90"].transform("mean")
    # Deviations from each team's own norm. A player's season dc90 already embeds his team's
    # TYPICAL possession and TYPICAL fixture difficulty, so the absolute levels mix the
    # fixture effect with between-team differences that have already been normalised away.
    df["wp_dev"] = df["win_prob"] - df.groupby("team_name")["win_prob"].transform("mean")
    if "possession" in df:
        df["poss_dev"] = df["possession"] - df.groupby("team_name")["possession"].transform("mean")

    print(f"{len(df):,} player-MATCHES | {df['player_id'].nunique()} players with "
          f">={min_matches} matches of >={MIN_MINUTES} min | {df['match_id'].nunique()} matches")
    print(f"win probability {df.win_prob.min():.2f}-{df.win_prob.max():.2f}"
          + (f" | possession {df.possession.min():.0f}-{df.possession.max():.0f}%"
             if "possession" in df else "") + "\n")

    for pos in ("DEF", "MID", "FWD"):
        sub = df[df["position"] == pos]
        r = fit(sub["win_prob"], sub["ratio"])
        if not r:
            continue
        slope, intercept, t, corr, n = r
        print(f"{pos}: n={n:>5}  vs WIN PROB   slope={slope:+.3f}  r={corr:+.3f}  t={t:+.1f}"
              f"   [20%->{slope*.2+intercept:.2f}x  80%->{slope*.8+intercept:.2f}x]")
        rw = fit(sub["wp_dev"], sub["ratio"])
        if rw:
            s4, i4, t4, c4, n4 = rw
            print(f"      {'':>5}  vs WIN PROB DEVIATION from team norm  slope={s4:+.3f}"
                  f"  r={c4:+.3f}  t={t4:+.1f}   [-0.20->{s4*-0.2+i4:.2f}x  "
                  f"+0.20->{s4*0.2+i4:.2f}x]")
        if "possession" in sub:
            rp = fit(sub["possession"], sub["ratio"])
            if rp:
                s2, i2, t2, c2, n2 = rp
                print(f"      {'':>5}  vs POSSESSION slope={s2:+.4f}/pt  r={c2:+.3f}  t={t2:+.1f}"
                      f"   [35%->{s2*35+i2:.2f}x  65%->{s2*65+i2:.2f}x]")
            # The specification that actually matches how dc90 is consumed. A player's
            # season dc90 already embeds his team's TYPICAL possession, so regressing on the
            # absolute level mixes the fixture effect with between-team style that has
            # already been normalised away. The deviation isolates the part we could act on:
            # when this team has more of the ball THAN USUAL, do its players contribute less?
            rd = fit(sub["poss_dev"], sub["ratio"])
            if rd:
                s3, i3, t3, c3, n3 = rd
                print(f"      {'':>5}  vs POSS DEVIATION from team norm  slope={s3:+.4f}/pt"
                      f"  r={c3:+.3f}  t={t3:+.1f}   [-15pts->{s3*-15+i3:.2f}x  "
                      f"+15pts->{s3*15+i3:.2f}x]")
        band = pd.cut(sub["win_prob"], [0, .25, .40, .55, .70, 1.0])
        obs = sub.groupby(band, observed=True)["ratio"].agg(["mean", "size"])
        print("      by win-prob band: " + "  ".join(
            f"{str(i).split(',')[1].strip(' ]')}={r['mean']:.2f}(n={int(r['size'])})"
            for i, r in obs.iterrows()) + "\n")

    out = os.path.join(config.OUTPUTS_DIR, "defcon_vs_odds.csv")
    cols = ["player_id", "position", "team_name", "opponent", "gameweek", "home",
            "win_prob", "minutes_played", "defensive_contributions", "dc_per90", "ratio"]
    df[[c for c in cols + ["possession"] if c in df]].to_csv(out, index=False)
    print(f"-> {os.path.relpath(out, config.ROOT)}")
    return df


def forecast_test(season="2025-2026", split=19, min_train=5):
    """Can we PREDICT the possession deviation, and does it improve DC forecasts?

    The effect being real is not the same as it being usable. Possession is only known after
    the match, so an adjustment is worth building only if a fixture's possession is
    predictable in advance from the two teams' styles.

    Strictly out of sample: everything — team possession baselines, the DC-vs-possession
    slope, each player's own dc90 — is fitted on gameweeks 1..split and evaluated on the
    rest. Anything fitted on the test half would flatter the adjusted model for free.

    Possession model, deliberately the simplest thing that could work:

        predicted home share = 50 + (home attack-the-ball baseline - away's)/2 + home edge

    where a team's baseline is its mean possession in training, and the home edge is the
    league-average home possession advantage. Then DC is predicted two ways — the player's
    own training dc90 flat, versus that same rate scaled by the predicted deviation — and
    compared on mean absolute error against what actually happened.
    """
    df = join_odds(per_match(season), match_odds(season))
    df["dc_per90"] = df["defensive_contributions"] / df["minutes_played"] * 90
    df = df[np.isfinite(df["dc_per90"]) & df["possession"].notna()].copy()
    train, test = df[df["gameweek"] <= split], df[df["gameweek"] > split]

    # --- team possession baselines and the home edge, from TRAIN only ---
    base = train.groupby("team_name")["possession"].mean()
    home_edge = (train.loc[train["home"] == 1, "possession"].mean()
                 - train.loc[train["home"] == 0, "possession"].mean()) / 2
    league = base.mean()

    def predict_possession(row):
        own, opp = base.get(row["team_name"], np.nan), base.get(row["opponent"], np.nan)
        if not np.isfinite(own) or not np.isfinite(opp):
            return np.nan
        edge = home_edge if row["home"] == 1 else -home_edge
        return league + (own - opp) / 2 + edge

    fixtures = test.drop_duplicates(subset=["match_id", "team_name"]).copy()
    fixtures["pred_poss"] = fixtures.apply(predict_possession, axis=1)
    ok = fixtures["pred_poss"].notna()
    r = np.corrcoef(fixtures.loc[ok, "pred_poss"], fixtures.loc[ok, "possession"])[0, 1]
    mae = (fixtures.loc[ok, "pred_poss"] - fixtures.loc[ok, "possession"]).abs().mean()
    naive = (league - fixtures.loc[ok, "possession"]).abs().mean()
    print(f"POSSESSION FORECAST (train GW1-{split}, test GW{split+1}+, "
          f"{int(ok.sum())} team-matches)")
    print(f"  r={r:+.3f}  MAE={mae:.1f} pts  vs {naive:.1f} assuming everyone gets 50% "
          f"({(naive-mae)/naive:+.0%})\n")

    # --- does that forecast improve DC prediction? ---
    train_dc = train.groupby("player_id")["dc_per90"].agg(["mean", "size"])
    train_dc = train_dc[train_dc["size"] >= min_train]["mean"]
    tbase = train.groupby("team_name")["possession"].transform("mean")
    tr = train.assign(poss_dev=train["possession"] - tbase,
                      own=train["player_id"].map(train_dc))
    tr = tr[tr["own"].notna() & (tr["own"] > 0)]
    tr["ratio"] = tr["dc_per90"] / tr["own"]
    slope_fit = fit(tr["poss_dev"], tr["ratio"])
    if not slope_fit:
        return
    slope, intercept = slope_fit[0], slope_fit[1]
    print(f"  slope fitted on TRAIN only: {slope:+.4f}/pt (t={slope_fit[2]:+.1f})\n")

    ev = test.copy()
    ev["own"] = ev["player_id"].map(train_dc)
    ev["pred_poss"] = ev.apply(predict_possession, axis=1)
    ev = ev[ev["own"].notna() & ev["pred_poss"].notna() & (ev["own"] > 0)]
    ev["pred_dev"] = ev["pred_poss"] - ev["team_name"].map(base)
    ev["flat"] = ev["own"]
    ev["adjusted"] = ev["own"] * (intercept + slope * ev["pred_dev"])
    # ceiling: what the adjustment would score with PERFECT possession knowledge
    ev["oracle"] = ev["own"] * (intercept + slope * (ev["possession"] - ev["team_name"].map(base)))

    print(f"DC FORECAST on {len(ev):,} held-out player-matches (mean actual "
          f"{ev['dc_per90'].mean():.2f} per 90):")
    flat_mae = (ev["flat"] - ev["dc_per90"]).abs().mean()
    for label, col in (("flat player average", "flat"),
                       ("possession-adjusted", "adjusted"),
                       ("oracle (true possession)", "oracle")):
        m = (ev[col] - ev["dc_per90"]).abs().mean()
        print(f"  {label:<26} MAE={m:.3f}   {(flat_mae-m)/flat_mae:+.2%} vs flat")
    for pos in ("DEF", "MID"):
        sub = ev[ev["position"] == pos]
        if len(sub) < 100:
            continue
        f_mae = (sub["flat"] - sub["dc_per90"]).abs().mean()
        a_mae = (sub["adjusted"] - sub["dc_per90"]).abs().mean()
        print(f"    {pos}: n={len(sub):<5} flat={f_mae:.3f}  adjusted={a_mae:.3f}  "
              f"{(f_mae-a_mae)/f_mae:+.2%}")
    print()


def validate(season="2025-2026"):
    """Prove the method can see fixture effects, so a DefCon null means something.

    A null is worthless if the machinery is broken — a bad odds join produces exactly the
    flat line we are testing for. So run the identical pipeline over stats that MUST depend
    on the fixture: goals conceded should fall as a team's win probability rises.
    """
    df = join_odds(per_match(season), match_odds(season))
    print("VALIDATION — effects that must appear if the odds join is sound:")
    for label, col in (("team goals conceded", "team_goals_conceded"),
                       ("goals scored", "goals_scored"), ("saves", "saves")):
        if col not in df:
            continue
        r = fit(df["win_prob"], pd.to_numeric(df[col], errors="coerce"))
        if r:
            slope, intercept, t, corr, n = r
            print(f"  {label:<22} slope={slope:+6.2f}  t={t:+6.1f}  n={n:<6}"
                  f" [10%->{slope*.1+intercept:.2f}  80%->{slope*.8+intercept:.2f}]")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--forecast", action="store_true",
                    help="out-of-sample test: can the possession deviation be predicted, "
                         "and does it improve DC forecasts?")
    args = ap.parse_args()
    if args.forecast:
        forecast_test(args.season)
    else:
        if not args.no_validate:
            validate(args.season)
        analyse(args.season, args.min_matches)
