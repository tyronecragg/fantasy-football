"""Pre-season bootstrap: build starting lineups and synthetic GW1/GW2 odds before
bookmakers price the matches and team news exists.

- inputs/starting_lineups.csv: best XI per team scored by last season's minutes (any
  club — players carry form to new clubs) blended with FPL launch price; formation
  chosen to maximise total score. Start probabilities tiered by how nailed the player
  looks (minutes-based; price-based for promoted teams).
- sportsbet/*.csv: model-derived odds in the exact scraper schemas. Win probabilities
  come from the season odds via the win-pred regression; player/team probabilities are
  last season's factors x the baselines; odds carry the standard margins so the
  pipeline's de-margining recovers the intended probabilities exactly.

Synthetic odds are placeholders: rerun sportsbet.py when real markets open (it
overwrites these files), and only run `--gw 1` archive recording on real odds.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, ingest, model, team_model  # noqa: E402
from fpl_pipeline.names import apply_player_names  # noqa: E402

FULL_SEASON_MINUTES = 34 * 90  # normalisation cap (nobody truly plays 38*90)
FORMATIONS = [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1)]
GW1_EPOCH, GW2_EPOCH = 1787425200, 1788030000  # 21 & 28 Aug 2026 (informational only)
PLAYER_MARGIN, WDW_MARGIN = 1.05, 1.03

# Measured from last season's REAL bookmaker odds (archive GW16-29, 4742 player-weeks):
# summing per-player assist lambdas gives ~1.13x the goal lambdas — the odds-market
# convention (longshot pricing, per-player framing) that the factors are calibrated on.
# Synthetic assist odds anchor to this so every team matches the real-market behaviour.
ASSISTS_PER_GOAL_ODDS = 1.132


def last_season_minutes():
    base = os.path.join(config.ROOT, "fpl_data", "FPL-Core-Insights", "data", "2025-2026")
    players = pd.read_csv(os.path.join(base, "players.csv"))
    stats = pd.read_csv(os.path.join(base, "playerstats.csv"))
    stats = stats.sort_values("gw").drop_duplicates(subset="id", keep="last")
    df = players.merge(stats[["id", "minutes"]], left_on="player_id", right_on="id", how="left")
    names = apply_player_names(df["first_name"] + " " + df["second_name"])
    return dict(zip(names, df["minutes"].fillna(0)))


def build_lineups(roster):
    minutes = last_season_minutes()
    r = roster.copy()

    # Injured/sold players (inputs/unavailable_players.csv) are removed before XI
    # selection so the algorithm re-picks the best available XI — this handles
    # cascading absences (e.g. three injured defenders at one club) that pairwise
    # lineup_overrides swaps cannot.
    unavailable_path = os.path.join(config.INPUTS_DIR, "unavailable_players.csv")
    if os.path.exists(unavailable_path):
        unavailable = pd.read_csv(unavailable_path)
        unknown = set(unavailable["Player"]) - set(r["name"])
        if unknown:
            print(f"  unavailable list: UNRECOGNISED names ignored: {sorted(unknown)}")
        n_before = len(r)
        r = r[~r["name"].isin(set(unavailable["Player"]))]
        print(f"  unavailable list: excluded {n_before - len(r)} players from XI selection")
    r["minutes"] = r["name"].map(minutes).fillna(0)
    r["min_score"] = (r["minutes"] / FULL_SEASON_MINUTES).clip(0, 1)
    r["price_score"] = r.groupby(["team", "position"], observed=True)["cost"].rank(pct=True)
    r["score"] = 0.65 * r["min_score"] + 0.35 * r["price_score"]

    rows = []
    for team, squad in r.groupby("team"):
        by_pos = {p: squad[squad["position"] == p].nlargest(20, "score") for p in ("GK", "DEF", "MID", "FWD")}
        best, best_total = None, -1
        for d, m, f in FORMATIONS:
            if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
                continue
            xi = pd.concat([by_pos["GK"].head(1), by_pos["DEF"].head(d),
                            by_pos["MID"].head(m), by_pos["FWD"].head(f)])
            if xi["score"].sum() > best_total:
                best, best_total = xi, xi["score"].sum()
        for _, p in best.iterrows():
            prob = 1.0 if p["minutes"] >= 2400 else (0.85 if p["minutes"] >= 1200 else 0.75)
            rows.append({"Player": p["name"], "Team": team,
                         **{f"F{k}": prob for k in range(1, 7)}})
    return pd.DataFrame(rows)


def apply_overrides(lineups, roster):
    """Apply inputs/lineup_overrides.csv (Player, start_prob): judgement calls for new
    signings the minutes+price algorithm can't see. An override either adjusts an
    existing XI member's probabilities or swaps the player in over the weakest
    droppable teammate (respecting formation minimums)."""
    path = os.path.join(config.INPUTS_DIR, "lineup_overrides.csv")
    if not os.path.exists(path):
        return lineups
    overrides = pd.read_csv(path)
    r = roster.set_index("name")
    prob_cols = [f"F{k}" for k in range(1, 7)]
    min_pos = {"DEF": 3, "MID": 2, "FWD": 1}

    for _, o in overrides.iterrows():
        name, prob = o["Player"], o["start_prob"]
        explicit_out = o.get("replaces")
        if name not in r.index:
            print(f"  override SKIPPED (not in FPL data): {name}")
            continue
        team, pos = r.loc[name, "team"], str(r.loc[name, "position"])
        if name in set(lineups["Player"]):
            lineups.loc[lineups["Player"] == name, prob_cols] = prob
            continue

        xi = lineups[lineups["Team"] == team].copy()
        xi["pos"] = xi["Player"].map(r["position"]).astype(str)
        xi["cost"] = xi["Player"].map(r["cost"])
        if pd.notna(explicit_out) and explicit_out in set(xi["Player"]):
            drop = xi[xi["Player"] == explicit_out].index[:1]
        elif pos == "GK":
            drop = xi[xi["pos"] == "GK"].index[:1]
        else:
            counts = xi["pos"].value_counts().to_dict()
            counts[pos] = counts.get(pos, 0) + 1  # pending addition
            droppable = xi[(xi["pos"] != "GK")
                           & xi["pos"].map(lambda p: counts.get(p, 0) > min_pos.get(p, 0))]
            drop = droppable.sort_values(["F1", "cost"]).index[:1]
        dropped = lineups.loc[drop, "Player"].iloc[0] if len(drop) else None
        lineups = lineups.drop(index=drop)
        lineups = pd.concat([lineups, pd.DataFrame([{
            "Player": name, "Team": team, **{c: prob for c in prob_cols}}])], ignore_index=True)
        print(f"  override: {name} ({pos}, {team}) in" + (f", {dropped} out" if dropped else ""))
    return lineups.sort_values("Team").reset_index(drop=True)


def _poisson_tail(lam, k):
    """P(X >= k) for Poisson(lam)."""
    from math import exp, factorial
    return 1.0 - sum(exp(-lam) * lam ** i / factorial(i) for i in range(k))

def _lambda_from_line(line, p_over):
    """Goal rate implied by a market total-goals line: P(X > line) = p_over."""
    k = int(line) + 1
    lo, hi = 0.01, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _poisson_tail(mid, k) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_real_gw1():
    """Real GW1 markets from inputs/gw1_match_odds.csv when present: 1X2 odds
    (three-way de-margined win probabilities) and per-team total-goals lines
    (Poisson goal rates). Returns None when the file doesn't exist."""
    path = os.path.join(config.INPUTS_DIR, "gw1_match_odds.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    inv = 1 / df["home_odds"] + 1 / df["draw_odds"] + 1 / df["away_odds"]
    df["p_home"] = (1 / df["home_odds"]) / inv
    df["p_away"] = (1 / df["away_odds"]) / inv

    def demargin(over, under):
        return (1 / over) / (1 / over + 1 / under)

    df["lam_home"] = [_lambda_from_line(r.home_line, demargin(r.home_over, r.home_under))
                      for r in df.itertuples()]
    df["lam_away"] = [_lambda_from_line(r.away_line, demargin(r.away_over, r.away_under))
                      for r in df.itertuples()]
    return df


def anchor_to_team_goals(p_cond, start, teams, lam_by_team):
    """Make player scoring odds consistent with the market's team-goal totals.

    p_cond are per-player P(score 1+) conditional on starting (factor x baseline —
    driven by win probabilities only). Each player's Poisson rate is scaled so the
    team's expected goals, sum(start_prob x rate), equals the market goal rate fitted
    from the total-goals lines. Relative shares between teammates (their factors) are
    preserved; teams without a market rate are left unscaled.

    Returns (anchored probabilities, per-row scale factors).
    """
    lam = -np.log(1.0 - p_cond.clip(0.01, 0.95))
    team_lam = (start * lam).groupby(teams).transform("sum")
    market = teams.map(lam_by_team)
    scale = (market / team_lam).where(market.notna() & (team_lam > 0), 1.0)
    return 1.0 - np.exp(-lam * scale), scale


def team_block_from_lambdas(real, epoch):
    """Clean-sheet and team-goals CSV blocks derived from market goal rates:
    P(team scores 2+/4+) from its own lambda, clean sheet = P(opponent scores 0)."""
    rows = pd.concat([
        real.assign(Team=real["home_team"], opp=real["away_team"],
                    lam_own=real["lam_home"], lam_opp=real["lam_away"]),
        real.assign(Team=real["away_team"], opp=real["home_team"],
                    lam_own=real["lam_away"], lam_opp=real["lam_home"]),
    ], ignore_index=True)
    p2 = pd.Series([_poisson_tail(l, 2) for l in rows["lam_own"]]).clip(0.02, 0.95)
    p4 = pd.Series([_poisson_tail(l, 4) for l in rows["lam_own"]]).clip(0.005, 0.6)
    p_cs = pd.Series([float(np.exp(-l)) for l in rows["lam_opp"]]).clip(0.02, 0.9)

    def to_odds(p):
        return (1 / (p * PLAYER_MARGIN)).round(2)

    cs = pd.DataFrame({"match_name": rows["Team"] + " v " + rows["opp"], "date": epoch,
                       "team_name": rows["Team"],
                       "clean_sheet_yes": to_odds(p_cs), "clean_sheet_no": to_odds(1 - p_cs)})
    tg = pd.DataFrame({"Match": rows["Team"] + " v " + rows["opp"], "Date": epoch,
                       "Team": rows["Team"], "Opponent": rows["opp"],
                       "Team_Over_1.5": to_odds(p2), "Team_Under_1.5": to_odds(1 - p2),
                       "Team_Over_3.5": to_odds(p4), "Team_Under_3.5": to_odds(1 - p4)})
    for c in ("Over_1.5", "Under_1.5", "Over_3.5", "Under_3.5"):
        tg[f"Opponent_Concedes_{c}"] = tg[f"Team_{c}"]
    return cs, tg


def fixture_win_probs(season, fixtures_gw):
    """Both teams' win probabilities for each fixture via the win-pred regression."""
    s = season.set_index("team")

    def side(team, opp, home):
        args = [pd.Series([s.loc[team, "title"]]), pd.Series([s.loc[team, "relegation"]]),
                pd.Series([s.loc[team, "top6"]]),
                pd.Series([s.loc[opp, "title"]]), pd.Series([s.loc[opp, "relegation"]]),
                pd.Series([s.loc[opp, "top6"]]), pd.Series([home])]
        return float(model.win_pred(*args).iloc[0])

    out = []
    for _, fx in fixtures_gw.iterrows():
        h, a = fx["home_team"], fx["away_team"]
        ph, pa = max(side(h, a, True), 0.05), max(side(a, h, False), 0.05)
        total = ph + pa
        if total > 0.87:            # leave a plausible draw share
            ph, pa = ph * 0.87 / total, pa * 0.87 / total
        out.append({"home_team": h, "away_team": a, "p_home": ph, "p_away": pa})
    return pd.DataFrame(out)


def _series(val, index):
    return pd.Series(val, index=index)


def synth_odds(season, lineups, fallback):
    fallback = fallback.copy()
    for col in fallback.columns:
        if col != "Player Name":            # workbook export left '#N/A' strings behind
            fallback[col] = pd.to_numeric(fallback[col], errors="coerce")
    sf = pd.read_csv(os.path.join(config.INPUTS_DIR, "season_fixtures.csv"))
    real_gw1 = load_real_gw1()
    gw1 = real_gw1 if real_gw1 is not None else fixture_win_probs(season, sf[sf["gameweek"] == 1])
    gw2 = fixture_win_probs(season, sf[sf["gameweek"] == 2])
    if real_gw1 is not None:
        print("  using REAL GW1 match odds (inputs/gw1_match_odds.csv) for win/team-goals/clean-sheet markets")

    # --- win-draw-win (GW1 block then GW2 block; draw odds enable 3-way de-margining)
    def wdw_rows(frame):
        if "home_odds" in frame.columns:  # real odds pass through verbatim
            return frame.rename(columns={"home_odds": "home_win_odds",
                                         "away_odds": "away_win_odds",
                                         "draw_odds": "draw_odds"})
        out = frame.copy()
        out["home_win_odds"] = (1 / (out["p_home"] * WDW_MARGIN)).round(2)
        out["away_win_odds"] = (1 / (out["p_away"] * WDW_MARGIN)).round(2)
        out["draw_odds"] = (1 / ((1 - out["p_home"] - out["p_away"]).clip(0.10) * WDW_MARGIN)).round(2)
        return out

    cols = ["home_team", "away_team", "home_win_odds", "away_win_odds", "draw_odds"]
    pd.concat([wdw_rows(gw1)[cols], wdw_rows(gw2)[cols]], ignore_index=True).to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_win_draw_win_odds.csv"), index=False)

    # --- per-player context for GW1
    fb = fallback.drop_duplicates(subset="Player Name").set_index("Player Name")
    pos_median = fallback.groupby(
        fallback["Player Name"].map(dict(zip(lineups["Player"], lineups["Player"]))).notna()).median(numeric_only=True)
    med = fallback.median(numeric_only=True)

    ctx = lineups.merge(
        pd.concat([gw1.rename(columns={"home_team": "Team", "away_team": "opp"}).assign(venue="H", p=lambda d: d.p_home, po=lambda d: d.p_away),
                   gw1.rename(columns={"away_team": "Team", "home_team": "opp"}).assign(venue="A", p=lambda d: d.p_away, po=lambda d: d.p_home)],
                  ignore_index=True)[["Team", "opp", "venue", "p", "po"]],
        on="Team", how="left")
    roster_pos = ingest.load_fpl_players().set_index("name")["position"]
    ctx["position"] = ctx["Player"].map(roster_pos).fillna("MID").astype(str)

    def factor(col):
        return ctx["Player"].map(fb[col]).fillna(med[col])

    win, opp, home = ctx["p"], ctx["po"], ctx["venue"] == "H"
    pos = ctx["position"]
    match_name = ctx["Team"] + " v " + ctx["opp"]
    mid = 90000000 + ctx.index

    def to_odds(p, margin=PLAYER_MARGIN):
        return (1 / (p.clip(0.01, 0.95) * margin)).round(2)

    outfield = pos != "GK"
    p_score = (factor("Score 1+ Factor") * model.baseline("score1", win, opp, pos, home)).clip(0.01, 0.9)
    p_assist = (factor("Assist Factor") * model.baseline("assist", win, opp, pos, home)).clip(0.01, 0.9)

    # Anchor scoring odds to the market team-goal rates (real GW1 total-goals lines);
    # assists share the same team scale so the model's assist/goal ratio is preserved.
    if real_gw1 is not None:
        lam_by_team = {**dict(zip(real_gw1["home_team"], real_gw1["lam_home"])),
                       **dict(zip(real_gw1["away_team"], real_gw1["lam_away"]))}
        # GKs never appear in scoring markets — they must not consume team goal budget
        start = ctx["F1"].astype(float).fillna(0).where(outfield, 0.0)
        p_score, scale = anchor_to_team_goals(p_score, start, ctx["Team"], lam_by_team)
        p_assist, _ = anchor_to_team_goals(
            p_assist, start, ctx["Team"],
            {t: ASSISTS_PER_GOAL_ODDS * l for t, l in lam_by_team.items()})
        by_team = scale.groupby(ctx["Team"].values).first().sort_values()
        print(f"  player odds anchored to market team goals: scale "
              f"{by_team.iloc[0]:.2f} ({by_team.index[0]}) .. "
              f"{by_team.iloc[-1]:.2f} ({by_team.index[-1]})")

    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": to_odds(p_score)})[outfield].to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_goalscorer_odds.csv"), index=False)
    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": to_odds(model.poisson_score2(p_score))})[outfield].to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_two_goals_odds.csv"), index=False)
    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": to_odds(p_assist)})[outfield].to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_assist_odds.csv"), index=False)
    p_yellow = factor("F1 Yellow Card Factor") * model.baseline("yellow", win, opp, pos, home)
    pd.DataFrame({"match_name": match_name, "date": GW1_EPOCH, "player_name": ctx["Player"],
                  "odds_decimal": to_odds(p_yellow)})[outfield].to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_booking_odds.csv"), index=False)

    gk = ctx[pos == "GK"]
    p3 = (gk["Player"].map(fb["F1 3+ Saves Factor"]).fillna(med["F1 3+ Saves Factor"])
          * model.baseline("saves3", gk["p"], gk["po"], _series("GK", gk.index), gk["venue"] == "H"))
    p6 = (gk["Player"].map(fb["F1 6+ Saves Factor"]).fillna(med["F1 6+ Saves Factor"])
          * model.baseline("saves6", gk["p"], gk["po"], _series("GK", gk.index), gk["venue"] == "H"))
    pd.DataFrame({"Match": gk["Team"] + " v " + gk["opp"], "Date": GW1_EPOCH, "Team": gk["Team"],
                  "Goalkeeper": gk["Player"], "3+ Saves": to_odds(p3), "6+ Saves": to_odds(p6)}).to_csv(
        os.path.join(config.SPORTSBET_DIR, "sportsbet_goalkeeper_saves_odds.csv"), index=False)

    # --- team-level markets for a gameweek block
    def team_block(gw, epoch):
        rows = pd.concat([
            gw.rename(columns={"home_team": "Team", "away_team": "opp"}).assign(venue="H", p=lambda d: d.p_home, po=lambda d: d.p_away),
            gw.rename(columns={"away_team": "Team", "home_team": "opp"}).assign(venue="A", p=lambda d: d.p_away, po=lambda d: d.p_home),
        ], ignore_index=True)
        anypos = _series("MID", rows.index)
        p_cs = model.baseline("clean_sheet", rows["p"], rows["po"], anypos, rows["venue"] == "H").clip(0.03, 0.7)
        # Team scores 2+/4+ == opponent concedes 2+/4+ (evaluated from the opponent's view)
        p2 = model.baseline("concede2", rows["po"], rows["p"], anypos, rows["venue"] != "H").clip(0.05, 0.9)
        p4 = model.baseline("concede4", rows["po"], rows["p"], anypos, rows["venue"] != "H").clip(0.01, 0.6)
        cs = pd.DataFrame({"match_name": rows["Team"] + " v " + rows["opp"], "date": epoch,
                           "team_name": rows["Team"],
                           "clean_sheet_yes": to_odds(p_cs), "clean_sheet_no": to_odds(1 - p_cs)})
        tg = pd.DataFrame({"Match": rows["Team"] + " v " + rows["opp"], "Date": epoch,
                           "Team": rows["Team"], "Opponent": rows["opp"],
                           "Team_Over_1.5": to_odds(p2), "Team_Under_1.5": to_odds(1 - p2),
                           "Team_Over_3.5": to_odds(p4), "Team_Under_3.5": to_odds(1 - p4)})
        for c in ("Over_1.5", "Under_1.5", "Over_3.5", "Under_3.5"):
            tg[f"Opponent_Concedes_{c}"] = tg[f"Team_{c}"]
        return cs, tg

    if real_gw1 is not None:
        cs1, tg1 = team_block_from_lambdas(real_gw1, GW1_EPOCH)
    else:
        cs1, tg1 = team_block(gw1, GW1_EPOCH)
    cs2, tg2 = team_block(gw2, GW2_EPOCH)
    cs1.to_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_clean_sheet_odds.csv"), index=False)
    cs2.to_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_clean_sheet_odds_f2.csv"), index=False)
    tg1.to_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_team_goals_odds.csv"), index=False)
    tg2.to_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_team_goals_odds_f2.csv"), index=False)

    with open(os.path.join(config.SPORTSBET_DIR, "SYNTHETIC_NOTE.txt"), "w", encoding="utf-8") as fh:
        real = "REAL (from inputs/gw1_match_odds.csv)" if real_gw1 is not None else "model-derived"
        fh.write(f"Generated by tools/build_preseason_data.py. GW1 match odds, team goals and "
                 f"clean sheets: {real}. GW2 markets and all PLAYER markets (goalscorer, "
                 f"assists, cards, saves): model-derived placeholders. Rerun sportsbet.py when "
                 f"player markets open (it overwrites these files), then delete this note. Do "
                 f"not record --gw archives while player odds are synthetic.\n")


FACTOR_STATS = {
    "Score 1+ Factor": ("score1", "F1 Score 1+"),
    "Assist Factor": ("assist", "F1 Assist"),
    "F1 Yellow Card Factor": ("yellow", "F1 Yellow Card"),
    "F1 Concede 2+ Goals Factor": ("concede2", "F1 Concede 2+ Goals"),
    "F1 Concede 4+ Goals Factor": ("concede4", "F1 Concede 4+ Goals"),
    "F1 3+ Saves Factor": ("saves3", "F1 3+ Saves"),
    "F1 6+ Saves Factor": ("saves6", "F1 6+ Saves"),
}


def rebuild_factors():
    """Regenerate inputs/fallback_factors.csv from the archive on the CURRENT
    coefficient scale.

    Factors are only meaningful relative to a baseline-coefficient set, and the
    workbook's Fallback Factors sheet predated a late-season coefficient update (its
    values are ~2x the current scale). The archived odds-implied probabilities are
    scale-free, so: factor = archived probability / current baseline at the archived
    context, median per player across all archived gameweeks."""
    hist = pd.read_csv(os.path.join(config.INPUTS_DIR, "historical_player_data.csv"), low_memory=False)
    for col in hist.columns:
        if col not in ("Season", "Player Name", "Position", "Team", "F1 Opponent", "F1 Venue"):
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    win, opp = hist["F1 Win"], hist["F1 Opponent Win"]
    pos, home = hist["Position"], hist["F1 Venue"] == "H"
    out = pd.DataFrame({"Player Name": hist["Player Name"]})
    for col, (stat, prob_col) in FACTOR_STATS.items():
        out[col] = hist[prob_col] / model.baseline(stat, win, opp, pos, home)

    factors = out.groupby("Player Name", sort=True).median().reset_index()
    factors.to_csv(os.path.join(config.INPUTS_DIR, "fallback_factors.csv"), index=False)
    print(f"fallback_factors.csv rebuilt on current coefficient scale "
          f"({len(factors)} players, median over archived gameweeks)")
    return factors


from fpl_pipeline.players import normalize_start_probs  # noqa: E402  (shared invariant)


def report_pool_depth(lineups, target=11.0):
    """Raw beliefs are stored un-normalized; warn when a team's pool can't reach 11
    starters even at full certainty (needs more depth players curated in)."""
    for team, grp in lineups.groupby("Team"):
        ceiling = grp["F1"].astype(float).clip(0, 1).count()
        if ceiling < target:
            print(f"  depth: {team} has only {ceiling} players listed — cannot field 11")


def patch_lineups(roster):
    """Default mode: the XIs in starting_lineups.csv are CURATED (picked with football
    judgement, not the deterministic algorithm) — preserve them, and just patch the
    feedback files on top: drop unavailable players (no auto-replacement; curation
    handles that), then apply overrides (set probability if present, append if not,
    drop an explicit `replaces` target if named)."""
    path = os.path.join(config.INPUTS_DIR, "starting_lineups.csv")
    lineups = pd.read_csv(path)
    prob_cols = [f"F{k}" for k in range(1, 7)]

    unavailable_path = os.path.join(config.INPUTS_DIR, "unavailable_players.csv")
    if os.path.exists(unavailable_path):
        banned = set(pd.read_csv(unavailable_path)["Player"])
        dropped = sorted(set(lineups["Player"]) & banned)
        if dropped:
            lineups = lineups[~lineups["Player"].isin(banned)]
            print(f"  patch: dropped unavailable {', '.join(dropped)} (curate replacements!)")

    overrides_path = os.path.join(config.INPUTS_DIR, "lineup_overrides.csv")
    if os.path.exists(overrides_path):
        r = roster.set_index("name")
        for _, o in pd.read_csv(overrides_path).iterrows():
            name, prob, out = o["Player"], o["start_prob"], o.get("replaces")
            if pd.notna(out) and out in set(lineups["Player"]):
                lineups = lineups[lineups["Player"] != out]
            if name in set(lineups["Player"]):
                lineups.loc[lineups["Player"] == name, prob_cols] = prob
            elif name in r.index:
                lineups = pd.concat([lineups, pd.DataFrame([{
                    "Player": name, "Team": r.loc[name, "team"],
                    **{c: prob for c in prob_cols}}])], ignore_index=True)
            else:
                print(f"  patch: override SKIPPED (not in FPL data): {name}")

    lineups = lineups.sort_values("Team").reset_index(drop=True)
    lineups.to_csv(path, index=False)
    report_pool_depth(lineups)
    print(f"starting_lineups.csv: {len(lineups)} players (curated beliefs, feedback patched)")
    return lineups


def main(rebuild_lineups=False, lineups_only=False):
    roster = ingest.load_fpl_players()

    if lineups_only:
        # In-season safe mode: apply unavailable_players/lineup_overrides to the
        # lineups and STOP — never touches the odds files (which hold real scraped
        # markets once the season is running).
        patch_lineups(roster)
        return

    inputs = ingest.load_inputs()
    season = team_model.season_probs(inputs)

    if rebuild_lineups:
        lineups = build_lineups(roster)
        lineups = apply_overrides(lineups, roster)
        lineups.to_csv(os.path.join(config.INPUTS_DIR, "starting_lineups.csv"), index=False)
        print(f"starting_lineups.csv: REGENERATED from the algorithm ({len(lineups)} players) "
              f"— this discards curated XIs")
    else:
        lineups = patch_lineups(roster)

    factors = rebuild_factors()
    synth_odds(season, normalize_start_probs(lineups), factors)
    print("Synthetic odds written to sportsbet/*.csv (see SYNTHETIC_NOTE.txt)")


if __name__ == "__main__":
    main(rebuild_lineups="--rebuild-lineups" in sys.argv,
         lineups_only="--lineups-only" in sys.argv)
