"""Backtest the F2-F6 projections against what actually happened to the odds.

For every pair of archived gameweeks (M, N=M+k), reconstruct the prediction the
pipeline would have made at GW M for GW N using only week-M information:

    factor_M x baseline(win_pred_M(N), opp_win_pred_M(N), position, venue_N)

and compare it against the odds-implied probability that materialised at GW N
(the archived F1 value). Everything reuses fpl_pipeline.model, so this scores the
real serving code, not a reimplementation.

Comparators per horizon:
- persistence: the player's F1 probability at M, unchanged (the dumbest honest forecast)
- position mean: mean F1 probability at M by position
- oracle odds: factor_M x baseline at GW N's *actual* match odds — isolates the two error
  sources (oracle ~ model => win-prediction error dominates; oracle << model error =>
  factor drift dominates)
- f2_formula (score, k=1 only): the Coefficients-sheet score model the pipeline uses for
  F2, vs the generic baseline used for F3+

Outputs: outputs/backtest_pairs.csv (every pair) + a console report.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model  # noqa: E402

ARCHIVE = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
PAIRS_OUT = os.path.join(config.OUTPUTS_DIR, "backtest_pairs.csv")

STATS = {
    "score1": "F1 Score 1+",
    "assist": "F1 Assist",
    "yellow": "F1 Yellow Card",
    "clean_sheet": "F1 Clean Sheet",
    "concede2": "F1 Concede 2+ Goals",
    "saves3": "F1 3+ Saves",
}
MAX_HORIZON = 5

NUMERIC = ["Gameweek", "Title", "Relegation", "Top 6", "F1 Win", "F1 Opponent Win"] \
    + list(STATS.values())


def load_archive():
    df = pd.read_csv(ARCHIVE, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Season" not in df.columns:
        df["Season"] = "unknown"
    return df


def team_probs_by_gw(archive):
    """Per gameweek: team -> (title, relegation, top6) season probabilities as of that week."""
    t = (archive.groupby(["Gameweek", "Team"])[["Title", "Relegation", "Top 6"]]
         .first().reset_index())
    return {gw: sub.set_index("Team") for gw, sub in t.groupby("Gameweek")}


def build_pairs(archive, factor_overrides=None):
    """Forecast-vs-actual pairs, built within each season (gameweek numbers recur).

    factor_overrides: optional {stat: {(player, gameweek): factor}} — replaces the
    default single-week odds/baseline factor where a value exists (used by
    tools/factor_experiment.py to evaluate alternative factor constructions)."""
    out = []
    for _, season_archive in archive.groupby("Season"):
        out.extend(_build_season_pairs(season_archive, factor_overrides))
    return pd.concat(out, ignore_index=True)


def _build_season_pairs(archive, factor_overrides=None):
    gws = sorted(archive["Gameweek"].dropna().unique())
    team_probs = team_probs_by_gw(archive)
    out = []

    base_cols = ["Player Name", "Position", "Team", "Title", "Relegation", "Top 6",
                 "F1 Win", "F1 Opponent Win", "F1 Venue"] + list(STATS.values())
    actual_cols = ["Player Name", "F1 Opponent", "F1 Venue", "F1 Win", "F1 Opponent Win"] \
        + list(STATS.values())

    for m in gws:
        at_m = archive[archive["Gameweek"] == m][base_cols]
        for k in range(1, MAX_HORIZON + 1):
            n = m + k
            if n not in team_probs:
                continue
            at_n = archive[archive["Gameweek"] == n][actual_cols]
            pair = at_m.merge(at_n, on="Player Name", suffixes=("_m", "_n"))
            pair = pair[pair["F1 Win_m"].notna() & pair["F1 Win_n"].notna()]
            if pair.empty:
                continue

            pos = pair["Position"]
            home_m = pair["F1 Venue_m"] == "H"
            home_n = pair["F1 Venue_n"] == "H"

            # Opponent-at-N's season probabilities as seen at M
            opp_at_m = team_probs[m].reindex(pair["F1 Opponent"].values)  # only in the N side, so unsuffixed
            known_opp = pd.Series(opp_at_m["Title"].notna().values, index=pair.index)

            win_hat = model.win_pred(
                pair["Title"], pair["Relegation"], pair["Top 6"],
                pd.Series(opp_at_m["Title"].values, index=pair.index),
                pd.Series(opp_at_m["Relegation"].values, index=pair.index),
                pd.Series(opp_at_m["Top 6"].values, index=pair.index), home_n)
            opp_hat = model.opp_win_pred(
                pair["Title"], pair["Relegation"],
                pd.Series(opp_at_m["Title"].values, index=pair.index),
                pd.Series(opp_at_m["Relegation"].values, index=pair.index), home_n)
            win_hat, opp_hat = model.scale_win_pair(win_hat.clip(0, 1), opp_hat.clip(0, 1))

            for stat, col in STATS.items():
                prob_m = pair[f"{col}_m"]
                actual = pair[f"{col}_n"]
                factor = prob_m / model.baseline(stat, pair["F1 Win_m"],
                                                 pair["F1 Opponent Win_m"], pos, home_m)
                if factor_overrides and stat in factor_overrides:
                    fmap = factor_overrides[stat]
                    override = pair["Player Name"].map(lambda p: fmap.get((p, m), np.nan))
                    factor = override.where(override.notna(), factor)

                pred = (factor * model.baseline(stat, win_hat, opp_hat, pos, home_n)).clip(0, 1)
                oracle = (factor * model.baseline(stat, pair["F1 Win_n"],
                                                  pair["F1 Opponent Win_n"], pos, home_n)).clip(0, 1)
                if stat == "score1" and k == 1:
                    f2_formula = model.f2_score1(factor, win_hat, opp_hat, home_n).clip(0, 1)
                    # same formula given N's *actual* odds — is the formula itself weak,
                    # or just starved of good win predictions?
                    f2_oracle = model.f2_score1(factor, pair["F1 Win_n"],
                                                pair["F1 Opponent Win_n"], home_n).clip(0, 1)
                else:
                    f2_formula = f2_oracle = np.nan
                pos_mean = prob_m.groupby(pos.values).transform("mean")

                rows = pd.DataFrame({
                    "stat": stat, "horizon": k, "gw_from": m, "gw_to": n,
                    "player": pair["Player Name"], "position": pos.values,
                    "predicted": pred, "oracle_odds": oracle, "persistence": prob_m.values,
                    "position_mean": pos_mean.values, "f2_formula": f2_formula,
                    "f2_oracle": f2_oracle, "actual": actual.values,
                    "valid_opp": known_opp.values,
                })
                out.append(rows[rows["actual"].notna() & rows["persistence"].notna()])

    return out


def _metrics(pred, actual):
    mask = pred.notna() & actual.notna() & np.isfinite(pred)
    if mask.sum() == 0:
        return None
    e = pred[mask] - actual[mask]
    return {"n": int(mask.sum()), "mae": e.abs().mean(), "bias": e.mean(),
            "corr": pred[mask].corr(actual[mask])}


def report(pairs):
    for stat in pairs["stat"].unique():
        print(f"\n=== {stat} ===")
        print(f"{'horizon':<9} {'n':>5} | {'model':>7} {'oracle':>7} {'persist':>7} "
              f"{'posmean':>7} {'f2form':>7} | {'bias':>7} {'corr':>6}")
        sub_all = pairs[(pairs["stat"] == stat) & pairs["valid_opp"]]
        for k in sorted(sub_all["horizon"].unique()):
            s = sub_all[sub_all["horizon"] == k]
            mm = _metrics(s["predicted"], s["actual"])
            oo = _metrics(s["oracle_odds"], s["actual"])
            pp = _metrics(s["persistence"], s["actual"])
            gg = _metrics(s["position_mean"], s["actual"])
            ff = _metrics(s["f2_formula"], s["actual"])
            ff_mae = f"{ff['mae']:>7.4f}" if ff else "      -"
            print(f"F{k+1:<8} {mm['n']:>5} | {mm['mae']:>7.4f} {oo['mae']:>7.4f} "
                  f"{pp['mae']:>7.4f} {gg['mae']:>7.4f} {ff_mae} | "
                  f"{mm['bias']:>+7.4f} {mm['corr']:>6.3f}")

        # Calibration of the model, pooled across horizons
        s = sub_all[sub_all["predicted"].notna()]
        buckets = pd.cut(s["predicted"], [0, .1, .2, .3, .4, .5, .6, 1.0])
        cal = s.groupby(buckets, observed=True).agg(
            n=("actual", "size"), mean_pred=("predicted", "mean"), mean_actual=("actual", "mean"))
        print("\n  calibration (all horizons pooled):")
        for b, row in cal.iterrows():
            drift = row["mean_pred"] - row["mean_actual"]
            print(f"    pred {str(b):<12} n={int(row['n']):>5}  "
                  f"avg pred {row['mean_pred']:.3f}  avg actual {row['mean_actual']:.3f}  "
                  f"({'+' if drift >= 0 else ''}{drift:.3f})")


def main():
    archive = load_archive()
    pairs = build_pairs(archive)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    pairs.to_csv(PAIRS_OUT, index=False)
    print(f"{len(pairs)} forecast-vs-actual pairs -> {os.path.relpath(PAIRS_OUT, config.ROOT)}")
    report(pairs)
    return pairs


if __name__ == "__main__":
    main()
