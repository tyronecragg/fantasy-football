# -*- coding: utf-8 -*-
"""Should I use my free transfer this week, or BANK it for an extra move next week?

    env/Scripts/python tools/transfer_bank_check.py [--free-transfers 1] [--bank 0.0] [--gain-now G]

A 2-period lookahead static check on the optimiser's recommendation. The optimiser treats a free
transfer as costless, so it grabs any positive gain (even +0.2). But a held FT has OPTION VALUE:
banking it gives an extra transfer NEXT week. This compares:

    gain_now      = horizon value of the best 1-transfer move THIS week   (or pass --gain-now to use
                    the optimiser's actual recommended-transfer gain)
    marginal_next = [best (ft+1)-move next week] - [best (ft)-move next week], on the SHIFTED horizon
                    (next week's F1 = this week's F2, ...) — the value of the extra FT banking buys

    -> BANK if marginal_next > gain_now (and you're below the 5-FT cap; at the cap banking is wasted).

Short + rolling by design, so it's robust to forced (injury/price) transfers — you re-decide every
week with fresh news rather than committing to a fragile long-range plan."""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.chip_history import (MASTER, FIX_COLS, HORIZON_W, MAX_FT,  # noqa: E402
                                best_squad, best_xi, current_squad_names)

# Next week's fixture columns = this week's shifted one on (F1<-F2 ... F8<-F8 steady-state hold)
FIX_COLS_NEXT = [f"F{k} XP" for k in range(2, 9)] + ["F8 XP"]


def value(m, xi_idx, fix_cols):
    """A squad-XI's weighted-horizon value + captaincy (best starter doubled each fixture)."""
    xi = m.loc[list(xi_idx)]
    body = float(sum(w * xi[c] for w, c in zip(HORIZON_W, fix_cols)).sum())
    cap = float(sum(w * xi[c].max() for w, c in zip(HORIZON_W, fix_cols)))
    return body + cap


def best_move_value(m, budget, current_idx, htotal_col, fix_cols, k):
    """Horizon value of the best squad reachable with <=k transfers (None if PuLP missing)."""
    xi = best_squad(m, budget, htotal_col, current_idx, k)
    return value(m, xi, fix_cols) if xi else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-transfers", type=int, default=1, help="free transfers you hold now")
    ap.add_argument("--bank", type=float, default=0.0, help="money in the bank (added to budget)")
    ap.add_argument("--gain-now", type=float, default=None,
                    help="override: the optimiser's actual recommended-transfer horizon gain")
    a = ap.parse_args()
    ft = a.free_transfers

    m = pd.read_csv(MASTER)[["Player Name", "Position", "Team", "Cost"] + FIX_COLS].copy()
    m["_htotal"] = sum(w * m[c] for w, c in zip(HORIZON_W, FIX_COLS))            # this week's horizon
    m["_htotal_next"] = sum(w * m[c] for w, c in zip(HORIZON_W, FIX_COLS_NEXT))  # next week's horizon
    names = current_squad_names()
    squad = m[m["Player Name"].isin(names)]
    cur = list(squad.index)
    budget = float(squad["Cost"].sum()) + a.bank

    # gain_now: horizon value of the best 1-transfer move this week vs holding
    v0 = value(m, best_xi(squad, "_htotal")[1], FIX_COLS)
    if a.gain_now is not None:
        gain_now = a.gain_now
    else:
        v1 = best_move_value(m, budget, cur, "_htotal", FIX_COLS, 1)
        if v1 is None:
            raise SystemExit("PuLP unavailable — run with env/Scripts/python")
        gain_now = v1 - v0

    # marginal value of the extra FT banking buys NEXT week: best(ft+1) - best(ft), shifted horizon
    n_lo = best_move_value(m, budget, cur, "_htotal_next", FIX_COLS_NEXT, min(ft, MAX_FT))
    n_hi = best_move_value(m, budget, cur, "_htotal_next", FIX_COLS_NEXT, min(ft + 1, MAX_FT))
    marginal_next = n_hi - n_lo

    at_cap = ft >= MAX_FT
    bank = (not at_cap) and marginal_next > gain_now

    print(f"\n2-period transfer check (you hold {ft} free transfer{'s' if ft != 1 else ''}):\n")
    print(f"  move now   : best 1-transfer gain THIS week (horizon) = +{gain_now:.2f}")
    if at_cap:
        print(f"  bank       : you're at the {MAX_FT}-FT cap — banking wastes next week's +1")
    else:
        print(f"  bank       : extra transfer NEXT week is worth +{marginal_next:.2f} "
              f"(best {min(ft+1,MAX_FT)}-move {n_hi:.1f} - best {min(ft,MAX_FT)}-move {n_lo:.1f})")
    verdict = "BANK IT" if bank else "MAKE THE MOVE"
    why = (f"banking's +{marginal_next:.2f} beats moving's +{gain_now:.2f}" if bank
           else (f"at the FT cap" if at_cap
                 else f"moving's +{gain_now:.2f} beats banking's +{marginal_next:.2f}"))
    print(f"\n  -> {verdict}  ({why})\n")
    print("  (rolling check — re-run each week with fresh injury news; captures option value while "
          "staying robust to forced transfers)")


if __name__ == "__main__":
    main()
