"""FPL selling-price mechanics.

You only bank half of a price rise: sell price = purchase price + 0.1 for every full
0.2 of rise (rounded down). Falls are borne in full until back at the purchase price.
A player bought at 8.0 whose market price is 8.6 is therefore an 8.3 asset — and the
transfer optimiser must budget owned players at SELL price, not market price.

Purchase prices live in inputs/purchase_prices.csv (Player, purchase_price), updated
whenever the squad changes. A squad player missing from the file is assumed to have
been bought at the current market price (with a warning), so a lapse degrades
gracefully rather than crashing.
"""
import os

import pandas as pd

from . import config

PURCHASE_PRICES_CSV = os.path.join(config.INPUTS_DIR, "purchase_prices.csv")


def sell_price(purchase, current):
    """FPL sell price, computed in tenths to avoid float artifacts."""
    p, c = round(purchase * 10), round(current * 10)
    if c <= p:
        return c / 10.0
    return (p + (c - p) // 2) / 10.0


def apply_sell_prices(df, current_team_names, purchase_csv=PURCHASE_PRICES_CSV):
    """Replace the Cost of currently-owned players with their SELL price, in place.

    Downstream this makes the optimiser exactly right: total resources = sum of owned
    players' sell prices (+ bank), keeping a player consumes his sell price, and buying
    anyone else costs market price."""
    if not os.path.exists(purchase_csv):
        print(f"  note: {os.path.basename(purchase_csv)} missing - owned players valued at market price")
        return df

    purchases = pd.read_csv(purchase_csv).drop_duplicates(subset="Player")
    purchase_of = dict(zip(purchases["Player"], purchases["purchase_price"]))

    adjustments = []
    for name in current_team_names:
        rows = df.index[df["Player Name"].str.strip() == str(name).strip()]
        if len(rows) == 0:
            continue
        market = float(df.loc[rows[0], "Cost"])
        if name not in purchase_of:
            print(f"  note: no purchase price recorded for {name} - assuming bought at {market}")
            continue
        sp = sell_price(purchase_of[name], market)
        if abs(sp - market) > 1e-9:
            df.loc[rows[0], "Cost"] = sp
            adjustments.append(f"{name} {market:.1f}->{sp:.1f}")
    if adjustments:
        print(f"  sell prices applied: {', '.join(adjustments)}")
    return df
