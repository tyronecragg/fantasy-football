import pandas as pd

from fpl_pipeline.prices import apply_sell_prices, sell_price


def test_sell_price_banks_half_the_rise():
    assert sell_price(8.0, 8.6) == 8.3   # the user's example
    assert sell_price(8.0, 8.2) == 8.1
    assert sell_price(8.0, 8.1) == 8.0   # first 0.1 banks nothing
    assert sell_price(8.0, 8.3) == 8.1   # 0.3 rise -> +0.1 (floor)
    assert sell_price(4.5, 4.7) == 4.6   # float-artifact regression check


def test_sell_price_falls_borne_in_full():
    assert sell_price(8.0, 7.7) == 7.7
    assert sell_price(5.5, 5.5) == 5.5


def test_apply_sell_prices(tmp_path, capsys):
    csv = tmp_path / "purchase_prices.csv"
    pd.DataFrame({"Player": ["A", "B"], "purchase_price": [8.0, 6.0]}).to_csv(csv, index=False)
    df = pd.DataFrame({"Player Name": ["A", "B", "C", "D"],
                       "Cost": [8.6, 5.7, 9.9, 4.4]})

    apply_sell_prices(df, ["A", "B", "D"], str(csv))
    out = capsys.readouterr().out
    assert df.loc[0, "Cost"] == 8.3      # owned, risen -> sell price
    assert df.loc[1, "Cost"] == 5.7      # owned, fallen -> full fall (unchanged market)
    assert df.loc[2, "Cost"] == 9.9      # not owned -> market untouched
    assert df.loc[3, "Cost"] == 4.4      # owned but no purchase record -> market + warning
    assert "no purchase price recorded for D" in out
    assert "A 8.6->8.3" in out
