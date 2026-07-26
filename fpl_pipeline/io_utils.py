"""Snapshot helpers: every pipeline stage writes its DataFrame to outputs/ for inspection."""
import os

from . import config

_counter = {"n": 0}


def snapshot(df, name):
    """Write a stage DataFrame to outputs/NN_name.csv and print a one-line summary."""
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    _counter["n"] += 1
    path = os.path.join(config.OUTPUTS_DIR, f"{_counter['n']:02d}_{name}.csv")
    df.to_csv(path, index=False)
    print(f"  [{_counter['n']:02d}] {name:<28} {df.shape[0]:>5} rows x {df.shape[1]:>3} cols -> {os.path.relpath(path, config.ROOT)}")
    return df


def reset_counter():
    _counter["n"] = 0


def vlookup(keys, table, key_col, val_col):
    """Excel VLOOKUP(..., FALSE) semantics: exact match, first occurrence wins, NaN if absent."""
    mapping = table.drop_duplicates(subset=key_col, keep="first").set_index(key_col)[val_col]
    return keys.map(mapping)
