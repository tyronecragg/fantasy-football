"""Snapshot helpers: every pipeline stage writes its DataFrame to outputs/ for inspection."""
import os

import pandas as pd

from . import config


def _fix_mojibake(val):
    """Undo UTF-8-read-as-cp1252 double encoding ('Touré' -> 'TourÃ©'). Excel shows a
    UTF-8 file's accents as mojibake and its ANSI save bakes them in; the reverse
    round-trip only succeeds for genuinely double-encoded text, so plain strings and
    correctly-typed cp1252 accents pass through untouched."""
    if not isinstance(val, str):
        return val
    try:
        return val.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return val


def repair_mojibake(df):
    df.columns = [_fix_mojibake(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(_fix_mojibake)
    return df


def read_csv_tolerant(path, **kwargs):
    """pd.read_csv that survives Excel's ANSI re-saves: falls back to cp1252, repairs
    double-encoded names, and re-writes the file as UTF-8 so the repo heals itself.
    Editing the input CSVs in Excel is a supported weekly workflow — Excel silently
    changes their encoding."""
    try:
        df = pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        df = repair_mojibake(pd.read_csv(path, encoding="cp1252", **kwargs))
        print(f"  note: {os.path.basename(path)} was not UTF-8 (Excel ANSI save) - "
              f"repaired and re-saved as UTF-8")
        df.to_csv(path, index=False)
    return df

_state = {"n": 0, "dir": None}


def snapshot(df, name):
    """Write a stage DataFrame to the active output dir and print a one-line summary."""
    out_dir = _state["dir"] or config.OUTPUTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    _state["n"] += 1
    path = os.path.join(out_dir, f"{_state['n']:02d}_{name}.csv")
    df.to_csv(path, index=False)
    print(f"  [{_state['n']:02d}] {name:<28} {df.shape[0]:>5} rows x {df.shape[1]:>3} cols -> {os.path.relpath(path, config.ROOT)}")
    return df


def reset_counter(subdir=None):
    """Reset the stage counter. Parity runs pass subdir='parity' so their snapshots
    never clobber outputs/ — the optimisers read outputs/13_players_master.csv."""
    _state["n"] = 0
    _state["dir"] = os.path.join(config.OUTPUTS_DIR, subdir) if subdir else None


def vlookup(keys, table, key_col, val_col):
    """Excel VLOOKUP(..., FALSE) semantics: exact match, first occurrence wins, NaN if absent."""
    mapping = table.drop_duplicates(subset=key_col, keep="first").set_index(key_col)[val_col]
    return keys.map(mapping)
