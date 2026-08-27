# -*- coding: utf-8 -*-
"""Serve the persisted forward-projection models inside the pipeline.

The models (tools/save_projection_models.py) predict the F2-F8 odds-derived value for the
deployable defensive markets (clean_sheet, concede2, saves3) from 19 objective features. This
module rebuilds those exact features at pipeline runtime and applies the models, so the master's
F2-F8 columns for those three components come from the model instead of factor x baseline.

Feature parity is guaranteed by reusing the SAME code the training builder uses: trailing form
via build_training_data._dense_trailing, and the diff/momentum block via build_training_data.
add_diffs. The only runtime-specific step is extending the on-disk archive with a placeholder row
for the (not-yet-archived) current gameweek so the dense reindex reaches it; trailing form at M
uses shift(1), so the placeholder's own (empty) values never enter the form.
"""
import os

import joblib
import pandas as pd

from . import config

MODELS_DIR = os.path.join(config.OUTPUTS_DIR, "models")


def _imports():
    """Deferred imports from tools/ (kept out of module import time so parity mode never needs them)."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.backtest_projections import STATS
    from tools.build_training_data import FEATURES, add_diffs, _dense_trailing
    from tools.train_projection_model import _prep
    return STATS, FEATURES, add_diffs, _dense_trailing, _prep


def load_models(models_dir=MODELS_DIR):
    """Return (models dict {stat: LGBMRegressor}, meta) or (None, None) if not present."""
    meta_path = os.path.join(models_dir, "proj_meta.joblib")
    if not os.path.exists(meta_path):
        return None, None
    meta = joblib.load(meta_path)
    models = {s: joblib.load(os.path.join(models_dir, f"proj_{s}.joblib")) for s in meta["deploy"]}
    return models, meta


class Server:
    """Holds the models + the trailing-form lookups (computed once per run) and predicts a
    horizon on demand from the master's current F{k} columns."""

    def __init__(self, master, archive, season, gameweek, models, meta):
        self.models, self.meta = models, meta
        STATS, FEATURES, add_diffs, dense, prep = _imports()
        self.STATS, self.FEATURES, self.add_diffs, self._prep = STATS, FEATURES, add_diffs, prep
        self.CATEGORICAL = meta.get("categorical", ["position", "venue"])
        # per-stat feature lists (assist adds `predicted`); fall back to the global set for old metas
        self._feats = meta.get("features_by_stat", {s: FEATURES for s in models})

        a = archive[archive["Season"] == season].copy()
        # placeholder current-GW rows so the dense reindex reaches M (values unused: form shift(1))
        ph = pd.DataFrame({"Season": season, "Player Name": master["Player Name"].values,
                           "Gameweek": gameweek})
        cols = ["F1 Win", "F1 Opponent Win"] + [STATS[s] for s in models]
        for c in cols:
            ph[c] = float("nan")
        ext = pd.concat([a, ph], ignore_index=True)

        self._form = {}                                              # per-stat own form at M
        for s in models:
            lut = dense(ext, STATS[s], (3,)).xs(gameweek, level="Gameweek")
            self._form[s] = lut.groupby(level="Player Name").last()  # (mean3, count3) by player
        self._win = dense(ext, "F1 Win", (3,)).xs(gameweek, level="Gameweek").groupby(level="Player Name").last()
        self._loss = dense(ext, "F1 Opponent Win", (3,)).xs(gameweek, level="Gameweek").groupby(level="Player Name").last()

    def predict_horizon(self, m, k):
        """dict {stat: Series aligned to m.index} of model predictions for fixture F{k}."""
        name = m["Player Name"]
        out = {}
        for s, model in self.models.items():
            df = pd.DataFrame(index=m.index)
            df["persistence"] = pd.to_numeric(m[self.STATS[s]], errors="coerce").values
            df["form3"] = name.map(self._form[s]["mean3"]).values
            df["form_n3"] = name.map(self._form[s]["count3"]).values
            df["form_win3"] = name.map(self._win["mean3"]).values
            df["form_loss3"] = name.map(self._loss["mean3"]).values
            df["own_title"], df["own_releg"], df["own_top6"] = m["Title"].values, m["Relegation"].values, m["Top 6"].values
            df["opp_title"] = pd.to_numeric(m[f"F{k} Opponent Title"], errors="coerce").values
            df["opp_releg"] = pd.to_numeric(m[f"F{k} Opponent Relegation"], errors="coerce").values
            df["opp_top6"] = pd.to_numeric(m[f"F{k} Opponent Top 6"], errors="coerce").values
            df["position"] = m["Position"].values
            df["venue"] = m[f"F{k} Venue"].values
            df["horizon"] = k - 1                                    # F{k} predicts M+(k-1)
            df = self.add_diffs(df)                                  # title_diff.../strength_diff/momentum
            feats = self._feats.get(s, self.FEATURES)
            if "predicted" in feats:
                # the pipeline's F{k} baseline (factor x baseline(win_pred)) is exactly the training
                # `predicted`; read it BEFORE this override replaces it (players.py sets it just above)
                base_col = f"F{k} {self.STATS[s][3:]}"               # STATS[s] = 'F1 <label>'
                df["predicted"] = pd.to_numeric(m[base_col], errors="coerce").values
            X = df[feats].copy()
            for c in feats:
                if c not in self.CATEGORICAL:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
            for c in self.CATEGORICAL:
                X[c] = X[c].astype("category")
            pred = model.predict(X)
            out[s] = pd.Series(pred, index=m.index).clip(0, 1)
        return out


def make_server(master, archive, season, gameweek, models_dir=MODELS_DIR):
    models, meta = load_models(models_dir)
    if models is None:
        return None
    return Server(master, archive, season, gameweek, models, meta)
