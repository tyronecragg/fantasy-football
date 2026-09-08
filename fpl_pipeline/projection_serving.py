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
import numpy as np
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

    def __init__(self, master, archive, season, gameweek, models, meta, rating=None):
        self.models, self.meta, self.rating = models, meta, rating
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

        self.windows = (3, 6, 10)
        self._form = {}                                              # per-stat own form at M
        for s in models:
            lut = dense(ext, STATS[s], self.windows).xs(gameweek, level="Gameweek")
            self._form[s] = lut.groupby(level="Player Name").last()  # (mean{w}, count{w}) by player
        self._win = dense(ext, "F1 Win", self.windows).xs(gameweek, level="Gameweek").groupby(level="Player Name").last()
        self._loss = dense(ext, "F1 Opponent Win", self.windows).xs(gameweek, level="Gameweek").groupby(level="Player Name").last()
        # real archived weeks of this season before M - gates the stacks (config.PROJECTION_STACK_MIN_WEEKS)
        gws = pd.to_numeric(a["Gameweek"], errors="coerce")
        self.weeks = int(gws[gws < gameweek].nunique())

    def predict_horizon(self, m, k):
        """dict {stat: Series aligned to m.index} of model predictions for fixture F{k}."""
        name = m["Player Name"]
        out = {}
        for s, model in self.models.items():
            df = pd.DataFrame(index=m.index)
            df["persistence"] = pd.to_numeric(m[self.STATS[s]], errors="coerce").values
            for w in self.windows:
                df[f"form{w}"] = name.map(self._form[s][f"mean{w}"]).values
                df[f"form_n{w}"] = name.map(self._form[s][f"count{w}"]).values
                df[f"form_win{w}"] = name.map(self._win[f"mean{w}"]).values
                df[f"form_loss{w}"] = name.map(self._loss[f"mean{w}"]).values
            df["own_title"], df["own_releg"], df["own_top6"] = m["Title"].values, m["Relegation"].values, m["Top 6"].values
            df["opp_title"] = pd.to_numeric(m[f"F{k} Opponent Title"], errors="coerce").values
            df["opp_releg"] = pd.to_numeric(m[f"F{k} Opponent Relegation"], errors="coerce").values
            df["opp_top6"] = pd.to_numeric(m[f"F{k} Opponent Top 6"], errors="coerce").values
            df["position"] = m["Position"].values
            df["venue"] = m[f"F{k} Venue"].values
            df["horizon"] = k - 1                                    # F{k} predicts M+(k-1)
            df = self.add_diffs(df)                                  # title_diff.../strength_diff/momentum
            feats = self._feats.get(s, self.FEATURES)
            if "rating_win" in feats:
                # the rolling rating's win / opponent-win for the F{k} fixture (fpl_pipeline/rating.py)
                # - the identical call the training builder makes (rating_backtest.rating_win_fn)
                if self.rating is not None:
                    rw, ro = self.rating.win_probs(m["Team"], m[f"F{k} Opponent"],
                                                   m[f"F{k} Venue"] == "H", k - 1)
                    df["rating_win"], df["rating_opp"] = rw.values, ro.values
                    if "rating_xg_for" in feats:
                        xf, xa = self.rating.expected_goals(m["Team"], m[f"F{k} Opponent"],
                                                            m[f"F{k} Venue"] == "H", k - 1)
                        df["rating_xg_for"], df["rating_xg_against"] = xf.values, xa.values
                else:                                            # no rating: the master's Lasso fallback
                    wc = f"F{k} Win" if f"F{k} Win" in m else f"F{k} Win Pred"
                    oc = f"F{k} Opponent Win" if f"F{k} Opponent Win" in m else f"F{k} Opponent Win Pred"
                    df["rating_win"] = pd.to_numeric(m[wc], errors="coerce").values
                    df["rating_opp"] = pd.to_numeric(m[oc], errors="coerce").values
                    if "rating_xg_for" in feats:
                        df["rating_xg_for"] = df["rating_xg_against"] = np.nan
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


def make_server(master, archive, season, gameweek, models_dir=MODELS_DIR, rating=None):
    models, meta = load_models(models_dir)
    if models is None:
        return None
    return Server(master, archive, season, gameweek, models, meta, rating=rating)
