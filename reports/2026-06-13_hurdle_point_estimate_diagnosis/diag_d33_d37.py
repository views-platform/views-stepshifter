"""
Story A diagnostic (issue #78) — D-33 + D-37 mechanism check for the HurdleModel
point-prediction estimator. READ-ONLY measurement; modifies no production code.

Run with an env that has an editable install of this repo + darts/lightgbm, e.g.:
    /home/simon/anaconda3/envs/views_pipeline/bin/python \
        reports/2026-06-13_hurdle_point_estimate_diagnosis/diag_d33_d37.py

What it shows:
  D-33  the binary stage's predict() returns hard {0,1} CLASS LABELS, not probabilities
        (a probability IS available via predict_likelihood_parameters=True but is discarded).
  D-37  the positive stage is trained on whole series with ANY positive value, zeros
        included, so target_pos is mostly zeros -> it estimates E[Y|unit ever positive],
        not E[Y|Y>0].

The real-world MAGNITUDE of these mechanisms is assessed separately against the six
production Hurdle constituents of `chunky_bunny` (see findings.md): predicted means were
cross-checked against models/<name>/data/generated/predictions_calibration_*.parquet and
matched the companion note to 3 s.f. The aggregate verdict (MCR ~ 0.75-1.29, magnitude-honest)
is in findings.md.
"""
import numpy as np
import pandas as pd


CFG = {
    "steps": [1, 2],
    "targets": ["target"],
    "target_transform": "identity",
    "parameters": {"clf": {"n_estimators": 5}, "reg": {"n_estimators": 5}},
    "sweep": False,
    "model_clf": "LGBMClassifier",
    "model_reg": "LGBMRegressor",
}
PART = {"train": [0, 10], "test": [11, 20]}


def _df(target_by_country):
    """21 months x len(target_by_country) countries; target_by_country[c] is a callable m->value."""
    n_c = len(target_by_country)
    idx = pd.MultiIndex.from_product([range(21), range(n_c)], names=["month_id", "country_id"])
    t = np.array([target_by_country[c](m) for (m, c) in idx], dtype=float)
    return pd.DataFrame({"f1": np.arange(len(idx), dtype=float), "target": t}, index=idx)


def intermittent_df():
    # country 0: all zero; countries 1,2: zero-inflated with a few spikes (mimics conflict DGP)
    spikes1 = {3: 5.0, 7: 2.0, 12: 9.0, 18: 1.0}
    spikes2 = {2: 7.0, 9: 3.0, 15: 4.0, 20: 11.0}
    return _df({0: lambda m: 0.0, 1: lambda m: spikes1.get(m, 0.0), 2: lambda m: spikes2.get(m, 0.0)})


def clean_event_df():
    # country 0: never event; countries 1,2: always event -> classifier must emit both classes
    return _df({0: lambda m: 0.0, 1: lambda m: 5.0, 2: lambda m: 3.0})


def d37(model):
    """Zero fraction of the positive-stage training subset target_pos (reproduces fit():116-122)."""
    tt = model._target_train
    pos = [t for t in tt if (t.values() > 0).any()]
    vals = np.concatenate([t.values().ravel() for t in pos])
    print(f"[D-37] series total={len(tt)}  series-with-any-positive={len(pos)}  "
          f"target_pos points={vals.size}  ZERO-fraction={float((vals == 0).mean()):.3f}")
    print("[D-37] -> positive stage trains on mostly zeros => E[Y|ever-positive], NOT E[Y|Y>0]")


def d33(model):
    """Domain of the binary stage's predictions + default(label) vs likelihood(probability)."""
    vals = []
    for step in model._steps:
        for seq in range(5):
            vals.append(model._predict_by_step(model._models[step][0], step, seq).iloc[:, 0].to_numpy())
    vals = np.concatenate(vals)
    integral = np.allclose(vals, np.round(vals))
    print(f"[D-33] binary preds n={vals.size} unique={np.unique(np.round(vals,6))} integral={integral} "
          f"=> {'LABELS {0,1}' if integral else 'NOT pure labels'}")
    bm = model._models[1][0]
    target = [s.slice(model._train_start, model._train_end + 1)[model._targets] for s in model._series]
    default = np.unique(np.concatenate([p.values().ravel()
              for p in bm.predict(n=1, series=target, past_covariates=model._past_cov, show_warnings=False)]))
    print(f"[D-33] DEFAULT predict() (used by code) values={default}")
    try:
        proba = np.concatenate([p.values().ravel() for p in bm.predict(
            n=1, series=target, past_covariates=model._past_cov,
            predict_likelihood_parameters=True, show_warnings=False)])
        print(f"[D-33] predict_likelihood_parameters=True (AVAILABLE, discarded by code): "
              f"min={proba.min():.4f} max={proba.max():.4f}")
    except Exception as e:  # pragma: no cover - darts-version dependent
        print(f"[D-33] likelihood path: {type(e).__name__}: {e}")


def main():
    from views_stepshifter.models.hurdle_model import HurdleModel

    print("== D-37 on a zero-inflated intermittent fixture ==")
    di = intermittent_df()
    print(f"[fixture] target zero-fraction={float((di['target'] == 0).mean()):.3f}")
    mi = HurdleModel(CFG, PART)
    mi.fit(di)
    d37(mi)

    print("\n== D-33 on a clean two-class fixture ==")
    mc = HurdleModel(CFG, PART)
    mc.fit(clean_event_df())
    d33(mc)


if __name__ == "__main__":
    main()
