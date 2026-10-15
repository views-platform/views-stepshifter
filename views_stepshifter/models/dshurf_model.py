from __future__ import annotations
import os

from views_pipeline_core.managers.model import ForecastingModelManager
from views_stepshifter.models import StepshifterModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from typing import List, Dict, Literal, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr

if TYPE_CHECKING:
    # Type-only imports for the PREDICTION_WRAPPERS annotations.
    from quantile_forest import RandomForestQuantileRegressor
    from views_stepshifter.models.distributional.drf import (
        DistributionalRandomForestRegressor,
    )
    from views_stepshifter.models.distributional.ngboost import NGBRegressorWrapper


logger = logging.getLogger(__name__)


def qrf_predict(qrf: RandomForestQuantileRegressor, X, sample_size:int) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X), 
    sample_size) for Quantile Random Forest."""
    y_hat = qrf.predict(
        X, quantiles=list(np.linspace(0, 1, sample_size)), interpolation="linear"
    ).round(0)
    return y_hat


def drf_predict(drf: DistributionalRandomForestRegressor, X, sample_size:int) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X), 
    sample_size) for Distributional Random Forest."""
    y_hat = drf.predict(X, functional="sample", n_samples=sample_size)
    y_hat = y_hat.squeeze() # type: ignore
    return y_hat


def ngb_predict(ngb: NGBRegressorWrapper, X, sample_size:int) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X),
    sample_size) for NGBoost."""
    y_hat = ngb.predict(X, sample_size=sample_size)
    # predicted output includes 0 - replace with 1 in line with the non-zero prediction logic
    # expected by the hurdle combination strategies.
    y_hat[y_hat < 1] = 1
    return y_hat

# unify predict signature across different quantile regressors
PREDICTION_WRAPPERS = {
    "NGBoostRegressor": ngb_predict,
    "RandomForestQuantileRegressor": qrf_predict,
    "DistributionalRandomForestRegressor": drf_predict,
}

def _parallel_worker(model_instance, seq, target):
    """Standalone worker function that ProcessPoolExecutor can easily serialize."""
    # print(f"[Core PID {os.getpid()}] -> Claimed Target: '{target}' | Sequence: {seq}...", flush=True)
    # sys.stdout.flush()
    
    result = model_instance.predict_sequence(seq, target=target)
    
    # print(f"[Core PID {os.getpid()}] -> COMPLETED Target: '{target}' | Sequence: {seq}", flush=True)
    # sys.stdout.flush()
    return result


class DshurfModel(StepshifterModel):
    """ Two-stage distributional hurdle model: a native tabular binary
    classifier for P(target > 0) and a natively distributional positive-stage
    regressor, combined per the configured ``hurdle_strategy``."""

    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]]):
        super().__init__(config, partitioner_dict)
        if self._target_transform_name != "identity":
            raise ValueError(
                f"{self.__class__.__name__} requires target_transform='identity'; "
                f"got '{self._target_transform_name}'"
            )
        params = self._get_parameters(config)
        self._clf_params = params["clf"]
        self._reg_params = params["reg"]

        # Unused: kept as a declared attribute for future reference. The
        # legacy log-space sampler stays off (D-27, gate-enforced).
        self._log_target = config["log_target"]
        self._pred_samples = config["pred_samples"]
        self._hurdle_strategy = config["hurdle_strategy"]
        self.rng = np.random.default_rng()

    def _inverse_transform_predictions(self, preds):
        """Inverse target-space transform for Shurf's list-valued predictions."""
        frames = preds if isinstance(preds, list) else [preds]
        for frame in frames:
            for target in self._target_names:
                col = f"pred_{target}"
                if col not in frame.columns:
                    continue
                frame[col] = frame[col].apply(
                    lambda v: (
                        self._inverse(np.asarray(v)).tolist()
                        if isinstance(v, (list, tuple, np.ndarray))
                        else self._inverse(v)
                    )
                )
        return preds
    
    def _resolve_clf_model(self, func_name: str):
        """Lookup table for supported classification models"""

        match func_name:
            case "XGBClassifier":
                from xgboost import XGBClassifier
                return XGBClassifier
            case "XGBRFClassifier":
                from xgboost import XGBRFClassifier
                return XGBRFClassifier
            case "LGBMClassifier":
                from lightgbm.sklearn import LGBMClassifier
                return LGBMClassifier
            case _:
                raise ValueError(
                    f"Model {func_name} is not a valid classifier model or is not supported now. "
                    f"Change the model in the config file."
                )
    
    def _resolve_distreg_model(self, func_name: str):
        """
        Lookup table for supported distributional regression models.

        The imports are lazy per branch (mirroring PREDICTION_WRAPPERS' key
        set): a backend module is only imported when a config actually
        selects it, at fit time.
        """

        match func_name:
            case "NGBoostRegressor":
                from views_stepshifter.models.distributional.ngboost import NGBRegressorWrapper
                return NGBRegressorWrapper
            case "RandomForestQuantileRegressor":
                from quantile_forest import RandomForestQuantileRegressor
                return RandomForestQuantileRegressor
            case "DistributionalRandomForestRegressor":
                from views_stepshifter.models.distributional.drf import (
                    DistributionalRandomForestRegressor,
                )
                return DistributionalRandomForestRegressor
            case _:
                raise ValueError(
                    f"Model {func_name} is not a valid forecasting model or is not supported now. "
                    f"Change the model in the config file."
                )

    def _prepare_X_y_step(
        self, step: int, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Horizon-aligned training matrix for one step, shared by both stages.

        X = the covariates at month ``t - step`` per unit (the only lag the
        stepshifter design uses), y = the target at month ``t``.

        Order of operations, explicit and load-bearing:
        1. shift the FULL processed panel's covariates by ``step`` — a label
           month ``t`` pairs with the covariate snapshot at calendar month
           ``t - step``, whatever the panel holds there;
        2. mask label rows to the train window ``[_train_start, _train_end]``
           and cut the first ``step`` panel months (no feature exists
           ``step`` months before the panel start);
        3. drop rows whose feature snapshot contains NaN — the estimators
           cannot train on incomplete covariates, and the paired label
           rows go with them to keep X and y row-aligned.

        Stage-specific label derivation (binary ``y > 0`` for the
        classifier, positive-row filtering for the regressor) deliberately
        lives in ``fit``, never here: it selects *label rows*, never
        covariate rows. Filtering the frame before shifting would instead
        pair a label month ``t`` with the covariates of the step-th
        previous *surviving* row.

        The shift assumes the panel rectangle guaranteed by
        ``_process_data`` (every unit present in every month), so a
        per-unit positional shift equals a calendar-month shift.

        Args:
            step: the forecast horizon (covariate lag) to shift by.
            target: the target column for y.

        Returns:
            (X, y): feature frame and label series, row-aligned on the
            training-window label months.
        """
        panel = self._df_processed

        X = panel[self._independent_variables].groupby(
            level=self._level, sort=False
        ).shift(step)
        y = panel[target]

        t = panel.index.get_level_values(self._time)
        month_first = t.min()
        keep = (
            (t >= month_first + step)
            & (t >= self._train_start)
            & (t <= self._train_end)
        )
        X_step, y_step = X.loc[keep], y.loc[keep]

        na_free = X_step.notna().all(axis=1)
        if not na_free.all():
            n_dropped = int((~na_free).sum())
            logger.info(
                f"Step {step}, target '{target}': dropped {n_dropped} of "
                f"{len(X_step)} training rows with NaN features."
            )
            X_step, y_step = X_step.loc[na_free], y_step.loc[na_free]
        if X_step.empty:
            raise ValueError(
                f"Step {step}, target '{target}': every training row has "
                f"NaN features — no complete rows remain to fit on."
            )
        return X_step, y_step

    @views_validate
    def fit(self, df: pd.DataFrame):
        """
        Train the binary classifier and distributional regressor per target
        and step.

        Both stages share the horizon-aligned training matrix from
        ``_prepare_X_y_step`` (features at ``t - step``, label at ``t``):
        the classifier is fit on ``1[y > 0]`` over all aligned rows; the
        distributional regressor is fit on the positive-label rows only.
        Trained ``(classifier, regressor)`` pairs are stored in
        ``_models_by_target[target][step]`` for ``predict_sequence``.
        """
        df = self._process_data(df)
        self._df_processed = df
        self._clf = self._resolve_clf_model(self._config["model_clf"])
        self._reg = self._resolve_distreg_model(self._config["model_reg"])

        self._models_by_target = {}

        for target in self._target_names:
            hurdle_component_models = {}
            for (
                step
            ) in self._steps:
                X_step, y_step = self._prepare_X_y_step(step, target)

                clf_model = self._clf(**self._clf_params)
                clf_model.fit(X_step, (y_step > 0).astype(int))

                positive = y_step > 0
                if not positive.any(): # TODO: add a threshold rather than any
                    raise ValueError(
                        f"No positive training observations for target "
                        f"'{target}' at step {step}; the positive stage "
                        f"cannot be fit."
                    )
                positive_model = self._reg(**self._reg_params)
                positive_model.fit(X_step.loc[positive], y_step.loc[positive])
                hurdle_component_models[step] = (clf_model, positive_model)

            self._models_by_target[target] = hurdle_component_models

        if len(self._target_names) == 1:
            self._models = self._models_by_target[self._targets]

        self.is_fitted_ = True

    def _predict_by_step_sklike(
        self,
        model,
        step: int,
        sequence_number: int,
        hurdle_step: Literal["clf", "reg"],
        target: Optional[str] = None,
        X: Optional[pd.DataFrame] = None,
    ) -> xr.DataArray:
        """Unified per-step prediction for one hurdle stage.

        Diagonal: for ``sequence_number`` ``s`` the rolling origin is month
        ``train_end + s``; the covariate snapshot at the origin is the
        feature matrix, and the output lands at ``origin + step`` — the
        same month for both stages.

        - ``hurdle_step="clf"``: the native classifier's
          ``predict_proba(X)[:, 1]`` (binary setting: class 1 is the event
          probability P(target > 0)), dims ``(month_id, level)``.
        - ``hurdle_step="reg"``: the distributional regressor's
          ``pred_samples`` draws, dims ``(month_id, level, draws)``.

        Args:
            model: the fitted stage model for this step.
            step: the forecast horizon.
            sequence_number: the rolling-origin sequence index.
            hurdle_step: which stage to predict, ``"clf"`` or ``"reg"``.
            target: the target the output is named after; defaults to the
                single target.
            X: optional pre-extracted feature snapshot at the origin, so
                ``predict_sequence`` can share one extraction between both
                stages.

        Returns:
            An ``xr.DataArray`` named ``pred_<target>``.
        """
        target = target or self._targets
        feature_month = self._train_end + sequence_number
        output_month = feature_month + step

        if X is None:
            X = self._df_processed.xs(
                feature_month, level=self._time, drop_level=False
            )[self._independent_variables]

        if hurdle_step == "reg":
            preds_raw = np.asarray(
                PREDICTION_WRAPPERS[self._config["model_reg"]](model, X, self._pred_samples)
            )
            if preds_raw.shape != (len(X), self._pred_samples):
                raise ValueError(
                    f"Distributional regressor for step {step} returned "
                    f"draws with shape {preds_raw.shape}; expected "
                    f"{(len(X), self._pred_samples)}."
                )

            preds = xr.DataArray(
                data=preds_raw[np.newaxis, :, :],
                dims=[self._time, self._level, "draws"],
                coords={
                    self._time: [output_month],
                    self._level: X.index.get_level_values(self._level).to_numpy(),
                    "draws": np.arange(self._pred_samples),
                },
                name=f"pred_{target}",
            )
        elif hurdle_step == "clf":
            preds_raw = np.asarray(model.predict_proba(X)[:, 1])

            preds = xr.DataArray(
                data=preds_raw[np.newaxis, :],
                dims=[self._time, self._level],
                coords={
                    self._time: [output_month],
                    self._level: X.index.get_level_values(self._level).to_numpy(),
                },
                name=f"pred_{target}",
            )
        else:
            raise ValueError(
                f'Argument <hurdle_step> param needs to be one of ["reg", "clf"], '
                f"got {hurdle_step}."
            )
        return preds

    def _hurdle_combination(
        self,
        pred_binary: xr.DataArray,
        pred_positive: xr.DataArray,
        target: Optional[str] = None,
    ) -> pd.DataFrame:
        """Combine the binary and positive stages into the per-sequence output.

        ``xr.apply_ufunc`` maps the per-unit hurdle combination over the
        (month, unit) coordinates with ``draws`` as the core dimension:
        one scalar event probability per coordinate, paired with the
        unit's distributional draw vector. Strategy dispatch on the
        configured ``hurdle_strategy``:

        - ``"probabilistic"``: each regression draw survives with
          probability ``p`` (uniform draw > ``p`` zeroes it) — ``p`` sets
          the zero share in expectation.
        - ``"shares"``: a fixed share of ``round(p * n_draws)`` draws is
          kept (sampled without replacement), the rest zeroed.

        The legacy ``"shurf"`` combination is currently disabled (dormant
        branch retained below); the gate rejects it at config-audit time.
        Each unit's result is cast to int and clipped to ``[0, None)``.

        This class controls the whole xarray workflow — both stage arrays
        are assembled by ``predict_sequence`` from one shared feature
        snapshot, so their (month, unit) coordinates are identical by
        construction; ``apply_ufunc`` matches by dimension name, so no
        transpose or positional pairing is needed. The per-draw sample
        count is already checked at the estimator boundary in
        ``_predict_by_step``.

        Args:
            pred_binary: the binary stage — event probabilities,
                dims (month_id, level).
            pred_positive: the positive stage draws,
                dims (month_id, level, draws).
            target: the target the output column is named after; defaults
                to the single target.

        Returns:
            A DataFrame indexed by (month_id, level), one column
            ``pred_<target>``, each cell the unit's combined draws as a plain
            list (uniform length across all cells).
        """
        strategy = self._hurdle_strategy
        rng = self.rng
        
        HURDLE_STRATEGIES = {"probabilistic", "shares"}
        if strategy not in HURDLE_STRATEGIES:
            raise ValueError(
                f"Unknown hurdle_strategy '{strategy}'. "
                f"Available: {HURDLE_STRATEGIES}"
            )


        def hurdle_draws(p: float, draws: np.ndarray) -> np.ndarray:
            p = float(np.clip(p, 0.0, 1.0))
            sample_size = len(draws)
            if strategy == "probabilistic":
                mask = rng.uniform(size=sample_size) > p
                combined = draws.copy()
                combined[mask] = 0
            else:  # shares
                n_non_zero = round(p * sample_size)
                n_zero = sample_size - n_non_zero
                zeroes = np.zeros(n_zero)
                non_zeroes = rng.choice(draws, size=n_non_zero, replace=False)
                combined = np.concatenate([zeroes, non_zeroes])
            # elif strategy == "shurf":  # TODO: re-enable and fix check
            #     clf_draws = rng.binomial(1, p, size=sample_size)
            #     combined = clf_draws * draws
            return np.clip(combined.astype(int), 0, None)

        combined = xr.apply_ufunc(
            hurdle_draws,
            pred_binary,
            pred_positive,
            input_core_dims=[(), ("draws",)],
            output_core_dims=[("draws",)],
            output_dtypes=[int],
            vectorize=True,
        )

        # stack appends the stacked dim at the end; transpose to
        # (cell, draws) so values rows pair with the cell MultiIndex.
        stacked = combined.stack(cell=(self._time, self._level)).transpose(
            "cell", "draws"
        )
        return pd.DataFrame(
            {f"pred_{target}": [row.tolist() for row in stacked.values]},
            index=stacked.indexes["cell"].set_names([self._time, self._level]),
        ).sort_index()

    def predict_sequence(self, sequence_number, target=None) -> pd.DataFrame:
        """
        Predict one evaluation sequence: hurdle-combined draws for every step.

        The rolling origin is month ``train_end + sequence_number``; each
        step's (classifier, regressor) pair reads the same covariate
        snapshot at the origin and emits its stage output for month
        ``origin + step``.

        Args:
            sequence_number: the rolling-origin sequence index.
            target: the target to predict; defaults to the single target.

        Returns:
            A DataFrame indexed by (month_id, level), one column
            ``pred_<target>``, each cell a list of ``pred_samples`` draws.
        """

        target = target or self._targets
        stepmodels = self._models_by_target[target]

        feature_month = self._train_end + sequence_number
        X = self._df_processed.xs(
            feature_month, level=self._time, drop_level=False
        )[self._independent_variables]

        pred_by_step_proba = []
        pred_by_step_positive = []
        for step in self._steps:
            pred_by_step_proba.append(
                self._predict_by_step_sklike(
                    stepmodels[step][0],
                    step,
                    sequence_number,
                    "clf",
                    target=target,
                    X=X,
                )
            )
            pred_by_step_positive.append(
                self._predict_by_step_sklike(
                    stepmodels[step][1],
                    step,
                    sequence_number,
                    "reg",
                    target=target,
                    X=X,
                )
            )

        # Simple concat along the time dim: the per-step arrays share their
        # (level, draws) coordinates by construction and the class controls
        # the whole workflow, so no coordinate inspection is needed. Step
        # order becomes time order; the output frame is sorted in
        # _hurdle_combination.
        pred_concat_proba = xr.concat(pred_by_step_proba, dim=self._time)
        pred_concat_positive = xr.concat(pred_by_step_positive, dim=self._time)

        # NOTE: log_target=True is gate-rejected for DshurfModel (D-27); the
        # hurdle combination operates entirely in raw target space.
        return self._hurdle_combination(
            pred_concat_proba, pred_concat_positive, target=target
        )

    def predict(self, run_type: str, eval_type: str = "standard") -> pd.DataFrame:
        """
        Predicts outcomes based on the run type.

        If 'forecasting', a single prediction is made for sequence 0.
        Otherwise, one prediction per evaluation sequence is made.

        Parameters:
        -----------
        run_type : str
            The type of run to perform. If 'forecasting', a single prediction is made.
            Otherwise, multiple predictions are made based on the evaluation sequence number.
        eval_type : str, optional
            The type of evaluation to perform. Default is "standard".

        Returns:
        --------
        pd.DataFrame or list[pd.DataFrame]
            The final predictions; a single frame for forecasting, one frame
            per evaluation sequence otherwise.
        """
        check_is_fitted(self, "is_fitted_")

        if run_type != "forecasting":
            if eval_type != "standard":
                raise ValueError(
                    f"{eval_type} is not supported now. Please use 'standard' evaluation type."
                )

            total_sequence_number = (
                ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
            )

            tasks = [
                (seq, target)
                for seq in range(total_sequence_number)
                for target in self._target_names
            ]
            per_target_preds = {
                target: [None] * total_sequence_number for target in self._target_names
            }
            with ProcessPoolExecutor(
                max_workers=min(len(tasks), os.cpu_count()),  # temporary, for easier debugging
                # "spawn" is required, not an optimization: the default
                # "fork" context deadlocks when a forked child calls a
                # fitted XGBoost estimator (OpenMP thread pool state does
                # not survive fork). LightGBM is unaffected, which is why
                # the Hurdle/Shurf pools work with the default context.
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(_parallel_worker, self, seq, target): (seq, target)
                    for seq, target in tasks
                }

                for future in tqdm(
                    as_completed(futures),
                    desc=f"Predicting {total_sequence_number} sequences x {len(self._target_names)} targets",
                    total=len(futures),
                    leave=True,
                ):
                    seq, target = futures[future]
                    try:
                        output_df = future.result()
                        per_target_preds[target][seq] = output_df
                    except Exception as exc:
                        logger.error(
                            f"Sequence {seq} for target {target} generated an exception: {exc}"
                        )
                        raise exc

            preds = []
            for sequence_number in range(total_sequence_number):
                frames = [
                    per_target_preds[target][sequence_number]
                    for target in self._target_names
                ]
                preds.append(pd.concat(frames, axis=1).sort_index())
            return self._inverse_transform_predictions(preds)

        sequence_number = 0
        frames = [
            self.predict_sequence(sequence_number, target=target)
            for target in self._target_names
        ]
        preds = pd.concat(frames, axis=1).sort_index()
        return self._inverse_transform_predictions(preds)
