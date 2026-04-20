from typing import Dict, Optional
from views_pipeline_core.managers.postprocessor.registry import register_postprocessor
from views_pipeline_core.managers.postprocessor.base import BasePostprocessor
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class StochasticSmootherConfig(BaseModel):
    option: int
    seed: Optional[int] = None


class StochasticSmoother(BasePostprocessor):
    def __init__(self, post_config: Dict, config: Dict):
        """Initialize smoother options and precompute the smoothing matrix."""
        self.postprocessor_config = StochasticSmootherConfig(**post_config)
        self.option = self.postprocessor_config.option
        self._rng = np.random.default_rng(self.postprocessor_config.seed)

        self.samples = config.get("pred_samples") * config.get("submodels_to_train")
        self.steps = config.get("time_steps")
        self.targets = config.get("regression_targets")
        self.smoothing_matrix = self._initialize_smoothing_matrix()

    def _initialize_smoothing_matrix(self) -> np.ndarray:
        """Create the stochastic smoothing weight matrix.

        Rows are destination steps (0-indexed), columns are source steps.
        Each row's non-zero weights specify which source steps contribute draws
        and in what proportion.  Weights are stored as raw floats; integer
        allocation is deferred to _allocate_counts, which normalises before
        scaling, so the absolute magnitudes do not affect the final distribution.

        Interior rows use a symmetric window centred on the diagonal. Boundary
        rows truncate the window on the right so the peak stays on the diagonal.
        """
        if self.steps < 10:
            raise ValueError(
                f"Current implementation only supports steps >= 10. Got {self.steps}."
            )

        smoothing_matrix = np.zeros((self.steps, self.steps), dtype=float)

        if self.option == 0:
            smoothing_matrix[0, 0:4] = [550, 250, 150, 50]
            smoothing_matrix[1, 0:5] = [200, 450, 200, 100, 50]
            smoothing_matrix[2, 0:6] = [50, 200, 400, 200, 100, 50]
            for i in range(3, self.steps-3):
                smoothing_matrix[i, i-3:i+4] = [50, 100, 200, 300, 200, 100, 50]
            smoothing_matrix[self.steps-3, self.steps-7:self.steps] = [50, 100, 200, 300, 200, 100, 50] # This looks weird to me, the peak is not on step - 3
            smoothing_matrix[self.steps-2, self.steps-6:self.steps] = [50, 100, 150, 200, 300, 200]
            smoothing_matrix[self.steps-1, self.steps-5:self.steps] = [50, 100, 150, 250, 450]
        
        elif self.option == 1:
            smoothing_matrix[0, 0:5] = [550, 200, 125, 75, 50]
            smoothing_matrix[1, 0:6] = [200, 350, 175, 125, 100, 50]
            smoothing_matrix[2, 0:8] = [100, 200, 325, 125, 100, 75, 50, 25]
            smoothing_matrix[3, 0:8] = [50, 100, 175, 325, 150, 100, 75, 25]
            for i in range(4, self.steps-4):
                smoothing_matrix[i, i-4:i+5] = [25, 75, 100, 150, 300, 150, 100, 75, 25]
            smoothing_matrix[self.steps-4, self.steps-9:self.steps] = [25, 75, 100, 150, 300, 150, 100, 75, 25] # Same as above
            smoothing_matrix[self.steps-3, self.steps-8:self.steps] = [25, 50, 100, 125, 300, 150, 150, 100] # Same as above
            smoothing_matrix[self.steps-2, self.steps-7:self.steps] = [25, 50, 75, 125, 175, 325, 225]
            smoothing_matrix[self.steps-1, self.steps-6:self.steps] = [25, 50, 100, 125, 275, 425]
        
        else:
            raise ValueError(f"Invalid option: {self.option}")

        return smoothing_matrix

    def _allocate_counts(self, raw_weights: np.ndarray, total: int) -> np.ndarray:
        """
        Allocate integer draw counts from weights so counts sum exactly to total.
        """
        if total <= 0 or raw_weights.sum() <= 0:
            return np.zeros_like(raw_weights, dtype=int)

        normalized = raw_weights / raw_weights.sum()
        scaled = normalized * total
        counts = np.floor(scaled).astype(int)
        remainder = total - counts.sum()
        if remainder > 0:
            fractional = scaled - counts
            order = np.argsort(-fractional)
            counts[order[:remainder]] += 1
        return counts

    def _prepare_transform_axes(self, df: pd.DataFrame):
        """Extract months/countries axes and validate step coverage."""
        months = df.index.get_level_values(0).unique().sort_values()
        countries = df.index.get_level_values(1).unique()
        if len(months) > self.steps:
            raise ValueError(
                f"Found {len(months)} unique months but time_steps is {self.steps}."
            )
        return months, countries

    def _build_source_cache(self, df: pd.DataFrame, pred_col: str) -> Dict:
        """Build fast (month, country) -> ndarray lookup for one prediction column."""
        series = df[pred_col]
        cache = {}
        for key, values in series.items():
            arr = np.asarray(values)
            if arr.size > 0:
                cache[key] = arr
        return cache

    def _sample_for_cell(
        self,
        source_cache: Dict,
        to_step: int,
        to_months,
        country,
        row_weights: np.ndarray,
    ) -> list:
        """Draw self.samples predictions for cell (to_months, country).

        Missing source months are skipped and their weight redistributed.
        Returns an empty list if no source months are available.
        """
        available_values = []
        available_weights = []

        for from_step, weight in enumerate(row_weights):
            if weight <= 0:
                continue
            src_month = to_months - to_step + from_step
            source_arr = source_cache.get((src_month, country))
            if source_arr is None:
                logger.warning(
                    f"No source values found for month={src_month} country={country}"
                )
                continue
            available_values.append(source_arr)
            available_weights.append(weight)

        if not available_values:
            return []

        counts = self._allocate_counts(
            np.asarray(available_weights, dtype=float), self.samples
        )
        sampled_predictions = []
        for values, count in zip(available_values, counts):
            sampled_predictions.extend(self._rng.choice(values, size=count, replace=True))
        return sampled_predictions

    def _transform(self, df: pd.DataFrame):
        """Build smoothed prediction columns by weighted stochastic resampling."""
        missing = [f"pred_{t}" for t in self.targets if f"pred_{t}" not in df.columns]
        if missing:
            raise KeyError(f"Prediction column(s) not found in dataframe: {missing}")

        smoothed_df = df.copy()
        months, countries = self._prepare_transform_axes(smoothed_df)

        for target in self.targets:
            pred_col = f"pred_{target}"
            source_cache = self._build_source_cache(smoothed_df, pred_col)

            for to_step, to_months in enumerate(months):
                row_weights = self.smoothing_matrix[to_step, :]
                for country in countries:
                    sampled_predictions = self._sample_for_cell(
                        source_cache,
                        to_step,
                        to_months,
                        country,
                        row_weights,
                    )
                    smoothed_df.at[(to_months, country), pred_col] = sampled_predictions

        return smoothed_df

    def _validate(self, smoothed_df: pd.DataFrame):
        """Validate smoothed prediction columns for completeness and sample count."""
        for target in self.targets:
            col = f"pred_{target}"
            series = smoothed_df[col]

            if series.isnull().all():
                raise ValueError(f"Transformation resulted in entirely null {col}.")

            empty_mask = series.apply(
                lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 0
            )
            if empty_mask.any():
                logger.warning(
                    f"{empty_mask.sum()} cell(s) in {col} have empty sample lists "
                    "(likely missing source data for those month/country combinations)."
                )

            lengths = series.dropna().apply(
                lambda v: len(v) if isinstance(v, (list, np.ndarray)) else None
            ).dropna()
            wrong_count = (lengths != self.samples).sum()
            if wrong_count:
                logger.warning(
                    f"{wrong_count} cell(s) in {col} have a sample count other than "
                    f"{self.samples} (source data was partially missing)."
                )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing transform and validate resulting prediction columns."""
        smoothed_df = self._transform(df)
        self._validate(smoothed_df)
        return smoothed_df


register_postprocessor("stochastic_smoother", StochasticSmoother)
