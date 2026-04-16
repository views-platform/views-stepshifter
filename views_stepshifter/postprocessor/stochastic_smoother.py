from typing import Dict
from views_pipeline_core.managers.postprocessor.registry import register_postprocessor
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class StochasticSmootherConfig(BaseModel):
    option: int
    samples: int

class StochasticSmoother:
    def __init__(self, post_config: Dict, config: Dict):
        """Initialize smoother options and precompute the smoothing matrix."""

        self.postprocessor_config = StochasticSmootherConfig(**post_config)
        self.option = self.postprocessor_config.option
        self.samples = self.postprocessor_config.samples

        self.steps = config.get("time_steps")
        self.targets = config.get("regression_targets")
        self.smoothing_matrix = self._initialize_smoothing_matrix()

    def _initialize_smoothing_matrix(self):
        """Create and scale the stochastic smoothing weight matrix."""
        ## Change to dynamic window later
        if self.steps < 10:
            raise ValueError(f"Current implementation only supports steps >= 10. Got {self.steps}.")

        smoothing_matrix = np.zeros((self.steps, self.steps))

        if self.option == 0:
            smoothing_matrix[0, 0:4] = [550, 250, 150, 50]
            smoothing_matrix[1, 0:5] = [200, 450, 200, 100, 50]
            smoothing_matrix[2, 0:6] = [50, 200, 400, 200, 100, 50]
            for i in range(3, self.steps-3):
                smoothing_matrix[i, i-3:i+4] = [50, 100, 200, 300, 200, 100, 50]
            smoothing_matrix[self.steps-3, self.steps-7:self.steps] = [50, 100, 200, 300, 200, 100, 50]
            smoothing_matrix[self.steps-2, self.steps-6:self.steps] = [50, 100, 150, 200, 300, 200]
            smoothing_matrix[self.steps-1, self.steps-5:self.steps] = [50, 100, 150, 250, 450]
        
        elif self.option == 1:
            smoothing_matrix[0, 0:5] = [550, 200, 125, 75, 50]
            smoothing_matrix[1, 0:6] = [200, 350, 175, 125, 100, 50]
            smoothing_matrix[2, 0:8] = [100, 200, 325, 125, 100, 75, 50, 25]
            smoothing_matrix[3, 0:8] = [50, 100, 175, 325, 150, 100, 75, 25]
            for i in range(4, self.steps-4):
                smoothing_matrix[i, i-4:i+5] = [25, 75, 100, 150, 300, 150, 100, 75, 25]
            smoothing_matrix[self.steps-4, self.steps-9:self.steps] = [25, 75, 100, 150, 300, 150, 100, 75, 25]
            smoothing_matrix[self.steps-3, self.steps-8:self.steps] = [25, 50, 100, 125, 300, 150, 150, 100]
            smoothing_matrix[self.steps-2, self.steps-7:self.steps] = [25, 50, 75, 125, 175, 325, 225]
            smoothing_matrix[self.steps-1, self.steps-6:self.steps] = [25, 50, 100, 125, 275, 425]
        
        else:
            raise ValueError(f"Invalid option: {self.option}")
        
        smoothing_matrix = (smoothing_matrix * (self.samples / 1000)).astype(int)
        return smoothing_matrix
        
    def _prepare_transform_axes(self, df: pd.DataFrame):
        """Extract months/countries axes and validate step coverage."""
        months = df.index.get_level_values(0).unique().sort_values()
        countries = df.index.get_level_values(1).unique()
        if len(months) > self.steps:
            raise ValueError(f"Found {len(months)} unique months but time_steps is {self.steps}.")
        return months, countries

    def _get_source_values(self, df: pd.DataFrame, from_months, country, pred_col):
        """Return source values for a source (month, country) cell if present."""
        key = (from_months, country)
        if key not in df.index:
            return None
        return df.loc[key, pred_col]

    def _transform(self, df: pd.DataFrame):
        """Build smoothed prediction columns by weighted stochastic resampling."""
        smoothed_df = df.copy()
        months, countries = self._prepare_transform_axes(smoothed_df)

        for target in self.targets:
            pred_col = f"pred_{target}"
            smoothed_col = f"smoothed_{pred_col}"
            smoothed_df[smoothed_col] = None

            for to_step, to_months in enumerate(months):
                row_weights = self.smoothing_matrix[to_step, :]
                for country in countries:
                    sampled_predictions = []
                    
                    for from_step, weight in enumerate(row_weights):
                        if weight <= 0:
                            continue
                        from_months = to_months - to_step + from_step
                        source_values = self._get_source_values(smoothed_df, from_months, country, pred_col)
                        if source_values is None:
                            continue
                        samples = np.random.choice(source_values, size=weight, replace=True)
                        sampled_predictions.extend(samples)
                    smoothed_df.at[(to_months, country), smoothed_col] = sampled_predictions

        return smoothed_df

    def _validate(self, smoothed_df: pd.DataFrame):
        """Fail if any target's smoothed prediction column is entirely empty."""
        for target in self.targets:
            if smoothed_df[f"smoothed_pred_{target}"].isnull().all():
                raise ValueError(f"Transformation resulted in empty smoothed_pred_{target}.")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing transform and validate resulting prediction columns."""
        smoothed_df = self._transform(df)
        self._validate(smoothed_df)
        return smoothed_df

register_postprocessor("stochastic_smoother", StochasticSmoother)



    