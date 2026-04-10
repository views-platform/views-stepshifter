from typing import Dict
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.postprocessor import PostprocessorManager
import pandas as pd
import numpy as np
from typing import Optional
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StochasticSmoother(PostprocessorManager):

    def __init__(self, path_manager: ModelPathManager, config: Dict):
        super().__init__(path_manager, config)
        self.steps = len(config.get("steps", [*range(1, 36 + 1, 1)]))
        self.option = config.get("option", 0)
        self.samples = config.get("samples", 1000)
        self.target = config.get("targets")[0] # Currently only supports one target
        self.pred_col = f"pred_{self.target}"
        
        self.df = None
        self.smoothed_df = None
        self.smoothing_matrix = self._initialize_smoothing_matrix()
        self.forecast_file_suffix = None
        
    def _extract_forecast_file_suffix(self, forecast_file_name: Optional[str]=None):
        if forecast_file_name:
            base_name = Path(forecast_file_name)
        else:
            base_path = self._model_path.get_latest_forecasting_file_path()
            base_name = Path(base_path.name)
        
        logger.info(f"Using forecast file {base_name}")
        match = re.search(r"(\d{8}_\d{6})", base_name.stem)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Invalid forecast file name: {forecast_file_name}")
            

    def _initialize_smoothing_matrix(self):
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
    
    def _read(self, forecast_file_name: Optional[str]=None):
        self.forecast_file_suffix = self._extract_forecast_file_suffix(forecast_file_name)
        self.df = pd.read_parquet(self._model_path.data_generated / f"predictions_forecasting_{self.forecast_file_suffix}.parquet")
        
    def _transform(self):
        self.smoothed_df = self.df.copy()
        self.smoothed_df['smoothed_predictions'] = None

        months = self.smoothed_df.index.get_level_values(0).unique().sort_values()
        countries = self.smoothed_df.index.get_level_values(1).unique()

        for to_step, to_months in enumerate(months):
            for country in countries:
                sampled_predictions = []
                row_weights = self.smoothing_matrix[to_step, :]
                
                for from_step, weight in enumerate(row_weights):
                    if weight <= 0:
                        continue
                    from_months = to_months - to_step + from_step
                    source_values = self.smoothed_df.loc[(from_months, country), self.pred_col]
                    samples = np.random.choice(source_values, size=weight, replace=True)
                    sampled_predictions.extend(samples)
                self.smoothed_df.at[(to_months, country), 'smoothed_predictions'] = sampled_predictions

        return self.smoothed_df

    def _validate(self):
        if self.smoothed_df['smoothed_predictions'].isnull().all():
            raise ValueError("Transformation resulted in empty smoothed_predictions.")
    
    def _save(self):
        self.smoothed_df.to_parquet(self._model_path.data_processed / f"smoothed_predictions_forecasting_{self.forecast_file_suffix}.parquet")
    
    def run(self, forecast_file_name: Optional[str]=None):
        self._read(forecast_file_name)
        self._transform()
        self._validate()
        self._save()