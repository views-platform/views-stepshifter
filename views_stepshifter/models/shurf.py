from views_pipeline_core.managers.model import ModelManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.hurdle_model import HurdleModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from typing import List, Dict
import numpy as np
import logging
from darts.models import RandomForest

logger = logging.getLogger(__name__)

class StepShiftedHurdleUncertainRF(HurdleModel):
    """
    Hurdle model for time series forecasting. The model consists of two stages:
    1. Binary stage: Predicts whether the target variable is 0 or > 0.
    2. Positive stage: Predicts the value of the target variable when it is > 0.

    Note:
    This algorithm uses a two-step approach. 

    **Step 1: Classification Stage**  
    In the first step, a regression model is used with a binary target (0 or 1), 
    indicating the absence or presence of violence. This stage functions similarly 
    to a linear probability model, estimating the likelihood of a positive outcome. 
    Since the model is a regression rather than a classification model, 
    these estimates are not strictly bounded between 0 and 1, 
    but this is acceptable for the purpose of this step.

    To determine whether an observation is classified as "positive," we apply a threshold. 
    The default threshold is 1, meaning that predictions above this value 
    are considered positive outcomes. This threshold can be adjusted as 
    a tunable hyperparameter to better suit specific requirements.

    **Step 2: Regression Stage**  
    In the second step, we use a regression model to predict a continuous or count value 
    (e.g., the expected number of conflict fatalities) for the selected time series. 
    We include the entire time series for countries or PRIO grids where the 
    classification stage yielded at least one "positive" prediction, 
    rather than limiting the regression to just the predicted positive values.
    """
    
    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]], threshold: float = 1.0):
        super().__init__(config, partitioner_dict, threshold)
        print(config)
        self._clf = RandomForest
        self._reg = RandomForest
        self._clf_params = self._get_parameters(config)['clf']
        self._reg_params = self._get_parameters(config)['reg']
        self._threshold = threshold

        self._submodel_list = []

        self._partitioner_dict = partitioner_dict
        self._submodels_to_train = config['submodels_to_train']
        # self._n_estimators = config['parameters']['n_estimators']
        self._max_features = config['max_features']
        self._max_depth = config['max_depth']
        self._max_samples = config['max_samples']
        self._pred_samples = config['pred_samples']
        self._draw_dist = config['draw_dist']
        self._draw_sigma = config['draw_sigma']
        self._geo_unit_samples = config['geo_unit_samples']
        self._n_jobs = config['n_jobs']

    @views_validate
    def fit(self, df: pd.DataFrame):
        df = self._process_data(df)
        self._prepare_time_series(df)

        target_binary = [s.map(lambda x: (x > self._threshold).astype(float)) for s in self._target_train]

        # Positive outcome (for cases where target > threshold)
        target_pos, past_cov_pos = zip(*[(t, p) for t, p in zip(self._target_train, self._past_cov)
                                         if (t.values() > self._threshold).any()])

        for i in range(self._submodels_to_train):
            logger.info(f"Training submodel {i+1}/{self._submodels_to_train}")
            for step in self._steps:
                logger.info(f"Training step {step}")
                # Fit binary-like stage using a regression model, but the target is binary (0 or 1)
                binary_model = self._clf(lags_past_covariates=[-step], **self._clf_params)
                binary_model.fit(target_binary, past_covariates=self._past_cov)

                # Fit positive stage using the regression model
                positive_model = self._reg(lags_past_covariates=[-step], **self._reg_params)
                positive_model.fit(target_pos, past_covariates=past_cov_pos)
                self._models[step] = (binary_model, positive_model)
            submodel_dict = {}
            submodel_dict['model_clf'] = binary_model
            submodel_dict['model_reg'] = positive_model
            self._submodel_list.append(submodel_dict)
            logger.info(f"Submodel {i+1}/{self._submodels_to_train} trained successfully")
        self.is_fitted_ = True