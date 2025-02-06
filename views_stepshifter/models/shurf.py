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
from tqdm import tqdm

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
        """
        Generate predictions using the trained submodels.
        This method performs the following steps:
        1. Prepares the data for classification and regression stages.
        2. Iterates over each submodel to generate predictions:
            - Predicts probabilities using the classification model.
            - Predicts target values using the regression model.
            - Handles infinite values in predictions.
        3. Draws samples from the distributions:
            - For each prediction sample, combines classification and regression predictions.
            - Applies binomial, Poisson, or lognormal distributions to generate final predictions.
        4. Aggregates the predictions from all submodels into a final DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the final set of predictions with indices set to 'draw'.
        """
        df = self._process_data(df)
        self._prepare_time_series(df)

        target_binary = [s.map(lambda x: (x > self._threshold).astype(float)) for s in self._target_train]

        # Positive outcome (for cases where target > threshold)
        target_pos, past_cov_pos = zip(*[(t, p) for t, p in zip(self._target_train, self._past_cov)
                                         if (t.values() > self._threshold).any()])

        for i in tqdm(range(self._submodels_to_train), desc="Training submodel"):
            # logger.info(f"Training submodel {i+1}/{self._submodels_to_train}")
            for step in tqdm(self._steps, desc=f"Steps for submodel {i+1}"):
                # logger.info(f"Training step {step}")
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

    @views_validate
    def predict(self, df: pd.DataFrame, run_type: str, eval_type: str = "standard") -> pd.DataFrame:
        """
        Predicts outcomes based on the provided DataFrame and run type.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data for making predictions.
        run_type : str
            The type of run to perform. If 'forecasting', a single prediction is made.
            Otherwise, multiple predictions are made based on the evaluation sequence number.
        eval_type : str, optional
            The type of evaluation to perform. Default is "standard".

        Returns:
        --------
        pd.DataFrame
            The final predictions as a DataFrame.
        """
        
        # Process the input data to ensure it is in the correct format
        df = self._process_data(df)
        # Check if the model has been fitted before making predictions
        check_is_fitted(self, 'is_fitted_')
        print('Parameters:', ' submodels:', self._submodels_to_train, ', samples within submodels: ', self._pred_samples, ', draw distribution: ', self._draw_dist, ', sigma: ', self._draw_sigma)

        # If the run type is not 'forecasting', perform multiple predictions
        if run_type != 'forecasting':
            # If the evaluation type is "standard", iterate over the evaluation sequence number
            submodel_preds = {} # Not sure this belongs here
            if eval_type == "standard":
                # Loop over the evaluation sequence number
                for sequence_number in tqdm(range(ModelManager._resolve_evaluation_sequence_number(eval_type)), desc=f"Sequence", leave=True):
                    print('sequence_number', sequence_number)
                    sample_number = 0
                    final_preds_samples = [] # This will hold predictions for all sub-models and all samples within sub-models
                    # Loop over submodels
                    for submodel_number in tqdm(range(self._submodels_to_train), desc=f"Predicting submodel: {run_type}", leave=True):
                        print('submodel_number', submodel_number)
                        # Predict binary outcomes for each step. Generates a list of dataframes, one df for each step.
                        pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, sequence_number) 
                                            for step in self._steps]
                        # Predict positive outcomes for each step. Generates a list of dataframes, one df for each step.
                        pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, sequence_number) 
                                                for step in self._steps]
                        pred_concat_binary = pd.concat(pred_by_step_binary, axis=0)
                        pred_concat_positive = pd.concat(pred_by_step_positive, axis=0)
                        print('pred_concat_binary')
                        print(pred_concat_binary)
                        print(pred_concat_binary.describe())
                        print('pred_concat_positive')
                        print(pred_concat_positive)
                        print(pred_concat_positive.describe())
                        # Dylan's code:
                        # Combine binary and positive predictions by multiplying them
#                        final_pred = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)
                        # New code:
                        # Extract a set, size self._pred_samples, of predictions for the current sequence number
                        # Check: will this cover all the columns, for each of the steps?
                        for subsample_number in range(self._pred_samples):
                            print('subsample_number', subsample_number)
                            intermediate_preds = []
                            # Draw samples from the classifier; 0/1 according to the classifier's prediction
                            pred_concat_binary_drawn = pred_concat_binary.where(np.random.binomial(n=1,p=pred_concat_binary)==1,0) 
                            if self._draw_dist == 'Poisson':
                                pred_concat_positive_drawn = np.random.poisson(final_pred)
                                print(self._draw_dist,'pred_concat_positive_drawn')
                            if self._draw_dist == 'Lognormal':
                                # Draw from lognormal only for those with predictions larger than zero
                                pred_concat_positive_drawn = np.where(pred_concat_positive>0, np.round(np.random.lognormal(np.log1p(pred_concat_positive), self._draw_sigma)-1), 0)
                                print(self._draw_dist,'pred_concat_positive_drawn')
                                print(pred_concat_positive_drawn)
                            if self._draw_dist == '':
                                # Use the predicted value without a random draw
                                pred_concat_positive_drawn = pred_concat_positive
                                print('No random draw', 'pred_concat_positive_drawn')
                                print(pred_concat_positive_drawn)
                            # Combine the binary and positive predictions
                            sample_pred = pred_concat_binary_drawn * pred_concat_positive_drawn
                            sample_pred['submodel'] = submodel_number
                            sample_pred['sample'] = sample_number 
                            # Add 'sample_number' to the indices of the sample predictions DataFrame:
                            
                            sample_pred.set_index(['sample'],append=True, inplace=True)
                            print('sample_pred')
                            print(sample_pred)
                            # From old SHURF code:
                            # Draw whether more than zero or not from the classifier, insert predictions if more than zero, set as zero if not:
                            #model['predictions_combined'][col] = model['predictions_reg'][col].where(np.random.binomial(n=1,p=model['predictions_clf'][col])==1,0) 
                            #if self.draw_dist == 'Poisson':
                            #    model['predictions_combined'][col] = np.random.poisson(model['predictions_combined'][col])
                            #if self.draw_dist == 'Lognormal':
                            #    # Draw only for those with predictions larger than zero
                            #    model['predictions_combined'][col] = np.where(model['predictions_combined'][col]>0, np.round(np.random.lognormal(np.log1p(model['predictions_combined'][col]), self.draw_sigma)-1), 0)

                            # Append the combined predictions to the predictions list for this sequence number
                            final_preds_samples.append(sample_pred)
                            sample_number += 1
                        
                        # Append the combined predictions to the final predictions list
#                        final_preds.append(intermediate_preds)
#                    submodel_preds[i] = final_preds
                    # Generate a DataFrame from the final predictions list for this sequence number
                    final_preds_full = pd.concat(final_preds_samples, axis=0)  
                    # Output the final predictions with samples as parquet  
                    final_preds_full.to_parquet(f'data/generated/final_pred_full_{run_type}_{eval_type}_{sequence_number}.parquet')
                    # Aggregate the predictions into point predictions
                    final_preds = final_preds_full.groupby(['month_id', 'country_id']).mean()
                    final_preds.pop('submodel')
                    # Output the final predictions as parquet
                    final_preds.to_parquet(f'data/generated/final_preds_{run_type}_{eval_type}_{sequence_number}_agg.parquet')
        else:
            # If the run type is 'forecasting', perform a single prediction
            submodel_preds = {}
            for i in tqdm(range(self._submodels_to_train), desc=f"Predicting submodel: {run_type}"):
                pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, 0)
                                    for step in self._steps]
                pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, 0)
                                        for step in self._steps]
                # Combine binary and positive predictions by multiplying them
                final_preds = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)
                submodel_preds[i] = final_preds

        # Return the final predictions as a DataFrame
        return final_preds