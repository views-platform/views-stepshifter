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
    
    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]], threshold: float = 0.1):
        super().__init__(config, partitioner_dict, threshold)
        print(config)
#        self._clf = RandomForest
#        self._reg = RandomForest
        self._clf_params = self._get_parameters(config)['clf']
        self._reg_params = self._get_parameters(config)['reg']
        self._threshold = threshold

        self._submodel_list = []

        self._partitioner_dict = partitioner_dict
        self._submodels_to_train = config['submodels_to_train']
        # self._n_estimators = config['parameters']['n_estimators']
        self.log_target = config['log_target']
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
            self._submodel_list.append(self._models)
            logger.info(f"Submodel {i+1}/{self._submodels_to_train} trained successfully")
        self.is_fitted_ = True
    
        
    def predict_sequence(self,run_type, eval_type, sequence_number) -> pd.DataFrame:
        """
        Predicts n draws of outcomes based on the provided DataFrame .

        Parameters:
        -----------
        self: StepShiftedHurdleUncertainRF
            The model object.
            
        run_type : str
            The type of run to perform. Currently it is unlikely to affect the behaviour of the function.
            
        eval_type : str 
            The type of evaluation to perform. Currently it is unlikely to affect the behaviour of the function.
            
        sequence_number : int
            The sequence number to predict outcomes for.
             
            
        Returns:
        --------
        pd.DataFrame
            The final predictions as a DataFrame.
        """
        
        sample_number = 0
        final_preds = [] # This will hold predictions for all sub-models and all samples within sub-models
        # Loop over submodels
        submodel_number = 0
        for submodel in tqdm(self._submodel_list, desc=f"Predicting submodel: {run_type}", leave=True):
#            print(submodel)
            pred_by_step_binary = [self._predict_by_step(submodel[step][0], step, sequence_number) 
                                for step in self._steps]
            pred_by_step_positive = [self._predict_by_step(submodel[step][1], step, sequence_number) 
                                    for step in self._steps]
            
#        for submodel_number in tqdm(range(self._submodels_to_train), desc=f"Predicting submodel: {run_type}", leave=True):
#            print('submodel_number', submodel_number)
            # Predict binary outcomes for each step. Generates a list of dataframes, one df for each step.
#            pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, sequence_number) 
#                                for step in self._steps]
            # Predict positive outcomes for each step. Generates a list of dataframes, one df for each step.
#            pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, sequence_number) 
#                                    for step in self._steps]
            pred_concat_binary = pd.concat(pred_by_step_binary, axis=0)
            
            pred_concat_binary.rename(columns={'step_combined':'Classification'}, inplace=True)
            pred_concat_positive = pd.concat(pred_by_step_positive, axis=0)
            pred_concat_positive.rename(columns={'step_combined':'Regression'}, inplace=True)
            pred_concat = pd.concat([pred_concat_binary, pred_concat_positive], axis=1)
            pred_concat['submodel'] = submodel_number
#            print(pred_concat.tail(12))
                
            # Append the combined predictions to the final predictions list
            final_preds.append(pred_concat)
            submodel_number += 1
#                    submodel_preds[i] = final_preds
        # Generate a DataFrame from the final predictions list for this sequence number
        final_preds_aslists = pd.concat(final_preds, axis=0)  
        # Creating index of samples
#        first_index = self._pred_samples * submodel_number + 1
#        last_index = self._pred_samples * submodel_number + self._pred_samples
#        print('Submodel',submodel_number,'First index',first_index,'Last',last_index)
#        final_preds_aslists['Draw'] = final_preds_aslists.apply(np.arange(first_index,last_index))
#        list(range(first_index,last_index))
        # Drawing samples from the classification model
        # Ensuring that the classification probabilities are between 0 and 1:
        final_preds_aslists['Classification'] = final_preds_aslists['Classification'].apply(lambda x: np.clip(x, 0, 1))
        final_preds_aslists['ClassificationSample'] = final_preds_aslists['Classification'].apply(lambda x: np.random.binomial(1, x, self._pred_samples))
        
        # Drawing samples from the regression model

        if self.log_target == True:
            if self._draw_dist == 'Poisson': # Note: this assumes a non-log-transformed target
                print('Poisson not implemented')
                final_preds_aslists['RegressionSample'] = final_preds_aslists['Regression']
            if self._draw_dist == 'Lognormal':
                # Draw from normal distribution for log-transformed outcomes, then exponentiate, then round to integer
                #pred_concat['RegressionSample'] = pred_concat['Regression'].apply(lambda x: np.random.normal(x, self._draw_sigma, self._pred_samples))
                final_preds_aslists['RegressionSample'] = final_preds_aslists['Regression'].apply(lambda x: np.abs(np.rint(np.expm1(np.random.normal(x, self._draw_sigma, self._pred_samples)))))
        if self.log_target == False:
            if self._draw_dist == 'Poisson': # Note: this assumes a non-log-transformed target
                print('Poisson not implemented')
                final_preds_aslists['RegressionSample'] = final_preds_aslists['Regression']
            if self._draw_dist == 'Lognormal':
                print('Draws for non-log-transformed target: first implementation' )
                final_preds_aslists['RegressionSample'] = final_preds_aslists['Regression'].apply(lambda x: np.abs(np.rint(np.expm1(np.random.normal(np.log1p(x), self._draw_sigma, self._pred_samples)))))
            
        if self._draw_dist == '':
            final_preds_aslists['RegressionSample'] = final_preds_aslists['Regression']
        print('final_preds_aslists contains the samples in list form. Shape:', final_preds_aslists.shape)
        print(final_preds_aslists.tail(20))
#        final_pred_full = final_preds_aslists.explode(['ClassificationSample','RegressionSample','Draw'])
        final_preds_full = final_preds_aslists.explode(['ClassificationSample','RegressionSample'])
        final_preds_full['Prediction'] = final_preds_full['ClassificationSample'] * final_preds_full['RegressionSample']
        # Ensuring that the final predictions are positive:
        final_preds_full['Prediction'] = final_preds_full['Prediction'].apply(lambda x: np.clip(x, 0, None))
        # Log-transforming the final predictions if necessary:
        final_preds_full['LogPrediction'] = np.log1p(final_preds_full['Prediction'])
        final_preds_full.drop(columns=['Classification','Regression','ClassificationSample','RegressionSample'],inplace=True)
#        print('final_preds_full is the end product of the predict sequence function. Shape:', final_preds_full.shape)
#        print(final_preds_full.tail(20))
        return final_preds_full

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
        print('Dependent variable:', self.depvar, 'Parameters:', 'Log target:', self.log_target, ' submodels:', self._submodels_to_train, ', samples within submodels: ', self._pred_samples, ', draw distribution: ', self._draw_dist, ', sigma: ', self._draw_sigma)

        # If the run type is not 'forecasting', perform multiple predictions
        if run_type != 'forecasting':
            preds = []  # D: List to collect predictions for each sequence
            # If the evaluation type is "standard", iterate over the evaluation sequence number
            submodel_preds = {} # Not sure this belongs here
            if eval_type == "standard":
                # Loop over the evaluation sequence number
                for sequence_number in tqdm(range(ModelManager._resolve_evaluation_sequence_number(eval_type)), desc=f"Sequence", leave=True):
#                    print('sequence_number', sequence_number)
                    final_preds_full = self.predict_sequence(run_type, eval_type, sequence_number)   
#                    print('final_preds_aslists')
#                    print(final_preds_aslists.describe())
                        
                    # Output the final predictions with samples as parquet  
                    final_preds_full.to_parquet(f'data/generated/final_pred_full_{run_type}_{eval_type}_{sequence_number}.parquet')
                    # Aggregate the predictions into point predictions
#                    final_preds.pop('LogPrediction')
                    final_preds = np.log1p(final_preds_full.groupby(['month_id', 'country_id']).mean())
#                    final_preds.rename(columns={'Prediction':'pred_ged_sb'}, inplace=True)   
                    final_preds.pop('submodel')
                    preds.append(final_preds) # D: Append the final predictions for this sequence number
                    # Output the final predictions as parquet
                    final_preds.to_parquet(f'data/generated/final_preds_{run_type}_{eval_type}_{sequence_number}_agg.parquet')
                return preds
        else:
            # If the run type is 'forecasting', perform a single prediction
            sequence_number = 0
            final_preds_full = self.predict_sequence(run_type, eval_type, sequence_number)           
#            print('final_preds_aslists')
#            print(final_preds_aslists.describe())
                
            # Output the final predictions with samples as parquet  
            final_preds_full.to_parquet(f'data/generated/final_pred_full_{run_type}_{eval_type}_{sequence_number}.parquet')
            # Aggregate the predictions into point predictions
            final_preds = final_preds_full.groupby(['month_id', 'country_id']).mean()
            final_preds.pop('submodel')
            # Output the final predictions as parquet
            final_preds['ged_sb_dep'] = final_preds['Prediction']
            final_preds.to_parquet(f'data/generated/final_preds_{run_type}_{eval_type}_{sequence_number}_agg.parquet')

            # Return the final predictions as a DataFrame
            return final_preds