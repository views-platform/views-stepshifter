from views_pipeline_core.managers.model import ModelManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from typing import List, Dict


class HurdleModel(StepshifterModel):
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
        super().__init__(config, partitioner_dict)
        self._clf = self._resolve_estimator(config['model_clf'])
        self._reg = self._resolve_estimator(config['model_reg'])
        self._clf_params = self._get_parameters(config)['clf']
        self._reg_params = self._get_parameters(config)['reg']
        self._threshold = threshold

    @views_validate
    def fit(self, df: pd.DataFrame):
        """
        Fits the hurdle model to the provided DataFrame.

        This method processes the input data, prepares the time series, and fits two stages of models:
        1. A binary outcome model to predict the occurrence of an event (binary classification).
        2. A positive outcome model to predict the magnitude of the event when it occurs (regression).

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to fit the model.

        Returns:
        None
        """
        df = self._process_data(df)
        self._prepare_time_series(df)

        # Binary outcome (event/no-event)
        # According to the DARTS doc, if timeseries uses a numeric type different from np.float32 or np.float64, not all functionalities may work properly.
        # So use astype(float) instead of astype(int) (we should have binary outputs 0,1 though)
        # Create binary targets by mapping each series in _target_train to 1 if greater than threshold, else 0
        target_binary = [s.map(lambda x: (x > self._threshold).astype(float)) for s in self._target_train]

        # Filter target and past covariates for positive outcomes (where target > threshold)
        target_pos, past_cov_pos = zip(*[(t, p) for t, p in zip(self._target_train, self._past_cov)
                         if (t.values > self._threshold).any()])

        # Iterate over each step in the model
        for step in self._steps:
            # Initialize the binary model with specified lags and parameters
            binary_model = self._clf(lags_past_covariates=[-step], **self._clf_params)
            # Fit the binary model using the binary targets and past covariates
            binary_model.fit(target_binary, past_covariates=self._past_cov)

            # Initialize the positive outcome model with specified lags and parameters
            positive_model = self._reg(lags_past_covariates=[-step], **self._reg_params)
            # Fit the positive outcome model using the filtered positive targets and past covariates
            positive_model.fit(target_pos, past_covariates=past_cov_pos)
            
            # Store the fitted models for the current step
            self._models[step] = (binary_model, positive_model)
        
        # Mark the model as fitted
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

        # If the run type is not 'forecasting', perform multiple predictions
        if run_type != 'forecasting':
            final_preds = []
            # If the evaluation type is "standard", iterate over the evaluation sequence number
            if eval_type == "standard":
                for sequence_number in range(ModelManager._resolve_evaluation_sequence_number(eval_type)):
                    # Predict binary outcomes for each step
                    pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, sequence_number) 
                                        for step in self._steps]
                    # Predict positive outcomes for each step
                    pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, sequence_number) 
                                            for step in self._steps]
                    # Combine binary and positive predictions by multiplying them
                    final_pred = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)
                    # Append the combined predictions to the final predictions list
                    final_preds.append(final_pred)

        else:
            # If the run type is 'forecasting', perform a single prediction
            pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, 0)
                                   for step in self._steps]
            pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, 0)
                                     for step in self._steps]
            # Combine binary and positive predictions by multiplying them
            final_preds = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)

        # Return the final predictions as a DataFrame
        return final_preds
