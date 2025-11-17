import xgboost as xgb
from darts.models import XGBModel, XGBClassifierModel

class XGBRFModel(XGBModel):
    """
    A Darts wrapper for the XGBoost Random Forest Regressor.
    
    This class inherits all functionality from darts.models.XGBModel
    and simply replaces the underlying model with 
    `xgb.XGBRFRegressor`.
    """

    @staticmethod
    def _create_model(**kwargs):
        """Create an XGBRFRegressor model."""
        return xgb.XGBRFRegressor(**kwargs)

class XGBRFClassifierModel(XGBClassifierModel):
    """
    A Darts wrapper for the XGBoost Random Forest Classifier.
    
    This class inherits all functionality from darts.models.XGBClassifierModel
    and simply replaces the underlying model with 
    `xgb.XGBRFClassifier`.
    """

    @staticmethod
    def _create_model(**kwargs):
        """Create an XGBRFClassifier model."""
        return xgb.XGBRFClassifier(**kwargs)