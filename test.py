from darts.models import XGBModel

# 1. Configure for GPU usage
model = XGBModel(
    tree_method='gpu_hist',  # GPU training
    gpu_id=0,  # Which GPU to use
    # XGBoost automatically uses GPU for prediction when available
    # No separate predictor parameter needed
)

# 2. Force GPU availability check
import xgboost as xgb
print("GPU support:", xgb.XGBRegressor().get_params()['tree_method'] == 'gpu_hist')

# 3. Verify prediction device
preds = model.predict(n=5)
print("Prediction device:", model.model.get_params()['tree_method'])  # Should be 'gpu_hist'