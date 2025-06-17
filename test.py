# Create a simple test to isolate the issue
import xgboost as xgb
import numpy as np

# Small test dataset
X_small = np.random.randn(1000, 10).astype(np.float32)
y_small = np.random.randn(1000).astype(np.float32)

# Fresh model
test_model = xgb.XGBRegressor(device='cuda', n_estimators=10)
test_model.fit(X_small, y_small)

print("Testing fresh model prediction...")
# Monitor nvidia-smi during this
predictions = test_model.predict(X_small)