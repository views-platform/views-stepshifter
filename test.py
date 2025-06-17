import xgboost as xgb
import numpy as np

# Create sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randn(1000).astype(np.float32)
X_test = np.random.randn(200, 10).astype(np.float32)

# Train model on GPU
model = xgb.XGBRegressor(device='cuda', n_estimators=50)
model.fit(X_train, y_train)

# Simple fix - use DMatrix
print("Testing GPU prediction...")
dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)

print("Done. Check nvidia-smi to see if GPU was used.")