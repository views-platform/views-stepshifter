import xgboost as xgb
import numpy as np
import time

# Create sample data
print("Creating test data...")
n_samples = 50000
n_features = 100
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randn(n_samples).astype(np.float32)

# Split data
X_train, X_test = X[:40000], X[40000:]
y_train, y_test = y[:40000], y[40000:]

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Train model on GPU
print("\nTraining model on GPU...")
model = xgb.XGBRegressor(
    device='cuda',
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)
print("Training complete")

# GPU prediction test
print("\n=== GPU PREDICTION TEST ===")
print(f"Model device: {model.get_params()['device']}")

print("\n>>> RUN 'nvidia-smi' IN ANOTHER TERMINAL NOW <<<")
print(">>> WATCH FOR GPU UTILIZATION SPIKE <<<")
time.sleep(3)

print("Starting prediction...")
start = time.time()
predictions = model.predict(X_test)
pred_time = time.time() - start

print(f"Prediction time: {pred_time:.3f}s")
print(f"Prediction shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")

print(f"\nXGBoost version: {xgb.__version__}")
print(f"CUDA support: {xgb.build_info()['USE_CUDA']}")
print("\nIf you saw GPU utilization spike in nvidia-smi, GPU prediction is working!")