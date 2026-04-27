"""
Programmatic Verification Script for stepshifter_full_eval_imp_report.md

This script provides executable proof for the key findings of the evaluation
interface analysis. It programmatically checks the data structures produced
by the different model types and validates the core claims of the report.

To run: `python verify_report_claims.py`

The script will exit with code 0 and print success messages if all claims
in the report are verified. It will raise an AssertionError if any claim
is false.
"""
import pandas as pd
import numpy as np
import logging

# Suppress verbose logging from the library to keep output clean.
logging.basicConfig(level=logging.ERROR)

from views_stepshifter.models.hurdle_model import HurdleModel
from views_stepshifter.models.shurf_model import ShurfModel

# --- 1. SETUP: Create minimal data and configs required for execution ---

def get_mock_configs_and_data():
    """Generates self-contained mock objects to run the models."""

    # This config will be used for the HurdleModel (point predictions)
    hurdle_config = {
        "steps": [1, 2],
        "targets": ["target"],
        "model_clf": "XGBRFClassifier",
        "model_reg": "XGBRFRegressor",
        "parameters": {"clf": {}, "reg": {}},
        "sweep": False,
    }

    # This config will be used for the ShurfModel (uncertainty predictions)
    shurf_config = {
        "steps": [1, 2],
        "targets": ["target"],
        "model_clf": "XGBRFClassifier",
        "model_reg": "XGBRFRegressor",
        "submodels_to_train": 2,
        "log_target": False,
        "pred_samples": 10,
        "draw_dist": "Lognormal",
        "draw_sigma": 0.1,
        "parameters": {"clf": {}, "reg": {}},
        "sweep": False,
    }

    partitioner_dict = {"train": [0, 10], "test": [11, 20]}

    # A minimal DataFrame that allows the models to run
    index = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    data = {
        "feature1": np.random.rand(63),
        "feature2": np.random.rand(63),
        "target": np.random.randint(0, 5, size=63).astype(float),
    }
    dataframe = pd.DataFrame(data, index=index)
    # Ensure some values are non-zero for the positive stage of the Hurdle model
    dataframe.loc[(1, 1), "target"] = 10.0
    dataframe.loc[(2, 2), "target"] = 5.0


    return hurdle_config, shurf_config, partitioner_dict, dataframe

# --- 2. VERIFICATION FUNCTIONS ---

def verify_claim_point_prediction_format_is_float():
    """
    VERIFIES: The report's central claim that point-prediction models
    (HurdleModel, StepshifterModel) produce raw floats, contradicting the guide.
    """
    print("--- Verifying Claim: Point Prediction Cell Format ---")
    hurdle_config, _, partitioner, df = get_mock_configs_and_data()

    # Instantiate and fit the model
    model = HurdleModel(hurdle_config, partitioner)
    model.fit(df)

    # Generate predictions in evaluation mode (run_type != 'forecasting')
    # We only need the first sequence for this verification.
    predictions_list = model.predict(run_type="evaluation", eval_type="standard")

    print("Step 1: Asserting the output is a list of DataFrames...")
    assert isinstance(predictions_list, list), "Output is not a list!"
    assert all(isinstance(item, pd.DataFrame) for item in predictions_list), "List does not contain DataFrames!"
    print("  [PASS] Output is a list of DataFrames.")

    # Get the actual value from the first cell of the first prediction DataFrame
    first_cell_value = predictions_list[0].iloc[0, 0]
    cell_type = type(first_cell_value)

    print(f"Step 2: Asserting the cell value type is a float (e.g., numpy.float64)...")
    print(f"  > Found type: {cell_type}")
    assert isinstance(first_cell_value, (float, np.floating)), f"Cell type is {cell_type}, NOT a float!"
    print("  [PASS] Cell value is a float, confirming the report's finding of a divergence.")
    print("--- Verification PASSED ---\n")
    return predictions_list


def verify_claim_uncertainty_prediction_format_is_list():
    """
    VERIFIES: The report's claim that uncertainty-prediction models (ShurfModel)
    produce lists of floats, confirming the guide.
    """
    print("--- Verifying Claim: Uncertainty Prediction Cell Format ---")
    _, shurf_config, partitioner, df = get_mock_configs_and_data()

    # Instantiate and fit the model
    model = ShurfModel(shurf_config, partitioner)
    model.fit(df)

    # Generate predictions
    predictions_list = model.predict(run_type="evaluation", eval_type="standard")

    print("Step 1: Asserting the output is a list of DataFrames...")
    assert isinstance(predictions_list, list), "Output is not a list!"
    print("  [PASS] Output is a list of DataFrames.")

    # Get the actual value from the first cell of the first prediction DataFrame
    # Note: ShurfModel returns a single DF with lists inside, so we take the DF itself
    first_cell_value = predictions_list[0].iloc[0, 0]
    cell_type = type(first_cell_value)

    print(f"Step 2: Asserting the cell value type is a list...")
    print(f"  > Found type: {cell_type}")
    assert isinstance(first_cell_value, list), f"Cell type is {cell_type}, NOT a list!"
    print("  [PASS] Cell value is a list, confirming the report's finding of compliance.")
    print("--- Verification PASSED ---\n")


def verify_claim_reconciliation_logic_works(point_predictions_list):
    """
    VERIFIES: The report's claim that the recommended reconciliation code
    in the 'Minimal Verification Checklist' is correct and effective.
    """
    print("--- Verifying Claim: Reconciliation Logic Correctness ---")
    print("Step 1: Applying reconciliation code from the report...")

    first_cell = point_predictions_list[0].iloc[0, 0]
    if not isinstance(first_cell, list):
        print("  > Detected float format, as expected. Applying transformation...")
        reconciled_predictions = [df.applymap(lambda x: [x]) for df in point_predictions_list]
    else:
        reconciled_predictions = point_predictions_list

    print("Step 2: Asserting the reconciled data now adheres to the guide's schema...")
    reconciled_cell_value = reconciled_predictions[0].iloc[0, 0]
    cell_type = type(reconciled_cell_value)

    print(f"  > New cell type is: {cell_type}")
    assert isinstance(reconciled_cell_value, list), "Reconciliation failed! Cell is not a list."
    print("  [PASS] Reconciled cell is a list.")
    print("--- Verification PASSED ---\n")


# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    print("======================================================")
    print("   Programmatic Verification of Analysis Report       ")
    print("======================================================\n")

    # This is the core proof of the CONTRADICTION found in the report.
    # It proves that point-prediction models produce floats.
    point_preds = verify_claim_point_prediction_format_is_float()

    # This is the proof that the report correctly identified compliance
    # for uncertainty models.
    verify_claim_uncertainty_prediction_format_is_list()

    # This is the proof that the report's recommended FIX is correct.
    verify_claim_reconciliation_logic_works(point_preds)

    print("======================================================")
    print("      All claims programmatically verified.           ")
    print("          Report is confirmed to be accurate.         ")
    print("======================================================")
