# Import python libraries
import os
import pandas as pd
import joblib
import json
import sys

# Loading the features I selected during model training
with open("model_output/selected_features.json", "r") as f:
    selected_features = json.load(f)

# Loading the test dataset (scaled features and actual labels)
X_test = pd.read_csv("X_test_scaled.csv")
y_test = pd.read_csv("y_test_creditcard.csv")

# From the full test set, get all rows labeled as 'fraud', i.e. Class = 1
fraud_indices = y_test[y_test["Class"] == 1].index.tolist()

# Manually selecting two fraud samples to test my models —
# one that I expect to pass and one that should fail (based on prior results)
selected_indices = [40, 56]  # These can be changed to test other cases

# Defining the models I’ve trained and saved previously
models_to_test = [
    "logistic_regression", "decision_tree", "knn",
    "svm", "naive_bayes", "random_forest"
]

# Defining how many models must be correct to count the case as a PASS
required_correct = 6

# Track all model predictions and summaries
all_results = []
case_summary = []

print("Running tests on 2 selected fraud samples...\n")

# Looping through the two selected fraud samples
for case_num, fraud_index in enumerate(selected_indices, start=1):
    print("=" * 50)
    print(f"Fraud Sample #{case_num} — Index: {fraud_index}")
    print("=" * 50)

    # Extracting the input features for this fraud transaction
    sample_input = X_test.loc[fraud_index].to_dict()
    expected_label = 1  # We know this is a fraud case
    correct_predictions = 0
    sample_results = []

    # Looping through all models and test for their prediction
    for model_name in models_to_test:
        try:
            # Loading the trained model from file
            model_path = f"model_output/{model_name}.pkl"
            model = joblib.load(model_path)

            # Format the input exactly as the model expects (selected features)
            input_df = pd.DataFrame([sample_input])[selected_features]

            # Making the predictions
            prediction = model.predict(input_df)
            is_correct = prediction[0] == expected_label

            if is_correct:
                correct_predictions += 1

            # Getting the predicted fraud probability if the model supports it
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            proba_str = f"{proba:.2f}" if proba is not None else "--"

            # Format the console output
            model_display = f"{model_name:<20}"
            pred_status = "✓" if is_correct else "✗"
            print(f"{model_display} | Prediction: {int(prediction[0])} | Prob: {proba_str} | {pred_status}")

            # Storing the result for this model/sample combo
            sample_results.append({
                "sample_index": fraud_index,
                "model": model_name,
                "prediction": int(prediction[0]),
                "expected": expected_label,
                "correct": is_correct,
                "fraud_probability": proba
            })

        except Exception as e:
            # Handling and logging any errors from model prediction
            print(f"Error with {model_name}: {e}")
            sample_results.append({
                "sample_index": fraud_index,
                "model": model_name,
                "prediction": None,
                "expected": expected_label,
                "correct": False,
                "fraud_probability": None,
                "error": str(e)
            })

    # Add this sample’s model results to the full list
    all_results.extend(sample_results)

    # Determine if this fraud sample passed (based on threshold)
    passed = correct_predictions >= required_correct
    result_text = f"{correct_predictions}/{len(models_to_test)} models correct"
    if passed:
        print(f"\n{result_text} → \033[92mPASS \033[0m")
    else:
        print(f"\n{result_text} → \033[91mFAIL \033[0m")

    # Track a summary for this fraud test case
    case_summary.append({
        "sample_index": fraud_index,
        "correct_predictions": correct_predictions,
        "total_models": len(models_to_test),
        "passed": passed
    })

# Save full model prediction results to CSV
pd.DataFrame(all_results).to_csv("prediction_results.csv", index=False)

# Save summary of the 2 test cases (useful for logs or pipeline checks)
pd.DataFrame(case_summary).to_csv("fraud_case_summary.csv", index=False)

# Final summary on console
total_passed = sum(1 for case in case_summary if case["passed"])
print("\n" + "=" * 50)
print(f"Completed {len(selected_indices)} fraud test cases")
print(f"Cases Passed: \033[92m{total_passed}/{len(selected_indices)}\033[0m")
print("=" * 50 + "\n")

# Azure DevOps or CI/CD pipeline exit code based on test pass/fail
if total_passed == len(selected_indices):
    sys.exit(0)  # All test cases passed — pipeline can continue
else:
    sys.exit(1)  # At least one failed — block pipeline or flag it
