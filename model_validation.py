import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ============================================
# PSI (POPULATION STABILITY INDEX)
# ============================================

def calculate_psi(expected, actual, buckets=10):
    """
    Compares two distributions of the same score/probability
    (e.g. predicted probabilities on train vs test) and measures how much
    the population has shifted.

    expected = baseline distribution (usually train)
    actual   = new distribution to check (usually test)

    Standard PSI interpretation:
    < 0.1   -> no significant population shift
    0.1-0.25 -> moderate shift, worth investigating
    > 0.25  -> significant shift, model may need retraining
    """
    # Build bucket edges from the EXPECTED (train) distribution only
    breakpoints = np.linspace(0, 1, buckets + 1)
    breakpoints[0] = -0.001  # so 0.0 values are included in the first bucket
    breakpoints[-1] = 1.001  # so 1.0 values are included in the last bucket

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero / log(0) for empty buckets
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_per_bucket = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = np.sum(psi_per_bucket)

    # Build a readable table
    psi_table = pd.DataFrame({
        'bucket': [f'{breakpoints[i]:.2f} - {breakpoints[i+1]:.2f}' for i in range(buckets)],
        'expected_pct': (expected_pct * 100).round(2),
        'actual_pct': (actual_pct * 100).round(2),
        'psi_contribution': psi_per_bucket.round(4)
    })

    return psi_total, psi_table


# Using the XGBoost model and probabilities from models.py as an example
# (swap in whichever model's predict_proba output you want to check)
from models import run_xgboost, X, y, test_loan_encoded
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model, _ = run_xgboost(X, y, X_test=None, show_chart=False)

train_probs = xgb_model.predict_proba(X_train)[:, 1]
test_probs = xgb_model.predict_proba(test_loan_encoded)[:, 1]

psi_score, psi_table = calculate_psi(train_probs, test_probs, buckets=10)

print(f"\nPSI Score: {psi_score:.4f}")
if psi_score < 0.1:
    print("No significant population shift.")
elif psi_score < 0.25:
    print("Moderate shift - worth investigating which segments changed.")
else:
    print("Significant shift - model may need retraining or investigation.")

print("\nPSI breakdown by bucket:")
print(psi_table.to_string(index=False))

psi_table.to_csv('psi_analysis.csv', index=False)
print("\nSaved to psi_analysis.csv")


# ============================================
# CALIBRATION CURVE
# ============================================

# Uses the validation split (y_val has true labels; test.csv likely doesn't)
val_probs = xgb_model.predict_proba(X_val)[:, 1]

prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10, strategy='uniform')

plt.figure(figsize=(7, 7))
plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Actual fraction of defaults')
plt.title('Calibration Curve - XGBoost')
plt.legend()
plt.grid(True)
plt.show()

# Print the underlying numbers too, not just the chart
calibration_table = pd.DataFrame({
    'predicted_probability': prob_pred.round(3),
    'actual_default_rate': prob_true.round(3),
    'gap': (prob_pred - prob_true).round(3)
})
print("\nCalibration table:")
print(calibration_table.to_string(index=False))