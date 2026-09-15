import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_loading import loan_train, test_loan, test_ids

# ============================================
# STEP 1: BIN THE CONTINUOUS VARIABLES
# ============================================

def create_bins(df, column, n_bins=5):
    """Splits a continuous column into n_bins quantile groups."""
    df[f'{column}_bin'] = pd.qcut(df[column], q=n_bins, duplicates='drop')
    return df

feature_binning = [
    'person_age', 'person_income', 'loan_int_rate',
    'loan_percent_income', 'person_emp_length', 'cb_person_cred_hist_length'
]

for feature in feature_binning:
    loan_train = create_bins(loan_train, feature, n_bins=5)

# ============================================
# STEP 2: CALCULATE WOE AND IV PER BIN
# ============================================

def calculate_woe_iv(df, feature, target):
    """
    Calculates Weight of Evidence and Information Value for each bin of a feature.
    """
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()

    grouped['bad_pct'] = (grouped['bad'] + 0.5) / (total_bad + 0.5)
    grouped['good_pct'] = (grouped['good'] + 0.5) / (total_good + 0.5)

    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
    grouped['iv_contribution'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']

    iv_total = grouped['iv_contribution'].sum()
    return grouped, iv_total

binned_features = [
    'person_age_bin', 'person_income_bin', 'loan_int_rate_bin',
    'loan_percent_income_bin', 'person_emp_length_bin', 'cb_person_cred_hist_length_bin'
]
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
all_features = binned_features + categorical_features

woe_tables = {}
iv_summary = {}

print("\n===== WOE / IV BY FEATURE =====\n")
for feature in all_features:
    woe_table, iv = calculate_woe_iv(loan_train, feature, 'loan_status')
    woe_tables[feature] = woe_table
    iv_summary[feature] = iv
    print(f"--- {feature} (IV = {iv:.4f}) ---")
    print(woe_table[['total', 'bad', 'good', 'woe']].round(3))
    print()

iv_ranking = pd.Series(iv_summary).sort_values(ascending=False)
print("===== FEATURE RANKING BY INFORMATION VALUE =====")
print(iv_ranking.round(4))

# ============================================
# STEP 3: TRANSFORM TRAIN DATA - REPLACE RAW VALUES WITH WOE
# ============================================

def apply_woe(df, feature, woe_table):
    """Maps each row's bin to its WOE value. .astype(float) avoids Categorical dtype issues."""
    woe_map = woe_table['woe'].to_dict()
    return df[feature].map(woe_map).astype(float)

X_woe_train = pd.DataFrame()
for feature in all_features:
    new_col_name = feature.replace('_bin', '') + '_woe'
    X_woe_train[new_col_name] = apply_woe(loan_train, feature, woe_tables[feature])

y_train_full = loan_train['loan_status']

print(X_woe_train.head())
print(X_woe_train.isnull().sum())

# ============================================
# STEP 4: FIT LOGISTIC REGRESSION ON WOE VALUES
# ============================================

scorecard_model = LogisticRegression(class_weight='balanced', random_state=42)
scorecard_model.fit(X_woe_train, y_train_full)

print('\nCoefficients:')
for feature, coef in zip(X_woe_train.columns, scorecard_model.coef_[0]):
    print(f'{feature}: {coef:.4f}')
print(f'Intercept: {scorecard_model.intercept_[0]:.4f}')

# ============================================
# STEP 5: CONVERT TO A POINTS-BASED SCORECARD
# ============================================

base_score = 600
pdo = 20
factor = pdo / np.log(2)
offset = base_score - factor * scorecard_model.intercept_[0]

print(f'\nFactor: {factor:.2f}, Offset: {offset:.2f}')

scorecard_points = []
for feature, coef in zip(X_woe_train.columns, scorecard_model.coef_[0]):
    original_feature = [f for f in all_features if f.replace('_bin', '') == feature.replace('_woe', '')][0]
    woe_table = woe_tables[original_feature]
    for bin_label, row in woe_table.iterrows():
        points = -(coef * row['woe']) * factor
        scorecard_points.append({
            'feature': feature,
            'bin': bin_label,
            'woe': round(row['woe'], 3),
            'points': round(points, 1)
        })

scorecard_df = pd.DataFrame(scorecard_points)
print('\n===== SCORECARD =====')
print(scorecard_df.to_string(index=False))

scorecard_df.to_csv('credit_scorecard.csv', index=False)
print(f'Saved scorecard with {len(scorecard_df)} rows to credit_scorecard.csv')

# ============================================
# STEP 6: APPLY THE SAME BINS + WOE TO TEST DATA
# ============================================

def apply_same_bins(df, column, train_bin_edges):
    """Applies the SAME bin edges learned from train onto test."""
    return pd.cut(df[column], bins=train_bin_edges, include_lowest=True)

bin_edges = {}
for col in feature_binning:
    _, edges = pd.qcut(loan_train[col], q=5, duplicates='drop', retbins=True)
    bin_edges[col] = edges

test_loan_binned = test_loan.copy()
for col, edges in bin_edges.items():
    test_loan_binned[f'{col}_bin'] = apply_same_bins(test_loan_binned, col, edges)

X_woe_test = pd.DataFrame()
for feature in all_features:
    new_col_name = feature.replace('_bin', '') + '_woe'
    X_woe_test[new_col_name] = apply_woe(test_loan_binned, feature, woe_tables[feature])

X_woe_test = X_woe_test.fillna(0)

# ============================================
# STEP 7: PREDICT ON TEST AND SAVE TO CSV
# ============================================

test_predictions = scorecard_model.predict(X_woe_test)
test_probabilities = scorecard_model.predict_proba(X_woe_test)[:, 1]

results = pd.DataFrame({
    'id': test_ids,
    'loan_status': test_predictions,
    'default_probability': test_probabilities.round(4)
})

results.to_csv('logisticregression_scorecard_test.csv', index=False)
print(f'\nSaved {len(results)} predictions to logisticregression_scorecard_test.csv')
print(f'Predicted default rate: {test_predictions.mean() * 100:.2f}%')