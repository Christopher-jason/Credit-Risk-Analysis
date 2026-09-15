import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import loan_train

sns.set_style("whitegrid")

# Basic structure
print("Rows, Columns:", loan_train.shape)
print(loan_train.columns.tolist())
loan_train.info()
print(loan_train.isnull().sum())
print("Number of duplicate rows:", loan_train.duplicated().sum())
print(loan_train.describe())

# Categorical columns
categorical_cols = loan_train.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", categorical_cols)
for col in categorical_cols:
    print(f"--- {col} ---")
    print(loan_train[col].value_counts())
    print()

# Target distribution
print(loan_train["loan_status"].value_counts())
print((loan_train["loan_status"].value_counts(normalize=True) * 100).round(2))

# Averages by loan status
print(loan_train.groupby('loan_status')['loan_int_rate'].mean())
print(loan_train.groupby('loan_status')['loan_grade'].value_counts(normalize=True).unstack())
print(loan_train.groupby('loan_status')['cb_person_default_on_file'].value_counts(normalize=True).unstack())

# Charts
loan_train["loan_status"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

numeric_cols = ["person_age", "person_income", "person_emp_length",
                "loan_amnt", "loan_int_rate", "loan_percent_income",
                "cb_person_cred_hist_length"]

loan_train[numeric_cols].hist(bins=30, figsize=(14, 10))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(data=loan_train, x="loan_status", y="loan_int_rate", ax=axes[0, 0])
axes[0, 0].set_title("Interest Rate vs Loan Status")
sns.boxplot(data=loan_train, x="loan_status", y="loan_percent_income", ax=axes[0, 1])
axes[0, 1].set_title("Loan % of Income vs Loan Status")
sns.boxplot(data=loan_train, x="loan_status", y="person_income", ax=axes[1, 0])
axes[1, 0].set_title("Income vs Loan Status")
axes[1, 0].set_ylim(0, 200000)
sns.boxplot(data=loan_train, x="loan_status", y="person_age", ax=axes[1, 1])
axes[1, 1].set_title("Age vs Loan Status")
plt.tight_layout()
plt.show()

# Default rate by category
for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
    print(f"--- Default rate by {col} ---")
    print(loan_train.groupby(col)["loan_status"].mean().round(3).sort_values(ascending=False))
    print()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = loan_train[numeric_cols + ["loan_status"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Sanity checks (run before cleaning, for reference on what was fixed)
print("Rows where emp_length looks impossible (>= age):",
      loan_train[loan_train["person_emp_length"] >= loan_train["person_age"]].shape[0])