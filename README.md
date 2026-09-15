# Credit-Risk-Analysis

# Credit Risk: Probability of Default (PD) Model

A credit risk portfolio project estimating Probability of Default (PD) on a loan dataset — includes exploratory data analysis, a full points-based credit scorecard (WOE/IV methodology), traditional ML models for comparison, and a SQL-based feature exploration layer.

## Project overview

This project treats loan default prediction the way a credit risk team would, not as a generic ML exercise:

- **Credit scorecard** built using industry-standard Weight of Evidence (WOE) / Information Value (IV) binning, converted into a points-based score (FICO-style methodology)
- **SQL layer** replicating key feature analysis in SQLite, alongside the pandas equivalent
- **Comparison models**: Logistic Regression, Random Forest, XGBoost
- **Data cleaning**: identified and corrected sentinel/outlier values in `person_age` and `person_emp_length`

## Setup

1. Clone this repo:

```bash
   git clone <your-repo-url>
   cd <repo-folder-name>
```

2. Create a virtual environment:

```bash
   python3 -m venv venv
   source venv/bin/activate       # Mac/Linux
   venv\Scripts\activate          # Windows
```

3. Install required packages:

```bash
   pip install -r requirements.txt
```

Mac users on Apple Silicon: XGBoost also needs the OpenMP runtime:

```bash
   brew install libomp
```

4. Place `train.csv` and `test.csv` inside a `./Data/` folder in the project root.

## Running

Open the main notebook:

```bash
jupyter notebook loan_data_eda.ipynb
```

Or run the scripts directly, in order:

```bash
python data_loading.py        # loads and cleans train/test data
python eda.py                 # exploratory analysis
python scorecard.py           # builds WOE/IV scorecard, predicts on test
python sql_exploration.py     # SQL-based feature analysis
python models.py              # Logistic Regression / Random Forest / XGBoost comparison
```

## Methodology

**Data cleaning**: `person_age` and `person_emp_length` contained a small number of sentinel-like outlier values (123), corrected via median imputation computed from train and applied consistently to test.

**Credit scorecard**: Continuous features were binned into quantile groups, WOE and IV calculated per bin to quantify each feature's predictive strength, then logistic regression was fit on the WOE-transformed features and converted into a points-based scorecard (base score 600, 20 points to double the odds — standard scorecard scaling conventions).

**SQL layer**: Feature-level default rate analysis (by loan grade, home ownership, loan intent, and combined segments) was replicated in SQLite, mirroring the pandas groupby analysis, to demonstrate the same logic in both tools.

**Model comparison**: Logistic Regression, Random Forest, and XGBoost were trained and validated on a held-out split of the training data, evaluated on accuracy, F1 score, and confusion matrix, given the ~86/14 class imbalance in the target.

## Project structure

.
├── Data/
│ ├── train.csv
│ └── test.csv
├── loan_data_eda.ipynb # exploratory data analysis
├── data_loading.py # load + clean train/test
├── scorecard.py # WOE/IV scorecard build + test predictions
├── sql_exploration.py # SQLite feature analysis
├── models.py # Logistic Regression / RF / XGBoost functions
├── credit_scorecard.csv # output: points scorecard by feature/bin
├── logisticregression_scorecard_test.csv # output: scorecard model predictions on test
├── risk_by_income_and_ownership.csv # output: SQL query - risk by income x ownership
├── top_10_riskiest_segments.csv # output: SQL query - riskiest grade/ownership combos
├── requirements.txt
└── README.md

