import sqlite3
import pandas as pd
from data_loading import loan_train

conn = sqlite3.connect('credit_risk.db')
loan_train.to_sql('loans', conn, if_exists='replace', index=False)

check = pd.read_sql("SELECT COUNT(*) as row_count FROM loans", conn)
print(check)

query1 = """
SELECT loan_status, COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loans), 2) as percentage
FROM loans GROUP BY loan_status
"""
print("--- Overall default rate ---")
print(pd.read_sql(query1, conn))

query2 = """
SELECT loan_grade, COUNT(*) as total_loans, SUM(loan_status) as defaults,
    ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
FROM loans GROUP BY loan_grade ORDER BY default_rate_pct DESC
"""
print("\n--- Default rate by loan grade ---")
print(pd.read_sql(query2, conn))

query3 = """
SELECT person_home_ownership, COUNT(*) as total_loans, SUM(loan_status) as defaults,
    ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
FROM loans GROUP BY person_home_ownership ORDER BY default_rate_pct DESC
"""
print("\n--- Default rate by home ownership ---")
print(pd.read_sql(query3, conn))

query4 = """
SELECT loan_intent, COUNT(*) as total_loans, SUM(loan_status) as defaults,
    ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
FROM loans GROUP BY loan_intent ORDER BY default_rate_pct DESC
"""
print("\n--- Default rate by loan intent ---")
print(pd.read_sql(query4, conn))

query5 = """
SELECT loan_status, ROUND(AVG(loan_int_rate), 2) as avg_interest_rate,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amount, ROUND(AVG(person_income), 2) as avg_income
FROM loans GROUP BY loan_status
"""
print("\n--- Averages by loan status ---")
print(pd.read_sql(query5, conn))

query6 = """
SELECT
    CASE
        WHEN person_income < 30000 THEN 'Low Income'
        WHEN person_income BETWEEN 30000 AND 70000 THEN 'Mid Income'
        ELSE 'High Income'
    END as income_bracket,
    person_home_ownership, COUNT(*) as total_loans,
    ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
FROM loans GROUP BY income_bracket, person_home_ownership
ORDER BY income_bracket, default_rate_pct DESC
"""
print("\n--- Default rate by income bracket AND home ownership ---")
print(pd.read_sql(query6, conn))

query7 = """
SELECT loan_grade, person_home_ownership, COUNT(*) as total_loans,
    ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
FROM loans GROUP BY loan_grade, person_home_ownership
HAVING COUNT(*) >= 50 ORDER BY default_rate_pct DESC LIMIT 10
"""
print("\n--- Top 10 riskiest grade+ownership combinations (min 50 loans) ---")
print(pd.read_sql(query7, conn))

income_ownership_risk = pd.read_sql(query6, conn)
income_ownership_risk.to_csv('risk_by_income_and_ownership.csv', index=False)
print(f"Saved {len(income_ownership_risk)} rows to risk_by_income_and_ownership.csv")

top_risk_segments = pd.read_sql(query7, conn)
top_risk_segments.to_csv('top_10_riskiest_segments.csv', index=False)
print(f"Saved {len(top_risk_segments)} rows to top_10_riskiest_segments.csv")