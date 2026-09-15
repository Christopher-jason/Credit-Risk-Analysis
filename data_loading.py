import pandas as pd

# ============================================
# LOAD AND CLEAN TRAINING DATA
# ============================================

loan_train = pd.read_csv('./Data/train.csv')


clean_age = loan_train.loc[loan_train['person_age'] <= 100, 'person_age'].median()
clean_emp = loan_train.loc[loan_train['person_emp_length'] <= 60, 'person_emp_length'].median()
loan_train.loc[loan_train['person_age'] > 100, 'person_age'] = clean_age
loan_train.loc[loan_train['person_emp_length'] > 60, 'person_emp_length'] = clean_emp

print('Training data ready:', loan_train.shape)

# ============================================
# LOAD AND CLEAN TEST DATA
# ============================================

test_df = pd.read_csv('./Data/test.csv')


test_df.loc[test_df['person_age'] > 100, 'person_age'] = clean_age
test_df.loc[test_df['person_emp_length'] > 60, 'person_emp_length'] = clean_emp

test_ids = test_df['id']
test_loan = test_df.drop(columns=['id'])

print('Test data ready:', test_loan.shape)