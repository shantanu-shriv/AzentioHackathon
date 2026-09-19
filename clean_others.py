import pandas as pd
import numpy as np

# ================================
# 1. CLEAN ACCOUNTS.CSV
# ================================
accounts = pd.read_csv('accounts.csv')

# Drop >50% nulls (except close_date which we handle manually)
null_pct_acc = (accounts.isnull().sum() / len(accounts)) * 100
cols_to_drop_acc = null_pct_acc[(null_pct_acc > 50.0) & (null_pct_acc.index != 'close_date')].index
accounts = accounts.drop(columns=cols_to_drop_acc)

# Standardize categorical strings
accounts_cat_cols = ['account_type', 'account_status', 'currency', 'branch_code', 'branch_city', 'overdraft_enabled', 'card_type', 'mobile_banking_enrolled', 'account_tier']
for col in accounts_cat_cols:
    if col in accounts.columns:
        accounts[col] = accounts[col].astype(str).str.strip().str.upper().str.replace(' ', '_').replace('NAN', np.nan)

# Special case for close_date
accounts['is_closed'] = accounts['close_date'].notnull().astype(int)
accounts['close_date'] = accounts['close_date'].fillna('NOT_CLOSED')

# Fill Categorical Nulls using Group-Based Mode (Group by customer_id)
for col in accounts_cat_cols:
    if col in accounts.columns:
        accounts[col] = accounts.groupby('customer_id')[col].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
        )
        accounts[col] = accounts[col].fillna('UNKNOWN') # Fallback

# Standardize Dates
accounts['open_date'] = pd.to_datetime(accounts['open_date'], errors='coerce')
accounts['last_login_date'] = pd.to_datetime(accounts['last_login_date'], errors='coerce')

# Sanity checks
accounts = accounts.drop_duplicates()
num_cols = ['current_balance', 'avg_monthly_balance_6m', 'credit_limit']
for col in num_cols:
    if col in accounts.columns:
        accounts[col] = accounts[col].fillna(accounts[col].median())

accounts.to_csv('accounts_cleaned.csv', index=False)
print("Accounts cleaned!")

# ================================
# 2. CLEAN CUSTOMERS.CSV
# ================================
customers = pd.read_csv('customers.csv')

# Drop >50% nulls
null_pct_cust = (customers.isnull().sum() / len(customers)) * 100
cols_to_drop_cust = null_pct_cust[null_pct_cust > 50.0].index
customers = customers.drop(columns=cols_to_drop_cust)

# Standardize categorical strings
customers_cat_cols = ['gender', 'city', 'state', 'country', 'occupation', 'marital_status', 'education_level', 'employment_status', 'customer_segment', 'kyc_status', 'risk_rating', 'preferred_channel', 'email_verified', 'phone_verified']
for col in customers_cat_cols:
    if col in customers.columns:
        customers[col] = customers[col].astype(str).str.strip().str.upper().str.replace(' ', '_').replace('NAN', np.nan)

# Fill Categorical Nulls 
for col in customers_cat_cols:
    if col in customers.columns:
        # Use overall mode since we can't group by customer_id
        if not customers[col].mode().empty:
            customers[col] = customers[col].fillna(customers[col].mode()[0])
        else:
            customers[col] = customers[col].fillna('UNKNOWN')

# Fill missing phone numbers
if 'phone_number' in customers.columns:
    customers['phone_number'] = customers['phone_number'].fillna('UNKNOWN')

# Fill numerical nulls
if 'annual_income' in customers.columns:
    customers['annual_income'] = customers['annual_income'].fillna(customers['annual_income'].median())
    customers['annual_income'] = customers['annual_income'].abs() # No negative incomes

# Standardize Dates
if 'date_of_birth' in customers.columns:
    customers['date_of_birth'] = pd.to_datetime(customers['date_of_birth'], errors='coerce')
if 'customer_since' in customers.columns:
    customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')

# Sanity checks
customers = customers.drop_duplicates()

customers.to_csv('customers_cleaned.csv', index=False)
print("Customers cleaned!")
