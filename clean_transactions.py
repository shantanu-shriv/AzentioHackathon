import pandas as pd
import numpy as np

df = pd.read_csv('transactions.csv')

# 1. Clean numerical column 'amount' (remove currency symbols, commas, spaces)
df['amount'] = df['amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

# 2. Standardize categorical string columns (uppercase, strip whitespace, replace spaces with underscores)
categorical_cols = ['transaction_type', 'channel', 'status', 'merchant_category', 'device_type']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.upper().str.replace(' ', '_').replace('NAN', np.nan)

# 3. Standardize boolean 'is_foreign_transaction'
bool_map = {'0': 0, '1': 1, 'FALSE': 0, 'TRUE': 1, 'N': 0, 'Y': 1, 'NO': 0, 'YES': 1}
df['is_foreign_transaction'] = df['is_foreign_transaction'].astype(str).str.strip().str.upper().map(bool_map)

# 4. Standardize datetime column
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')

# 5. Calculate and print null percentages
null_pct = (df.isnull().sum() / len(df)) * 100
print("--- Percentage of Null Values Before Handling ---")
print(null_pct[null_pct > 0].sort_values(ascending=False))

# 6. Drop columns with > 50% missing values
threshold = 50.0
cols_to_drop = null_pct[null_pct > threshold].index
df = df.drop(columns=cols_to_drop)
print(f"Dropped columns due to high missing values (> {threshold}%): {list(cols_to_drop)}")

# 7. Handle remaining Null Values
# Fill customer_id first (fallback)
df['customer_id'] = df['customer_id'].fillna('UNKNOWN')

# Fill categorical columns with Group-Based Mode (per customer)
categorical_nulls = ['auth_method', 'device_type', 'merchant_category', 'device_id', 
                    'transaction_type', 'channel', 'merchant_city', 'status']
for col in categorical_nulls:
    if col in df.columns:
        df[col] = df.groupby('customer_id')[col].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
        )
        # Fallback to 'UNKNOWN' if the customer has entirely missing values for this column
        df[col] = df[col].fillna('UNKNOWN')

# Clever imputation for amount_to_account_avg_ratio
account_avg_amount = df.groupby('account_id')['amount'].transform('mean')
df['amount_to_account_avg_ratio'] = df['amount_to_account_avg_ratio'].fillna(
    df['amount'] / account_avg_amount
)

# Fill remaining numerical columns with median (fallback for ratio)
numerical_nulls = ['time_since_prev_txn_mins', 'amount', 'amount_to_account_avg_ratio']
for col in numerical_nulls:
    df[col] = df[col].fillna(df[col].median())

# Fill datetime column using forward fill, then backward fill for any remaining
df['transaction_timestamp'] = df['transaction_timestamp'].ffill().bfill()

# 8. Final Sanity Checks (Duplicates and Anomalies)
# Drop exact duplicate rows
df = df.drop_duplicates()

# Handle negative or zero amounts (likely data entry errors, we take the absolute value for negatives and drop zeros if they are invalid, or just take absolute value)
df['amount'] = df['amount'].abs()
# Let's drop transactions where amount is exactly 0 as they are usually failed/invalid pings
df = df[df['amount'] > 0]

print("\n--- Max Missing Values After Handling ---")
print(df.isnull().sum().max())

# Save cleaned data
df.to_csv('transactions_cleaned.csv', index=False)
print("Data cleaned and saved to transactions_cleaned.csv")
