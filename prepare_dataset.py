import pandas as pd
import json
import random
from sklearn.ensemble import IsolationForest

# 1. Load Cleaned Data
print("Loading cleaned datasets...")
transactions = pd.read_csv('transactions_cleaned.csv')
accounts = pd.read_csv('accounts_cleaned.csv')
customers = pd.read_csv('customers_cleaned.csv')

# 2. Merge Data for Context
print("Merging datasets...")
df = transactions.merge(accounts, on=['account_id', 'customer_id'], how='left')
df = df.merge(customers, on='customer_id', how='left')

# 3. Unsupervised Anomaly Detection (Isolation Forest)
print("Extracting features for Isolation Forest...")
# We select numerical columns that might indicate fraudulent behavior
features = ['amount', 'distance_from_home_km', 'time_since_prev_txn_mins', 
            'txn_count_last_24h', 'amount_to_account_avg_ratio']

# Convert booleans/strings to numeric where necessary
df['is_foreign_numeric'] = df['is_foreign_transaction'].apply(lambda x: 1 if str(x).upper() in ['1', 'TRUE', 'YES'] else 0)
df['is_new_device_numeric'] = df['is_new_device'].apply(lambda x: 1 if str(x).upper() in ['1', 'TRUE', 'YES'] else 0)
features.extend(['is_foreign_numeric', 'is_new_device_numeric'])

# Impute any remaining NaNs with 0 to prevent ML crashes
X = df[features].fillna(0)

print("Training Isolation Forest (Contamination = 2%)...")
# contamination=0.02 means we expect the top 2% weirdest transactions to be fraud
model = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
# Returns -1 for anomalies, 1 for normal
anomaly_preds = model.fit_predict(X)

# Map -1 to True (Fraud), 1 to False (Normal)
df['is_fraud'] = [True if pred == -1 else False for pred in anomaly_preds]

# Provide justification
df['justification'] = df['is_fraud'].apply(
    lambda x: "Transaction flagged as a multi-dimensional mathematical anomaly." if x 
    else "Transaction behavior matches standard account profile."
)

# Calculate a mock confidence score based on the anomaly score
# anomaly_score is negative for anomalies, positive for normal
scores = model.decision_function(X)
# Normalize scores to a pseudo-confidence between 0.70 and 0.99
def calc_confidence(score, is_fraud):
    if is_fraud:
        # Lower score = more anomalous = higher confidence
        return round(min(0.99, max(0.85, 0.85 + abs(score) * 0.5)), 2)
    else:
        # Higher score = more normal = higher confidence
        return round(min(0.99, max(0.70, 0.70 + score * 0.5)), 2)

df['confidence'] = [calc_confidence(s, f) for s, f in zip(scores, df['is_fraud'])]

fraud_count = df['is_fraud'].sum()
print(f"Isolation Forest identified {fraud_count} anomalous (fraud) transactions out of {len(df)}.")

# 4. Generate JSONL Training Dataset for Fine-Tuning
print("Generating JSONL Training Dataset...")
train_data = []

prompt_template = """You are an AI Fraud Sentinel. Analyze the following transaction and output a precise JSON risk profile.
Transaction Details:
Amount: {amount} {currency}
Type: {txn_type}
Foreign: {foreign}
Device: {device} (New: {is_new})
Auth: {auth}
Account Avg Ratio: {ratio}
"""

for _, row in df.iterrows():
    input_text = prompt_template.format(
        amount=row['amount'], currency=row.get('currency_x', row.get('currency', 'USD')), txn_type=row['transaction_type'],
        foreign=row['is_foreign_transaction'], device=row['device_type'], is_new=row['is_new_device'],
        auth=row['auth_method'], ratio=round(row['amount_to_account_avg_ratio'], 2) if pd.notnull(row['amount_to_account_avg_ratio']) else 1.0
    )
    
    output_json = {
        "transaction_id": str(row['transaction_id']),
        "is_fraud": bool(row['is_fraud']),
        "confidence": float(row['confidence']),
        "justification": str(row['justification'])
    }
    
    llama_format = {
        "messages": [
            {"role": "system", "content": "You are a fraud detection SLM that only outputs strict JSON."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": json.dumps(output_json)}
        ]
    }
    train_data.append(llama_format)

with open('train_dataset.jsonl', 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + '\n')

print(f"Saved to: train_dataset.jsonl")
