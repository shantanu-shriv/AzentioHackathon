import pandas as pd
import json
import random

# 1. Load Cleaned Data
transactions = pd.read_csv('transactions_cleaned.csv')
accounts = pd.read_csv('accounts_cleaned.csv')
customers = pd.read_csv('customers_cleaned.csv')

# 2. Merge Data for Context
# Merge transactions with accounts, then with customers
df = transactions.merge(accounts, on=['account_id', 'customer_id'], how='left')
df = df.merge(customers, on='customer_id', how='left')

# 3. Create Synthetic Labels (Since no is_fraud column exists in the dataset)
# We use banking heuristics to simulate fraud flags for training the SLM
def flag_fraud(row):
    # Rule 1: Abnormally high transaction ratio and high amount
    if row['amount_to_account_avg_ratio'] > 5 and row['amount'] > 5000:
        return True, "Extremely high amount relative to account history."
    # Rule 2: Foreign transaction on a PIN-less channel
    if row['is_foreign_transaction'] in ['1', 1, True, 'TRUE'] and row['auth_method'] == 'NONE':
        return True, "Foreign transaction executed without authentication."
    # Rule 3: High amount from a new device in a different country
    if row['is_new_device'] in ['1', 1, True, 'TRUE'] and row['amount'] > 3000:
        return True, "Large transaction from an unrecognized device."
    
    return False, "Transaction behavior matches standard account profile."

fraud_labels = df.apply(flag_fraud, axis=1)
df['is_fraud'] = [x[0] for x in fraud_labels]
df['justification'] = [x[1] for x in fraud_labels]

# Calculate a mock confidence score
df['confidence'] = df['is_fraud'].apply(lambda x: round(random.uniform(0.85, 0.99), 2) if x else round(random.uniform(0.70, 0.99), 2))

# 4. Generate JSONL Training Dataset for Fine-Tuning
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
    # Build the input prompt
    input_text = prompt_template.format(
        amount=row['amount'], currency=row.get('currency_x', row.get('currency', 'USD')), txn_type=row['transaction_type'],
        foreign=row['is_foreign_transaction'], device=row['device_type'], is_new=row['is_new_device'],
        auth=row['auth_method'], ratio=round(row['amount_to_account_avg_ratio'], 2) if pd.notnull(row['amount_to_account_avg_ratio']) else 1.0
    )
    
    # Build the strict JSON output
    output_json = {
        "transaction_id": str(row['transaction_id']),
        "is_fraud": bool(row['is_fraud']),
        "confidence": float(row['confidence']),
        "justification": str(row['justification'])
    }
    
    # Format for Llama 3 Chat Template (Instruct format)
    llama_format = {
        "messages": [
            {"role": "system", "content": "You are a fraud detection SLM that only outputs strict JSON."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": json.dumps(output_json)}
        ]
    }
    train_data.append(llama_format)

# Save to JSONL
with open('train_dataset.jsonl', 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + '\n')

print(f"Generated synthetic training dataset with {len(train_data)} examples!")
print("Saved to: train_dataset.jsonl")
