import pandas as pd
import json
from inference_pipeline import FraudSentinelPipeline

def main():
    print("Loading cleaned datasets for final inference...")
    # Load and merge just like we did for training
    transactions = pd.read_csv('transactions_cleaned.csv')
    accounts = pd.read_csv('accounts_cleaned.csv')
    customers = pd.read_csv('customers_cleaned.csv')

    df = transactions.merge(accounts, on=['account_id', 'customer_id'], how='left')
    df = df.merge(customers, on='customer_id', how='left')
    
    # Initialize the SLM Pipeline
    # Attempt to load the fine-tuned model first. If it doesn't exist, fallback to the base model.
    import os
    if os.path.exists("fraud_sentinel_final"):
        print("Found fine-tuned model! Loading...")
        model_to_load = "fraud_sentinel_final"
    else:
        print("WARNING: Fine-tuned model not found. Falling back to the Raw Base Model (this may cause hallucinations!)")
        model_to_load = "models/Llama-3.2-1B"
        
    sentinel = FraudSentinelPipeline(model_path=model_to_load)
    
    print(f"Loaded {len(df)} transactions. Starting processing on GPU...")
    
    final_results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing TXN: {row['transaction_id']} ({idx + 1}/{len(df)})...")
        
        # Convert row to dictionary format expected by the pipeline
        txn_data = row.to_dict()
        
        # Run through the SLM
        result = sentinel.analyze_transaction(txn_data)
        final_results.append(result)
        
    print("\nBatch Processing Complete!")
    
    # Save to strict JSON format
    output_file = "final_predictions.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Saved strict JSON output to {output_file}")
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()
