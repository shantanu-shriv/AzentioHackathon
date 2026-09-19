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
    # Note: We are using the base model since fine-tuning requires a CUDA GPU.
    sentinel = FraudSentinelPipeline(model_path="models/Llama-3.2-1B")
    
    print(f"Loaded {len(df)} transactions. Starting batch processing...")
    
    final_results = []
    
    # Process the first 5 transactions as a proof-of-concept for the hackathon
    # (Running all rows on a CPU would take hours)
    for idx, row in df.head(5).iterrows():
        print(f"Processing TXN: {row['transaction_id']}...")
        
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
