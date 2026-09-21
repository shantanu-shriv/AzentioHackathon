import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("Loading Ground Truth from train_dataset.jsonl...")
    ground_truth = {}
    try:
        with open("train_dataset.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                # The assistant's response contains the ground truth JSON string
                assistant_content = data["messages"][2]["content"]
                gt_json = json.loads(assistant_content)
                ground_truth[gt_json["transaction_id"]] = gt_json["is_fraud"]
    except Exception as e:
        print(f"Error loading train_dataset.jsonl: {e}")
        return

    print("Loading Predictions from final_predictions.json...")
    predictions = {}
    try:
        with open("final_predictions.json", "r") as f:
            preds_list = json.load(f)
            for p in preds_list:
                predictions[p["transaction_id"]] = p["is_fraud"]
    except Exception as e:
        print(f"Error loading final_predictions.json: {e}")
        return

    y_true = []
    y_pred = []
    
    # Align the lists
    for txn_id, pred_is_fraud in predictions.items():
        if txn_id in ground_truth:
            y_true.append(ground_truth[txn_id])
            y_pred.append(pred_is_fraud)
            
    if not y_true:
        print("No matching transactions found to evaluate!")
        return
        
    print(f"\nSuccessfully matched {len(y_true)} transactions for evaluation.\n")
    
    print("=== Evaluation Results ===")
    print(classification_report(y_true, y_pred, target_names=["Normal (False)", "Fraud (True)"]))
    
    # Calculate specific metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"F1-Score (Fraud Class): {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"True Negatives (Normal correctly flagged as Normal): {tn}")
    print(f"False Positives (Normal incorrectly flagged as Fraud): {fp}")
    print(f"False Negatives (Fraud missed by AI): {fn}")
    print(f"True Positives (Fraud correctly caught by AI): {tp}")

if __name__ == "__main__":
    main()
