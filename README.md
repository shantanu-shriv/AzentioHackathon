# AI Engineering Hackathon: Fraud Sentinel Pipeline

This repository contains an end-to-end Python pipeline for the AI Engineering Hackathon. It processes messy relational data, synthesizes fraud labels using Machine Learning anomaly detection, fine-tunes a Small Language Model (SLM) via QLoRA, and deploys a strict JSON inference pipeline with adversarial prompt injection defenses.

## 🚀 How to Run the Project

This pipeline is designed to be executed sequentially. Follow the steps below:

### 1. Environment Setup
Install the required dependencies. 

```bash
pip install pandas numpy torch transformers accelerate peft bitsandbytes trl datasets huggingface_hub scikit-learn
```

### 2. Download the SLM
We use `meta-llama/Llama-3.2-1B` as our Tiny/Small Language Model (< 3B parameters). Because this is a gated model, you will need to provide your Hugging Face Access Token when prompted.

```bash
python download_slm.py
```
*(This will securely download the ~2.5GB model weights into the local `models/Llama-3.2-1B` directory).*

### 3. Clean the Relational Data
The raw data is messy, incomplete, and contains anomalies. We must clean and merge it.

```bash
# Clean the transactions dataset (drops >50% nulls, imputes categoricals, fixes anomalies)
python clean_transactions.py

# Clean the accounts and customers datasets
python clean_others.py
```
*(This produces `transactions_cleaned.csv`, `accounts_cleaned.csv`, and `customers_cleaned.csv`).*

### 4. Synthesize the Training Dataset
Because the raw data lacks an explicit `is_fraud` label, we dynamically synthesize labels using advanced **Unsupervised Machine Learning**. We use Scikit-Learn's `IsolationForest` to mathematically detect and flag the weirdest 2% of anomalous transactions.

```bash
python prepare_dataset.py
```
*(This merges the cleaned datasets, runs anomaly detection, and outputs a Llama-3 instruction-formatted JSONL file: `train_dataset.jsonl`).*

### 5. Fine-Tune the SLM (Requires CUDA GPU)
To teach the model how to output strict JSON and understand the fraud heuristics, we fine-tune the 1B parameter model using **QLoRA (4-bit quantization)**. 
*Note: This script requires a native CUDA-enabled GPU and bitsandbytes. If your local machine is CPU-only, you must run this script in Google Colab or Kaggle.*

```bash
python finetune.py
```
*(This uses the `trl` SFTTrainer to fine-tune the SLM and saves the final weights to the `fraud_sentinel_final/` directory).*

### 6. Run the Final Inference Pipeline
The final deliverable. This pipeline intercepts raw transaction JSON, sanitizes it to neutralize adversarial prompt injections (e.g., "ignore previous instructions"), and queries the SLM to output a strict, parsable JSON risk profile.

```bash
python batch_inference.py
```
*(This will process your transactions and output the final `final_predictions.json`. **Note**: If run on a CPU-only machine without the fine-tuned adapter weights from Step 5, the Base Model will hallucinate instead of returning strict JSON. This correctly demonstrates the necessity of the fine-tuning pipeline!)*

---
*Built for the AI Engineering Hackathon - Relational Data Wrangler & Fraud Sentinel Challenge.*
