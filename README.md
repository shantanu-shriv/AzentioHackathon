# AI Engineering Hackathon: Fraud Sentinel Pipeline

This repository contains an end-to-end Python pipeline for the AI Engineering Hackathon. It processes messy relational data, synthesizes fraud labels using banking heuristics, fine-tunes a Small Language Model (SLM) via QLoRA, and deploys a strict JSON inference pipeline with adversarial prompt injection defenses.

## 🚀 How to Run the Project

This pipeline is designed to be executed sequentially. Follow the steps below:

### 1. Environment Setup
Install the required dependencies. Note that to run the fine-tuning script, you **must** have a CUDA-enabled GPU and the appropriate PyTorch build installed.

```bash
pip install pandas numpy torch transformers accelerate peft bitsandbytes trl datasets huggingface_hub
```

### 2. Download the SLM
We use `meta-llama/Llama-3.2-1B` as our Tiny/Small Language Model (< 3B parameters). Because this is a gated model, you will need to provide your Hugging Face Access Token when prompted.

```bash
python download_slm.py
```
*(This will securely download the model weights into the local `models/Llama-3.2-1B` directory).*

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
Because the raw data lacks an explicit `is_fraud` label, we dynamically synthesize labels using advanced banking heuristics (e.g., abnormally high transaction ratios, foreign/unauthenticated transactions).

```bash
python prepare_dataset.py
```
*(This merges the three cleaned datasets and outputs a Llama-3 instruction-formatted JSONL file: `train_dataset.jsonl`).*

### 5. Fine-Tune the SLM (Requires GPU)
To improve performance, we fine-tune the 1B parameter model using **QLoRA (4-bit quantization)**. This allows the model to train efficiently on consumer hardware (e.g., a 6GB RTX 3060).

```bash
python finetune.py
```
*(This uses the `trl` SFTTrainer to fine-tune the SLM and saves the final weights to the `fraud_sentinel_final/` directory. **Note:** If your environment lacks CUDA toolkit access, run this script in Google Colab or Kaggle).*

### 6. Run the Inference Pipeline
The final deliverable. This pipeline intercepts raw transaction JSON, sanitizes it to neutralize adversarial prompt injections (e.g., "ignore previous instructions"), and queries the SLM to output a strict, parsable JSON risk profile.

```bash
python inference_pipeline.py
```
*(This will output the final JSON schema containing `transaction_id`, `is_fraud`, `confidence`, and `justification`).*

---
*Built for the AI Engineering Hackathon - Relational Data Wrangler & Fraud Sentinel Challenge.*
