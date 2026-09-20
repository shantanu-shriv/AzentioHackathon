# AI Engineering Hackathon: Fraud Sentinel Pipeline

This repository contains an end-to-end Python pipeline for the AI Engineering Hackathon. It processes messy relational data, synthesizes fraud labels using Machine Learning anomaly detection, fine-tunes a Small Language Model (SLM) via QLoRA, and deploys a strict JSON inference pipeline with adversarial prompt injection defenses.

## 🧠 Architecture Overview: Hybrid Fraud Detection

This project uses a clever hybrid architecture to solve the "Cold Start" problem:
1. **Unsupervised ML for Ground Truth:** Because the raw datasets lacked an explicit `is_fraud` label, we use Scikit-Learn's **Isolation Forest** (an unsupervised algorithm). It mathematically analyzes the data to detect the weirdest outliers (e.g. extreme amounts, foreign devices) and generates synthetic `is_fraud` labels.
2. **SLM for Reasoning & Justification:** While the Isolation Forest is great at math, it can't read text or explain *why* something is fraud. We use the synthetic labels to fine-tune a Llama-3 1B SLM. During inference, the SLM acts as the final judge—reading user notes, understanding context, neutralizing prompt injections, and generating a **human-readable justification** in strict JSON format.

## 🚀 How to Run the Project

This pipeline is designed to be executed sequentially. Follow the steps below:

### 1. Lightning-Fast Environment Setup (Using `uv`)
We highly recommend using `uv` to manage the environment and ensure the correct CUDA versions are installed, especially on Windows.

```bash
# 1. Install uv (if you don't have it)
irm https://astral.sh/uv/install.ps1 | iex

# 2. Create a virtual environment with Python 3.12
uv venv venv --python 3.12

# 3. Activate the environment
.\venv\Scripts\activate   # (PowerShell)
# OR .\venv\Scripts\activate.bat (CMD)

# 4. Install all dependencies from requirements.txt
uv pip install -r requirements.txt

# 5. Install PyTorch with CUDA 12.1 support (Required for RTX/NVIDIA GPUs)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
This runs the Isolation Forest logic to mathematically flag anomalies and synthesize our "ground truth" for the SLM.

```bash
python prepare_dataset.py
```
*(This merges the cleaned datasets, runs anomaly detection, and outputs a Llama-3 instruction-formatted JSONL file: `train_dataset.jsonl`).*

### 5. Fine-Tune the SLM (Requires CUDA GPU)
To teach the model how to output strict JSON and understand the fraud heuristics, we fine-tune the 1B parameter model using **QLoRA (4-bit quantization)**. 

```bash
python finetune.py
```
*(This uses the `trl` SFTTrainer to fine-tune the SLM on your GPU and saves the final weights to the `fraud_sentinel_final/` directory).*

### 6. Run the Final GPU Inference Pipeline
The final deliverable. This pipeline iterates through the entire dataset, sanitizes transaction notes to neutralize adversarial prompt injections (e.g., "ignore previous instructions"), and queries the GPU-accelerated SLM to output a strict, parsable JSON risk profile.

```bash
python batch_inference.py
```
*(This will rapidly process your transactions on the GPU and output the final predictions to `final_predictions.json`.)*

---
*Built for the AI Engineering Hackathon - Relational Data Wrangler & Fraud Sentinel Challenge.*
