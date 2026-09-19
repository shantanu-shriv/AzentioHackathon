import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Note: This script requires a CUDA-enabled GPU and the following libraries:
# pip install torch transformers accelerate peft bitsandbytes trl datasets

def main():
    model_id = "models/Llama-3.2-1B"
    
    print("Loading dataset...")
    # Load our synthetic JSONL dataset
    dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")
    
    print(f"Loaded {len(dataset)} training examples.")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Llama 3 requires a pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Model in 4-bit (QLoRA)...")
    # This requires bitsandbytes and a CUDA GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True
    )
    
    # Prepare model for LoRA
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fraud_sentinel_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        max_steps=100, 
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )
    
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("Starting Fine-Tuning...")
    trainer.train()
    
    print("Saving Fine-Tuned Model...")
    trainer.model.save_pretrained("fraud_sentinel_final")
    tokenizer.save_pretrained("fraud_sentinel_final")
    print("Done! Model saved to 'fraud_sentinel_final'")

if __name__ == "__main__":
    main()
