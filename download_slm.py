import os
import getpass
from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-3.2-1B"
local_dir = os.path.join(os.getcwd(), "models", "Llama-3.2-1B")

print("Because Llama 3.2 is a gated model, you must provide your HuggingFace token.")
print("You can get one from: https://huggingface.co/settings/tokens")
hf_token = getpass.getpass("Paste your HF Token (it will be hidden as you type) and press Enter: ")

print(f"\nDownloading {model_id} to {local_dir}...")
print("This will download ~2.5GB of data. Please wait...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, 
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "coreml/*", "*.onnx", "*.pt"] 
    )
    print(f"\nDownload complete! Model saved perfectly in {local_dir}")
except Exception as e:
    print(f"\nFailed to download: {e}")
