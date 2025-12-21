import os

# Directory to cache Hugging Face models and datasets
HF_CACHE_DIR = "/home/vincent/storage/huggingface/cache"

if not os.path.exists(HF_CACHE_DIR):
    HF_CACHE_DIR = None