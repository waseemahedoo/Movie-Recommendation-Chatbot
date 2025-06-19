from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    local_dir=os.getenv('MODEL_PATH'),
    local_dir_use_symlinks=False
)
