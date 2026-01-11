# hfcopy.py
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="Jonnester/Baseline",
    filename="ckpt_step:38146.pt",
    local_dir="",
    local_dir_use_symlinks=False
)
print("Downloaded to:", file_path)