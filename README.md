# [Unified Residual Learning and Exogenous Anchors](#)

## Download  Repository

  ```bash
  git clone https://github.com/jon123boss/ExoFormer
  cd ExoFormer
  ```

## Prerequisites

Install required dependencies via pip:
```bash
pip install flash-attn --no-build-isolation
pip install tiktoken
pip install huggingface-hub
pip install lm_eval
pip install hf_transfer
pip install wandb  # Optional, for experiment tracking
```

## Data Preparation

Download and preprocess the GPT-2 tokenized FinewebEDU10B dataset:
```bash
python prepdata.py
```

## Training Configuration

1. Edit hyperparameters in `train.py` (default settings are for E-ExoFormer)
2. Launch training:
   ```bash
   python train.py
   ```
   *Note: You will be prompted to log in to Weights & Biases (optional) if you turn it on*

## Evaluation

### Validation Set Evaluation
To evaluate on the full validation set:
1. In `train.py`, modify:
    - `eval_steps = 3052`
   - `eval_only = True`
   - `init_from = 'resume'`
   - `ckpt_file_name = 'out/ckpt_step:38146.pt'` (replace with your checkpoint)

2. Run evaluation:
   ```bash
   python train.py
   ```

### Downstream Task Evaluation
Evaluate on benchmark tasks using:
```bash
python run_eval.py --ckpts out/ckpt_step:38146.pt 
```

## Pre-trained Models

Pre-trained models from the paper are available on Hugging Face Hub:

**Repository:** [https://huggingface.co/Jonnester](https://huggingface.co/Jonnester)

### Download Instructions

1. Use the provided `hfcopy.py` script to download models:
   ```python
   # hfcopy.py
   from huggingface_hub import hf_hub_download

   file_path = hf_hub_download(
       repo_id="Jonnester/Baseline",
       filename="ckpt_step:38146.pt",
       local_dir="",
       local_dir_use_symlinks=False
   )
   print(f"Downloaded to: {file_path}")
   ```

2. Execute the script:
   ```bash
   python hfcopy.py
   ```

3. Move the downloaded checkpoint to the `out/` directory for evaluation.

*Note: Replace checkpoint filenames and model names with your specific paths and desired models from the repository.*
