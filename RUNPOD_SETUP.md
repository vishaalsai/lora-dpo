# 🚀 Runpod Setup Guide — Project 4

## Step 1: Create a Runpod Account
1. Go to https://runpod.io
2. Sign up and add credits ($10 will cover the entire project)

## Step 2: Launch a GPU Pod

1. Click **"Deploy"** → **"GPU Pod"**
2. Select this template: **`RunPod Pytorch 2.1`**
3. Select GPU: **A100 SXM 40GB** (or 80GB if available)
   - Cost: ~$1.99/hr (A100 40GB)
   - You need roughly 2–3 hours total across all training runs
4. Set storage: **30 GB container disk** (needed for model weights)
5. Click **"Deploy On-Demand"**

## Step 3: Open JupyterLab

1. Once pod status is **"Running"**, click **"Connect"**
2. Click **"Connect to Jupyter Lab"**
3. You'll see a JupyterLab interface in your browser

## Step 4: Upload Project Files

In JupyterLab, open a Terminal and run:
```bash
git clone https://github.com/YOUR_USERNAME/project4-lora-dpo.git
cd project4-lora-dpo
```

OR upload files directly via the JupyterLab file browser (drag and drop).

## Step 5: Get Your W&B API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key
3. Paste it into `00_setup.ipynb` alongside your HuggingFace token

## Step 6: Run Notebooks IN ORDER

```
00_setup.ipynb           ← Run first, verify GPU shows A100
01_data_prep.ipynb       ← Generates train/test data files
02_baseline_eval.ipynb   ← CRITICAL: saves baseline_metrics.json
03_sft_qlora.ipynb       ← ~30-45 min training
04_dpo_train.ipynb       ← ~20-30 min training
05_final_comparison.ipynb ← Generates your comparison table + README
```

## ⚠️ Important: Stop Your Pod When Done!

Runpod charges by the hour. When you're done training:
1. Download your `./outputs/` folder and `./data/` folder to your local machine
2. Click **"Stop Pod"** in Runpod dashboard
3. Your files persist in pod storage even when stopped (you only pay for storage, not GPU)

## Cost Estimate

| Task | Time | Cost (A100 40GB @ $1.99/hr) |
|---|---|---|
| Baseline eval | ~15 min | ~$0.50 |
| SFT training | ~45 min | ~$1.50 |
| DPO training | ~30 min | ~$1.00 |
| **Total** | **~1.5 hrs** | **~$3.00** |
