# 🧠 Fine-Tuning Qwen2.5-7B for Structured JSON Extraction
### LoRA + QLoRA Supervised Fine-Tuning → DPO Preference Tuning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![TRL](https://img.shields.io/badge/TRL-0.9.6-green)
![WandB](https://img.shields.io/badge/Tracked-W%26B-ffbe00?logo=weightsandbiases)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A40%2048GB-76b900?logo=nvidia)

---

## 🎯 What This Project Does

Takes **messy, unstructured natural language text** and fine-tunes a 7B parameter LLM to reliably extract **clean, schema-compliant JSON** — a task where even carefully prompted base models fail ~61% of the time.

**Example:**
```
Input:  "Please add Alice Johnson to the mailing list. City: Austin,
         Age: 32, Email address is alice@example.com."

Output: {"name": "Alice Johnson", "age": 32,
         "email": "alice@example.com", "city": "Austin"}
```

---

## 📊 Results: 3-Way Comparison

| Model | JSON Validity | Exact Match | Field Coverage |
|---|:---:|:---:|:---:|
| 🔴 Base Model (Qwen2.5-7B-Instruct) | 100% | 39% | 80.2% |
| 🔵 SFT Model (QLoRA fine-tuned) | 100% | **100%** | **100%** |
| 🟢 DPO Model (SFT + DPO) | 100% | **100%** | **100%** |

> **Key finding:** Exact match accuracy jumped from **39% → 100%** after QLoRA fine-tuning — a +61% improvement. The base model understood JSON structure but consistently failed on value accuracy, field naming consistency, and data types.

![Comparison Chart](data/comparison_chart.png)

---

## 📈 Training Curves (SFT Run)

The SFT training loss curve shows smooth, consistent convergence over 315 steps across 3 epochs:

- **Train loss:** 2.0 → 0.0275 (smooth decline, no instability)
- **Eval loss:** 0.038 → 0.028 (tracks train loss — no overfitting)
- **Grad norm:** stabilized quickly after initial steps
- **Learning rate:** cosine decay from 2e-4 → 0.0

> Both train and eval loss converge together — a sign the model is generalizing, not memorizing.

---

## 🏗️ Architecture & Approach

### Why Fine-Tune at All?
The base model with careful prompt engineering achieved only 39% exact match. Common failure modes:
- Wrapping JSON in markdown code fences (` ```json `) — breaks downstream parsers
- Adding explanatory text before/after the JSON object
- Inconsistent field names (`product_name` vs `productName`)
- Wrong data types (`"true"` string instead of `true` boolean)
- Hallucinating fields not present in the input text

Fine-tuning on 1,687 task-specific examples eliminated all of these failure modes.

### Why QLoRA?
Full fine-tuning of a 7B model requires ~80GB+ VRAM. QLoRA (4-bit quantization + LoRA adapters) achieves the same task-specific improvement using only ~22GB VRAM on a single A40 — making it accessible and cost-effective.

### Why DPO After SFT?
DPO teaches the model to **prefer** correct outputs over flawed ones by showing it explicit comparisons. In this project, SFT achieved 100% exact match leaving minimal room for DPO to show incremental gains — but the preference training pipeline is fully implemented and documented for real-world application on noisier datasets.

---

## 🗂️ Project Structure

```
├── 00_setup.ipynb              # Environment setup, auth, GPU verification
├── 01_data_prep.ipynb          # Dataset generation: 1,875 SFT + 1,215 DPO examples
├── 02_baseline_eval.ipynb      # Base model evaluation — Row 1 of comparison table
├── 03_sft_qlora.ipynb          # QLoRA fine-tuning — Row 2
├── 04_dpo_train.ipynb          # DPO preference tuning — Row 3
├── 05_final_comparison.ipynb   # 3-way comparison table + bar chart
├── data/
│   ├── baseline_metrics.json   # Base model scores
│   ├── sft_metrics.json        # SFT model scores
│   ├── dpo_metrics.json        # DPO model scores
│   └── comparison_chart.png    # Visual comparison
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool | Details |
|---|---|---|
| Base Model | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B params, strong structured output capability |
| Fine-Tuning | QLoRA (4-bit NF4) | rank=16, alpha=32, dropout=0.05 |
| Target Modules | q/k/v/o projections + MLP | All attention + feed-forward layers |
| SFT Framework | HuggingFace TRL SFTTrainer | 3 epochs, lr=2e-4, cosine decay |
| DPO Framework | HuggingFace TRL DPOTrainer | 1 epoch, beta=0.1, lr=5e-5 |
| Compute | NVIDIA A40 48GB (Runpod) | ~$3 total for full training run |
| Experiment Tracking | Weights & Biases | Loss curves, eval metrics, hyperparams |

---

## 📦 Dataset

**SFT Dataset:** 1,875 synthetic JSON extraction examples across 4 real-world schemas:
- `person` — name, age, email, city
- `product` — product_name, price, category, in_stock
- `event` — event_name, date, location, organizer
- `invoice` — invoice_number, amount, due_date, client_name

Each schema has multiple messy text templates to ensure distribution diversity.

**DPO Dataset:** 1,215 preference pairs (prompt, chosen, rejected) derived from SFT data.
Rejected outputs simulate real model failure modes:
- Invalid JSON (single quotes instead of double)
- Missing fields
- Extra markdown wrapping
- Wrong nested structure

**Split:** 90% train / 10% test, held out before any training.

---

## 🔧 LoRA Configuration

```python
LoraConfig(
    r=16,                    # Rank — balances capacity vs VRAM
    lora_alpha=32,           # Scaling = 2x rank (standard practice)
    lora_dropout=0.05,       # Light regularization
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"        # MLP
    ]
)
# Trainable parameters: ~40M / 7B total (0.57%)
```

---

## 🐛 What Went Wrong & How I Fixed It

This section documents the real engineering challenges — because debugging is what actual ML work looks like.

### 1. Dependency conflicts on Runpod
`wandb`, `pydantic` and `typing_extensions` had version mismatches causing import errors on the pre-built PyTorch image. Fixed by upgrading all three packages before running any notebooks.

### 2. SFTTrainer rejecting message format
`SFTTrainer` expected a `str` column but received a list of message dicts. Fixed by applying the tokenizer's chat template during dataset preparation to convert messages → formatted strings, then setting `dataset_text_field="text"`.

### 3. TRL DPOTrainer collator bug with Qwen tokenizer
`DPODataCollatorWithPadding` produced `None` token IDs with Qwen2.5's tokenizer regardless of TRL version (0.7.11, 0.8.6, 0.9.6 all affected). Diagnosed with a custom tokenization check — data was clean, bug was inside TRL's collator. Fixed by implementing a `SafeDPOCollator` with explicit padding and the exact key names TRL's `concatenated_inputs` expects (`chosen_input_ids`, `rejected_input_ids`, etc.).

### 4. DPO loss collapsed to zero after step 50
SFT achieved 100% exact match, leaving no preference signal for DPO to learn from on this synthetic dataset. This is expected behaviour — the fix for real-world deployment is to use a harder, noisier dataset where SFT alone doesn't saturate performance. The full DPO pipeline is correctly implemented and validated.

### 5. GitHub push protection blocked secrets
HuggingFace token was accidentally committed inside notebook cells. Fixed by using `sed` to replace tokens with placeholders, amending the commit, and force-pushing. Token was also regenerated immediately.

---

## 🚀 Reproducing This Project

```bash
# 1. Clone
git clone https://github.com/vishaalsai29/lora-dpo.git
cd lora-dpo

# 2. Launch on GPU (Runpod A40 recommended)
# Use template: RunPod PyTorch 2.1

# 3. Install dependencies
pip install transformers==4.44.0 trl==0.9.6 peft==0.12.0 \
    bitsandbytes==0.43.3 accelerate==0.33.0 datasets wandb rich matplotlib

# 4. Add your tokens to each notebook
HF_TOKEN = "hf_..."
WANDB_API_KEY = "..."

# 5. Run notebooks in order: 00 → 01 → 02 → 03 → 04 → 05
```

**Estimated cost:** ~$3 on Runpod A40 (~2 hours total GPU time)

---

## 📚 References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [DPO Paper](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl)
