<h1 align="center">LoRA: Low-Rank Adaptation of Large Language Models</h1>
<h3 align="center">A Complete Deep Dive — From Theory to Production</h3>

<p align="center">
  <img src="https://img.shields.io/badge/chapters-7-blue?style=for-the-badge" alt="Chapters"/>
  <img src="https://img.shields.io/badge/reading_time-~3_hours-orange?style=for-the-badge" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/SVG_diagrams-33-green?style=for-the-badge" alt="Diagrams"/>
  <img src="https://img.shields.io/badge/code_examples-20+-purple?style=for-the-badge" alt="Code Examples"/>
</p>

<p align="center">
  <img src="./01-Introduction/images/lora-core-architecture.svg" alt="LoRA Architecture Overview" width="100%"/>
</p>

<p align="center"><i>The most comprehensive open-source guide to LoRA — covering intuition, rigorous mathematics, 16 variants, from-scratch PyTorch code, and production deployment.</i></p>

---

## Why This Blog Exists

Fine-tuning a **7-billion parameter** model the traditional way needs **~120 GB of GPU memory**. Most teams can't afford that. LoRA changes the equation:

| Metric | Full Fine-Tuning | LoRA | QLoRA |
|--------|:---:|:---:|:---:|
| GPU Memory (7B) | ~120 GB | ~32 GB | **~6 GB** |
| Trainable Params | 100% | 0.1% | 0.1% |
| Adapter Size | 14 GB | **10 MB** | **10 MB** |
| Inference Overhead | 0 | **0** | **0** |
| Quality vs Full FT | Baseline | ~99% | ~98% |

This blog takes you from **"what is LoRA?"** to **"I deployed 100 LoRA adapters in production"** — with every step explained.

---

## Who Is This For?

- **ML Engineers** wanting to fine-tune LLMs on limited hardware
- **Researchers** seeking deep understanding of the math behind LoRA and its variants
- **Students** learning about parameter-efficient methods
- **Practitioners** looking for copy-paste training recipes and production patterns

---

## Chapters

| # | Chapter | What You'll Learn | Reading Time |
|:---:|---------|-------------------|:---:|
| 1 | [**Introduction**](./01-Introduction/README.md) | What problem LoRA solves, how it works at a high level, why it's effective | 10 min |
| 2 | [**Mathematics**](./02-Mathematics/README.md) | SVD, Eckart-Young theorem, gradient derivations, convergence proofs — every equation explained | 45 min |
| 3 | [**Types of LoRA**](./03-Types-of-LoRA/README.md) | 16 variants: QLoRA, DoRA, AdaLoRA, LoRA+, PiSSA, O-LoRA, and more — when to use each | 30 min |
| 4 | [**Implementation**](./04-Implementation/README.md) | From-scratch PyTorch code (Linear, Embedding, Conv2d), HuggingFace PEFT, line-by-line explanations | 40 min |
| 5 | [**Training Guide**](./05-Training-Guide/README.md) | Hyperparameter selection, memory optimization, debugging, task-specific recipes | 20 min |
| 6 | [**Advanced Topics**](./06-Advanced-Topics/README.md) | Adapter merging, multi-LoRA serving, diffusion models, continual learning, LoRA arithmetic | 25 min |
| 7 | [**References**](./07-References/README.md) | 30+ papers, code repositories, tutorials, and BibTeX citations | 5 min |

---

## Quick Decision Guide

Not sure where to start? Use this:

```
I want to...
│
├── "Understand what LoRA is"
│   └── Start with Chapter 1
│
├── "Fine-tune a model RIGHT NOW"
│   └── Jump to the Quick Start below, then Chapter 5
│
├── "Understand the math deeply"
│   └── Chapter 2 (full proofs and derivations)
│
├── "Know which LoRA variant to pick"
│   └── Chapter 3 (comparison table + decision flowchart)
│
├── "Build LoRA from scratch in PyTorch"
│   └── Chapter 4 (line-by-line implementation)
│
├── "Deploy LoRA adapters in production"
│   └── Chapter 6 (merging, serving, multi-adapter)
│
└── "Find the original papers"
    └── Chapter 7 (references and BibTeX)
```

---

## Quick Start (5 Minutes)

### Install

```bash
pip install peft transformers accelerate bitsandbytes
```

### Train a LoRA Adapter

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                          # Rank — start here
    lora_alpha=32,                 # Scaling factor = alpha/r = 2.0
    target_modules="all-linear",   # Apply to every linear layer
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,107,200 || all params: 6,751,322,112 || trainable%: 0.19%

# Train with any standard trainer...
# model.save_pretrained("./my-adapter")  # Saves ~50 MB adapter
```

### Use the Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "./my-adapter")

# Option A: Keep adapter separate (multi-task serving)
output = model.generate(input_ids)

# Option B: Merge for zero-overhead inference
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

---

## The One-Sentence Summary of Each Chapter

1. **Introduction** — LoRA adds a tiny trainable branch (`BA`) alongside each frozen weight, achieving 99% of full fine-tuning quality at 0.1% of the parameter cost.
2. **Mathematics** — The weight update `ΔW` has low intrinsic rank, proven by SVD and the Eckart-Young theorem, with stable gradients guaranteed by careful initialization.
3. **Types** — 16 variants solve specific problems: QLoRA for memory, DoRA for quality, AdaLoRA for efficiency, O-LoRA for continual learning.
4. **Implementation** — A complete LoRA library in ~300 lines of PyTorch, plus production recipes with HuggingFace PEFT.
5. **Training** — Start with `r=16, alpha=32, lr=2e-4, target_modules="all-linear"` and adjust from there.
6. **Advanced** — Merge adapters via task arithmetic, serve 100+ adapters from one GPU, compose skills without retraining.
7. **References** — 30+ papers with links, 15+ code repositories, 5+ tutorials.

---

## Prerequisites

| Topic | Level Required |
|-------|:---:|
| Linear Algebra | Matrix multiplication, rank, eigenvalues |
| Deep Learning | Transformers, attention, backpropagation |
| PyTorch | Basic model training loop |
| Python | Intermediate |

---



<p align="center"><b>Author:</b> Gaurav Goswami &nbsp;|&nbsp; <b>Last updated:</b> February 2026</p>
<p align="center">Start reading → <a href="./01-Introduction/README.md"><b>Chapter 1: Introduction</b></a></p>
