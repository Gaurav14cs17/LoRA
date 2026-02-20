# Chapter 7: References & Further Reading

> Papers, codebases, tutorials, and resources for going deeper.

---

### How to Use This Chapter

- **New to LoRA?** Read the **Core Paper** first, then follow with QLoRA and DoRA
- **Implementing from scratch?** Check **Code Repositories** → HuggingFace PEFT and Microsoft LoRA
- **Writing a paper?** Use the **BibTeX Citations** at the bottom
- **Need a quick refresher?** See the **Glossary** at the end

---

## Core Papers

### The LoRA Paper

| | |
|---|---|
| **Title** | LoRA: Low-Rank Adaptation of Large Language Models |
| **Authors** | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| **Affiliation** | Microsoft |
| **Year** | 2021 |
| **Link** | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| **Key Contribution** | Low-rank decomposition of weight updates for parameter-efficient fine-tuning with zero inference latency overhead |

---

## LoRA Variants

| Paper | Year | Key Idea | Link |
|-------|------|----------|------|
| **QLoRA** — Efficient Finetuning of Quantized Language Models | 2023 | 4-bit NF4 quantization + LoRA | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) |
| **AdaLoRA** — Adaptive Budget Allocation for PEFT | 2023 | Dynamic rank allocation via importance scoring | [arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512) |
| **LoRA+** — Efficient Low Rank Adaptation of Large Models | 2024 | Asymmetric learning rates for A and B | [arxiv.org/abs/2402.12354](https://arxiv.org/abs/2402.12354) |
| **DoRA** — Weight-Decomposed Low-Rank Adaptation | 2024 | Magnitude-direction decomposition | [arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353) |
| **GLoRA** — Generalized LoRA for Parameter-Efficient Fine-tuning | 2024 | Generalized adapter search space | [arxiv.org/abs/2306.07967](https://arxiv.org/abs/2306.07967) |
| **LongLoRA** — Efficient Fine-tuning of Long-Context LLMs | 2024 | Shifted sparse attention for long context | [arxiv.org/abs/2309.12307](https://arxiv.org/abs/2309.12307) |
| **S-LoRA** — Serving Thousands of Concurrent LoRA Adapters | 2023 | Unified paging for multi-adapter serving | [arxiv.org/abs/2311.03285](https://arxiv.org/abs/2311.03285) |
| **LoRA-FA** — Memory-efficient Low-rank Adaptation | 2023 | Freeze A matrix, train only B | [arxiv.org/abs/2308.03303](https://arxiv.org/abs/2308.03303) |
| **Delta-LoRA** — Fine-Tuning High-Rank Parameters with Low-Rank Deltas | 2023 | Propagate LoRA deltas to base weights | [arxiv.org/abs/2309.02411](https://arxiv.org/abs/2309.02411) |
| **LoRAHub** — Efficient Cross-Task Generalization | 2024 | Compose pre-trained LoRA adapters for new tasks | [arxiv.org/abs/2307.13269](https://arxiv.org/abs/2307.13269) |
| **MELoRA** — Mini-Ensemble Low-Rank Adapters | 2024 | Ensemble of small LoRA modules | [arxiv.org/abs/2402.17263](https://arxiv.org/abs/2402.17263) |
| **rsLoRA** — Rank Stabilization Scaling Factor | 2024 | Replace α/r with α/√r scaling | [arxiv.org/abs/2312.03732](https://arxiv.org/abs/2312.03732) |
| **PiSSA** — Principal Singular Values Adaptation | 2024 | SVD-based initialization | [arxiv.org/abs/2404.02948](https://arxiv.org/abs/2404.02948) |
| **O-LoRA** — Orthogonal Low-Rank Adaptation | 2023 | Orthogonal adapters for continual learning | [arxiv.org/abs/2312.02151](https://arxiv.org/abs/2312.02151) |
| **OLoRA** — Orthonormal Low-Rank Adaptation | 2024 | QR-based orthonormal initialization | [arxiv.org/abs/2406.01775](https://arxiv.org/abs/2406.01775) |

---

## Foundational Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| **Intrinsic Dimensionality Explains Effectiveness of Language Model Fine-Tuning** (Aghajanyan et al.) | 2020 | Theoretical foundation for why LoRA works | 
| **Attention Is All You Need** (Vaswani et al.) | 2017 | The Transformer architecture that LoRA adapts |
| **Parameter-Efficient Transfer Learning for NLP** (Houlsby et al.) | 2019 | Serial adapter modules (predecessor) |
| **Prefix-Tuning** (Li & Liang) | 2021 | Continuous prompt-based PEFT |
| **The Power of Scale for Parameter-Efficient Prompt Tuning** (Lester et al.) | 2021 | Soft prompt tuning |
| **BitFit** (Zaken et al.) | 2022 | Bias-only training |
| **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning** (IA3, Liu et al.) | 2022 | Learned rescaling vectors |

---

## Merging and Composition Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| **Editing Models with Task Arithmetic** (Ilharco et al.) | 2023 | Add/subtract task vectors |
| **TIES-Merging** (Yadav et al.) | 2023 | Trim, elect sign, merge |
| **DARE** (Yu et al.) | 2024 | Drop and rescale for merging |
| **Model Soups** (Wortsman et al.) | 2022 | Average multiple fine-tuned models |

---

## Code Repositories

### Training Libraries

| Repository | Description | Link |
|-----------|-------------|------|
| **HuggingFace PEFT** | Production-ready LoRA, QLoRA, AdaLoRA, DoRA, etc. | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| **Microsoft LoRA** | Original LoRA implementation | [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA) |
| **Unsloth** | 2x faster LoRA training with custom kernels | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| **Axolotl** | Easy-to-use fine-tuning framework with LoRA | [github.com/OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) |
| **LLaMA-Factory** | All-in-one LLM fine-tuning with LoRA | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **TRL** | Transformer Reinforcement Learning (SFT, DPO with LoRA) | [github.com/huggingface/trl](https://github.com/huggingface/trl) |

### Serving Frameworks

| Repository | Description | Link |
|-----------|-------------|------|
| **vLLM** | High-throughput serving with LoRA support | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **TGI** | HuggingFace Text Generation Inference | [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| **S-LoRA** | Serve thousands of LoRA adapters | [github.com/S-LoRA/S-LoRA](https://github.com/S-LoRA/S-LoRA) |

### Diffusion Model LoRA

| Repository | Description | Link |
|-----------|-------------|------|
| **Diffusers** | LoRA training and inference for Stable Diffusion | [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) |
| **kohya-ss/sd-scripts** | Popular SD LoRA training toolkit | [github.com/kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) |
| **CivitAI** | Community hub for sharing LoRA models | [civitai.com](https://civitai.com) |

---

## Tutorials and Guides

| Resource | Type | Link |
|----------|------|------|
| HuggingFace PEFT Documentation | Official Docs | [huggingface.co/docs/peft](https://huggingface.co/docs/peft) |
| Fine-tune LLaMA 2 with LoRA | Tutorial | [huggingface.co/blog/llama2](https://huggingface.co/blog/llama2) |
| Sebastian Raschka's LoRA Blog | Technical Blog | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com) |
| Maxime Labonne's LLM Course | Course | [github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course) |
| Weights & Biases LoRA Guide | Guide | [wandb.ai/docs](https://wandb.ai/docs) |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Rank (r)** | Dimensionality of the low-rank decomposition; controls expressiveness vs. parameter count |
| **Alpha (α)** | Scaling numerator; effective scale is α/r (standard) or α/√r (rsLoRA) |
| **Target modules** | Which linear layers receive LoRA adapters (e.g., `q_proj`, `v_proj`, or `"all-linear"`) |
| **Merging** | Combining `BA` into `W₀` for zero-overhead inference: `W' = W₀ + (α/r)·BA` |
| **Adapter** | The saved LoRA weights — just the `B` and `A` matrices plus a config JSON |
| **PEFT** | Parameter-Efficient Fine-Tuning — the broader field LoRA belongs to |
| **NF4** | NormalFloat 4-bit — the quantization format used in QLoRA |
| **Intrinsic dimensionality** | The effective dimensionality of the fine-tuning subspace (typically much smaller than full parameter count) |
| **Task vector** | The difference `W_finetuned - W_base`, used in adapter arithmetic |
| **TIES** | Trim, Elect Sign, Merge — a conflict-aware adapter merging strategy |
| **DARE** | Drop And REscale — randomly drop adapter weights before merging to reduce interference |
| **Orthogonal constraint** | Forcing new adapters to be perpendicular to old ones (prevents forgetting in continual learning) |

---

## BibTeX Citations

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized Language Models},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}

@article{zhang2023adalora,
  title={Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2303.10512},
  year={2023}
}

@article{hayou2024lora+,
  title={LoRA+: Efficient Low Rank Adaptation of Large Models},
  author={Hayou, Soufiane and Ghosh, Nikhil and Yu, Bin},
  journal={arXiv preprint arXiv:2402.12354},
  year={2024}
}

@article{kalajdzievski2023rslora,
  title={A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA},
  author={Kalajdzievski, Damjan},
  journal={arXiv preprint arXiv:2312.03732},
  year={2023}
}
```

---

## Navigation

| Previous | Up |
|----------|------|
| [← Chapter 6: Advanced Topics](../06-Advanced-Topics/README.md) | [Home](../README.md) |
