# 🔥 Pyroton

> A fine-tuned Python code generation model built on top of Qwen2.5-Coder-0.5B-Instruct.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📖 Overview

**Pyroton** is a lightweight Python code generation model fine-tuned from [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) using supervised fine-tuning (SFT) on the [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) dataset.

The goal of this project is to produce a small, efficient model that generates clean Python code from natural language instructions.

---

## ✨ Features

- 🐍 **Python-focused** — trained exclusively on Python instruction-output pairs
- ⚡ **Lightweight** — 0.5B parameters, runs on free-tier GPUs
- 🧠 **Instruction-following** — understands natural language prompts
- 🔁 **Chunked training** — resilient to cloud session interruptions
- ☁️ **Auto-saved to HuggingFace** — model pushed after every training chunk

---

## 🏗️ Project Structure

```
pyroton/
├── notebooks/
│   └── llm_final.ipynb        # Full training notebook
├── scripts/
│   └── Modelfile              # Ollama integration file
├── assets/
│   └── (diagrams, images)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🚀 Quick Start

### Run with Ollama (Local)

```bash
# 1. Pull the model (after GGUF conversion)
ollama create pyroton -f Modelfile

# 2. Run it
ollama run pyroton
```

### Run with HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_HF_USERNAME/pyroton",
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/pyroton")

inputs = tokenizer(
    "### Instruction:\nWrite a Python function to reverse a string\n\n### Response:\n",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.3,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🧪 Example Output

**Prompt:**
```
Write a Python function to check if a number is prime
```

**Pyroton Output:**
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

---

## 🛠️ Training Details

| Setting | Value |
|---|---|
| Base Model | Qwen2.5-Coder-0.5B-Instruct |
| Dataset | python_code_instructions_18k_alpaca |
| Dataset Size | 18,612 examples |
| Training Strategy | Chunked SFT (5 chunks) |
| Epochs per Chunk | 1 |
| Batch Size | 2 |
| Gradient Accumulation | 8 steps |
| Learning Rate | 2e-4 |
| Precision | BFloat16 |
| Max Sequence Length | 512 tokens |
| Final Training Loss | ~0.57 |
| Token Accuracy | ~84.5% |

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/pyroton.git
cd pyroton
pip install -r requirements.txt
```

---

## 🤗 HuggingFace Model

The fine-tuned model is available on HuggingFace:
👉 [YOUR_HF_USERNAME/pyroton](https://huggingface.co/YOUR_HF_USERNAME/pyroton)

---

## 📋 Requirements

See `requirements.txt` for the full list. Main dependencies:

- `transformers`
- `datasets`
- `trl`
- `peft`
- `bitsandbytes`
- `accelerate`

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.

Base model (Qwen2.5-Coder) is also Apache 2.0. Attribution to Alibaba Cloud / Qwen Team.

---

## 🙏 Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for the base model
- [iamtarun](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) for the Python dataset
- [HuggingFace](https://huggingface.co) for the model hub and libraries

---

<p align="center">Made with 🔥 by <strong>YOUR_NAME</strong></p>
