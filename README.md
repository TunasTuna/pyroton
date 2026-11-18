<div align="center">

# 🔥 Pyroton

> A lightweight Python code generation model built on top of Qwen2.5-Coder-0.5B-Instruct.

</div>

---
## Discontinued!
I will not be continuing this project anymore due to the files that I lost.

---

## Overview

**Pyroton** is a lightweight Python-focused code generation project built from [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).

The goal of the project is to create a small, efficient coding model that can handle easy to medium Python tasks while remaining practical for free-tier GPUs and lightweight deployment.

Pyroton was originally trained with supervised fine-tuning (SFT) on Python instruction-style datasets, then improved with targeted repair finetunes focused on correctness bugs.

---

## Run Pyroton locally

The published [Q4_K_M GGUF](https://huggingface.co/shohuu/Pyroton/blob/7b617718c20aed8410a74c7f0fbbf4bef5f763b6/pyroton-q4.gguf) is about 397 MB. From the repository root, with [Ollama](https://docs.ollama.com/) running:

```sh
curl -fL https://huggingface.co/shohuu/Pyroton/resolve/7b617718c20aed8410a74c7f0fbbf4bef5f763b6/pyroton-q4.gguf -o scripts/pyroton-q4.gguf
ollama create pyroton -f scripts/Modelfile
ollama run pyroton "Write a Python function named reverse_string that reverses a string. Return only Python code."
```

In Windows PowerShell, use `curl.exe`. The published model card does not identify the adapter revision merged into this GGUF.

To try the Transformers adapters with Python 3.12 on CPU or CUDA:

```sh
python -m pip install -r requirements-inference.txt
python scripts/generate.py --model primefix "Write a Python function to check whether an integer is prime."
```

Use `--model pyroton` for the original adapter or `--model qwen` for stock Qwen.

---

## Current Model Variants

### Base project adapter
- **Hugging Face:** [`shohuu/pyroton`](https://huggingface.co/shohuu/pyroton)
- General Python instruction-tuned adapter

### Prime-fix patched adapter
- **Hugging Face:** [`shohuu/pyroton-primefix-v3`](https://huggingface.co/shohuu/pyroton-primefix-v3)
- Adds targeted repair finetuning for prime-checker correctness issues
- Best choice if you want the latest corrected adapter from this project iteration

---

## Features

- **Python-focused** — trained on Python-style instruction/output examples
- **Lightweight** — based on a 0.5B parameter coder model
- **Instruction-following** — responds to natural-language coding prompts
- **Chunked training workflow** — practical for Colab/Kaggle interruptions
- **Repair finetuning workflow** — supports targeted bug fixing without retraining from scratch
- **Mobile-friendly direction** — suitable for GGUF export and lightweight deployment workflows

---

## Project Structure

```text
pyroton/
  notebooks/
    train.ipynb        # Main training notebook
  scripts/
    generate.py            # CPU/CUDA inference CLI
    Modelfile              # Ollama integration file
  benchmarks/              # Python correctness tasks and evaluator
  README.md
  requirements.txt
  .gitignore
  LICENSE
```

---

## Training Summary

### Base model
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`

### Main data sources used during project iteration
- `iamtarun/python_code_instructions_18k_alpaca`
- `sahil2801/CodeAlpaca-20k`
- `TokenBender/code_instructions_122k_alpaca_style`

### Data pipeline used
- Combined multiple instruction-style coding datasets
- Reformatted them into Alpaca-style text:
  - `### Instruction:`
  - optional `### Input:`
  - `### Response:`
- Applied a Python-focused filtering pass
- Applied a targeted cleanup step to remove samples using `math.` without the required import

### Important cleanup result
A targeted cleanup removed **570 rows** from the Python-filtered dataset because they used `math.` without `import math` / `from math import ...`.

This was important because the model had learned a recurring bug where it generated `math.sqrt(...)` without importing `math`.

---

## Fine-tuning Strategy

### Main SFT pass
Typical settings used during the cleaned-data retrain:

| Setting | Value |
|---|---|
| Base Model | Qwen2.5-Coder-0.5B-Instruct |
| Training Strategy | Chunked SFT |
| Number of Chunks | 5 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Learning Rate | 1e-4 |
| Precision | BFloat16 |
| Max Length | 512 |

### Repair finetunes
After the main retrain, additional targeted repair finetunes were applied to fix specific correctness bugs in prime-checking outputs:
- missing import / `math.sqrt` issues
- incorrect handling of even numbers
- crashes for negative inputs due to `n**0.5`

The latest targeted repair adapter from this process is:
- `shohuu/pyroton-primefix-v3`

---

## Evaluation Notes

A small execution-based harness was used to evaluate generated `is_prime()` functions against test cases such as:
- `-1`
- `0`
- `1`
- `2`
- `3`
- `4`
- `6`
- `9`
- `17`
- `49`

### Observed result after primefix-v3
- **Greedy decoding (`do_sample=False`)**: **5/5 passing samples** in the prime-checking harness
- **Sampled decoding (`temperature=0.3`)**: improved, but still not fully stable

This means the model is currently most reliable in deterministic correctness-focused decoding mode.

---

## Benchmark correctness and speed

The [benchmark](benchmarks/evaluate.py) scores the first completion for each of 40 short Python tasks. A task passes when its generated function passes every test case. The same prompts and token budget are used across models, with the native chat format for stock Qwen and the Alpaca format for the adapters and GGUF.

Install the inference dependencies for the three Transformers variants. Scoring also requires Docker running Linux containers. Pull the scoring image once, then run the evaluator from the repo root:

```sh
docker pull python:3.12-slim
python benchmarks/evaluate.py --model qwen --output qwen.json
python benchmarks/evaluate.py --model pyroton --output pyroton.json
python benchmarks/evaluate.py --model primefix --output primefix.json
python benchmarks/evaluate.py --model gguf --output gguf.json
```

The JSON reports contain completions, pass counts, model provenance, and throughput. Ollama reports decode-only throughput, while the Transformers measurement includes prompt processing. Use `--limit 1` for a smoke run or `--generate-only` to collect completions without executing them. Run comparisons on the same hardware and record peak RAM separately. Compare quantization levels built from the same adapter revision to isolate the effect of quantization.


## Known Limitations

- The model is still a **0.5B** model, so it can degrade on harder tasks or complex reasoning chains.
- Sampling can still reintroduce mistakes that disappear in greedy mode.
- The project has been most thoroughly tested on short Python coding tasks; broader evaluation across more libraries/tasks is still needed.
- Repair finetuning improved specific failure modes, but it does not replace a large benchmark suite.

---

## Installation

```bash
git clone https://github.com/TunasTuna/pyroton.git
cd pyroton
pip install -r requirements.txt
```

---

## Requirements

Main dependencies used in this project include:

- `transformers`
- `datasets`
- `trl`
- `peft`
- `bitsandbytes`
- `accelerate`
- `torchao`

---

## License

This project is licensed under the **Apache 2.0 License** — see [LICENSE](LICENSE) for details.

Base model (`Qwen2.5-Coder`) is also Apache 2.0. Attribution belongs to Alibaba Cloud / Qwen Team.

---

## Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for the base model
- [iamtarun](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) for the original Python dataset source used early in the project
- [Hugging Face](https://huggingface.co/) for the model hub and training ecosystem
- My friend [Yumi](https://www.tiktok.com/@yumi_naomi6?_r=1&_t=ZS-96ok4qcUKij) for the name of this project
