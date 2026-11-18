"""Compare first-generation Python correctness and generation throughput."""

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen

from tasks import TASKS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from generate import BASE, MODELS, Generator, alpaca_prompt

IMAGE = "python:3.12-slim"
WORKER = (HERE / "worker.py").read_text(encoding="utf-8")


def extract_code(answer):
    block = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", answer, re.IGNORECASE | re.DOTALL)
    return (block.group(1) if block else answer).strip()


def score(answer, name, cases, image):
    code = base64.b64encode(extract_code(answer).encode()).decode()
    cmd = [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=8m", "--memory=128m",
        "--cpus=1", "--pids-limit=32", "--cap-drop=ALL", "--log-driver=none",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        image, "timeout", "-k", "1s", "3s", "python", "-I", "-B", "-c",
        WORKER, code, name, json.dumps(cases),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
    )
    if result.returncode in (125, 126, 127):
        raise RuntimeError(f"Could not start the scoring container (exit {result.returncode}).")
    return result.returncode == 0


def ollama_request(path, payload):
    request = Request(
        f"http://127.0.0.1:11434/api/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def generate_gguf(model, instruction, budget):
    result = ollama_request("generate", {
        "model": model, "prompt": alpaca_prompt(instruction), "raw": True,
        "stream": False, "options": {
            "temperature": 0, "repeat_penalty": 1.0,
            "num_ctx": 2048, "num_predict": budget, "seed": 0,
        },
    })
    return result["response"], result["eval_count"], result["eval_duration"] / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "gguf"), required=True)
    parser.add_argument("--ollama-model", default="pyroton")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=len(TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= len(TASKS) or args.max_new_tokens < 1:
        parser.error("Choose a valid task limit and a positive output token budget.")

    image = None
    if not args.generate_only:
        if not shutil.which("docker"):
            parser.error("Docker is required for scoring. Use --generate-only to save responses.")
        image = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True,
        ).strip()

    if args.model == "gguf":
        details = ollama_request("show", {"model": args.ollama_model})
        provenance = {"model": args.ollama_model, "details": details["details"],
                      "modelfile": details["modelfile"]}
        def generate(prompt, budget):
            return generate_gguf(args.ollama_model, prompt, budget)
    else:
        from huggingface_hub import model_info

        base_rev = model_info(BASE).sha
        adapter_rev = model_info(MODELS[args.model]).sha if args.model != "qwen" else None
        model = Generator(args.model, base_rev, adapter_rev)
        generate = model.generate
        provenance = {
            "base_revision": base_rev, "adapter_revision": adapter_rev,
            "device": str(model.model.device), "dtype": str(model.model.dtype),
            "packages": {name: version(name) for name in ("torch", "transformers", "peft")},
        }

    results = []
    for name, description, cases in TASKS[:args.limit]:
        prompt = f"Write a Python function named {name}. {description} Return only Python code."
        answer, tokens, seconds = generate(prompt, args.max_new_tokens)
        passed = score(answer, name, cases, image) if image else None
        results.append({
            "task": name, "prompt": prompt, "answer": answer,
            "tokens": tokens, "generation_seconds": seconds, "passed": passed,
        })
        print(f"{name}: {'generated' if passed is None else 'pass' if passed else 'fail'}",
              flush=True)

    elapsed = sum(row["generation_seconds"] for row in results)
    suite = (HERE / "tasks.py").read_text(encoding="utf-8")
    report = {
        "variant": args.model, "max_new_tokens": args.max_new_tokens,
        "task_suite_sha256": hashlib.sha256(suite.encode()).hexdigest(),
        "provenance": provenance, "scoring_image": image,
        "passed": sum(row["passed"] for row in results) if image else None,
        "total": len(results),
        "tokens_per_second": sum(row["tokens"] for row in results) / elapsed if elapsed else None,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
