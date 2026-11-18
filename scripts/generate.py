"""Run stock Qwen or a Pyroton adapter without the training dependencies."""

import argparse
import time

BASE = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODELS = {
    "qwen": BASE,
    "pyroton": "shohuu/Pyroton",
    "primefix": "shohuu/pyroton-primefix-v3",
}


def alpaca_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


class Generator:
    def __init__(self, variant, base_revision=None, adapter_revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.variant = variant
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE, revision=base_revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        tokenizer_revision = base_revision
        if variant != "qwen":
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, MODELS[variant], revision=adapter_revision,
            )
            tokenizer_revision = adapter_revision
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODELS[variant], revision=tokenizer_revision,
        )

    def generate(self, instruction, max_new_tokens):
        prompt = alpaca_prompt(instruction)
        if self.variant == "qwen":
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).to(self.model.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, inputs.input_ids.shape[-1]:]
        seconds = time.perf_counter() - start
        return self.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens), seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--model", choices=MODELS, default="primefix")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    answer, _, _ = Generator(args.model).generate(args.instruction, args.max_new_tokens)
    print(answer)


if __name__ == "__main__":
    main()
