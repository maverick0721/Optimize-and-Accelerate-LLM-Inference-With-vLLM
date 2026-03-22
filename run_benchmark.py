#!/usr/bin/env python3
"""Deterministic benchmark runner for HF vs vLLM TinyLlama inference."""

import argparse
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass

# Set vLLM runtime behavior before importing vLLM.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_V1_INPROC", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPTS = [
    "Explain transformers in simple words.",
    "What is gradient descent?",
    "Explain overfitting in ML.",
    "What is KV cache?",
    "Describe neural networks.",
]


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class GenResult:
    latency: float
    tokens: int
    text: str


class HFRunner:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(end - start, len(new_tokens), gen_text)


class VLLMRunner:
    def __init__(self, model_name: str, tokenizer: AutoTokenizer, seed: int, max_tokens: int):
        self.tokenizer = tokenizer
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            enforce_eager=True,
            disable_log_stats=True,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )

    def build_chat_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str) -> GenResult:
        text = self.build_chat_text(prompt)

        start = time.perf_counter()
        outputs = self.llm.generate([text], self.sampling_params, use_tqdm=False)
        end = time.perf_counter()

        gen_text = outputs[0].outputs[0].text
        tokens = len(self.tokenizer(gen_text, add_special_tokens=False)["input_ids"])
        return GenResult(end - start, tokens, gen_text)

    def batch_throughput(self, prompts: list[str]) -> float:
        texts = [self.build_chat_text(p) for p in prompts]
        start = time.perf_counter()
        self.llm.generate(texts, self.sampling_params, use_tqdm=False)
        end = time.perf_counter()
        return len(prompts) / (end - start)

    def shutdown(self) -> None:
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


def gpu_peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def configure_logging(quiet: bool) -> None:
    if not quiet:
        return

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("vllm").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)


def run(args: argparse.Namespace) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    set_reproducible(args.seed)
    print("CUDA device:", torch.cuda.get_device_name(0))

    hf = HFRunner(MODEL_NAME)
    vllm = VLLMRunner(MODEL_NAME, hf.tokenizer, args.seed, args.max_new_tokens)

    prompt = "Explain KV cache in simple words."
    hf_single = hf.generate(prompt, args.max_new_tokens)
    vllm_single = vllm.generate(prompt)
    print("SINGLE REQUEST")
    print(f"HF Latency: {hf_single.latency:.4f}s")
    print(f"vLLM Latency: {vllm_single.latency:.4f}s")
    print(f"HF Tokens/sec: {hf_single.tokens / hf_single.latency:.2f}")
    print(f"vLLM Tokens/sec: {vllm_single.tokens / vllm_single.latency:.2f}")

    hf_latencies = []
    vllm_latencies = []
    hf_tps = []
    vllm_tps = []

    for p in DEFAULT_PROMPTS:
        hf_out = hf.generate(p, args.max_new_tokens)
        vllm_out = vllm.generate(p)
        hf_latencies.append(hf_out.latency)
        vllm_latencies.append(vllm_out.latency)
        hf_tps.append(hf_out.tokens / hf_out.latency)
        vllm_tps.append(vllm_out.tokens / vllm_out.latency)

    print("AVERAGE LATENCY")
    print("HF Avg Latency:", sum(hf_latencies) / len(hf_latencies))
    print("vLLM Avg Latency:", sum(vllm_latencies) / len(vllm_latencies))

    print("TOKENS PER SECOND")
    print("HF Avg Tokens/sec:", sum(hf_tps) / len(hf_tps))
    print("vLLM Avg Tokens/sec:", sum(vllm_tps) / len(vllm_tps))

    results = []
    for batch in args.batch_sizes:
        prompts = [DEFAULT_PROMPTS[0]] * batch

        start = time.perf_counter()
        for p in prompts:
            hf.generate(p, args.max_new_tokens)
        end = time.perf_counter()
        hf_throughput = batch / (end - start)

        vllm_throughput = vllm.batch_throughput(prompts)
        results.append((batch, hf_throughput, vllm_throughput))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hf.generate("Test memory.", args.max_new_tokens)
    hf_mem_peak = gpu_peak_memory_mb()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vllm.generate("Test memory.")
    vllm_mem_peak = gpu_peak_memory_mb()

    print("GPU PEAK MEMORY (MB)")
    print("HF Peak Memory:", hf_mem_peak)
    print("vLLM Peak Memory:", vllm_mem_peak)

    df = pd.DataFrame(results, columns=["Batch Size", "HF Throughput", "vLLM Throughput"])
    df["HF Avg Latency"] = sum(hf_latencies) / len(hf_latencies)
    df["vLLM Avg Latency"] = sum(vllm_latencies) / len(vllm_latencies)
    df["HF Avg Tokens/sec"] = sum(hf_tps) / len(hf_tps)
    df["vLLM Avg Tokens/sec"] = sum(vllm_tps) / len(vllm_tps)
    df["HF Peak Memory MB"] = hf_mem_peak
    df["vLLM Peak Memory MB"] = vllm_mem_peak
    vllm.shutdown()
    cleanup_distributed()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF vs vLLM deterministic benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output", default="vllm_vs_hf_results.csv")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--quiet", action="store_true", help="Reduce non-critical runtime logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.quiet)
    df = run(args)
    df.to_csv(args.output, index=False)
    print("Saved:", args.output)
    cleanup_distributed()


if __name__ == "__main__":
    main()
