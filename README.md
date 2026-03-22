# 🚀 Accelerating Large Language Model Inference with vLLM  
### A Systems-Level Benchmark on NVIDIA H100 PCIe

> This work presents a systems-focused benchmarking study comparing HuggingFace Transformers and vLLM on NVIDIA H100 PCIe hardware, evaluating latency, throughput scaling, KV cache efficiency, and GPU utilization under controlled experimental conditions.  
> We quantify the impact of continuous batching and PagedAttention on modern Hopper (SM90) architecture.  
> Results provide reproducible insights when run with fixed seeds and deterministic decode settings.

---

## 1. Introduction

As large language models transition from research prototypes to production systems, inference efficiency has become a primary bottleneck.

Modern deployment constraints require:

- Low end-to-end latency  
- High throughput under concurrent workloads  
- Stable and predictable GPU memory behavior  
- Efficient hardware utilization  

While HuggingFace provides a robust and flexible inference stack, it is not explicitly optimized for high-concurrency production serving.

vLLM introduces architectural innovations designed to address these limitations:

- Continuous batching
- PagedAttention-based KV cache management
- Optimized GPU scheduling

This study evaluates the practical performance implications of these architectural differences on NVIDIA H100 PCIe hardware.

---

## 2. Experimental Setup

| Component | Specification |
|------------|--------------|
| GPU | NVIDIA H100 PCIe |
| Architecture | Hopper (SM90) |
| Memory | 80GB HBM3 |
| Precision | FP16 |
| Model | TinyLlama-1.1B-Chat |
| Max Generation Tokens | 100 |
| Batch Sizes | 1, 4, 8, 16, 32 |

All latency measurements were conducted using synchronized GPU timing:

```python
import time
torch.cuda.synchronize()
start = time.perf_counter()

# inference call

torch.cuda.synchronize()
end = time.perf_counter()
latency = end - start
```

This ensures accurate measurement of kernel execution time rather than asynchronous dispatch overhead.

---

## 3. Architectural Comparison

### 3.1 HuggingFace Baseline

```
Prompt → Tokenizer → model.generate() → GPU → Output
```

Characteristics:

- Sequential request execution  
- Static KV cache allocation  
- No dynamic scheduling layer  
- Limited batch scaling  

---

### 3.2 vLLM Inference Engine

```
Prompt → Scheduler → Continuous Batching
       → PagedAttention → KV Cache Manager
       → GPU Execution → Output
```

Key differences:

- Dynamic request aggregation  
- Memory-paged KV cache to reduce fragmentation  
- Improved GPU occupancy  
- Reduced idle cycles during decoding  

---

## 4. Benchmarks Conducted

### 4.1 Single-Request Latency

Measured:

- End-to-end generation time  
- Tokens generated  
- Tokens per second  

---

### 4.2 Average Latency Across Prompts

Multiple prompts were evaluated to reduce variance and improve reliability.

---

### 4.3 Decoding Throughput (Tokens/sec)

```python
tokens_per_second = generated_tokens / latency
```

This isolates decoding performance from prompt formatting overhead.

---

### 4.4 Batch Throughput Scaling

Batch sizes tested:

```python
[1, 4, 8, 16, 32]
```

Throughput metric:

```python
throughput = batch_size / total_time
```

Comparison:

- HuggingFace → Sequential loop execution  
- vLLM → True batched decoding  

---

### 4.5 GPU Memory Behavior

Measured using:

```python
torch.cuda.memory_allocated()
```

Evaluated:

- Allocation stability  
- Fragmentation behavior  
- Memory scaling under load  

---

## 5. Observations

### 5.1 Single Request

On H100 hardware:

- Latency differences between HF and vLLM are modest at batch size = 1  
- Compute throughput of Hopper architecture minimizes small-scale differences  

---

### 5.2 Tokens per Second

vLLM consistently achieves higher decoding throughput due to:

- Efficient KV cache reuse  
- Reduced memory allocation overhead  
- Higher GPU occupancy  

---

### 5.3 Batch Scaling

HuggingFace:

- Throughput plateaus as batch increases  
- Sequential execution limits scalability  

vLLM:

- Scales efficiently with batch size  
- Maintains superior request throughput  
- Better utilizes Hopper memory bandwidth  

---

### 5.4 Memory Stability

vLLM demonstrates:

- Reduced fragmentation  
- More stable allocation patterns  
- Improved behavior under concurrent load  

---

## 6. Why vLLM Aligns with Hopper Architecture

The NVIDIA H100 (SM90) provides:

- High memory bandwidth  
- Larger L2 cache  
- Optimized FP16 compute throughput  

vLLM’s architectural design:

- Continuous batching  
- PagedAttention  
- Optimized scheduling  

aligns closely with Hopper’s execution model, improving overall inference efficiency.

---

## 7. Outputs Generated

- Throughput vs Batch Size plot  
- CSV benchmark results (`vllm_vs_hf_results.csv`)  
- Tokens/sec comparison  
- Latency comparison  

---

## 8. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA wheel for PyTorch on your system, install that wheel first,
then run `pip install -r requirements.txt`.

---

## 9. Quick Run (Deterministic)

Validate the environment:

```bash
python validate_setup.py
```

Run the benchmark and export CSV:

```bash
python run_benchmark.py --seed 42 --max-new-tokens 100 --output vllm_vs_hf_results.csv
```

The notebook `vLLM.ipynb` is also updated to use deterministic settings and the same dependency file.

---

## 10. Reproducibility

Steps:

1. Set the same seed for Python, NumPy, and PyTorch.
2. Use greedy/de-determinized-off decoding (`do_sample=False` for HF, `temperature=0.0` for vLLM).
3. Load TinyLlama model.
4. Initialize HuggingFace baseline.
5. Initialize vLLM engine.
6. Execute latency and throughput benchmarks.
7. Export CSV results.
8. Generate scaling visualizations.

With those controls, repeated runs should be statistically stable on the same hardware/software stack.

---

## 11. Conclusion

This study demonstrates that:

- vLLM significantly improves batch throughput under concurrent workloads  
- Continuous batching increases GPU utilization  
- PagedAttention improves KV cache memory stability  
- On modern Hopper hardware, architectural alignment with GPU design materially impacts inference efficiency  

For production-grade LLM serving, vLLM provides measurable systems-level advantages.

---

## 12. Future Directions

- Multi-GPU tensor parallel benchmarking  
- BF16 vs FP16 comparison on Hopper  
- FlashAttention vs PagedAttention evaluation  
- Quantized inference benchmarking  
- Concurrent API stress testing  
