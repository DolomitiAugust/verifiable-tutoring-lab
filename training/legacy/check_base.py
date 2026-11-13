"""Load the verified base model from disk and run a single CPU forward pass."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "base")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT.parent / ".runtime" / "hf-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads((ROOT / "model" / "source.json").read_text(encoding="utf-8"))
    weight_file = args.model / "model.safetensors"
    with weight_file.open("rb") as weights:
        actual = hashlib.file_digest(weights, "sha256").hexdigest()
    if actual != source["files"]["model.safetensors"][1]:
        raise ValueError("Weight hash does not match the pinned official manifest")
    with safe_open(weight_file, framework="pt", device="cpu") as tensors:
        storage_dtype = tensors.get_slice("model.embed_tokens.weight").get_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    load_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    load_seconds = time.perf_counter() - load_at
    inputs = tokenizer("Python 中 print(2 + 3) 的结果是", return_tensors="pt")
    forward_at = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
    forward_seconds = time.perf_counter() - forward_at
    if model.__class__.__name__ != "Qwen2ForCausalLM" or not torch.isfinite(logits).all().item():
        raise ValueError("Unexpected model type or nonfinite logits")
    result = {
        "model_id": source["model_id"], "revision": source["revision"], "weight_sha256": actual,
        "weight_file_bytes": weight_file.stat().st_size, "embedding_storage_dtype": storage_dtype,
        "architecture": model.__class__.__name__, "unique_parameters": sum(p.numel() for p in model.parameters()),
        "hidden_size": model.config.hidden_size, "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size, "tokenizer": tokenizer.__class__.__name__,
        "input_tokens": inputs.input_ids.shape[1], "logits_shape": list(logits.shape), "finite_logits": True,
        "device": str(logits.device), "loaded_dtype": str(next(model.parameters()).dtype),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch": torch.__version__,
        "load_seconds": round(load_seconds, 3), "short_forward_seconds": round(forward_seconds, 3),
        "python": sys.version.split()[0], "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"), "accelerate": importlib.metadata.version("accelerate"),
    }
    report = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
