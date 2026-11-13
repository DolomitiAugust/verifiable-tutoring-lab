"""Pinned local inference. Base+SFT and official Instruct stay distinct."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_instruct_source(model_path):
    folder = Path(model_path)
    manifest = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != "Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("Instruct identity mismatch")
    if len(manifest["revision"]) != 40 or "model.safetensors" not in manifest["files"]:
        raise ValueError("Incomplete pinned source manifest")
    for name, (size, digest) in manifest["files"].items():
        path = (folder / name).resolve()
        if path.parent != folder.resolve() or path.stat().st_size != size or file_sha(path) != digest:
            raise ValueError("Model manifest mismatch: " + name)
    return manifest


class LocalQwenBackend:
    is_real_model = True

    def __init__(self, model_path, adapter_path=None, kind="sft", threads=6):
        import torch
        self.torch = torch
        self.model_path = Path(model_path).resolve()
        torch.set_num_threads(threads)
        if kind == "instruct":
            if adapter_path is not None:
                raise ValueError("Instruct diagnostic must not silently load a base-model adapter")
            source = verify_instruct_source(self.model_path)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True,
                            torch_dtype=torch.float32, trust_remote_code=False, attn_implementation="eager")
            identity = {"model_id": source["model_id"], "revision": source["revision"],
                        "source_manifest_sha256": file_sha(self.model_path / "source.json"),
                        "checkpoint_stage": "official_instruct_diagnostic", "adapter_sha256": None}
        elif kind == "sft":
            from training.legacy.train import load_base, apply_adapter
            if adapter_path is None:
                raise ValueError("SFT requires an explicit adapter path")
            checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
            if checkpoint["stage"] != "sft" or not checkpoint["finished"] or checkpoint["step"] < 1:
                raise ValueError("A completed SFT checkpoint is required")
            self.model, self.tokenizer = load_base(self.model_path, "float32", threads, "cpu")
            apply_adapter(self.model, adapter_path)
            identity = {"model_id": "Qwen/Qwen2.5-0.5B", "revision": self.model._training_revision,
                        "base_hash": self.model._training_base_hash, "checkpoint_stage": "sft",
                        "adapter_sha256": file_sha(adapter_path), "adapter_path": str(Path(adapter_path).resolve())}
        else:
            raise ValueError("kind must be sft or instruct")
        self.kind = kind
        self.model.requires_grad_(False)
        self.model.eval()
        self.terminal_ids = list(dict.fromkeys([self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id]))
        self.metadata = {**identity, "kind": "real_local_llm", "diagnostic_only": kind == "instruct",
                         "model_path": str(self.model_path), "threads": threads, "device": "cpu", "dtype": "float32",
                         "context_limit": 4096, "do_sample": False, "seed": 20261107,
                         "independently_trained_agents": False, "roles_share_weights": True}

    def generate(self, prompt, role, max_new_tokens):
        if self.kind == "sft":
            from training.legacy.data_io import encode_prompt
            prefix = encode_prompt(self.tokenizer, prompt)
            serialization = "training_encode_prompt"
        else:
            prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            serialization = "official_tokenizer_chat_template"
        if len(prefix) + max_new_tokens > 4096:
            raise ValueError("Context would exceed 4096 tokens; no silent truncation")
        torch = self.torch
        torch.manual_seed(20261107)
        x = torch.tensor([prefix], dtype=torch.long)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=self.terminal_ids,
                pad_token_id=self.terminal_ids[0], use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
        ids = output[0, len(prefix):].tolist()
        stopped = bool(ids and ids[-1] in self.terminal_ids)
        return {"raw_response": self.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "generated_token_ids": ids, "prompt_token_ids": prefix, "serialization": serialization,
                "prompt_tokens": len(prefix), "new_tokens": len(ids), "stop_reason": "eos" if stopped else "max_new_tokens",
                "max_new_tokens": max_new_tokens, "elapsed_seconds": round(time.perf_counter() - started, 4),
                "role": role, "real_model_call": True}
