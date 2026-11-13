"""One pinned local Qwen SFT checkpoint, called independently for three roles."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]


class LocalQwenBackend:
    is_real_model = True

    def __init__(self, model_path=None, adapter_path=None, threads=6, max_new_tokens=96):
        import torch
        from training.legacy.train import apply_adapter, load_base

        if not 1 <= max_new_tokens <= 512:
            raise ValueError("Generation budget must be in 1..512 tokens.")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_path = Path(model_path or ROOT / "training/legacy/model/base").resolve()
        self.adapter_path = Path(adapter_path or ROOT / "training/legacy/runs/sft/adapter.pt").resolve()
        checkpoint = torch.load(self.adapter_path, map_location="cpu", weights_only=True)
        if checkpoint["stage"] != "sft" or not checkpoint["finished"] or checkpoint["step"] < 1:
            raise ValueError("tutor requires the completed training SFT adapter.")
        self.model, self.tokenizer = load_base(self.model_path, "float32", threads, "cpu")
        apply_adapter(self.model, self.adapter_path)
        self.model.requires_grad_(False)
        self.model.eval()
        self.terminal_ids = list(dict.fromkeys([
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id]))
        with self.adapter_path.open("rb") as stream:
            adapter_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {
            "kind": "real_local_llm", "model_id": "Qwen/Qwen2.5-0.5B",
            "checkpoint_stage": "sft", "adapter_sha256": adapter_hash,
            "base_hash": self.model._training_base_hash, "base_revision": self.model._training_revision,
            "model_path": str(self.model_path), "adapter_path": str(self.adapter_path),
            "device": "cpu", "dtype": "float32", "threads": threads,
            "max_new_tokens": max_new_tokens, "seed": 20261104,
            "role_weight_sharing": "planner, teacher and reviewer use the same weights in independent calls",
            "independently_trained_agents": False,
        }

    def generate(self, prompt, role, max_new_tokens=None):
        from training.legacy.data_io import encode_prompt

        budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= budget <= 512:
            raise ValueError("Invalid generation budget.")
        prefix = encode_prompt(self.tokenizer, prompt)
        if len(prefix) + budget > min(4096, self.model.config.max_position_embeddings):
            raise ValueError("Role context exceeds the 4096-token CPU budget; no silent truncation.")
        torch = self.torch
        torch.manual_seed(20261104)
        x = torch.tensor([prefix], dtype=torch.long)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, num_beams=1,
                max_new_tokens=budget, eos_token_id=self.terminal_ids,
                pad_token_id=self.terminal_ids[0], use_cache=True,
                repetition_penalty=1.0, no_repeat_ngram_size=0)
        ids = output[0, len(prefix):].tolist()
        stopped = bool(ids and ids[-1] in self.terminal_ids)
        visible = ids[:-1] if stopped else ids
        return {
            "raw_response": self.tokenizer.decode(visible, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False),
            "generated_token_ids": ids, "prompt_tokens": len(prefix),
            "new_tokens": len(ids), "stop_reason": "eos" if stopped else "max_new_tokens",
            "max_new_tokens": budget,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "role": role, "real_model_call": True,
        }
