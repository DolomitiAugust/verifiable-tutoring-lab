"""Actual Qwen2.5-0.5B base -> SFT -> DPO training with last-block Q/V LoRA."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .data_io import encode_prompt, encode_response, file_hash, read_synthesis, select_examples, write_json
else:
    from data_io import encode_prompt, encode_response, file_hash, read_synthesis, select_examples, write_json

BASE_ID = "Qwen/Qwen2.5-0.5B"
FORMAT = "training-qwen-last-qv-lora-v1"
CHAT_FORMAT = "training-chatml-v1:user+assistant; no BOS/system; completion includes im_end"


def load_base(model_path, dtype, threads, device="cpu"):
    """Load only an existing local model/snapshot. The trainer never downloads weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it.")
    if dtype == "float16" and device == "cpu":
        raise ValueError("Use float32 or bfloat16 for CPU training.")
    torch.set_num_threads(threads)
    path = Path(model_path)
    if not path.is_dir() and str(model_path) == BASE_ID:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(BASE_ID, local_files_only=True))
    if not path.is_dir() or "instruct" in str(path).lower() or (path / "adapter_config.json").exists():
        raise ValueError(f"Expected a local unmodified {BASE_ID} base snapshot: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    for marker in ("<|im_start|>", "<|im_end|>"):
        if marker not in tokenizer.get_vocab() or tokenizer.encode(marker, add_special_tokens=False) != [
            tokenizer.convert_tokens_to_ids(marker)
        ]:
            raise ValueError(f"Model tokenizer lacks the expected atomic ChatML marker {marker}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|im_end|>"  # Masked right-padding only; never counted as response text.
    precision = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=precision, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa")
    config = model.config
    if (config.model_type, config.hidden_size, config.num_hidden_layers) != ("qwen2", 896, 24):
        raise ValueError("Mismatched model architecture; expected Qwen2.5-0.5B base.")
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("No base weights found in the model directory.")
    source_files = weights + [path / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    model._training_sources = {p.name: file_hash(p) for p in source_files}
    source = json.loads((Path(__file__).resolve().parent / "base_source.json").read_text(encoding="utf-8"))
    if source["model_id"] != BASE_ID or any(source["files"].get(name, [None, None])[1] != digest
            for name, digest in model._training_sources.items()):
        raise ValueError("Base weights or tokenizer differ from the pinned official training/legacy/model/source.json.")
    model._training_revision = source["revision"]
    model._training_base_hash = hashlib.sha256(json.dumps(
        model._training_sources, sort_keys=True).encode("utf-8")).hexdigest()
    model.config.use_cache = False
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, tokenizer


class LoRAProjection(nn.Module):
    """Frozen original layer plus alpha/r * B(A(x)); B starts at zero."""
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original
        self.scale = alpha / rank
        options = {"device": original.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features, **options))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, **options))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.original(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return base + (self.scale * delta).to(base.dtype)


def install_adapter(model, settings):
    previous = getattr(model, "_training_lora", None)
    if previous:
        if previous != settings:
            raise ValueError("Another adapter layout is already installed on this model.")
        return
    blocks = model.model.layers
    if not (0 < settings["last_layers"] <= len(blocks) and settings["rank"] > 0 and settings["alpha"] > 0):
        raise ValueError("Invalid LoRA layer count, rank or scale.")
    model.requires_grad_(False)
    for block in blocks[-settings["last_layers"]:]:
        for name in ("q_proj", "v_proj"):
            original = getattr(block.self_attn, name)
            setattr(block.self_attn, name, LoRAProjection(original, settings["rank"], settings["alpha"]))
    model._training_lora = dict(settings)


def trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def weights_of(model):
    return {name: p.detach().cpu().clone() for name, p in trainable(model).items()}


def set_weights(model, weights):
    parameters = trainable(model)
    if parameters.keys() != weights.keys():
        raise ValueError("Checkpoint parameters differ from the trainable adapter.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != weights[name].shape:
                raise ValueError(f"Adapter tensor shape differs: {name}")
            parameter.copy_(weights[name])


def apply_adapter(model, checkpoint_path):
    """Install exactly one adapter on an untouched base; usable by independent evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != FORMAT or checkpoint["base_hash"] != model._training_base_hash:
        raise ValueError("Adapter format or base weights/tokenizer fingerprint mismatch.")
    install_adapter(model, checkpoint["settings"])
    set_weights(model, checkpoint["weights"])
    return model


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def response_scores(model, examples, pad_id, head_chunk=64, checkpoint_head=False):
    """Sum exact causal log probabilities over each response, in example/response order."""
    sequences = [(sample["prefix"], response) for sample in examples for response in sample["responses"]]
    n = len(sequences)
    width = max(len(prefix) + len(response) for prefix, response in sequences)
    device = next(model.parameters()).device
    ids = torch.full((n, width), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (prefix, response) in enumerate(sequences):
        end = len(prefix) + len(response)
        ids[i, :end] = torch.tensor(prefix + response, device=device)
        mask[i, :end] = 1
        labels[i, len(prefix):end] = ids[i, len(prefix):end]
    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    targets = labels[:, 1:]
    active = targets.ne(-100)
    owners = active.nonzero(as_tuple=True)[0]
    vectors, tokens = hidden[:, :-1][active], targets[active]

    def log_probabilities(states, gold):
        # Only response prediction positions reach the large frozen vocabulary head.
        logits = model.lm_head(states).float()
        return -F.cross_entropy(logits, gold, reduction="none")

    pieces = []
    for start in range(0, len(tokens), head_chunk):
        h = vectors[start:start + head_chunk]
        gold = tokens[start:start + head_chunk]
        if checkpoint_head and h.requires_grad and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            pieces.append(checkpoint(log_probabilities, h, gold, use_reentrant=False, preserve_rng_state=False))
        else:
            pieces.append(log_probabilities(h, gold))
    sums = torch.zeros(n, dtype=torch.float32, device=device).index_add(0, owners, torch.cat(pieces))
    return sums, active.sum(dim=1)


def batch_at(examples, step, batch_size):
    start = step * batch_size
    return [examples[(start + i) % len(examples)] for i in range(batch_size)]


def reference_scores(model, examples, args, signature, tokenizer, out):
    """Cache the immutable SFT scores before applying or resuming any DPO update."""
    key = {"sft_sha256": signature["sft_sha256"], "base_hash": model._training_base_hash,
           "selection": signature["selection"], "dtype": args.dtype, "device": args.device,
           "batch_size": args.batch_size}
    path = out / "reference.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"identity": key, "scores": {}}
    if cache["identity"] != key:
        raise ValueError("Cached reference scores do not match this SFT checkpoint or training selection.")
    scores = cache["scores"]
    with torch.no_grad():
        for step in range(args.steps):
            batch = batch_at(examples, step, args.batch_size)
            if all(ex["candidate_id"] in scores for ex in batch):
                continue
            values, _ = response_scores(model, batch, tokenizer.pad_token_id, args.head_chunk)
            for ex, values_pair in zip(batch, values.reshape(-1, 2).cpu().tolist()):
                scores.setdefault(ex["candidate_id"], values_pair)
            write_json(path, cache)
            if step == 0 or (step + 1) % 10 == 0:
                print(f"SFT reference: {len(scores)} pairs cached", flush=True)
    return scores


def objective(model, batch, stage, args, tokenizer, references):
    values, counts = response_scores(model, batch, tokenizer.pad_token_id, args.head_chunk, args.checkpoint_head)
    stats = {"response_tokens": int(counts.sum())}
    if stage == "sft":
        loss = -values.sum() / counts.sum()
        stats["answer_nll"] = float(loss.detach())
    else:
        values = values.view(-1, 2)
        ref = values.new_tensor([references[ex["candidate_id"]] for ex in batch])
        policy_ratio = values[:, 0] - values[:, 1]
        ref_ratio = ref[:, 0] - ref[:, 1]
        relative_margin = policy_ratio - ref_ratio
        loss = -F.logsigmoid(args.beta * relative_margin).mean()
        stats.update(policy_chosen_logp=float(values[:, 0].mean().detach()),
                     policy_rejected_logp=float(values[:, 1].mean().detach()),
                     reference_chosen_logp=float(ref[:, 0].mean()),
                     reference_rejected_logp=float(ref[:, 1].mean()),
                     policy_logratio=float(policy_ratio.mean().detach()), ref_logratio=float(ref_ratio.mean()),
                     relative_margin=float(relative_margin.mean().detach()),
                     dpo_scaled_margin=float((args.beta * relative_margin).mean().detach()))
    return loss, stats


def distance(current, baseline):
    return math.sqrt(sum(float((current[k] - baseline[k]).square().sum()) for k in current))


def write_history(out, history):
    path = out / "metrics.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history), encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    import transformers

    if min(args.steps, args.batch_size, args.threads, args.max_tokens, args.head_chunk) <= 0 or args.limit < 0:
        raise ValueError("Steps, batch size, threads and token bounds must be positive; limit may be zero.")
    if not all(math.isfinite(v) and v > 0 for v in (args.beta, args.lr, args.grad_clip)):
        raise ValueError("Beta, learning rate and gradient clipping threshold must be finite and positive.")
    if (args.stage == "dpo") != bool(args.sft_checkpoint):
        raise ValueError("DPO requires --sft-checkpoint; SFT starts from the base model.")
    out = Path(args.out).resolve()
    manifest_path = out / "run.json"
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError(f"Non-empty output directory: {out}. Use --resume or a new output location.")
    if args.resume and not manifest_path.exists():
        raise ValueError("No previous run manifest for --resume.")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rows, metadata, hashes = read_synthesis(args.dataset, args.stage)
    model, tokenizer = load_base(args.model, args.dtype, args.threads, args.device)
    selected, skipped = select_examples(rows, metadata, tokenizer, args.stage, args.seed, args.limit, args.max_tokens)
    if args.batch_size > len(selected):
        raise ValueError("Batch size exceeds the number of selected, intact examples.")
    if args.max_tokens > model.config.max_position_embeddings:
        raise ValueError("Configured sequence bound exceeds the model context.")
    sft_hash = None
    if args.initial_sft:
        if args.stage != "sft":
            raise ValueError("--initial-sft is only for the continued-SFT control.")
        parent = torch.load(args.initial_sft, map_location="cpu", weights_only=True)
        if parent["stage"] != "sft" or not parent["finished"] or parent["dataset_hashes"] != hashes:
            raise ValueError("Control requires finished SFT from the same data release.")
        sft_hash = file_hash(args.initial_sft)
        apply_adapter(model, args.initial_sft)
        settings = parent["settings"]
        for key, value in settings.items():
            if getattr(args, key) is not None and getattr(args, key) != value:
                raise ValueError("Control adapter layout must inherit its SFT parent.")
            setattr(args, key, value)
    if args.stage == "sft":
        settings = {"rank": 8 if args.rank is None else args.rank, "alpha": 16 if args.alpha is None else args.alpha,
                    "last_layers": 2 if args.last_layers is None else args.last_layers}
        install_adapter(model, settings)
    else:
        sft_hash = file_hash(args.sft_checkpoint)
        checkpoint = torch.load(args.sft_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["stage"] != "sft" or checkpoint["step"] < 1 or not checkpoint["finished"]:
            raise ValueError("DPO requires a finished, trained SFT checkpoint.")
        if checkpoint["dataset_hashes"] != hashes or checkpoint["chat_format"] != CHAT_FORMAT:
            raise ValueError("DPO and SFT differ in synthesis data or chat serialization.")
        settings = checkpoint["settings"]
        for name in settings:
            requested = getattr(args, name)
            if requested is not None and requested != settings[name]:
                raise ValueError(f"DPO must inherit SFT adapter parameter {name}.")
        apply_adapter(model, args.sft_checkpoint)
    selection = [{**{k: ex[k] for k in ("candidate_id", "export_line", "family", "task", "level")},
                  "prompt_tokens": len(ex["prefix"]), "response_tokens": list(map(len, ex["responses"]))}
                 for ex in selected]
    signature = {"stage": args.stage, "base_id": BASE_ID, "base_hash": model._training_base_hash,
                 "sft_sha256": sft_hash, "dataset_hashes": hashes, "chat_format": CHAT_FORMAT,
                 "training_sources": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("data_io.py"))},
                 "dtype": args.dtype, "device": args.device, "threads": args.threads, "seed": args.seed,
                 "settings": settings, "batch_size": args.batch_size, "lr": args.lr, "beta": args.beta,
                 "grad_clip": args.grad_clip, "head_chunk": args.head_chunk, "checkpoint_head": args.checkpoint_head,
                 "max_tokens": args.max_tokens, "limit": args.limit, "selection": selection}
    if args.resume and json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Resume configuration, checkpoint or dataset differs from the original run.")
    write_json(manifest_path, {"signature": signature, "steps_requested": args.steps,
               "base_revision": model._training_revision, "base_files_sha256": model._training_sources,
               "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()) if args.sft_checkpoint else None,
               "initial_sft": str(args.initial_sft.resolve()) if args.initial_sft else None,
               "experiment_role": "continued_sft_control" if args.initial_sft else args.stage,
               "source_rows": len(rows), "skipped_length_before_limit": skipped,
               "trainable_parameters": sum(p.numel() for p in trainable(model).values()),
               "model_parameters_including_lora": sum(p.numel() for p in model.parameters()),
               "torch": str(torch.__version__), "transformers": transformers.__version__})
    print(f"{args.stage}: {len(selected)}/{len(rows)} rows selected, {sum(p.numel() for p in trainable(model).values())} trainable parameters", flush=True)
    initial = weights_of(model)
    references = reference_scores(model, selected, args, signature, tokenizer, out) if args.stage == "dpo" else {}
    optimizer = torch.optim.AdamW(trainable(model).values(), lr=args.lr, weight_decay=0)
    state_path = out / "state.pt"
    history = []
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        set_weights(model, state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        if len(history) > args.steps:
            raise ValueError("Requested steps precede the resumed checkpoint.")
        write_history(out, history)
    for index in range(len(history), args.steps):
        started = time.perf_counter()
        batch = batch_at(selected, index, args.batch_size)
        before = weights_of(model)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = objective(model, batch, args.stage, args, tokenizer, references)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {index + 1}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable(model).values(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        weights = weights_of(model)
        change = distance(weights, before)
        if not math.isfinite(change):
            raise FloatingPointError(f"Nonfinite adapter update at step {index + 1}")
        record = {"step": index + 1, "ids": [ex["candidate_id"] for ex in batch], "loss": float(loss.detach()),
                  **stats, "grad_norm_before_clip": float(grad), "adapter_step_l2": change,
                  "adapter_delta_from_stage_start_l2": distance(weights, initial),
                  "seconds": round(time.perf_counter() - started, 3)}
        history.append(record)
        atomic_torch(state_path, {"weights": weights, "optimizer": optimizer.state_dict(), "history": history})
        write_history(out, history)
        print(json.dumps(record), flush=True)
    # The optimizer state is the atomic resume record. Rebuild this export even after a resume.
    final_weights = weights_of(model)
    delta = distance(final_weights, initial)
    if delta <= 0:
        raise RuntimeError("No adapter weight changed during this training stage.")
    adapter = {"format": FORMAT, "stage": args.stage, "step": len(history), "finished": True,
               "settings": settings, "weights": final_weights, "base_hash": model._training_base_hash,
               "base_id": BASE_ID, "chat_format": CHAT_FORMAT, "dataset_hashes": hashes,
               "parent_sft_sha256": sft_hash}
    atomic_torch(out / "adapter.pt", adapter)
    first = batch_at(selected, 0, args.batch_size)
    with torch.no_grad():
        probe_loss, probe_stats = objective(model, first, args.stage, args, tokenizer, references)
    write_json(out / "summary.json", {"stage": args.stage, "steps": len(history), "selected": len(selected),
               "unique_training_examples": len({x for r in history for x in r["ids"]}),
               "total_response_tokens_processed": sum(r["response_tokens"] for r in history),
               "training_step_seconds": sum(r["seconds"] for r in history), "adapter_delta_l2": delta,
               "sha256": file_hash(out / "adapter.pt"), "parent_sft_sha256": sft_hash,
               "first_training_batch_before": history[0],
               "same_training_batch_after": {"loss": float(probe_loss), **probe_stats}})
    print(f"Saved {args.stage}: {out / 'adapter.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    parser.add_argument("--model", required=True, help="Downloaded local Qwen2.5-0.5B base model directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "synthesis" / "dataset")
    parser.add_argument("--sft-checkpoint", type=Path)
    parser.add_argument("--initial-sft", type=Path, help="Continue SFT from this completed SFT adapter for the matched-step control")
    parser.add_argument("--limit", type=int, default=0, help="Maximum selected intact examples; 0 uses all eligible")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1536, help="Skip and log longer rows; never truncate")
    parser.add_argument("--head-chunk", type=int, default=64)
    parser.add_argument("--checkpoint-head", action="store_true", help="Recompute response logits in backward to reduce memory")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--last-layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true", help="Continue same run; --steps is the total target")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 5e-4 if args.stage == "sft" else 2e-4
    run(args)


if __name__ == "__main__":
    main()
