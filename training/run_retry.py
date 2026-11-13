"""Exploratory retry; pinned inputs, fail-closed resume, complete evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED_DATA = {'audit/candidates.jsonl': 'c3f13d0416c69c17514602651bd1f8cda7261cf67a3641b0bd1a7064f05b2d62', 'dev.jsonl': 'dc48241119b9a3162bb9f261bea5ef744adb43b2ff3dcc878be79802ec455cea', 'dev_metadata.jsonl': '233b27b37abf463842ea007b55faeb20d5b90c3e38d2ecd756c585417a3884fc', 'evaluation.jsonl': '83e018cfbfc3fe1703271b73435e20f75b937b6a4afa3e07b27de2cf8c11776c', 'independent_test.jsonl': '86b1376a7b55515c5fcd57a6b0018900aa48c4cb78fd2301f321468cc896c54d', 'independent_test_metadata.jsonl': 'd683c7fad438ff15094b2c8c039afbe3f8c9eb687eb9b93d0cf2f24411382c04', 'preference.jsonl': '72ebaac846744c05b7cffb52879fd3f050805dbcdc8b7614d734eb87ffdb80f1', 'preference_eval.jsonl': '49561cf8a17af7dd4355ec9d6147945d0fd3b8fdad3ee6a07428168c97be6e34', 'preference_eval_metadata.jsonl': '6949d78db15c715b4a23cb3fdbf05a0fdcda616a0cc433e19adb75abada97b39', 'sft.jsonl': 'b76286fa1f0ce5750caf7f7b99dc8ec49e3390b37661e026290c83373d8b3b12', 'stats.json': '05360989c424b6ea14a5fd75a1fd5f58abf0bb2e5cf06921e1ccc4f94c7e488f', 'split_manifest.json': 'f3331fb4c4583133d86a611b2c5ae8b390112de06419c86006b1b40c8409e2b5', 'independent_test_manifest.json': '3daf0da01beee624512a3dcb90d78c701650b8bfda351a232d674246b128e20e'}
EXPECTED_SCRIPTS = {'train.py': '94126615bee4d4f110c68b6ff64b65b8326eb540f28394a82c81ca2108f3481b', 'evaluate.py': '1f84416e2ad50acd1d11715fab31160315c8740951a002824be31b0dc9f09c7a', 'data_io.py': '4a618d87b2bb563b5a6332afcc96382d7695889bce24b279cf79927a3c65c1ad', 'base_source.json': '3063676ebd45b850baac1cd79c987ae7f3d03800bc2b6ad3f3911a9a10f3ba4f'}
PLAN_SHA256 = '45186c543c2a9f8b2d0e81f916a1c3849e970206a73874341b01b97f198dd85a'
SETTINGS = {"rank": 16, "alpha": 32, "last_layers": 24}
BASE_ID = "Qwen/Qwen2.5-0.5B"
CHAT = "training-chatml-v1:user+assistant; no BOS/system; completion includes im_end"
FORMAT = "training-qwen-last-qv-lora-v1"


def fail(condition, message):
    if not condition:
        raise ValueError(message + "; existing evidence is preserved, not overwritten")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]


def fresh_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def seal_path(path):
    return path.with_name(path.name + ".sha256")


def check_seal(path):
    digest = seal_path(path)
    fail(path.is_file() and digest.is_file(), f"Missing complete artifact/hash pair: {path}")
    fail(digest.read_text(encoding="ascii").strip() == sha(path), f"Artifact changed: {path}")


def seal_new(path):
    with seal_path(path).open("x", encoding="ascii") as f:
        f.write(sha(path) + "\n")


def publish_json(path, data):
    if path.exists():
        check_seal(path)
        fail(read(path) == data, f"Existing JSON differs: {path}")
    else:
        fail(not seal_path(path).exists(), f"Orphan hash: {path}")
        fresh_json(path, data)
        seal_new(path)


def copy_new(source, target):
    if target.exists():
        check_seal(target)
        fail(sha(source) == sha(target), f"Existing checkpoint differs: {target}")
    else:
        fail(not seal_path(target).exists(), f"Orphan hash: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        seal_new(target)


def check_expected(folder, expected):
    fail(bool(expected), "Missing frozen input manifest")
    actual = {}
    for name, digest in expected.items():
        path = folder / name
        fail(path.is_file() and not path.is_symlink(), f"Missing/nonregular input: {path}")
        actual[name] = sha(path)
        fail(actual[name] == digest, f"Frozen input hash mismatch: {path}")
    return actual


def input_record(model, data):
    scripts = check_expected(ROOT, EXPECTED_SCRIPTS)
    scripts[Path(__file__).name] = sha(Path(__file__))
    datasets = check_expected(data, EXPECTED_DATA)
    source = read(ROOT / "base_source.json")
    fail(model.is_dir() and "instruct" not in str(model).lower(), "Official base model directory is required")
    names = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    model_hashes = check_expected(model, {n: source["files"][n][1] for n in names})
    base_hash = hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()
    return {"scripts": scripts, "dataset": datasets, "model_files": model_hashes,
            "base_hash": base_hash, "model_revision": source["revision"], "plan_sha256": PLAN_SHA256}


def gpu_record(device):
    import torch
    info = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "torch": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
            "requested_device": device, "cuda_available": torch.cuda.is_available(), "gpus": []}
    if device == "cuda":
        fail(torch.cuda.is_available(), "CUDA requested but unavailable")
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i),
                                "total_memory_bytes": prop.total_memory,
                                "compute_capability": list(torch.cuda.get_device_capability(i))})
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False)
            info["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            info["nvidia_smi"] = {"unavailable": str(error)}
    return info


def tensor_shapes():
    return {f"model.layers.{i}.self_attn.{proj}.lora_{part}": shape
            for i in range(24) for proj, width in (("q_proj", 896), ("v_proj", 128))
            for part, shape in (("A", (16, 896)), ("B", (width, 16)))}


def check_weights(weights):
    import torch
    expected = tensor_shapes()
    fail(set(weights) == set(expected), "Adapter tensors do not match all 24 Q/V layers")
    for name, shape in expected.items():
        value = weights[name]
        fail(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and
             value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"Invalid adapter tensor: {name}")


def checkpoint(path, ctx, stage, step, parent=None, immutable=True):
    import torch
    if immutable:
        check_seal(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"format": FORMAT, "stage": stage, "step": step, "finished": True,
                "settings": SETTINGS, "base_hash": ctx["inputs"]["base_hash"], "base_id": BASE_ID,
                "chat_format": CHAT, "dataset_hashes": ctx["stats"]["sha256"], "parent_sft_sha256": parent}
    for key, value in expected.items():
        fail(obj.get(key) == value, f"Checkpoint {path}: mismatched {key}")
    check_weights(obj["weights"])
    return obj


def parent_path(ctx):
    return ctx["out"] / "checkpoints" / "selected_sft.pt"


def run_signature(ctx, name):
    parent = sha(parent_path(ctx)) if name != "sft" else None
    return {"stage": "dpo" if name == "dpo" else "sft", "base_id": BASE_ID,
            "base_hash": ctx["inputs"]["base_hash"], "sft_sha256": parent,
            "dataset_hashes": ctx["stats"]["sha256"], "chat_format": CHAT,
            "training_sources": {n: ctx["inputs"]["scripts"][n] for n in ("train.py", "data_io.py")},
            "dtype": "float32", "device": ctx["device"], "threads": ctx["threads"], "seed": 20261107,
            "settings": SETTINGS, "batch_size": 1, "lr": 0.0001 if name == "sft" else 0.00001,
            "beta": 0.1, "grad_clip": 1.0, "head_chunk": 64, "checkpoint_head": False,
            "max_tokens": 1536, "limit": 0}


def check_run(ctx, name, maximum, require_complete=False):
    import torch
    directory = ctx["out"] / "runs" / name
    if not directory.exists():
        fail(not require_complete, f"Missing completed run: {name}")
        return
    manifest = directory / "run.json"
    fail(manifest.exists(), f"Run directory has no manifest: {name}")
    run = read(manifest)
    sig = run["signature"]
    for k, value in run_signature(ctx, name).items():
        fail(sig.get(k) == value, f"Resume signature differs: {name}/{k}")
    expected_count = 500 if name == "dpo" else 1000
    fail(len(sig["selection"]) == expected_count and run["skipped_length_before_limit"] == [], "Incomplete intact sample coverage")
    ids = [r["candidate_id"] for r in sig["selection"]]
    fail(len(set(ids)) == expected_count, "Duplicate selected examples")
    state_path = directory / "state.pt"
    history = []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        check_weights(state["weights"])
        history = state["history"]
        fail(0 < len(history) <= maximum, f"Invalid resumed step count: {name}")
        for i, item in enumerate(history):
            fail(item["step"] == i + 1 and item["ids"] == [ids[i % len(ids)]] and math.isfinite(item["loss"]), "History step/order/finite-loss violation")
        recorded = rows(directory / "metrics.jsonl") if (directory / "metrics.jsonl").exists() else []
        # The atomic state is authoritative; interrupted metric publication may lag one step.
        fail(recorded == history[:len(recorded)] and len(history) - len(recorded) <= 1, "Metrics differ from atomic optimizer state")
        fail(bool(state.get("optimizer", {}).get("state")), "Missing optimizer state")
    else:
        fail(not (directory / "adapter.pt").exists() and not (directory / "metrics.jsonl").exists(), "Model/history without optimizer state")
    summary = directory / "summary.json"
    if summary.exists():
        obj = read(summary)
        fail(0 < obj["steps"] <= len(history) and obj["sha256"] == sha(directory / "adapter.pt"), "Summary/checkpoint digest or step mismatch")
        parent = run_signature(ctx, name)["sft_sha256"]
        checkpoint(directory / "adapter.pt", ctx, "dpo" if name == "dpo" else "sft", obj["steps"], parent, immutable=False)
        fail(obj["parent_sft_sha256"] == parent, "Summary parent differs")
        if obj["steps"] == len(history):
            adapter = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True)
            fail(all(torch.equal(state["weights"][k], v) for k, v in adapter["weights"].items()), "Adapter differs from final optimizer weights")
    if require_complete:
        fail(summary.exists() and read(summary)["steps"] == maximum and len(history) == maximum, "Training is incomplete")


def eval_spec(kind):
    if kind == "dev":
        return ["dev.jsonl", "dev_metadata.jsonl"], 36, 0
    if kind == "bridge":
        return ["evaluation.jsonl"], 60, 0
    return ["independent_test.jsonl", "independent_test_metadata.jsonl", "preference_eval.jsonl", "preference_eval_metadata.jsonl"], 24, 24


def check_eval(path, ctx, kind, adapter=None, sealed=True):
    if sealed:
        check_seal(path)
    result = read(path)
    names, count, generation = eval_spec(kind)
    expected_inputs = {str(ctx["data"] / n): ctx["inputs"]["dataset"][n] for n in names}
    fail(result["inputs_sha256"] == expected_inputs, f"Evaluation input identity differs: {path}")
    fail(result["source_sha256"] == ctx["inputs"]["scripts"]["evaluate.py"] and
         result["base_hash"] == ctx["inputs"]["base_hash"], "Evaluation source/base differs")
    fail(result["adapter_sha256"] == (sha(adapter) if adapter else None), "Evaluation adapter digest differs")
    if adapter:
        import torch
        ck = torch.load(adapter, map_location="cpu", weights_only=True)
        fail(result["adapter_step"] == ck["step"] and result["adapter_stage"] == ck["stage"] and result["parent_sft_sha256"] == ck.get("parent_sft_sha256"), "Evaluation ancestry differs")
    else:
        fail(result["adapter_step"] == 0 and result["adapter_stage"] == "base" and result["parent_sft_sha256"] is None, "Invalid base evaluation identity")
    protocol = result["protocol"]
    fail(protocol["generation_limit"] == generation and protocol["max_new_tokens"] == 384 and protocol["do_sample"] is False and protocol["reference_truncation"] is False, "Evaluation protocol differs")
    summary = result["summary"]
    refs, gens, prefs = result["reference_nll"], result["generation"], result["preferences"]
    fail(len(refs) == count and len(gens) == generation and len(prefs) == (100 if kind == "independent" else 0), "Evaluation is incomplete")
    import evaluate
    source_rows = evaluate.load_examples(ctx["data"] / names[0], ctx["data"] / names[1] if len(names) > 1 else None)
    fail([r["candidate_id"] for r in refs] == [r["candidate_id"] for r in source_rows], "Reference case order differs")
    for r in refs + prefs:
        fail(all(isinstance(n, int) and n > 0 for n in r["response_tokens"]) and all(math.isfinite(x) for x in r["logp_sum"]), "Invalid score/token count")
    nll = sum(-r["logp_sum"][0] for r in refs) / sum(r["response_tokens"][0] for r in refs)
    fail(math.isfinite(summary["reference_nll"]) and abs(summary["reference_nll"] - nll) < 1e-9, "NLL summary mismatch")
    fail(summary["reference_cases"] == count and summary["generation_cases"] == generation and summary["preference_pairs"] == len(prefs), "Evaluation summary coverage differs")
    for original, generated in zip(source_rows, gens):
        fail(generated["candidate_id"] == original["candidate_id"] and generated["prompt"] == original["prompt"] and generated["reference"] == original["answers"][0], "Generated prompt/reference differs")
        n = generated["generated_tokens"]
        fail(0 < n <= 384 and len(generated["generated_token_ids"]) == n and generated["stop_reason"] in ("eos", "max_new_tokens"), "Invalid generation length/stop")
        fail(generated["stop_reason"] != "max_new_tokens" or n == 384, "Inconsistent truncation reason")
    fail(summary["stop_counts"] == dict(Counter(r["stop_reason"] for r in gens)), "Stop count mismatch")
    if prefs:
        source_pairs = evaluate.load_examples(ctx["data"] / names[2], ctx["data"] / names[3])
        fail([r["candidate_id"] for r in prefs] == [r["candidate_id"] for r in source_pairs], "Preference identity/order differs")
    return result


def check_existing(ctx):
    out = ctx["out"]
    for name, digest in ctx["inputs"]["scripts"].items():
        saved = out / "sources" / name
        check_seal(saved)
        fail(sha(saved) == digest, f"Archived source differs: {name}")
    for sidecar in out.rglob("*.sha256"):
        check_seal(sidecar.with_name(sidecar.name[:-7]))
    for epoch in range(1, 4):
        snapshot = out / "checkpoints" / f"sft_epoch_{epoch}.pt"
        dev = out / "evaluation" / f"sft_epoch_{epoch}_dev.json"
        if snapshot.exists():
            checkpoint(snapshot, ctx, "sft", epoch * 1000)
        if dev.exists():
            fail(snapshot.exists(), "Dev result has no checkpoint")
            check_eval(dev, ctx, "dev", snapshot)
    selected = parent_path(ctx)
    selection = out / "selection.json"
    fail(selected.exists() == selection.exists(), "Incomplete selected-checkpoint publication")
    if selected.exists():
        check_seal(selection)
        candidates = []
        for epoch in range(1, 4):
            snap = out / "checkpoints" / f"sft_epoch_{epoch}.pt"
            dev = out / "evaluation" / f"sft_epoch_{epoch}_dev.json"
            fail(snap.exists() and dev.exists(), "Selection without all three dev candidates")
            candidates.append({"epoch": epoch, "checkpoint": str(snap), "checkpoint_sha256": sha(snap), "dev_nll": read(dev)["summary"]["reference_nll"]})
        best = min(candidates, key=lambda x: (x["dev_nll"], x["epoch"]))
        fail(read(selection) == {"candidates": candidates, "selected": best}, "Selection not minimum dev NLL")
        checkpoint(selected, ctx, "sft", best["epoch"] * 1000)
        fail(sha(selected) == best["checkpoint_sha256"], "Selected checkpoint differs from chosen epoch")
    check_run(ctx, "sft", 3000)
    for name in ("dpo", "continued_sft"):
        if (out / "runs" / name).exists():
            fail(selected.exists(), "Child run without selected SFT")
            check_run(ctx, name, 500)
    adapters = {"base": None, "sft": selected, "dpo": out / "runs/dpo/adapter.pt", "continued_sft": out / "runs/continued_sft/adapter.pt"}
    for name, adapter in adapters.items():
        for suffix, kind in (("independent", "independent"), ("bridge_nll", "bridge")):
            result = out / "evaluation" / f"{name}_{suffix}.json"
            if result.exists():
                fail(adapter is None or adapter.exists(), "Evaluation missing adapter")
                check_eval(result, ctx, kind, adapter)


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf8") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf8", errors="replace")
        try:
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line.rstrip(), flush=True)
            fail(process.wait() == 0, f"Subprocess failed; inspect {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            raise


def export(out, complete):
    target = out.parent / ("training_v2_retry_results.zip" if complete else "training_v2_retry_partial_" + uuid.uuid4().hex[:12] + ".zip")
    members = {str(p.relative_to(out.parent)).replace("\\", "/"): p for p in out.rglob("*") if p.is_file()}
    fail(all(not p.is_symlink() for p in members.values()), "Refuse symlink in result archive")
    if target.exists():
        with zipfile.ZipFile(target) as z:
            fail(set(z.namelist()) == set(members), "Existing archive members differ")
            for name, path in members.items():
                fail(hashlib.sha256(z.read(name)).hexdigest() == sha(path), "Existing archive differs from validated results")
    else:
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
            for name, path in sorted(members.items()):
                z.write(path, name)
    print("Results archive:", target, flush=True)
    return target


def require_complete(ctx):
    check_existing(ctx)
    for name, total in (("sft", 3000), ("dpo", 500), ("continued_sft", 500)):
        check_run(ctx, name, total, require_complete=True)
    for name in ("base", "sft", "dpo", "continued_sft"):
        for suffix in ("independent", "bridge_nll"):
            fail((ctx["out"] / "evaluation" / f"{name}_{suffix}.json").exists(), "Final evaluation missing")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT.parent / "synthesis/dataset")
    p.add_argument("--out", type=Path, default=ROOT / "retry_results")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--epochs", type=int, choices=(3,), default=3)
    args = p.parse_args()
    fail(args.threads > 0, "Threads must be positive")
    args.out, args.dataset, args.model = (x.resolve() for x in (args.out, args.dataset, args.model))
    fail(args.out.name == "retry_results", "Use a dedicated directory named retry_results")
    inputs = input_record(args.model, args.dataset)
    stats = read(args.dataset / "stats.json")
    fail((stats["selected"]["sft"]["count"], stats["selected"]["preference"]["count"]) == (1000, 500), "Frozen counts differ")
    protocol = {"version": "exploratory-retry-2", "base": BASE_ID, "epochs": 3, "seed": 20261107, "adapter": SETTINGS,
                "sft_learning_rate": 0.0001, "dpo_learning_rate": 0.00001, "beta": 0.1, "batch_size": 1,
                "device": args.device, "dtype": "float32", "threads": args.threads,
                "evaluation_status": "exploratory; chosen after observing first-run test outputs; not a blind test",
                "selection": "minimum 36-case dev NLL after three complete SFT epochs; ties choose earlier epoch",
                "control": "500 added updates and shared parent; tokens and compute are not matched",
                "generation": {"max_new_tokens": 384, "do_sample": False, "cases": 24}, "final_evaluation_required": True}
    ctx = {"out": args.out, "data": args.dataset, "inputs": inputs, "stats": stats, "device": args.device, "threads": args.threads}
    existing = args.out.exists() and any(args.out.iterdir())
    if existing:
        for name, expected in (("protocol.json", protocol), ("inputs.json", inputs)):
            check_seal(args.out / name)
            fail(read(args.out / name) == expected, "Existing protocol/input hashes differ")
        check_existing(ctx)  # Before any mutating trainer or appended log.
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        publish_json(args.out / "protocol.json", protocol)
        publish_json(args.out / "inputs.json", inputs)
        for name in inputs["scripts"]:
            copy_new(ROOT / name, args.out / "sources" / name)
    if (args.out / "completed.json").exists():
        check_seal(args.out / "completed.json")
        complete = read(args.out / "completed.json")
        fail(complete.get("completed") is True and complete.get("final_evaluation_run") is True, "Invalid completion record")
        require_complete(ctx); export(args.out, True); return
    attempt = args.out / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    publish_json(attempt / "hardware.json", gpu_record(args.device))
    mutable = {str(f.relative_to(args.out)): sha(f) for f in (args.out / "runs").rglob("*") if f.is_file()} if (args.out / "runs").exists() else {}
    publish_json(attempt / "preflight.json", {"inputs": inputs, "existing_artifacts_validated": existing, "mutable_files_observed_sha256": mutable})
    common = [sys.executable, "-u", ROOT / "train.py", "--model", args.model, "--dataset", args.dataset, "--limit", "0", "--batch-size", "1", "--device", args.device, "--dtype", "float32", "--threads", str(args.threads), "--seed", "20261107", "--max-tokens", "1536"]
    evaluate = [sys.executable, "-u", ROOT / "evaluate.py", "--model", args.model, "--device", args.device, "--dtype", "float32", "--threads", str(args.threads)]
    start = time.perf_counter()
    try:
        candidates = []
        sft_dir = args.out / "runs/sft"
        for epoch in range(1, 4):
            snap = args.out / "checkpoints" / f"sft_epoch_{epoch}.pt"
            dev = args.out / "evaluation" / f"sft_epoch_{epoch}_dev.json"
            if not snap.exists():
                check_run(ctx, "sft", epoch * 1000)
                cmd = common + ["--stage", "sft", "--out", sft_dir, "--steps", str(epoch * 1000), "--rank", "16", "--alpha", "32", "--last-layers", "24", "--lr", "0.0001"]
                if (sft_dir / "run.json").exists(): cmd += ["--resume"]
                run(cmd, args.out / "logs/sft.log"); check_run(ctx, "sft", epoch * 1000, True)
                copy_new(sft_dir / "adapter.pt", snap)
            checkpoint(snap, ctx, "sft", epoch * 1000)
            if not dev.exists():
                run(evaluate + ["--adapter", snap, "--data", args.dataset / "dev.jsonl", "--metadata", args.dataset / "dev_metadata.jsonl", "--out", dev], args.out / "logs/dev.log")
                check_eval(dev, ctx, "dev", snap, sealed=False); seal_new(dev)
            check_eval(dev, ctx, "dev", snap)
            candidates.append({"epoch": epoch, "checkpoint": str(snap), "checkpoint_sha256": sha(snap), "dev_nll": read(dev)["summary"]["reference_nll"]})
        best = min(candidates, key=lambda x: (x["dev_nll"], x["epoch"]))
        selected = parent_path(ctx)
        copy_new(Path(best["checkpoint"]), selected)
        publish_json(args.out / "selection.json", {"candidates": candidates, "selected": best})
        for name in ("dpo", "continued_sft"):
            directory = args.out / "runs" / name
            check_run(ctx, name, 500)
            if not (directory / "summary.json").exists():
                cmd = common + ["--stage", "dpo" if name == "dpo" else "sft", "--out", directory, "--steps", "500", "--lr", "0.00001", "--sft-checkpoint" if name == "dpo" else "--initial-sft", selected]
                if (directory / "run.json").exists(): cmd += ["--resume"]
                run(cmd, args.out / "logs" / f"{name}.log")
            check_run(ctx, name, 500, True)
        models = {"base": None, "sft": selected, "dpo": args.out / "runs/dpo/adapter.pt", "continued_sft": args.out / "runs/continued_sft/adapter.pt"}
        for name, adapter in models.items():
            for suffix, kind in (("independent", "independent"), ("bridge_nll", "bridge")):
                output = args.out / "evaluation" / f"{name}_{suffix}.json"
                if not output.exists():
                    if kind == "independent":
                        cmd = evaluate + ["--data", args.dataset / "independent_test.jsonl", "--metadata", args.dataset / "independent_test_metadata.jsonl", "--preferences", args.dataset / "preference_eval.jsonl", "--preference-metadata", args.dataset / "preference_eval_metadata.jsonl", "--generation-limit", "24", "--max-new-tokens", "384", "--out", output]
                    else:
                        cmd = evaluate + ["--data", args.dataset / "evaluation.jsonl", "--out", output]
                    if adapter: cmd += ["--adapter", adapter]
                    run(cmd, args.out / "logs" / f"{name}_{suffix}.log")
                    check_eval(output, ctx, kind, adapter, sealed=False); seal_new(output)
                check_eval(output, ctx, kind, adapter)
        require_complete(ctx)
        publish_json(args.out / "completed.json", {"completed": True, "final_evaluation_run": True, "elapsed_seconds_this_attempt": time.perf_counter()-start, "selected_epoch": best["epoch"], "all_inputs_and_outputs_validated": True})
        export(args.out, True)
    except BaseException as error:
        fresh_json(attempt / "failure.json", {"type": type(error).__name__, "message": str(error), "completed": False})
        export(args.out, False)
        raise


if __name__ == "__main__":
    main()
