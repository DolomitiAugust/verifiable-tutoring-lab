"""Run the prespecified base -> SFT -> DPO experiment and a matched-update SFT control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf8") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf8", errors="replace")
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line.rstrip(), flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed; preserve log {log}")


def export(results):
    target = results.parent / "training_v2_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(results.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(results.parent))
    print(f"Results archive: {target}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT.parent / "synthesis" / "dataset")
    p.add_argument("--out", type=Path, default=ROOT / "cloud_results")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-final-eval", action="store_true")
    args = p.parse_args()
    if args.epochs < 1:
        raise ValueError("At least one complete SFT epoch is required.")
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.dataset
    stats = read(data / "stats.json")
    sft_count, pref_count = (stats["selected"][x]["count"] for x in ("sft", "preference"))
    if (sft_count, pref_count) != (1000, 500):
        raise ValueError("This frozen protocol expects 1000 SFT samples and 500 preference pairs.")
    # Batch 1 avoids padded repetition at epoch boundaries and fits CPU/T4 without mixed-precision scaling.
    common = [sys.executable, "-u", ROOT / "train.py", "--model", args.model, "--dataset", data,
              "--limit", "0", "--batch-size", "1", "--device", args.device, "--dtype", "float32",
              "--threads", str(args.threads), "--seed", "20261107", "--max-tokens", "1536"]
    evaluation = [sys.executable, "-u", ROOT / "evaluate.py", "--model", args.model,
                  "--device", args.device, "--dtype", "float32", "--threads", str(args.threads)]
    protocol = {"base": "Qwen/Qwen2.5-0.5B", "epochs": args.epochs, "seed": 20261107,
                "adapter": {"rank": 16, "alpha": 32, "last_layers": 8},
                "sft_learning_rate": 0.0002, "dpo_learning_rate": 0.00001, "beta": 0.1,
                "scheduler": "constant (same verified optimizer as v1)", "batch_size": 1,
                "sft_selection": "lowest reference NLL on separate dev.jsonl after each full epoch; ties choose earlier epoch",
                "control": "same SFT parent, learning rate, batch size and number of additional optimizer updates as DPO; response tokens and wall time are not identical",
                "generation": {"max_new_tokens": 384, "sampling": False, "limit": 24},
                "device": args.device, "dtype": "float32",
                "script_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT / "train.py", ROOT / "evaluate.py", Path(__file__))}}
    plan = args.out / "protocol.json"
    if plan.exists() and read(plan) != protocol:
        raise ValueError("Existing output has a different protocol. Use another output directory.")
    plan.write_text(json.dumps(protocol, indent=2), encoding="utf8")
    started = time.perf_counter()
    epoch_results = []
    try:
        sft_dir = args.out / "runs" / "sft"
        for epoch in range(1, args.epochs + 1):
            snapshot = args.out / "checkpoints" / f"sft_epoch_{epoch}.pt"
            dev = args.out / "evaluation" / f"sft_epoch_{epoch}_dev.json"
            if not snapshot.exists():
                command = common + ["--stage", "sft", "--out", sft_dir, "--steps", str(sft_count * epoch),
                                    "--rank", "16", "--alpha", "32", "--last-layers", "8", "--lr", "0.0002"]
                if (sft_dir / "run.json").exists():
                    command += ["--resume"]
                run(command, args.out / "logs" / "sft.log")
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sft_dir / "adapter.pt", snapshot)
            if not dev.exists():
                run(evaluation + ["--adapter", snapshot, "--data", data / "dev.jsonl",
                                  "--metadata", data / "dev_metadata.jsonl", "--out", dev], args.out / "logs" / "dev.log")
            epoch_results.append({"epoch": epoch, "checkpoint": str(snapshot), "dev_nll": read(dev)["summary"]["reference_nll"]})
            export(args.out)
        best = min(epoch_results, key=lambda x: (x["dev_nll"], x["epoch"]))
        selected = args.out / "checkpoints" / "selected_sft.pt"
        shutil.copy2(best["checkpoint"], selected)
        (args.out / "selection.json").write_text(json.dumps({"candidates": epoch_results, "selected": best}, indent=2), encoding="utf8")
        for stage in ("dpo", "continued_sft"):
            directory = args.out / "runs" / stage
            if not (directory / "summary.json").exists():
                command = common + ["--stage", "dpo" if stage == "dpo" else "sft", "--out", directory,
                                    "--steps", str(pref_count), "--lr", "0.00001"]
                command += ["--sft-checkpoint" if stage == "dpo" else "--initial-sft", selected]
                if (directory / "run.json").exists():
                    command += ["--resume"]
                run(command, args.out / "logs" / f"{stage}.log")
                export(args.out)
        if not args.skip_final_eval:
            models = {"base": None, "sft": selected, "dpo": args.out / "runs" / "dpo" / "adapter.pt",
                      "continued_sft": args.out / "runs" / "continued_sft" / "adapter.pt"}
            for name, adapter in models.items():
                output = args.out / "evaluation" / f"{name}_independent.json"
                if not output.exists():
                    command = evaluation + ["--data", data / "independent_test.jsonl", "--metadata", data / "independent_test_metadata.jsonl",
                                            "--out", output, "--generation-limit", "24", "--max-new-tokens", "384"]
                    if adapter:
                        command += ["--adapter", adapter]
                    if (data / "preference_eval.jsonl").exists():
                        command += ["--preferences", data / "preference_eval.jsonl"]
                        if (data / "preference_eval_metadata.jsonl").exists():
                            command += ["--preference-metadata", data / "preference_eval_metadata.jsonl"]
                    run(command, args.out / "logs" / f"{name}_evaluation.log")
                bridge = args.out / "evaluation" / f"{name}_bridge_nll.json"
                if not bridge.exists():
                    command = evaluation + ["--data", data / "evaluation.jsonl", "--out", bridge]
                    if adapter:
                        command += ["--adapter", adapter]
                    run(command, args.out / "logs" / f"{name}_bridge.log")
                export(args.out)
        (args.out / "completed.json").write_text(json.dumps({"completed": True, "elapsed_seconds": time.perf_counter()-started,
                  "final_evaluation_run": not args.skip_final_eval}), encoding="utf8")
    finally:
        export(args.out)


if __name__ == "__main__":
    main()
