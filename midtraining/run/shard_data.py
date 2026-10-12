#!/usr/bin/env python3
"""shard_data.py — split each dataset's data.jsonl into N shards for datatrove.

Why this exists: datatrove's readers shard work *by file* — with n_tasks tasks,
input file i is processed by task (i % n_tasks). prepare_data.py writes a
single <Name>/data.jsonl per dataset, so a tokenize run on it gives the whole
file to ONE task while the other n_tasks-1 tasks "complete" instantly with
empty .ds outputs: tokenization runs on a single core no matter how large
--n-tasks is. Splitting each dataset into shards first restores real
parallelism.

Sharding is a separate step (not part of prepare_data.py) because the shard
count should match the machine that runs tokenize_data.sh: pick ~the same
value as the --n-tasks you will pass there (this machine has 64 cores; both
this script and tokenize_data.sh default to 32 so ~half the cores tokenize
while the rest stay free for I/O and other jobs). More files than tasks is
harmless — datatrove just round-robins them; fewer files than tasks means
idle tokenize workers.

Per <jsonl-root>/<Name>/:
  * data.jsonl present    -> split into data.shard-000.jsonl ... and delete
                             data.jsonl (existing shards are replaced; the
                             file is reproducible via prepare_data.py).
  * only shards present   -> re-shard from them. Shards are contiguous,
                             line-aligned byte ranges of the original file,
                             so concatenating them in index order reproduces
                             data.jsonl exactly and re-sharding is lossless.
  * already N shards      -> skipped (idempotent).
  * nothing found         -> warning (failure if the dataset was named
                             explicitly via --datasets).

Examples:
    python shard_data.py                            # every dataset dir, 32 shards
    python shard_data.py --n-shards 64              # e.g. --n-tasks 64 on tokenize
    python shard_data.py --datasets FineMath4Plus,MegaMathQA --n-shards 48
    JSONL_ROOT=/path/to/jsonl python shard_data.py  # same env var as tokenize_data.sh
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSONL_ROOT = REPO_ROOT / "midtraining" / "data" / "jsonl"
DEFAULT_N_SHARDS = 32
IO_BUFFER = 1 << 20  # 1 MiB


def out_paths(dataset_dir: Path, n_shards: int) -> list[Path]:
    return [dataset_dir / f"data.shard-{i:03d}.jsonl" for i in range(n_shards)]


def split_dataset(dataset_dir: Path, n_shards: int) -> tuple[str, str]:
    """Shard one dataset dir. Returns (status, message); status in ok/skip/error."""
    single = dataset_dir / "data.jsonl"
    old_shards = sorted(dataset_dir.glob("data.shard-*.jsonl"))
    # clean leftovers from an interrupted previous run (inputs are rebuilt below)
    for tmp in dataset_dir.glob("data.shard-*.jsonl.tmp"):
        tmp.unlink()

    if single.exists():
        inputs, input_kind = [single], "data.jsonl"
        if old_shards:
            print(f"    note: data.jsonl and {len(old_shards)} stale shard(s) coexist; "
                  f"sharding the fresh data.jsonl and dropping the stale shards")
    elif old_shards:
        if len(old_shards) == n_shards:
            return "skip", f"already sharded into {n_shards} files"
        inputs, input_kind = old_shards, "shards"
    else:
        return "absent", "no data.jsonl and no data.shard-*.jsonl found"

    total = sum(os.path.getsize(p) for p in inputs)
    boundaries = [total * i // n_shards for i in range(1, n_shards)]
    outs = out_paths(dataset_dir, n_shards)
    tmps = [Path(str(p) + ".tmp") for p in outs]
    handles = [open(t, "wb", buffering=IO_BUFFER) for t in tmps]
    counts = [0] * n_shards
    n_lines = 0
    try:
        shard_idx = 0
        written = 0
        next_boundary = boundaries[0] if boundaries else None
        for src in inputs:
            with open(src, "rb", buffering=IO_BUFFER) as fh:
                for line in fh:
                    while next_boundary is not None and written >= next_boundary:
                        shard_idx += 1
                        next_boundary = (boundaries[shard_idx]
                                         if shard_idx < len(boundaries) else None)
                    handles[shard_idx].write(line)
                    written += len(line)
                    counts[shard_idx] += 1
                    n_lines += 1
    finally:
        for h in handles:
            h.close()

    # drop shards of a previous layout that the new one does not rewrite
    # (in re-shard mode this happens only after every input has been read)
    keep = {outs[i] for i in range(n_shards) if counts[i] > 0}
    for p in old_shards:
        if p not in keep:
            p.unlink()
    for i, tmp in enumerate(tmps):
        if counts[i] > 0:
            os.replace(tmp, outs[i])  # atomic on the same filesystem
        else:
            tmp.unlink()  # fewer lines than shards: trailing shard is empty
    if input_kind == "data.jsonl":
        single.unlink()

    n_kept = sum(1 for c in counts if c > 0)
    return "ok", (f"{n_kept} shard(s), {n_lines:,} lines, {total / 1e9:.2f} GB "
                  f"(from {input_kind})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl-root", type=Path,
                    default=Path(os.environ.get("JSONL_ROOT", DEFAULT_JSONL_ROOT)),
                    help="dataset dir root (default: midtraining/data/jsonl; env JSONL_ROOT)")
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated dir names to shard (default: every subdir)")
    ap.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS,
                    help=f"shard count per dataset (default: {DEFAULT_N_SHARDS}; "
                         "match tokenize_data.sh --n-tasks on the tokenize machine)")
    args = ap.parse_args()

    root: Path = args.jsonl_root
    if not root.is_dir():
        print(f"ERROR: jsonl root {root} does not exist", file=sys.stderr)
        sys.exit(2)

    if args.n_shards < 1:
        print("ERROR: --n-shards must be >= 1", file=sys.stderr)
        sys.exit(2)

    if args.datasets:
        requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
        targets = [root / d for d in requested]
        explicit = True
    else:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
        explicit = False

    failures = []
    for target in targets:
        name = target.name
        if not target.is_dir():
            print(f"[{name}] ERROR: not a dataset dir under {root}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"[{name}] sharding into {args.n_shards} ...")
        t0 = time.time()
        try:
            status, msg = split_dataset(target, args.n_shards)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(name)
            continue
        elapsed = round(time.time() - t0, 1)
        if status == "error":
            print(f"    ERROR: {msg}", file=sys.stderr)
            failures.append(name)
        elif status == "absent":
            # nothing to shard here: only an error when the dataset was named
            # explicitly; harmless when scanning every dir in the root
            print(f"    skip: {msg}")
            if explicit:
                failures.append(name)
        elif status == "skip":
            print(f"    skip: {msg}")
            if explicit:
                failures.append(name)  # named explicitly but nothing to do
        else:
            print(f"    OK: {msg} ({elapsed}s)")

    print()
    if failures:
        print(f"FAILED/skipped datasets: {failures}", file=sys.stderr)
        sys.exit(1)
    print("all datasets sharded.")


if __name__ == "__main__":
    main()
