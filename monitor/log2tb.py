#!/usr/bin/env python3
"""log2tb.py — tail a nanotron / verl training log and mirror it to TensorBoard.

Reads per-step lines that nanotron (LoggerWriter) or verl (LocalLogger) print to
the console, and writes every metric as TB scalars plus a JSONL dump:

  * TensorBoard event files  -> <tb-dir>/events.out.tfevents.*   (pure `tensorboard`
    pip package, no torch needed)
  * metrics dump             -> <tb-dir>/metrics.jsonl           (one JSON per step)

Run `python3 log2tb.py --help` for the CLI. Designed to run under the miniforge
base python; it only reads the log file and never touches the training process.
"""
import argparse
import json
import math
import os
import re
import signal
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------
# TensorBoard writer backend (no torch).
# --------------------------------------------------------------------------
_TB_BACKEND = None
EventFileWriter = None
event_pb2 = None
summary_pb2 = None


def _init_tb_backend():
    """Import a torch-free EventFileWriter. Prefer `tensorboard`, fall back to
    `tensorboardX`. Exits with a hint if neither is importable."""
    global _TB_BACKEND, EventFileWriter, event_pb2, summary_pb2
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter as _EFW
        from tensorboard.compat.proto import event_pb2 as _epb2
        from tensorboard.compat.proto import summary_pb2 as _spb2
        EventFileWriter, event_pb2, summary_pb2 = _EFW, _epb2, _spb2
        _TB_BACKEND = "tensorboard"
        return
    except ImportError:
        pass
    try:
        from tensorboardX.event_file_writer import EventFileWriter as _EFW  # type: ignore
        from tensorboardX.proto import event_pb2 as _epb2  # type: ignore
        from tensorboardX.proto import summary_pb2 as _spb2  # type: ignore
        EventFileWriter, event_pb2, summary_pb2 = _EFW, _epb2, _spb2
        _TB_BACKEND = "tensorboardX"
        return
    except ImportError:
        pass
    sys.stderr.write(
        "ERROR: neither `tensorboard` nor `tensorboardX` is importable.\n"
        "Install one into the base python, e.g.:\n"
        "  /home/lawson/miniforge3/bin/python3 -m pip install -i "
        "https://pypi.tuna.tsinghua.edu.cn/simple tensorboard\n")
    sys.exit(2)


# --------------------------------------------------------------------------
# Number parsing: nanotron human_format() -> K/M/G/T/P/E suffixes (1000-based),
# scientific notation, plain ints/floats.
# --------------------------------------------------------------------------
_SUFFIX_MULT = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18}
_NUM_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([KMGTPEkMgtpE])?$")


def parse_human_num(tok):
    """Parse a nanotron-style human-formatted number. Returns float or None."""
    tok = tok.strip()
    if not tok:
        return None
    m = _NUM_RE.match(tok)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    suffix = m.group(2)
    if suffix:
        val *= _SUFFIX_MULT[suffix.upper()]
    return val


def _is_finite(x):
    return x is not None and math.isfinite(x)


# --------------------------------------------------------------------------
# nanotron parser.
# Line (loguru, world rank 0 only):
#   10/10/2026 14:23:01 [INFO|DP=0|PP=0|TP=0]: iteration: 12 / 4624 | consumed_tokens: 51.4M | ...
# --------------------------------------------------------------------------
_NANO_HEADER_RE = re.compile(r"\[(?:INFO|WARNING)\|DP=(\d+)\|PP=\d+\|TP=\d+[^\]]*\]")
_NANO_ITER_RE = re.compile(r"iteration:\s*(\d+)\s*/\s*(\d+)")
_NANO_TAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NANO_TIME_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{4}[ T]\d{2}:\d{2}:\d{2})")


def parse_nanotron_line(line):
    """Return dict(step, total, metrics, wall_time) or None for non-matching lines."""
    hdr = _NANO_HEADER_RE.search(line)
    if hdr and int(hdr.group(1)) != 0:
        return None  # only DP rank 0 logs the training line
    it = _NANO_ITER_RE.search(line)
    if not it:
        return None
    step, total = int(it.group(1)), int(it.group(2))
    metrics = {}
    for item in line[it.end():].split("|"):
        if ":" not in item:
            continue
        key, _, val = item.partition(":")
        key = key.strip()
        if not _NANO_TAG_RE.match(key):
            continue
        num = parse_human_num(val)
        if num is not None:
            metrics[key] = num
    wall = None
    tm = _NANO_TIME_RE.match(line)
    if tm:
        for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                wall = datetime.strptime(tm.group(1), fmt).timestamp()
                break
            except ValueError:
                continue
    return {"step": step, "total": total, "metrics": metrics,
            "wall_time": wall or time.time(), "source": "nanotron"}


# --------------------------------------------------------------------------
# verl parser.
# RayPPOTrainer logs through verl.utils.tracking.Tracking -> LocalLogger:
#   verl/utils/logger/aggregate_logger.py: concat_dict_to_str()
#       output = [f'step:{step}'] + [f'{k}:{v:.3f}' for numeric v]
#       output_str = ' - '.join(output)
# Real line looks like:
#   step:12 - critic/kl:0.005 - actor/pg_loss:-0.012 - actor/grad_norm:2.345 - response_length/mean:512.345 - Tokens/Sec/GPU:1234.567
# Validation metrics come through the same format (val/test_score/... keys).
# --------------------------------------------------------------------------
_VERL_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\b")
_VERL_KV_RE = re.compile(
    r"^([A-Za-z0-9_/]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


def parse_verl_line(line):
    """Return dict(step, total=None, metrics, wall_time) or None."""
    m = _VERL_STEP_RE.search(line)
    if not m:
        return None
    step = int(m.group(1))
    rest = line[m.end():].strip()
    if rest.startswith("-"):
        rest = rest[1:].strip()
    metrics = {}
    if rest:
        for item in rest.split(" - "):
            kv = _VERL_KV_RE.match(item.strip())
            if kv:
                try:
                    metrics[kv.group(1)] = float(kv.group(2))
                except ValueError:
                    continue
    if not metrics:
        return None
    return {"step": step, "total": None, "metrics": metrics,
            "wall_time": time.time(), "source": "verl"}


PARSERS = {"nanotron": parse_nanotron_line, "verl": parse_verl_line}


def parse_any_line(line, fmt):
    """Dispatch to the configured parser; 'auto' tries nanotron then verl."""
    if fmt in PARSERS:
        return PARSERS[fmt](line)
    rec = parse_nanotron_line(line)
    if rec is not None:
        return rec
    return parse_verl_line(line)


# --------------------------------------------------------------------------
# Derived metrics (nanotron only) + live summary formatting.
# --------------------------------------------------------------------------
MFU_PEAK_TFLOPS = 312.0  # A100-PCIE-40GB bf16 dense peak
EMA_ALPHA = 0.1


def derive_nanotron(rec, args, state):
    """Add ppl / mfu / eta / EMA entries into rec['metrics'] in place."""
    m = rec["metrics"]
    if _is_finite(m.get("lm_loss")):
        m["ppl"] = math.exp(m["lm_loss"])
    if _is_finite(m.get("model_tflops_per_gpu")):
        m["mfu"] = m["model_tflops_per_gpu"] / MFU_PEAK_TFLOPS
    tps = m.get("tokens_per_sec")
    if _is_finite(tps):
        state["tps_ema"] = tps if state.get("tps_ema") is None else \
            EMA_ALPHA * tps + (1.0 - EMA_ALPHA) * state["tps_ema"]
        m["tokens_per_sec_ema"] = state["tps_ema"]
    # ETA: remaining tokens / throughput
    total_tokens = args.eta_total_tokens
    if not total_tokens and rec.get("total"):
        tps_step = args.tokens_per_step
        if not tps_step:
            gbs = m.get("global_batch_size")
            if _is_finite(gbs) and args.seq_len:
                tps_step = gbs * args.seq_len
        if tps_step:
            total_tokens = rec["total"] * tps_step
    consumed = m.get("consumed_tokens")
    if total_tokens and _is_finite(consumed) and _is_finite(tps) and tps > 0:
        m["progress"] = min(max(consumed / float(total_tokens), 0.0), 1.0)
        m["eta_hours"] = max(total_tokens - consumed, 0.0) / tps / 3600.0


def fmt_duration(seconds):
    if seconds is None:
        return "?"
    seconds = float(seconds)
    if seconds < 0:
        return "?"
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def fmt_live_line(fmt, rec):
    m = rec["metrics"]
    step = rec["step"]
    if fmt == "nanotron":
        total = rec.get("total") or "?"
        eta_s = m.get("eta_hours", 0) * 3600.0 if "eta_hours" in m else None
        parts = [f"step {step}/{total}"]
        if "lm_loss" in m:
            parts.append(f"loss {m['lm_loss']:.3f}")
        if "ppl" in m:
            parts.append(f"ppl {m['ppl']:.1f}")
        if "lr" in m:
            parts.append(f"lr {m['lr']:.1e}")
        if "grad_norm" in m:
            parts.append(f"gnorm {m['grad_norm']:.2f}")
        if "tokens_per_sec_per_gpu" in m:
            parts.append(f"{m['tokens_per_sec_per_gpu']:.0f} tok/s/gpu")
        if "mfu" in m:
            parts.append(f"mfu {100.0 * m['mfu']:.1f}%")
        if eta_s is not None:
            parts.append(f"eta {fmt_duration(eta_s)}")
        return " ".join(parts)
    parts = [f"step {step}"]
    for key, label in (("critic/kl", "kl"), ("critic/rewards/mean", "reward"),
                       ("response_length/mean", "resp_len"), ("actor/pg_loss", "pg_loss"),
                       ("actor/grad_norm", "gnorm"), ("actor/entropy_loss", "entropy"),
                       ("val/test_score/avg_acc", "val_acc")):
        if key in m:
            parts.append(f"{label} {m[key]:.3f}")
    if "Tokens/Sec/GPU" in m:
        parts.append(f"{m['Tokens/Sec/GPU']:.0f} tok/s/gpu")
    if "timing_s/step" in m:
        parts.append(f"step {fmt_duration(m['timing_s/step'])}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# tail -F style line reader (handles rotation / truncation by path).
# --------------------------------------------------------------------------
def iter_lines(path, stop_flag, once=False):
    while not os.path.exists(path):
        if once:
            raise SystemExit(f"[log2tb] ERROR: log file not found: {path}")
        if stop_flag():
            return
        time.sleep(1.0)  # training may not have created the log yet
    fh = open(path, "r", errors="replace")
    inode = os.fstat(fh.fileno()).st_ino
    buf = ""
    while True:
        chunk = fh.read()
        if chunk:
            buf += chunk
            lines = buf.split("\n")
            buf = lines.pop()  # trailing partial line stays buffered
            for line in lines:
                yield line
            if once:
                break
            continue
        if once:
            if buf:
                yield buf  # last line without trailing newline
            break
        try:
            st = os.stat(path)
            if st.st_ino != inode or st.st_size < fh.tell():
                fh.close()
                fh = open(path, "r", errors="replace")
                inode = os.fstat(fh.fileno()).st_ino
                buf = ""
        except FileNotFoundError:
            pass  # keep waiting for the file to reappear
        if stop_flag():
            break
        time.sleep(1.0)
    fh.close()


# --------------------------------------------------------------------------
# TB + JSONL sink.
# --------------------------------------------------------------------------
class MonitorSink:
    """Writes new (step, key) pairs to a TB event file and metrics.jsonl."""

    def __init__(self, tb_dir):
        self.tb_dir = tb_dir
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = EventFileWriter(tb_dir)
        self.jsonl_path = os.path.join(tb_dir, "metrics.jsonl")
        self.jsonl = open(self.jsonl_path, "a", buffering=1)
        self.seen = set()  # (step, key) already written
        self._load_seen()

    def _load_seen(self):
        if not os.path.exists(self.jsonl_path):
            return
        try:
            with open(self.jsonl_path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = rec.get("step")
                    for key in (rec.get("metrics") or {}):
                        self.seen.add((step, key))
        except OSError:
            pass

    def add(self, rec, prefix=""):
        """Write only unseen (step, key) pairs. Returns the new metrics dict."""
        step = rec["step"]
        wall = rec.get("wall_time") or time.time()
        new = {}
        for key, val in rec["metrics"].items():
            if not _is_finite(val):
                continue
            if (step, key) in self.seen:
                continue
            self.seen.add((step, key))
            new[key] = val
        if not new:
            return {}
        summary = summary_pb2.Summary(value=[
            summary_pb2.Summary.Value(tag=f"{prefix}{key}", simple_value=float(val))
            for key, val in new.items()
        ])
        self.writer.add_event(event_pb2.Event(wall_time=wall, step=step,
                                              summary=summary))
        self.jsonl.write(json.dumps({
            "step": step,
            "wall_time": wall,
            "total": rec.get("total"),
            "metrics": new,
        }) + "\n")
        return new

    def flush(self):
        try:
            self.writer.flush()
        except Exception:
            pass
        try:
            self.jsonl.flush()
        except Exception:
            pass

    def close(self):
        self.flush()
        try:
            self.writer.close()
        except Exception:
            pass
        try:
            self.jsonl.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tail a nanotron/verl training log into TensorBoard events.")
    ap.add_argument("--log", required=True, help="training log file to tail")
    ap.add_argument("--tb-dir", required=True,
                    help="output dir for TB events + metrics.jsonl")
    ap.add_argument("--format", choices=["nanotron", "verl", "auto"], default="auto")
    ap.add_argument("--seq-len", type=int, default=8192,
                    help="sequence length used for ETA token accounting")
    ap.add_argument("--tokens-per-step", type=int, default=None,
                    help="explicit tokens per step (overrides global_batch_size*seq_len)")
    ap.add_argument("--eta-total-tokens", type=int, default=None,
                    help="explicit total token budget for ETA")
    ap.add_argument("--once", action="store_true",
                    help="parse the whole file once and exit")
    args = ap.parse_args(argv)

    _init_tb_backend()
    sink = MonitorSink(args.tb_dir)
    state = {"tps_ema": None}
    stop = {"flag": False}

    def _sig_handler(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    print(f"[log2tb] backend={_TB_BACKEND} log={args.log} tb_dir={args.tb_dir} "
          f"format={args.format}{' (once)' if args.once else ''}", flush=True)

    def prefix_for(line, rec):
        # nanotron metrics are flat -> group them under train/; verl metrics
        # already carry their own critic/ actor/ timing_s/ val/ grouping.
        if args.format == "verl":
            return ""
        if args.format == "nanotron":
            return "train/"
        return "train/" if rec.get("source") == "nanotron" else ""

    n_steps = 0
    try:
        for line in iter_lines(args.log, lambda: stop["flag"], once=args.once):
            if stop["flag"]:
                break
            if not line.strip():
                continue
            try:
                rec = parse_any_line(line, args.format)
            except Exception as exc:  # never die on a bad line
                print(f"[log2tb] WARN: unparsable line skipped: {exc}: {line[:120]}",
                      file=sys.stderr, flush=True)
                continue
            if rec is None or not rec["metrics"]:
                continue
            try:
                prefix = prefix_for(line, rec)
                rec = dict(rec, metrics=dict(rec["metrics"]))
                if prefix == "train/":
                    derive_nanotron(rec, args, state)
                new = sink.add(rec, prefix=prefix)
                if new:
                    n_steps += 1
                    print("[log2tb] " + (f"{prefix[:-1]} " if prefix else "") +
                          fmt_live_line("nanotron" if prefix else "verl", rec),
                          flush=True)
                    if n_steps % 10 == 0:
                        sink.flush()
            except Exception as exc:
                print(f"[log2tb] WARN: failed to record step: {exc}", file=sys.stderr,
                      flush=True)
                continue
            if args.once and stop["flag"]:
                break
    finally:
        sink.close()
    print(f"[log2tb] done: {n_steps} new step record(s) -> {args.tb_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
