#!/usr/bin/env python3
"""test_parse.py — sanity tests for monitor/log2tb.py.

Run with the miniforge BASE python (has tensorboard installed):
    /home/lawson/miniforge3/bin/python3 monitor/tests/test_parse.py
"""
import json
import math
import os
import sys
import tempfile
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import log2tb as m  # noqa: E402

NANO_LOG = os.path.join(HERE, "nanotron_sample.log")
VERL_LOG = os.path.join(HERE, "verl_sample.log")


def _close(a, b, rel=1e-6, abs_=1e-9):
    return abs(a - b) <= max(abs_ , rel * max(abs(a), abs(b)))


def test_parse_human_num():
    cases = {
        "51.4M": 51.4e6, "1M": 1e6, "166.7K": 166.7e3, "4.32K": 4320.0,
        "7.28T": 7.28e12, "20.1G": 20.1e9, "3e-05": 3e-05, "2.9e-05": 2.9e-05,
        "528": 528.0, "45.2": 45.2, "2.345": 2.345, "0.000123": 1.23e-4,
        "999K": 999e3, "-0.012": -0.012,
    }
    for tok, expected in cases.items():
        got = m.parse_human_num(tok)
        assert got is not None and _close(got, expected), (tok, got, expected)
    for bad in ["", "abc", "1.2.3", "12MB", "N/A", "--"]:
        assert m.parse_human_num(bad) is None, bad
    print("PASS test_parse_human_num")


def test_nanotron_parse():
    records = []
    with open(NANO_LOG) as fh:
        for line in fh:
            rec = m.parse_nanotron_line(line)
            if rec is not None:
                records.append(rec)
    # 33 data lines parsed (step 20 duplicated, DP=1 line filtered out)
    assert len(records) == 33, len(records)
    steps = [r["step"] for r in records]
    assert len(set(steps)) == 32, steps            # duplicate step 20 dedups later
    assert 31 not in steps                          # DP=1 line filtered out
    assert 32 in steps and 33 in steps              # headerless + memory line parsed
    by_step = {r["step"]: r for r in records}
    r12 = by_step[12]
    assert r12["total"] == 40
    mm = r12["metrics"]
    assert _close(mm["consumed_tokens"], 51.4e6)
    assert _close(mm["tokens_per_sec"], 1e6)
    assert _close(mm["tokens_per_sec_per_gpu"], 166.7e3)
    assert _close(mm["elapsed_time_per_iteration_ms"], 4320.0)
    assert _close(mm["global_batch_size"], 528.0)
    assert _close(mm["lr"], 3e-05)
    assert _close(mm["lm_loss"], 2.345)
    assert _close(mm["grad_norm"], 1.234)
    assert _close(mm["model_tflops_per_gpu"], 45.2)
    assert _close(mm["hardware_tflops_per_gpu"], 89.1)
    # K/M/G/T suffix expansion on the memory line
    mm33 = by_step[33]["metrics"]
    assert _close(mm33["cuda_memory_allocated"], 20.1e9)
    assert _close(mm33["cuda_max_memory_reserved"], 27.5e9)
    assert _close(mm33["hd_total_memory_tb"], 7.28e12)
    assert _close(mm33["hd_free_memory_tb"], 3.64e12)
    # missing grad_norm tolerated (step 15)
    assert "grad_norm" not in by_step[15]["metrics"]
    # wall time taken from the asctime prefix
    assert r12["wall_time"] > 0
    print("PASS test_nanotron_parse")


def _args(**kw):
    base = dict(seq_len=8192, tokens_per_step=None, eta_total_tokens=None)
    base.update(kw)
    return Namespace(**base)


def test_derived_metrics():
    state = {"tps_ema": None}
    rec = m.parse_nanotron_line(
        "10/10/2026 08:00:52 [INFO|DP=0|PP=0|TP=0]: iteration: 12 / 40 | "
        "consumed_tokens: 51.4M | elapsed_time_per_iteration_ms: 4.32K | "
        "tokens_per_sec: 1M | tokens_per_sec_per_gpu: 166.7K | "
        "global_batch_size: 528 | lm_loss: 2.345 | lr: 3e-05 | "
        "model_tflops_per_gpu: 45.2 | hardware_tflops_per_gpu: 89.1 | grad_norm: 1.234")
    args = _args(eta_total_tokens=173_015_040)  # 40 * 528 * 8192
    m.derive_nanotron(rec, args, state)
    mm = rec["metrics"]
    assert _close(mm["ppl"], math.exp(2.345)), mm["ppl"]
    assert _close(mm["mfu"], 45.2 / 312.0), mm["mfu"]
    assert _close(mm["tokens_per_sec_ema"], 1e6)          # first EMA = raw value
    assert _close(mm["progress"], 51.4e6 / 173_015_040)
    expected_eta_h = (173_015_040 - 51.4e6) / 1e6 / 3600.0
    assert _close(mm["eta_hours"], expected_eta_h), (mm["eta_hours"], expected_eta_h)
    # second record: EMA blends; tokens_per_step inferred from gbs * seq_len
    rec2 = m.parse_nanotron_line(
        "10/10/2026 08:00:57 [INFO|DP=0|PP=0|TP=0]: iteration: 13 / 40 | "
        "consumed_tokens: 56.2M | elapsed_time_per_iteration_ms: 4.33K | "
        "tokens_per_sec: 2M | tokens_per_sec_per_gpu: 333.3K | "
        "global_batch_size: 528 | lm_loss: 2.38 | lr: 3e-05 | "
        "model_tflops_per_gpu: 45.0 | hardware_tflops_per_gpu: 88.9 | grad_norm: 1.10")
    m.derive_nanotron(rec2, _args(), state)
    mm2 = rec2["metrics"]
    assert _close(mm2["tokens_per_sec_ema"], 0.1 * 2e6 + 0.9 * 1e6)
    total = 40 * 528 * 8192
    assert _close(mm2["eta_hours"], (total - 56.2e6) / 2e6 / 3600.0)
    # live line renders without crashing
    line = m.fmt_live_line("nanotron", rec)
    assert "step 12/40" in line and "ppl" in line and "mfu" in line and "eta" in line
    print("PASS test_derived_metrics")


def test_verl_parse():
    records = []
    with open(VERL_LOG) as fh:
        for line in fh:
            rec = m.parse_verl_line(line)
            if rec is not None:
                records.append(rec)
    # 7 step:N lines (train + val at step 4, step 5 duplicated);
    # pprint dicts / tqdm / banners ignored
    assert len(records) == 7, len(records)
    steps = [r["step"] for r in records]
    assert steps == [1, 2, 3, 4, 4, 5, 5], steps
    r1 = records[0]
    mm = r1["metrics"]
    assert _close(mm["critic/kl"], 0.005)
    assert _close(mm["actor/pg_loss"], -0.012)
    assert _close(mm["actor/grad_norm"], 2.345)
    assert _close(mm["critic/rewards/mean"], 0.125)
    assert _close(mm["response_length/mean"], 512.345)
    assert _close(mm["timing_s/step"], 45.123)
    assert _close(mm["Tokens/Sec/GPU"], 1234.567)
    assert _close(mm["total_num_tokens"], 100000.0)
    # validation record carries val/ keys at its own step
    val_recs = [r for r in records if any(k.startswith("val/") for k in r["metrics"])]
    assert len(val_recs) == 1 and val_recs[0]["step"] == 4
    assert _close(val_recs[0]["metrics"]["val/test_score/avg_acc"], 0.5)
    line = m.fmt_live_line("verl", r1)
    assert "step 1" in line and "kl" in line and "reward" in line
    print("PASS test_verl_parse")


def test_auto_dispatch():
    nano = open(NANO_LOG).readlines()[2]
    verl = "step:7 - critic/kl:0.030 - Tokens/Sec/GPU:1300.000"
    junk = "this is not a metrics line at all"
    assert m.parse_any_line(nano, "auto")["source"] == "nanotron"
    assert m.parse_any_line(verl, "auto")["source"] == "verl"
    assert m.parse_any_line(junk, "auto") is None
    assert m.parse_any_line(verl, "verl")["step"] == 7
    assert m.parse_any_line(nano, "verl") is None
    print("PASS test_auto_dispatch")


def test_sink_dedup_and_tb():
    m._init_tb_backend()
    with tempfile.TemporaryDirectory() as tmp:
        sink = m.MonitorSink(tmp)
        rec = m.parse_nanotron_line(
            "10/10/2026 08:00:52 [INFO|DP=0|PP=0|TP=0]: iteration: 12 / 40 | "
            "consumed_tokens: 51.4M | tokens_per_sec: 1M | global_batch_size: 528 | "
            "lm_loss: 2.345 | lr: 3e-05 | model_tflops_per_gpu: 45.2 | "
            "hardware_tflops_per_gpu: 89.1 | grad_norm: 1.234")
        m.derive_nanotron(rec, _args(eta_total_tokens=173_015_040), {"tps_ema": None})
        new = sink.add(rec, prefix="train/")
        assert "ppl" in new and "mfu" in new and "lm_loss" in new
        again = sink.add(rec, prefix="train/")   # exact duplicate -> nothing new
        assert again == {}
        # a new record at the same step but with different keys still lands
        val_rec = {"step": 12, "total": None, "wall_time": 1.0,
                   "metrics": {"val/acc": 0.5}, "source": "verl"}
        assert sink.add(val_rec, prefix="") == {"val/acc": 0.5}
        sink.close()
        # metrics.jsonl: 2 lines (train rec + val rec), duplicate suppressed
        with open(sink.jsonl_path) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert len(lines) == 2, lines
        assert lines[0]["step"] == 12 and "train/lm_loss" not in lines[0]["metrics"]  # raw jsonl keys unprefixed
        # TB roundtrip
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(tmp)
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        assert "train/lm_loss" in tags and "train/ppl" in tags and "val/acc" in tags, tags
        ev = ea.Scalars("train/lm_loss")[0]
        assert ev.step == 12 and _close(ev.value, 2.345, rel=1e-5)
        evs = ea.Scalars("train/ppl")
        assert len(evs) == 1 and _close(evs[0].value, math.exp(2.345), rel=1e-5)
    print("PASS test_sink_dedup_and_tb (backend=%s)" % m._TB_BACKEND)


def main():
    tests = [test_parse_human_num, test_nanotron_parse, test_derived_metrics,
             test_verl_parse, test_auto_dispatch, test_sink_dedup_and_tb]
    for t in tests:
        t()
    print("ALL %d TESTS PASSED" % len(tests))


if __name__ == "__main__":
    main()
