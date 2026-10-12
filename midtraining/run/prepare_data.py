#!/usr/bin/env python3
"""prepare_data.py — build nanotron-ready JSONL files for OctoThinker mid-training.

Reads the raw dataset uploads under --datasets-root (parquet / jsonl / json),
sniffs the actual schema of whatever files are present, formats each dataset
according to the OctoThinker paper (arXiv:2506.20512, Section 3), and writes
one UTF-8 jsonl per dataset with a single {"text": ...} column:

    <output-root>/<DatasetName>/data.jsonl

That single jsonl is the canonical prepared artifact. Before tokenizing, run
shard_data.py to replace it with data.shard-*.jsonl — datatrove shards work by
file, so tokenizing one big jsonl would use a single task no matter how many
workers are requested (see shard_data.py's docstring).

Per-dataset behaviour (paper Section 3):
  * Web corpora (FineMath-4plus, MegaMath-Web-Pro, MegaMath-Web-Pro-Max):
    single text column; documents shorter than WEB_MIN_CHARS (50) are dropped.
  * MegaMath-QA: question + "\n" + answer; rows with a question shorter than
    QA_MIN_QUESTION_CHARS (100) are dropped. The official MegaMath-QA upload
    pre-renders each pair as "Question: ... Answer: ..." in a single text
    column; that layout is recognised too and the text is kept as-is.
  * OpenR1-Math-220K: problem + "\n" + thinking wrapped in <think></think>
    (kept as-is if the field already carries the tags).
  * Instruction data (TULU3-sft persona-IF subset, UltraChat-220K, WildChat-1M):
    multi-turn conversations formatted as "User:{content}\nAssistant:{content}"
    per turn; WildChat keeps GPT-4-derived conversations only.

The uploads may be incomplete or still in progress (files can be truncated);
unreadable files are skipped with a warning, and a dataset fails loudly with a
schema report only when nothing usable could be read.

At the end a summary.json with per-dataset row/token counts is written to
<output-root>/summary.json, and the exit code is non-zero if any requested
dataset failed.

Examples:
    python prepare_data.py                                   # all 8 datasets
    python prepare_data.py --datasets MegaMathQA,OpenR1Math220K
    python prepare_data.py --max-samples-per-dataset 2000    # smoke-test subset
"""

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_ROOT = Path("/hdd2/xiaolirui/octoThinker-Datasets")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "midtraining" / "data" / "jsonl"

# --- filter thresholds (see module docstring) -------------------------------
WEB_MIN_CHARS = 50
QA_MIN_QUESTION_CHARS = 100
INSTRUCTION_MIN_CHARS = 50

# --- candidate column names, in priority order ------------------------------
WEB_TEXT_COLS = ["text", "content", "contents", "body", "document"]
QA_QUESTION_COLS = ["question", "problem", "prompt", "query", "instruction", "question_title"]
QA_ANSWER_COLS = [
    "answer",
    "solution",
    "response",
    "output",
    "completion",
    "explained_answer",
    "ground_truth",
    "short_answer",
]
OPENR1_PROBLEM_COLS = ["problem", "question", "prompt"]
OPENR1_THINKING_COLS = [
    "solution",
    "generation",
    "generations",
    "thinking",
    "reasoning",
    "response",
    "generated_solution",
]
MESSAGES_COLS = ["messages", "conversation", "conversations", "turns"]
SOURCE_COLS = ["source", "dataset", "origin"]
MODEL_COLS = ["model", "model_name", "modelName"]

# role normalisation: anything not user/assistant is dropped from the transcript
USER_ROLES = {"user", "human", "prompter"}
ASSISTANT_ROLES = {"assistant", "gpt", "agent", "model", "bot"}


class SchemaError(RuntimeError):
    """Raised when no usable schema could be matched for a dataset."""


def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _as_text(value):
    """Best-effort conversion of a cell to a clean string (None on failure)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("text", "content", "value"):
            if k in value:
                return _as_text(value[k])
        return None
    if isinstance(value, (list, tuple)):
        # e.g. OpenR1 'generations': take the first non-empty entry
        for item in value:
            t = _as_text(item)
            if t:
                return t
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
    return str(value)


def _clean(text):
    if not text:
        return None
    text = text.strip()
    return text if text else None


# ---------------------------------------------------------------------------
# row formatters: row(dict, columns already sniffed) -> text or None (skip)
# ---------------------------------------------------------------------------

def make_web_formatter(text_col):
    def fmt(row):
        text = _clean(_as_text(row.get(text_col)))
        if text is None or len(text) < WEB_MIN_CHARS:
            return None
        return text

    return fmt


def make_qa_formatter(q_col, a_col):
    def fmt(row):
        q = _clean(_as_text(row.get(q_col)))
        a = _clean(_as_text(row.get(a_col)))
        if q is None or len(q) < QA_MIN_QUESTION_CHARS or a is None:
            return None
        return f"{q}\n{a}"

    return fmt


# MegaMath-QA ships its pairs pre-rendered into a single text column:
# "Question: ...\n\nAnswer: ..." (plus domain/url metadata columns).
_QA_TEXT_ANSWER_MARKER = re.compile(r"^\s*Answer\s*:", re.M)


def make_qa_text_formatter(text_col):
    """Format a QA dataset stored as one 'Question: ... Answer: ...' text column.

    The text is kept as-is; only the question-length filter is applied,
    measured on the part before the 'Answer:' marker.
    """

    def fmt(row):
        text = _clean(_as_text(row.get(text_col)))
        if text is None:
            return None
        m = _QA_TEXT_ANSWER_MARKER.search(text)
        question = text[: m.start()] if m else text
        if len(question) < QA_MIN_QUESTION_CHARS:
            return None
        return text

    return fmt


def make_openr1_formatter(q_col, t_col):
    def fmt(row):
        q = _clean(_as_text(row.get(q_col)))
        t = _clean(_as_text(row.get(t_col)))
        if q is None or len(q) < QA_MIN_QUESTION_CHARS or t is None:
            return None
        thinking = t if "<think>" in t else f"<think>\n{t}\n</think>"
        return f"{q}\n{thinking}"

    return fmt


def _format_messages(messages):
    """['messages']-style list[{'role','content'}] -> 'User:..\nAssistant:..' text."""
    if not isinstance(messages, (list, tuple)):
        return None
    parts = []
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).lower()
        content = _clean(_as_text(turn.get("content")))
        if content is None:
            continue
        if role in USER_ROLES:
            parts.append(f"User:{content}")
        elif role in ASSISTANT_ROLES:
            parts.append(f"Assistant:{content}")
        # system / tool / other roles are skipped (paper format is User/Assistant)
    text = "\n".join(parts)
    if len(text) < INSTRUCTION_MIN_CHARS or "Assistant:" not in text:
        return None
    return text


def make_messages_formatter(m_col, source_col=None):
    persona_only = None  # decided lazily from the first rows that have a source

    def fmt(row):
        nonlocal persona_only
        text = _format_messages(row.get(m_col))
        if text is None:
            return None
        if source_col is not None:
            src = _as_text(row.get(source_col)) or ""
            if persona_only is None and "persona" in src.lower():
                persona_only = True  # dataset carries non-persona rows -> filter them out
            if persona_only and "persona" not in src.lower():
                return None
        return text

    return fmt


def make_wildchat_formatter(m_col, model_col):
    def fmt(row):
        model = (_as_text(row.get(model_col)) or "").lower()
        # paper (Section 3): keep only GPT-4-derived WildChat conversations
        if "gpt-4" not in model and "gpt4" not in model:
            return None
        return _format_messages(row.get(m_col))

    return fmt


# ---------------------------------------------------------------------------
# dataset registry: canonical name -> (source dir candidates, kind)
# ---------------------------------------------------------------------------

DATASETS = {
    "FineMath4Plus": {
        "dirs": ["FineMath-4plus", "FineMath-4+"],
        "kind": "web",
    },
    "MegaMathWebPro": {
        "dirs": ["MegaMath-Web-Pro"],
        "kind": "web",
    },
    # "MegaMathWebProMax": {
    #     "dirs": ["MegaMath-Web-Pro-Max", "MegaMath-Web-Pro-Max-100B"],
    #     "kind": "web",
    # },
    "MegaMathQA": {
        "dirs": ["MegaMath-QA", "MegaMath_QA"],
        "kind": "qa",
    },
    "OpenR1Math220K": {
        "dirs": ["OpenR1-Math-220K", "OpenR1-Math-220k"],
        "kind": "openr1",
    },
    "TULU3SFT": {
        "dirs": ["TULU3-sft", "tulu3-sft", "TULU3-SFT"],
        "kind": "messages",
    },
    "UltraChat220K": {
        "dirs": ["UltraChat-220K", "UltraChat-220k"],
        "kind": "messages",
    },
    "WildChat1M": {
        "dirs": ["WildChat-1M", "WildChat_1M"],
        "kind": "wildchat",
    },
}

DATA_EXTENSIONS = (".parquet", ".jsonl", ".jsonl.gz", ".json.gz", ".json")


# ---------------------------------------------------------------------------
# file reading helpers
# ---------------------------------------------------------------------------

def find_data_files(src_dir: Path):
    files = []
    for ext in DATA_EXTENSIONS:
        files.extend(sorted(src_dir.rglob(f"*{ext}")))
    # de-dup, keep a stable order, skip hidden files
    seen, out = set(), []
    for f in files:
        if any(part.startswith(".") for part in f.parts):
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def iter_rows(path: Path):
    """Yield row dicts from a parquet / jsonl / json file. Raises on unreadable file."""
    name = path.name.lower()
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif name.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("data", [])
        yield from data
    else:
        raise ValueError(f"unsupported file type: {path}")


def sniff_formatter(kind, columns):
    """Build a row formatter from the *actual* columns present; raise SchemaError."""
    colset = set(columns)
    if kind == "web":
        col = _pick(colset, WEB_TEXT_COLS)
        if col:
            return make_web_formatter(col), {kind: col}
    elif kind == "qa":
        q = _pick(colset, QA_QUESTION_COLS)
        a = _pick(colset, QA_ANSWER_COLS)
        if q and a:
            return make_qa_formatter(q, a), {"question": q, "answer": a}
        # QA pairs pre-rendered into a single text column (e.g. MegaMath-QA)
        col = _pick(colset, WEB_TEXT_COLS)
        if col:
            return make_qa_text_formatter(col), {"question_answer_text": col}
    elif kind == "openr1":
        q = _pick(colset, OPENR1_PROBLEM_COLS)
        t = _pick(colset, OPENR1_THINKING_COLS)
        if q and t:
            return make_openr1_formatter(q, t), {"problem": q, "thinking": t}
    elif kind == "messages":
        m = _pick(colset, MESSAGES_COLS)
        if m:
            src = _pick(colset, SOURCE_COLS)
            return make_messages_formatter(m, src), {"messages": m, "source": src}
    elif kind == "wildchat":
        m = _pick(colset, MESSAGES_COLS)
        model = _pick(colset, MODEL_COLS)
        if m and model:
            return make_wildchat_formatter(m, model), {"conversation": m, "model": model}
    raise SchemaError(
        f"could not match a {kind!r} schema. columns present: {sorted(columns)}; "
        f"tried candidates: web={WEB_TEXT_COLS}, qa=({QA_QUESTION_COLS} + {QA_ANSWER_COLS}), "
        f"openr1=({OPENR1_PROBLEM_COLS} + {OPENR1_THINKING_COLS}), "
        f"messages={MESSAGES_COLS}, model={MODEL_COLS}"
    )


def build_dataset(name, spec, src_dir: Path, out_file: Path, max_samples):
    files = find_data_files(src_dir)
    if not files:
        raise SchemaError(f"no data files ({DATA_EXTENSIONS}) found under {src_dir}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "source_dir": str(src_dir),
        "output_file": str(out_file),
        "files_read": [],
        "files_failed": [],
        "rows_read": 0,
        "rows_written": 0,
        "rows_filtered": 0,
        "chars": 0,
        "est_tokens": 0,
        "schema": None,
    }

    formatter = None
    written = 0
    with open(out_file, "w", encoding="utf-8") as out:
        for f in files:
            if max_samples is not None and written >= max_samples:
                break
            try:
                file_rows = 0
                for row in iter_rows(f):
                    stats["rows_read"] += 1
                    if formatter is None:
                        formatter, schema = sniff_formatter(spec["kind"], row.keys())
                        stats["schema"] = schema
                        print(f"    [{name}] matched schema {schema} from {f.name}")
                    try:
                        text = formatter(row)
                    except Exception:  # noqa: BLE001 — one bad row must not kill a dataset
                        text = None
                    if text is None:
                        stats["rows_filtered"] += 1
                    else:
                        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        stats["chars"] += len(text)
                        written += 1
                        stats["rows_written"] = written
                    file_rows += 1
                    if max_samples is not None and written >= max_samples:
                        break
                stats["files_read"].append(str(f))
            except Exception as e:  # noqa: BLE001 — truncated/corrupt uploads etc.
                stats["files_failed"].append({"file": str(f), "error": str(e)})
                print(f"    [{name}] WARNING: skipping unreadable file {f}: {e}", file=sys.stderr)
                continue

    if formatter is None:
        # nothing readable: build the loudest possible schema report
        report = [f"no readable rows in any of the {len(files)} candidate file(s)."]
        for f in files[:5]:
            try:
                if f.name.lower().endswith(".parquet"):
                    import pyarrow.parquet as pq

                    report.append(f"  {f.name}: columns={pq.ParquetFile(f).schema_arrow.names}")
                else:
                    with open(f, encoding="utf-8", errors="replace") as fh:
                        first = fh.readline()
                    report.append(f"  {f.name}: first line = {first[:200]!r}")
            except Exception as e:  # noqa: BLE001
                report.append(f"  {f.name}: unreadable ({e})")
        raise SchemaError("\n".join(report))

    stats["est_tokens"] = stats["chars"] // 4  # cheap chars/4 approximation
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT,
                    help="root directory of the raw dataset uploads")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help="where to write <DatasetName>/data.jsonl")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS.keys()),
                    help="comma-separated canonical dataset names (default: all 8)")
    ap.add_argument("--max-samples-per-dataset", type=int, default=None,
                    help="cap rows written per dataset (smoke tests)")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in requested if d not in DATASETS]
    if unknown:
        print(f"ERROR: unknown dataset(s) {unknown}; valid: {list(DATASETS)}", file=sys.stderr)
        sys.exit(2)

    args.output_root.mkdir(parents=True, exist_ok=True)
    results, failures = {}, []
    for name in requested:
        spec = DATASETS[name]
        src_dir = next((args.datasets_root / d for d in spec["dirs"] if (args.datasets_root / d).is_dir()), None)
        print(f"[{name}] processing ...")
        t0 = time.time()
        if src_dir is None:
            msg = f"source directory not found under {args.datasets_root} (tried {spec['dirs']})"
            print(f"    ERROR: {msg}", file=sys.stderr)
            results[name] = {"status": "error", "error": msg}
            failures.append(name)
            continue
        try:
            stats = build_dataset(name, spec, src_dir, args.output_root / name / "data.jsonl",
                                  args.max_samples_per_dataset)
            stats["status"] = "ok"
            stats["elapsed_sec"] = round(time.time() - t0, 1)
            results[name] = stats
            print(f"    OK: {stats['rows_written']} rows written "
                  f"({stats['rows_filtered']} filtered, ~{stats['est_tokens']:,} tokens, "
                  f"{stats['elapsed_sec']}s)")
        except SchemaError as e:
            print(f"    SCHEMA ERROR: {e}", file=sys.stderr)
            results[name] = {"status": "error", "error": str(e), "source_dir": str(src_dir)}
            failures.append(name)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results[name] = {"status": "error", "error": f"{type(e).__name__}: {e}", "source_dir": str(src_dir)}
            failures.append(name)

    summary_path = args.output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"datasets_root": str(args.datasets_root), "results": results}, fh, indent=2, ensure_ascii=False)
    print(f"\nsummary written to {summary_path}")

    if failures:
        print(f"FAILED datasets: {failures} (see summary.json for schema reports)", file=sys.stderr)
        sys.exit(1)
    print("all requested datasets prepared successfully.")


if __name__ == "__main__":
    main()
