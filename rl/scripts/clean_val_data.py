#!/usr/bin/env python3
"""Filter the 18 dummy 'pad' rows out of the MATH8K validation parquet.

rl/data/math.8k/test.parquet (SimpleRL-Zoo MATH8K) has 272 rows, of which the
last 18 are padding rows (data_source == 'pad', dummy prompt "1+1"). They add
noise to validation metrics, so this script writes a cleaned copy:

    rl/data/math.8k/test.parquet       -> 272 rows (untouched)
    rl/data/math.8k/test.clean.parquet -> 254 rows (created here)

Usage (from anywhere):
    python rl/scripts/clean_val_data.py
"""
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(HERE, "..", "data", "math.8k", "test.parquet")
OUT_PATH = os.path.join(HERE, "..", "data", "math.8k", "test.clean.parquet")


def is_pad(row) -> bool:
    """A pad row is flagged by data_source=='pad' or extra_info.split=='pad'.

    In the current file all 18 pad rows have data_source == 'pad' exactly
    (extra_info.split stays 'test'), but we check both defensively.
    """
    if str(row.get("data_source", "")) == "pad":
        return True
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, str):
        # tolerate stringified dicts
        return "'split': 'pad'" in extra_info or '"split": "pad"' in extra_info
    return extra_info.get("split") == "pad"


def main():
    table = pq.read_table(TEST_PATH)
    before = table.num_rows
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    keep = [i for i in range(before) if not is_pad({name: vals[i] for name, vals in cols.items()})]

    cleaned = pa.table({name: [vals[i] for i in keep] for name, vals in cols.items()}, schema=table.schema)
    pq.write_table(cleaned, OUT_PATH)

    after = cleaned.num_rows
    print(f"before: {before} rows")
    print(f"after : {after} rows (removed {before - after} pad rows)")
    print(f"wrote : {os.path.abspath(OUT_PATH)}")
    assert before - after == 18, f"expected to remove exactly 18 pad rows, removed {before - after}"


if __name__ == "__main__":
    main()
