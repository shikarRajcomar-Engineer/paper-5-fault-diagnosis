"""Convert the original TEP test RData and healthy CSV to verified Parquet.

Requires: Rscript, R data.table, Python pandas/pyarrow.
The R stage loads the 798 MB compressed RData once and writes one fault
at a time to local scratch CSV. Python streams CSV chunks to Parquet.
No training data, scalers, autoencoders, or frozen V2.5 settings are changed.

Run from repo root:
    python -u 6.2_Convert_TEP_Testing_To_Parquet.py

On an interrupted Python stage, rerun with --skip-extract if all
fault_XX.csv scratch files were successfully extracted.
"""
from pathlib import Path
import argparse
import gc
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
COLUMNS = ["faultNumber", "simulationRun", "sample", *SENSORS]
SCHEMA = pa.schema(
    [pa.field("faultNumber", pa.int16()),
     pa.field("simulationRun", pa.int16()),
     pa.field("sample", pa.int16())]
    + [pa.field(s, pa.float64()) for s in SENSORS]
)
EXPECTED_RUNS = 500
EXPECTED_SAMPLES = 960
CHUNK_ROWS = 50000


def check_batch(chunk, expected_fault, observed, where):
    if list(chunk.columns) != COLUMNS:
        raise ValueError(f"{where}: unexpected or reordered columns: {list(chunk.columns)}")
    if chunk.isna().any().any():
        raise ValueError(f"{where}: missing values detected")
    for key in ("faultNumber", "simulationRun", "sample"):
        vals = chunk[key].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all() or not np.equal(vals, np.floor(vals)).all():
            raise ValueError(f"{where}: invalid noninteger {key}")
    fault = chunk["faultNumber"].to_numpy(dtype=np.int64)
    if not np.all(fault == expected_fault):
        raise ValueError(f"{where}: expected faultNumber={expected_fault}")
    run = chunk["simulationRun"].to_numpy(dtype=np.int64)
    sample = chunk["sample"].to_numpy(dtype=np.int64)
    if ((run < 1) | (run > EXPECTED_RUNS)).any():
        raise ValueError(f"{where}: run outside 1..{EXPECTED_RUNS}")
    if ((sample < 1) | (sample > EXPECTED_SAMPLES)).any():
        raise ValueError(f"{where}: sample outside 1..{EXPECTED_SAMPLES}")
    # Detect duplicates even if a run straddles consecutive CSV chunks.
    flat = (run - 1) * EXPECTED_SAMPLES + (sample - 1)
    np.add.at(observed, flat, 1)
    if observed.max() > 1:
        raise ValueError(f"{where}: duplicate (simulationRun, sample)")
    chunk["faultNumber"] = fault.astype(np.int16)
    chunk["simulationRun"] = run.astype(np.int16)
    chunk["sample"] = sample.astype(np.int16)


def write_stream(sources, target, expected_faults):
    """One Parquet file with fixed 55-column schema; verify all run grids."""
    scratch_target = target.with_suffix(".parquet.part")
    if scratch_target.exists():
        scratch_target.unlink()
    writer = None
    total_rows = 0
    try:
        for source, fault_id in sources:
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty source: {source}")
            seen = np.zeros(EXPECTED_RUNS * EXPECTED_SAMPLES, dtype=np.uint8)
            count = 0
            print(f"Converting {source.name} (fault={fault_id})", flush=True)
            for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS):
                check_batch(chunk, fault_id, seen, source.name)
                table = pa.Table.from_pandas(
                    chunk, schema=SCHEMA, preserve_index=False, safe=True
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(scratch_target), SCHEMA, compression="snappy"
                    )
                writer.write_table(table)
                count += len(chunk)
                total_rows += len(chunk)
                del table, chunk
            if count != EXPECTED_RUNS * EXPECTED_SAMPLES or not np.all(seen == 1):
                raise ValueError(
                    f"{source.name}: invalid run grid: {count} rows; "
                    f"expected {EXPECTED_RUNS * EXPECTED_SAMPLES}"
                )
            print(f"  Validated 500 runs x 960 samples ({count:,} rows)", flush=True)
            del seen
            gc.collect()
        if writer is None:
            raise ValueError("No input batches written")
        writer.close()
        writer = None
        expected_rows = EXPECTED_RUNS * EXPECTED_SAMPLES * expected_faults
        if total_rows != expected_rows:
            raise ValueError(f"Expected {expected_rows:,} rows, got {total_rows:,}")
        meta = pq.ParquetFile(scratch_target)
        if meta.metadata.num_rows != total_rows or meta.schema_arrow != SCHEMA:
            raise ValueError("Parquet metadata/schema check failed")
        # Only expose completed, verified output; never overwrite an existing file.
        if target.exists():
            raise FileExistsError(f"Existing output will not be overwritten: {target}")
        scratch_target.replace(target)
        print(f"SAVED {target} | rows={total_rows:,} | "
              f"size={target.stat().st_size / 1048576:.1f} MiB", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if scratch_target.exists():
            scratch_target.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/content/drive/MyDrive/Paper 5/TEP")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/PHD/Paper5_Datasets")
    parser.add_argument("--scratch", default="/content/tep_testing_conversion_scratch")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Reuse successfully extracted fault_XX.csv slices")
    parser.add_argument("--keep-csv", action="store_true",
                        help="Retain large local scratch CSV files")
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    scratch = Path(args.scratch)
    repo = Path(__file__).resolve().parent
    normal_csv = source / "TEP_FaultFree_Testing.csv"
    faulty_rdata = source / "TEP_Faulty_Testing.RData"
    normal_out = output / "TEP_FaultFree_Testing.parquet"
    fault_out = output / "TEP_Faulty_Testing.parquet"
    if not normal_csv.exists() or not faulty_rdata.exists():
        raise FileNotFoundError("Required healthy CSV or faulty RData not found")
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    # In case output already exists, avoid replacing files silently.
    print("=" * 84, flush=True)
    print("TEP TESTING CONVERSION — ORIGINAL DATA ONLY", flush=True)
    print("=" * 84, flush=True)
    print("Expected grid per fault: 500 runs x 960 samples", flush=True)
    if not normal_out.exists():
        write_stream([(normal_csv, 0)], normal_out, expected_faults=1)
    else:
        print(f"Normal Parquet exists; leaving untouched: {normal_out}", flush=True)

    if fault_out.exists():
        print(f"Fault Parquet exists; leaving untouched: {fault_out}", flush=True)
        print("Conversion is finished; no files overwritten.", flush=True)
        return

    if not args.skip_extract:
        r_script = repo / "6.1_Extract_TEP_Faulty_Testing.R"
        if not r_script.exists():
            raise FileNotFoundError(r_script)
        print("Starting R extraction; this stage can use several GB of RAM.",
              flush=True)
        subprocess.run(
            ["Rscript", "--vanilla", str(r_script), str(faulty_rdata), str(scratch)],
            check=True
        )
        print("R extraction completed successfully.", flush=True)
    parts = sorted(scratch.glob("fault_*.csv"))
    if not parts:
        raise FileNotFoundError("No extracted fault_XX.csv files found")
    fault_ids = []
    for part in parts:
        text = part.stem.split("_")
        if len(text) != 2 or not text[1].isdigit():
            raise ValueError(f"Unexpected slice name: {part}")
        fault_ids.append(int(text[1]))
    if len(set(fault_ids)) != len(fault_ids) or min(fault_ids) != 1 or (
        sorted(fault_ids) != list(range(1, max(fault_ids) + 1))
    ):
        raise ValueError(f"Unexpected/incomplete fault IDs: {fault_ids}")
    print(f"Found {len(parts)} fault files: {fault_ids}", flush=True)
    write_stream(list(zip(parts, fault_ids)), fault_out, expected_faults=len(parts))
    if not args.keep_csv:
        for part in parts:
            part.unlink()
        print("Removed temporary fault CSV slices after verified conversion.",
              flush=True)
    print("DONE: test Parquet files are separate from training files.", flush=True)
    print("Next: use FROZEN normal-training scaler and saved autoencoders; "
          "do not fit a scaler on testing data.", flush=True)


if __name__ == "__main__":
    main()
