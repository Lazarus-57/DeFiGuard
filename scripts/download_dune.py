import argparse
import os
from pathlib import Path

import pandas as pd
from dune_client.client import DuneClient


REQUIRED_COLUMNS = ["tx_hash", "from", "to", "amount", "block_number", "block_time"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	mapping = {}
	if "tx_hash" not in df.columns and "hash" in df.columns:
		mapping["hash"] = "tx_hash"
	if "amount" not in df.columns and "value" in df.columns:
		mapping["value"] = "amount"
	if mapping:
		df = df.rename(columns=mapping)
	return df


def _validate_dataset(
	df: pd.DataFrame,
	*,
	expected_rows: int,
	expected_year: int,
	strict_rows: bool,
) -> None:
	missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
	if missing:
		raise ValueError(f"Dataset missing required columns: {missing}")

	if strict_rows and len(df) != expected_rows:
		raise ValueError(f"Expected exactly {expected_rows} rows, got {len(df)}")

	for col in REQUIRED_COLUMNS:
		nulls = int(df[col].isna().sum())
		if nulls > 0:
			raise ValueError(f"Column '{col}' has {nulls} null values")

	amount = pd.to_numeric(df["amount"], errors="coerce")
	if int(amount.isna().sum()) > 0:
		raise ValueError("amount contains non-numeric values")
	if int((amount < 0).sum()) > 0:
		raise ValueError("amount contains negative values")

	block_number = pd.to_numeric(df["block_number"], errors="coerce")
	if int(block_number.isna().sum()) > 0:
		raise ValueError("block_number contains non-numeric values")

	block_time = pd.to_datetime(df["block_time"], errors="coerce", utc=True)
	if int(block_time.isna().sum()) > 0:
		raise ValueError("block_time contains invalid timestamp values")

	years = set(block_time.dt.year.unique().tolist())
	if years != {expected_year}:
		raise ValueError(f"Expected only year {expected_year} in block_time, got years={sorted(years)}")

	dup_hash = int(df["tx_hash"].astype(str).duplicated().sum())
	if dup_hash > 0:
		raise ValueError(f"tx_hash has {dup_hash} duplicate values")

	print(f"Validation passed: rows={len(df)}, year={expected_year}")
	print(f"block_time range: {block_time.min()} -> {block_time.max()}")


def _build_2024_sql_contiguous(start_time: str, end_time: str, limit: int) -> str:
	return f"""
SELECT
  hash AS tx_hash,
  \"from\",
  \"to\",
  value AS amount,
  block_number,
  block_time
FROM ethereum.transactions
WHERE block_time >= timestamp '{start_time}'
  AND block_time < timestamp '{end_time}'
  AND hash IS NOT NULL
  AND \"from\" IS NOT NULL
  AND \"to\" IS NOT NULL
ORDER BY block_time ASC
LIMIT {limit}
""".strip()


def _build_2024_sql_hourly_cap(start_time: str, end_time: str, limit: int, per_hour_cap: int) -> str:
	return f"""
WITH base AS (
	SELECT
		hash AS tx_hash,
		\"from\",
		\"to\",
		value AS amount,
		block_number,
		block_time,
		date_trunc('hour', block_time) AS hour_bucket
	FROM ethereum.transactions
	WHERE block_time >= timestamp '{start_time}'
		AND block_time < timestamp '{end_time}'
		AND hash IS NOT NULL
		AND \"from\" IS NOT NULL
		AND \"to\" IS NOT NULL
), ranked AS (
	SELECT
		tx_hash,
		\"from\",
		\"to\",
		amount,
		block_number,
		block_time,
		row_number() OVER (PARTITION BY hour_bucket ORDER BY tx_hash) AS rn
	FROM base
)
SELECT
	tx_hash,
	\"from\",
	\"to\",
	amount,
	block_number,
	block_time
FROM ranked
WHERE rn <= {per_hour_cap}
ORDER BY block_time ASC
LIMIT {limit}
""".strip()


def main() -> None:
	parser = argparse.ArgumentParser(description="Download and validate Ethereum transactions for AML pipeline.")
	parser.add_argument("--mode", choices=["sql", "query"], default="sql")
	parser.add_argument("--sql-strategy", choices=["hourly_cap", "contiguous"], default="hourly_cap")
	parser.add_argument("--query-id", type=int, default=6702728)
	parser.add_argument("--start-time", default="2024-01-02 00:00:00")
	parser.add_argument("--end-time", default="2024-01-16 00:00:00")
	parser.add_argument("--limit", type=int, default=25000)
	parser.add_argument("--per-hour-cap", type=int, default=120)
	parser.add_argument("--expected-rows", type=int, default=25000)
	parser.add_argument("--expected-year", type=int, default=2024)
	parser.add_argument("--strict-rows", action="store_true")
	parser.add_argument("--output", default="eth_transfers_25k_2024.csv")
	args = parser.parse_args()

	api_key = os.getenv("DUNE_API_KEY")
	if not api_key:
		raise ValueError("Missing DUNE_API_KEY environment variable.")

	dune = DuneClient(api_key)
	if args.mode == "sql":
		if args.sql_strategy == "hourly_cap":
			sql = _build_2024_sql_hourly_cap(
				args.start_time,
				args.end_time,
				args.limit,
				args.per_hour_cap,
			)
		else:
			sql = _build_2024_sql_contiguous(args.start_time, args.end_time, args.limit)
		result = dune.run_sql(sql, is_private=True, archive_after=False, name="DeFIGuard 25k 2024 pull")
		df = pd.DataFrame(result.result.rows)
	else:
		result = dune.get_latest_result(args.query_id)
		df = pd.DataFrame(result.result.rows)

	df = _normalize_columns(df)
	_validate_dataset(
		df,
		expected_rows=args.expected_rows,
		expected_year=args.expected_year,
		strict_rows=args.strict_rows,
	)
	df = df[REQUIRED_COLUMNS].copy()

	project_root = Path(__file__).resolve().parent.parent
	out_dir = project_root / "data" / "raw"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_file = out_dir / args.output
	df.to_csv(out_file, index=False)

	print(f"Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
	main()
