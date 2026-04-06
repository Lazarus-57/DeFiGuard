import argparse
import os
from pathlib import Path

import pandas as pd
from dune_client.client import DuneClient


def main() -> None:
	parser = argparse.ArgumentParser(description="Download latest result set from a Dune query.")
	parser.add_argument("--query-id", type=int, default=6702728)
	parser.add_argument("--output", default="eth_transfers_2024_01_01_1hr.csv")
	args = parser.parse_args()

	api_key = os.getenv("DUNE_API_KEY")
	if not api_key:
		raise ValueError("Missing DUNE_API_KEY environment variable.")

	dune = DuneClient(api_key)
	result = dune.get_latest_result(args.query_id)
	df = pd.DataFrame(result.result.rows)

	project_root = Path(__file__).resolve().parent.parent
	out_dir = project_root / "data" / "raw"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_file = out_dir / args.output
	df.to_csv(out_file, index=False)

	print(f"Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
	main()
