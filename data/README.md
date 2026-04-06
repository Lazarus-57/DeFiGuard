# Data Directory

This repository is private-ready and does not track large dataset artifacts.

## What is excluded

The project `.gitignore` excludes common data outputs, including CSV/JSON/parquet artifacts under data folders and generated report files.

## How to get data

1. Set your Dune API key in the shell environment.

```powershell
$env:DUNE_API_KEY="<your_key>"
```

2. Download raw data via script.

```powershell
python scripts/download_dune.py --query-id 6702728 --output eth_transfers_2024_01_01_1hr.csv
```

3. Build processed datasets.

```powershell
python scripts/graph_pipeline.py
python scripts/model_prep.py
```

## Expected local structure

```text
data/
├── raw/
│   └── <downloaded raw files>
└── processed/
    └── <generated modeling datasets>
```

## Notes

- Do not commit generated data artifacts.
- Reproducibility comes from scripts + config, not committed data files.
