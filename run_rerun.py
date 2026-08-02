import subprocess
import sys
import os

commands = [
    # Step 1
    ["python", "scripts/100k/graph_pipeline_100k_peel.py"],
    ["python", "scripts/model_prep.py", "--transactions", "data/processed/100k/peel_only/augmented_transactions.csv", "--labels", "data/processed/100k/peel_only/transaction_labels.csv", "--output", "data/processed/100k/peel_only/modeling_dataset.csv", "--summary", "data/processed/100k/peel_only/modeling_split_summary.csv", "--metadata", "data/processed/100k/peel_only/model_prep_metadata.json"],
    ["python", "scripts/100k/phase1_model_selection.py", "--data", "data/processed/100k/peel_only/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step2_xgb_gnn_comparison.py", "--data", "data/processed/100k/peel_only/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step3_layer3_comparison.py", "--data", "data/processed/100k/peel_only/modeling_dataset.csv"],

    # Step 2
    ["python", "scripts/100k/graph_pipeline_100k_peel_smurf.py"],
    ["python", "scripts/model_prep.py", "--transactions", "data/processed/100k/peel_smurf/augmented_transactions.csv", "--labels", "data/processed/100k/peel_smurf/transaction_labels.csv", "--output", "data/processed/100k/peel_smurf/modeling_dataset.csv", "--summary", "data/processed/100k/peel_smurf/modeling_split_summary.csv", "--metadata", "data/processed/100k/peel_smurf/model_prep_metadata.json"],
    ["python", "scripts/100k/phase1_model_selection.py", "--data", "data/processed/100k/peel_smurf/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step2_xgb_gnn_comparison.py", "--data", "data/processed/100k/peel_smurf/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step3_layer3_comparison.py", "--data", "data/processed/100k/peel_smurf/modeling_dataset.csv", "--include-gnn-embeddings"], # We know GNN wins here so include it

    # Step 3
    ["python", "scripts/100k/graph_pipeline_100k_all.py"],
    ["python", "scripts/model_prep.py", "--transactions", "data/processed/100k/all_patterns/augmented_transactions.csv", "--labels", "data/processed/100k/all_patterns/transaction_labels.csv", "--output", "data/processed/100k/all_patterns/modeling_dataset.csv", "--summary", "data/processed/100k/all_patterns/modeling_split_summary.csv", "--metadata", "data/processed/100k/all_patterns/model_prep_metadata.json"],
    ["python", "scripts/100k/phase1_model_selection.py", "--data", "data/processed/100k/all_patterns/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step2_xgb_gnn_comparison.py", "--data", "data/processed/100k/all_patterns/modeling_dataset.csv"],
    ["python", "scripts/100k/phase1_step3_layer3_comparison.py", "--data", "data/processed/100k/all_patterns/modeling_dataset.csv"],
]

# Write output to a central log
with open("rerun_log.txt", "w") as f:
    for i, cmd in enumerate(commands):
        print(f"Running command {i+1}/{len(commands)}: {' '.join(cmd)}")
        f.write(f"\n{'='*80}\nCommand: {' '.join(cmd)}\n{'='*80}\n")
        f.flush()
        
        # Use the local venv python
        if cmd[0] == "python":
            cmd[0] = r".\.venv\Scripts\python.exe"
            
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")
            sys.exit(1)

print("All commands completed successfully.")
