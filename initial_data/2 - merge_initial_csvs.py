import os
import pandas as pd
import logging
import argparse
from datetime import datetime

# === Setup logging ===
log_filename = f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def merge_inputs_outputs(root_dir):
    output_files = []

    for folder, _, files in os.walk(root_dir):
        if "initial_inputs.csv" in files and "initial_outputs.csv" in files:
            inputs_path = os.path.join(folder, "initial_inputs.csv")
            outputs_path = os.path.join(folder, "initial_outputs.csv")

            try:
                df_inputs = pd.read_csv(inputs_path, header=None)
                df_outputs = pd.read_csv(outputs_path, header=None)

                if len(df_inputs) != len(df_outputs):
                    logging.warning(
                        f"Row mismatch in {folder}: inputs={len(df_inputs)}, outputs={len(df_outputs)}"
                    )
                    continue

                df_merged = pd.concat([df_inputs, df_outputs], axis=1)

                folder_name = os.path.basename(os.path.normpath(folder))
                merged_path = os.path.join(folder, f"{folder_name}.csv")
                df_merged.to_csv(merged_path, index=False, header=False)

                output_files.append(merged_path)
                logging.info(f"Merged: {merged_path}")

            except Exception as e:
                logging.error(f"Failed to merge in {folder}: {e}")

        else:
            if "initial_inputs.csv" in files or "initial_outputs.csv" in files:
                logging.warning(f"Incomplete pair in {folder}. Skipped.")
    
    return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge initial_inputs.csv and initial_outputs.csv in subfolders.")
    parser.add_argument("--folder", type=str, required=True, help="Root directory to search")

    args = parser.parse_args()
    root_directory = args.folder

    logging.info(f"Starting merge in root directory: {root_directory}")
    merged_csv_files = merge_inputs_outputs(root_directory)
    logging.info(f"Completed merge. Total files created: {len(merged_csv_files)}")
