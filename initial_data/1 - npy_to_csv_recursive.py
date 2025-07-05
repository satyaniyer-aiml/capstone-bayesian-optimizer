import os
import numpy as np
import argparse

def convert_all_npy_to_csv(root_dir):
    converted_files = []

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(folder, file)
                csv_path = os.path.splitext(npy_path)[0] + ".csv"

                try:
                    data = np.load(npy_path)
                    np.savetxt(csv_path, data, delimiter=",", fmt="%.8f")
                    converted_files.append(csv_path)
                    print(f"? Converted: {npy_path} ? {csv_path}")
                except Exception as e:
                    print(f"? Failed to convert {npy_path}: {e}")

    print(f"\n? Finished. {len(converted_files)} files converted.")
    return converted_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively convert .npy files to .csv.")
    parser.add_argument("--folder", type=str, required=True, help="Root folder to scan for .npy files")

    args = parser.parse_args()
    convert_all_npy_to_csv(args.folder)
