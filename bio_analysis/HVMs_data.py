import pandas as pd
import numpy as np
import os
import platform


def get_joint_data_and_individual_hvms():
    # --- 1. Path & Cohort Configurations ---
    base_dir = r"D:\Workspace\python\Stacking-main" if platform.system() == "Windows" else "/root/autodl-tmp/Stacking-main"
    data_dir = os.path.join(base_dir, "data")
    hvm_file = os.path.join(base_dir, "Results", "pre", "stacking", "Joint pre", "HVMs_Analysis",
                            "Universal_HVMs_Intersection.txt")
    save_dir = os.path.join(base_dir, "data", "Joint_HVMs_4347")

    os.makedirs(save_dir, exist_ok=True)

    # --- 2. Load Universal HVM Loci (Target: m = 4347) ---
    if not os.path.exists(hvm_file):
        raise FileNotFoundError(f"Error: Intersection loci file not found at '{hvm_file}'")

    print(">>> Loading intersection loci subset...")
    with open(hvm_file, 'r') as f:
        common_hvms = [line.strip() for line in f if line.strip()]

    tissues = ["BL", "SA", "BU", "BR"]
    cohorts = [
        {"name": "IMAGE", "prefix": "Gau"},
        {"name": "AMAZE", "prefix": "Jap"}
    ]

    # --- 3. High-Dimensional Matrix Filtering & Cohort Merging ---
    for tissue in tissues:
        print(f"\n[Tissue: {tissue}] Processing data arrays...")
        dfs_to_combine = []

        for ch in cohorts:
            file_name = f"{ch['prefix']}_{tissue}.txt"
            file_path = os.path.join(data_dir, ch['name'], file_name)

            if not os.path.exists(file_path):
                print(f"  Warning: Target file missing -> {file_path}")
                continue

            print(f"  Filtering: {ch['name']} -> {file_name}")
            df = pd.read_csv(file_path, sep='\t', index_col=0)

            # Step A: Filter rows matching the 4,347 target HVMs (Loci x Samples)
            filtered_df = df.loc[common_hvms]

            # Save individual cohort-specific HVM subset
            indiv_name = f"{ch['prefix']}_{tissue}_HVMs.txt"
            filtered_df.to_csv(os.path.join(save_dir, indiv_name), sep='\t')
            print(f"    -> Individual HVM file saved: {indiv_name} | Dim: {filtered_df.shape}")

            # Step B: Transpose to (Samples x Loci) for subsequent vertical concatenation
            dfs_to_combine.append(filtered_df.T)

        # --- 4. Cross-Cohort Concatenation & Joint Matrix Generation ---
        if len(dfs_to_combine) == 2:
            # Concatenate samples across cohorts row-wise (e.g., 21 + 19 = 40 samples)
            joint_samples_first = pd.concat(dfs_to_combine, axis=0)

            # Transpose back to native biological representation (Loci x Samples)
            joint_final = joint_samples_first.T
            print(f"  Integration Complete! Joint Matrix Shape (Loci, Samples): {joint_final.shape}")

            joint_name = f"{tissue}_Joint.txt"
            joint_final.to_csv(os.path.join(save_dir, joint_name), sep='\t')
            print(f"  -> Joint output file saved: {joint_name}")
        else:
            print(f"  Notice: Incomplete tissue sets for {tissue}. Skipping joint dataset generation.")

    print(f"\nPipeline successfully executed!\nAll downstream files saved to: '{save_dir}'")


if __name__ == "__main__":
    get_joint_data_and_individual_hvms()