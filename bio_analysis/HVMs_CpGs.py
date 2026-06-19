import pandas as pd
import numpy as np
import datetime
from pathlib import Path


def find_hvms(file_path, sd_threshold=0.15):
    """Loads array matrix, computes locus-specific standard deviations, and returns HVM indices."""
    print(f"Processing: {file_path.name} ...")
    # Native biological matrix: rows represent CpG loci, columns represent samples
    df = pd.read_csv(file_path, sep='\t', index_col=0).astype(np.float32)

    # Calculate standard deviation along columns (axis=1) for each locus across samples
    sd_series = df.std(axis=1)

    # Isolate highly variable methylation sites (HVMs)
    hvms = sd_series[sd_series > sd_threshold].index.tolist()
    return set(hvms)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # Smart path resolving: adaptable for both Windows and Linux environments natively
    base_dir = Path("D:/Workspace/python/myDNAm/data") if Path("D:/").exists() else Path("/root/autodl-tmp/myDNAm/data")

    datasets = {
        "IMAGE": ["Gau_BL.txt", "Gau_SA.txt", "Gau_BU.txt", "Gau_BR.txt"],
        "AMAZE": ["Jap_BL.txt", "Jap_SA.txt", "Jap_BU.txt", "Jap_BR.txt"]
    }

    results_dir = base_dir.parent / "Results" / "pre" / "stacking" / "Joint pre" / "HVMs_Analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_hvm_sets = []
    summary_report = [
        "Highly Variable Methylation Sites (HVMs) Analysis Report",
        f"Threshold: SD > {0.15}",
        f"Timestamp: {start_time}",
        "-" * 60
    ]

    # --- Step 1: Individual Cohort & Tissue Screening ---
    for ds_name, files in datasets.items():
        ds_path = base_dir / ds_name
        for f_name in files:
            full_path = ds_path / f_name

            if full_path.exists():
                hvm_set = find_hvms(full_path)
                all_hvm_sets.append(hvm_set)

                log_msg = f"Dataset: {f_name} | HVMs Count: {len(hvm_set)}"
                print(log_msg)
                summary_report.append(log_msg)

                # Export unique HVM tracking subset list per tissue file
                (results_dir / f"HVMs_List_{f_name}").write_text("\n".join(hvm_set))
            else:
                print(f"Warning: Target matrix missing at path -> {full_path}")

    # --- Step 2: Cross-Cohort Array Intersection (Target: 8 Entities) ---
    if len(all_hvm_sets) == 8:
        print("\nComputing universal intersection matrix across all 8 tissue datasets...")
        intersection_hvms = set.intersection(*all_hvm_sets)

        intersect_msg = [
            "-" * 60,
            f"Universal 8-Tissue Intersecting HVMs Count: {len(intersection_hvms)}",
            "-" * 60
        ]
        summary_report.extend(intersect_msg)
        for line in intersect_msg:
            print(line)

        # Export final intersecting biological markers
        (results_dir / "Universal_HVMs_Intersection.txt").write_text("\n".join(sorted(list(intersection_hvms))))
    else:
        print("Error: Missing expected files. Complete intersection vector cannot be generated.")

    # --- Step 3: Write Summary Manifest ---
    (results_dir / "HVMs_Summary_Report.txt").write_text("\n".join(summary_report))

    print(f"\nAnalysis completed successfully! Elapsed time: {datetime.datetime.now() - start_time}")
    print(f"Downstream report exported to: '{results_dir / 'HVMs_Summary_Report.txt'}'")