import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from matplotlib.font_manager import FontProperties


def run_differential_methylation_analysis():
    # --- 1. System Paths & Environment Setup ---
    # Smart path resolving configuration
    base_dir = Path("D:\Workspace\python\Stacking-main") if Path("D:/").exists() else Path("/root/autodl-tmp/Stacking-main")

    font_path = base_dir /  "times+simsun.ttf"
    font = FontProperties(fname=str(font_path), size=12)

    # Configure global publishing-grade plot aesthetics
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. Data Loading & Array Intersection ---
    print(">>> Loading native genomic methylation matrices...")
    # Original orientation: rows present CpG loci, columns present samples
    gau_br = pd.read_csv(base_dir / "data" / "Gau_data" / "Gau_BR.txt", sep='\t', index_col=0).T
    jap_br = pd.read_csv(base_dir / "data" / "Japan-data" / "Jap_BR.txt", sep='\t', index_col=0).T

    print(f"  IMAGE (Gau_BR) Matrix Shape: {gau_br.shape}")
    print(f"  AMAZE (Jap_BR) Matrix Shape: {jap_br.shape}")

    common_sites = gau_br.columns.intersection(jap_br.columns).tolist()
    print(f"  Intercepted common CpG loci count: {len(common_sites)}")

    # Isolate overlapping feature space (Samples x Loci)
    df_gau = gau_br[common_sites].astype(np.float32)
    df_jap = jap_br[common_sites].astype(np.float32)

    # --- 3. High-Performance Vectorized Statistical Testing ---
    print("\n>>> Performing vectorized independent two-sample t-tests...")
    # Calculate means and t-statistics across the sample dimension (axis=0) concurrently
    mean_gau = df_gau.mean(axis=0)
    mean_jap = df_jap.mean(axis=0)

    # Log-Fold Change representation (Group Mean Disparity)
    log_fc = mean_jap - mean_gau

    # Compute p-values across all columns via vectorized arrays
    t_stats, p_values = stats.ttest_ind(df_jap, df_gau, axis=0, equal_var=False, nan_policy='omit')

    # --- 4. Multiple Testing Correction & Phenotypic Categorization ---
    results_df = pd.DataFrame({
        'Site': common_sites,
        'logFC': log_fc,
        'P.Value': p_values
    }).dropna(subset=['P.Value'])

    # Apply False Discovery Rate (FDR) Benjamini-Hochberg adjustment
    results_df['adj.P.Val'] = multipletests(results_df['P.Value'], method='fdr_bh')[1]

    # Assign categorical annotations based on strict statistical thresholds
    results_df['Methylation_Status'] = 'Non-significant'
    results_df.loc[(results_df['adj.P.Val'] < 0.05) & (
                results_df['logFC'] > 0), 'Methylation_Status'] = 'Significantly upregulated (AMAZE vs IMAGE)'
    results_df.loc[(results_df['adj.P.Val'] < 0.05) & (
                results_df['logFC'] < 0), 'Methylation_Status'] = 'Significantly downregulated (AMAZE vs IMAGE)'

    # Display execution summary diagnostics
    print("\n>>> Differential Methylation Status Inventory:")
    for status, count in results_df['Methylation_Status'].value_counts().items():
        print(f"  {status}: {count}")

    # --- 5. Publishing-Grade Volcano Plot Visualization ---
    print("\n>>> Generating publication-quality volcano plot...")
    plt.figure(figsize=(10, 7))

    status_palette = {
        'Significantly upregulated (AMAZE vs IMAGE)': '#FF3B30',  # Pure red
        'Significantly downregulated (AMAZE vs IMAGE)': '#007AFF',  # Pure blue
        'Non-significant': '#8E8E93'  # Neutral grey
    }

    sns.scatterplot(
        data=results_df, x='logFC', y=-np.log10(results_df['adj.P.Val']),
        hue='Methylation_Status', palette=status_palette,
        alpha=0.6, s=35, edgecolor='none'
    )

    # Statistical significance ceiling bounds (FDR = 0.05)
    plt.axhline(-np.log10(0.05), color='#000000', linestyle='--', linewidth=1)
    plt.text(
        x=results_df['logFC'].max() * 0.85, y=-np.log10(0.05) + 0.1,
        s="FDR = 0.05", color='#000000', fontproperties=font, ha='right'
    )

    # Set axis properties & typography configurations
    plt.xlabel("logFC (AMAZE vs IMAGE)", fontsize=12, fontproperties=font)
    plt.ylabel("-log10 (FDR)", fontsize=12, fontproperties=font)

    # Configure graph layout frames
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.legend(
        title="The Methylation Status", title_fontsize=12,
        loc='upper right', frameon=True, framealpha=0.5, prop=font
    ).get_title().set_fontproperties(font)

    plt.tight_layout()

    # Export publishing high-density workspace array file
    output_fig = base_dir / "Results" / "pre" / "stacking" / "Joint pre" / "HVMs_Analysis" / "Volcano_Plot_DMA.tif"
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=400, format="tif")
    print(f"Figure successfully exported to: '{output_fig}'")


if __name__ == "__main__":
    run_differential_methylation_analysis()