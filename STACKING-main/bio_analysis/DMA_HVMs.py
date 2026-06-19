import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_hvm_volcano_analysis():
    # --- 1. Global Font & Aesthetic Configurations ---
    font_path = r'D:\Workspace\python\Stacking-main\times+simsun.ttf'
    font = FontProperties(fname=font_path, size=12)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. Data Loading ---
    input_path = r'D:\Workspace\python\Stacking-main\Differential_Methylation_Results(4347).txt'
    print(">>> Loading differential methylation matrix for the 4,347 HVM subset...")
    results_df = pd.read_csv(input_path, sep='\t')

    # --- 3. Statistical Boundary Partitioning ---
    results_df['Methylation_Status'] = 'Non-significant'
    results_df.loc[(results_df["adj.P.Val"] < 0.05) & (
                results_df["logFC"] > 0), 'Methylation_Status'] = "Significantly upregulated（AMAZE vs IMAGE）"
    results_df.loc[(results_df["adj.P.Val"] < 0.05) & (
                results_df["logFC"] < 0), 'Methylation_Status'] = "Significantly downregulated（AMAZE vs IMAGE）"

    # Print clean diagnostics summary to console
    print("\n>>> Selected HVM Differential Status Inventory:")
    for status, count in results_df['Methylation_Status'].value_counts().items():
        print(f"  {status}: {count}")

    # --- 4. Publishing-Grade Volcano Plot Visualization ---
    print("\n>>> Generating publication-quality HVM volcano plot...")
    plt.figure(figsize=(10, 7))

    status_palette = {
        "Significantly upregulated（AMAZE vs IMAGE）": "red",
        "Significantly downregulated（AMAZE vs IMAGE）": "blue",
        "Non-significant": "gray"
    }

    sns.scatterplot(
        data=results_df, x="logFC", y=-np.log10(results_df["adj.P.Val"]),
        hue="Methylation_Status", palette=status_palette,
        alpha=0.6, edgecolor='none', s=40
    )

    # Statistical significance boundaries (FDR = 0.05)
    plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    plt.text(
        x=results_df["logFC"].max() * 0.85, y=-np.log10(0.05) + 0.1,
        s="FDR = 0.05", color='black', fontproperties=font, ha='right'
    )

    # Axis labels & layout frames
    plt.xlabel("logFC(AMAZE vs IMAGE)", fontsize=12, labelpad=10, fontproperties=font)
    plt.ylabel("-log10 (FDR)", fontsize=12, labelpad=10, fontproperties=font)
    plt.title(" ", fontsize=12, pad=15, fontproperties=font)

    plt.legend(
        title="The Methylation Status", title_fontsize=12,
        loc='upper right', bbox_to_anchor=(1.0, 1),
        frameon=True, framealpha=0.5, prop=font
    ).get_title().set_fontproperties(font)

    # Strip redundant upper/right margins
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Save as high-density TIF image for publication
    plt.savefig('333Fig8.tif', dpi=400, format="tif")
    print(">>> Volcano plot successfully exported as '333Fig8.tif'.")


if __name__ == "__main__":
    plot_hvm_volcano_analysis()