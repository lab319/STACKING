import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# YlOrRd GnBu_r
# IMAGE
CURRENT_PALETTE = 'YlOrRd'


result_dir = r"D:\Workspace\python\myDNAm\model\stacking\bio_analysis\Top-N_data\result"
input_file = os.path.join(result_dir, "EWAS_Mapped_Brain_Diseases(IMAGE).csv")

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Please check the path : {input_file}")

df = pd.read_csv(input_file)

df['-log10(P-value)'] = -np.log10(df['p_value'] + 1e-300)

df = df.sort_values(by='R2_Multi_Tissue', ascending=False)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(12, 7), dpi=300)

scatter = sns.scatterplot(
    data=df,
    x='cpg_id',
    y='trait',
    size='-log10(P-value)',
    hue='-log10(P-value)',
    palette=CURRENT_PALETTE,
    sizes=(80, 400),
    edgecolor='#4A4A4A',
    linewidth=0.8,
    alpha=0.85
)


plt.title("The Association Between Top Predicted CpGs and Brain Disorders",
          fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
plt.xlabel("Predicted Core CpG Sites (Sorted by Model R²)", fontsize=11, fontweight='semibold', labelpad=10)
plt.ylabel("Neurological Traits / Diseases", fontsize=11, fontweight='semibold', labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9.5)
plt.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='-log10(P-value)')
plt.tight_layout()


output_svg = os.path.join(result_dir, "EWAS_Diseases_Bubble_Plot(IMAGE).svg")
plt.savefig(output_svg, bbox_inches='tight')
print(f" SVG : {output_svg}")
plt.show()