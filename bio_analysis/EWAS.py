import pandas as pd
import os

# --- 1. Explicit Path Configuration ---
result_dir = r"D:\Workspace\python\myDNAm\model\stacking\bio_analysis\Top-N_data\result"
top_n_file = os.path.join(result_dir, "AMAZE_stacking_top_500_loci.txt")
ewas_db_file = r"D:\Workspace\python\myDNAm\model\stacking\bio_analysis\Top-N_data\result\EWAS_Atlas_associations.tsv"

# --- 2. Safety Checks ---
if not os.path.exists(top_n_file):
    raise FileNotFoundError(f"Error: Target feature loci file not found: '{top_n_file}'")
if not os.path.exists(ewas_db_file):
    raise FileNotFoundError(f"Error: Downloaded TSV database file not found at path: '{ewas_db_file}'")

# --- 3. Data Loading ---
print(">>> Loading Top-N core feature loci...")
my_best_loci = pd.read_csv(top_n_file, sep='\t')

print(">>> Loading EWAS Atlas association matrix (using ISO-8859-1 encoding)...")
ewas_atlas_db = pd.read_csv(ewas_db_file, sep='\t', low_memory=False, encoding='ISO-8859-1')

# --- 4. Brain/Neurological Disease Keyword Filtering ---
print(">>> Filtering brain and neuropsychiatric disease keywords...")
brain_diseases = ['Alzheimer', 'Schizophrenia', 'Parkinson', 'Brain', 'Depression', 'Autism', 'Cognitive']
ewas_atlas_db['trait'] = ewas_atlas_db['trait'].astype(str)
ewas_brain_db = ewas_atlas_db[ewas_atlas_db['trait'].str.contains('|'.join(brain_diseases), case=False, na=False)].copy()

# --- 5. Loci ID Mapping (Inner Merge) ---
print(">>> Performing Loci ID mapping (aligning cpg_id with probe_ID)...")

# Unify to string format and strip whitespaces to ensure precise alignment
my_best_loci['cpg_id'] = my_best_loci['cpg_id'].astype(str).str.strip()
ewas_brain_db['probe_ID'] = ewas_brain_db['probe_ID'].astype(str).str.strip()

# Core merge logic
mapped_results = pd.merge(
    my_best_loci,
    ewas_brain_db,
    left_on='cpg_id',
    right_on='probe_ID',
    how='inner'
)

# --- 6. P-value Cleaning & Statistical Significance Sorting ---
print(">>> Cleaning data and sorting by statistical significance (p_value)...")
mapped_results['p_value'] = pd.to_numeric(mapped_results['p_value'], errors='coerce')
mapped_results = mapped_results.dropna(subset=['p_value'])
mapped_results = mapped_results.sort_values(by='p_value')

# --- 7. Exporting Final Mapping Report ---
output_path = os.path.join(result_dir, "EWAS_Mapped_Brain_Diseases(AMAZE).csv")
mapped_results.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n{'='*50}")
print("EWAS mapping simulation successfully completed!")
print(f"Successfully mapped {len(mapped_results)} high-performance loci to known neuropsychiatric phenotypes.")
print(f"Final analysis report saved to: '{output_path}'")
print(f"{'='*50}")

# Print the top 5 records for verification
if len(mapped_results) > 0:
    print("\nMapping Results Preview (Top 5 rows ranked by significance):")
    preview_cols = ['cpg_id', 'R2_Multi_Tissue', 'R2_Gain', 'trait', 'p_value', 'PMID']
    print(mapped_results[preview_cols].head(5))
else:
    print("\nNotice: The intersection result is empty.")
    print("This indicates that your Top-500 loci have no direct matches under the current brain disease keyword constraints.")
    print("Consider expanding the Top-N threshold (e.g., N=2000) or relaxing the disease keyword filters.")