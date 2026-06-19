# A Stacking-Based Ensemble Method for Predicting Brain DNA Methylation by Fusing Multi-Peripheral Tissue Features
A Stacking-based ensemble learning framework for high-precision, non-invasive brain DNA methylation prediction by fusing heterogeneous multi-peripheral tissue features (Blood, Saliva, Buccal).

## Datasets & Data Availability

We acquired three datasets from the Gene Expression Omnibus (GEO) database at the National Center for Biotechnology Information (NCBI):

| Dataset Name | GEO Accession | Platform | Matched Tissue Types | Official Link |
| :--- | :--- | :--- | :--- | :--- |
| **MRC** | `GSE59685` | Illumina 450K | Peripheral Blood (BL), Cerebellum (CER), Entorhinal Cortex (EC), Prefrontal Cortex (PFC), Superior Temporal Gyrus (STG) | [GSE59685](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59685) |
| **IMAGE** | `GSE111165` | Illumina EPIC (850K) | Matched Blood (BL), Saliva (SA), Buccal (BU), and Brain (BR) from the same individual | [GSE111165](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111165) |
| **AMAZE** | `GSE214901` | Illumina EPIC (850K) | Matched Blood (BL), Saliva (SA), Buccal (BU), and Brain (BR) from the same individual | [GSE214901](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214901) |

## Demo Data for Quick Verification

To facilitate rapid testing of the training and feature fusion pipeline, we have provided a synchronized sample subset inside the `demo data/` directory. 

**Data Provenance & Biology Note:**
These demonstration files were explicitly derived and filtered from the **AMAZE dataset (GSE214901)** cohort. Specifically, we isolated exactly **4,347 Highly Variable Methylation sites (HVMs)** that maintain persistent hyper-variability across all individual human tissues.

## Running the Stacking Pipeline with Demo Data

If you want to quickly verify the core Stacking ensemble network using the provided demo files, you need to configure the file tracking pointers inside your `stacking.py` (or your main execution script). 

Open your `stacking.py` and modify the input data path configurations to look like this:

```python
# =====================================================================
# DATA PATH CONFIGURATION (Switch to Demo Data for Rapid Verification)
# =====================================================================
# Comment out the global full-matrix dataset path
# data_dir = r"D:\Workspace\python\Stacking-main\data" 

# Point directly to your local demo data workspace
demo_data_dir = r"./demo data" 

# Load the demo HVM matrix files
df_blood  = pd.read_csv(os.path.join(demo_data_dir, "Jap_BL_HVMs.txt"), sep='\t', index_col=0)
df_saliva = pd.read_csv(os.path.join(demo_data_dir, "Jap_SA_HVMs.txt"), sep='\t', index_col=0)
df_buccal = pd.read_csv(os.path.join(demo_data_dir, "Jap_BU_HVMs.txt"), sep='\t', index_col=0)
df_brain  = pd.read_csv(os.path.join(demo_data_dir, "Jap_BR_HVMs.txt"), sep='\t', index_col=0)
```

Then run the main Stacking training script via your terminal command line:
```bash
python stacking.py
