import pandas as pd
from sklearn.metrics import (mean_squared_error, mean_absolute_error, roc_auc_score,
                             matthews_corrcoef, accuracy_score, confusion_matrix)
import numpy as np
import datetime
import os
import platform



def binarize_results(results, threshold=0.5):
    return (np.array(results) >= threshold).astype(int)

def calculate_comprehensive_metrics(y_true_df, y_pred_df, output_file):

    yt_np = y_true_df.values
    yp_np = y_pred_df.values

    print("Step 1: Calculating Regression Metrics (Row-wise Mean)...")
    row_mae = np.abs(yt_np - yp_np).mean(axis=1)
    row_mse = ((yt_np - yp_np) ** 2).mean(axis=1)
    row_rmse = np.sqrt(row_mse)
    row_r2 = y_true_df.corrwith(y_pred_df, axis=1, method='pearson').apply(np.square)
    final_mae, final_mse, final_rmse, final_r2 = row_mae.mean(), row_mse.mean(), row_rmse.mean(), row_r2.mean()

    print("Step 2: Calculating Classification Metrics (Global Flatten)...")
    y_true_bin = binarize_results(yt_np.flatten())
    y_pred_bin = binarize_results(yp_np.flatten())
    if len(np.unique(y_true_bin)) < 2 or len(np.unique(y_pred_bin)) < 2:
        SE, SP, ACC, MCC, AUC = 0, 0, 0, 0, 0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        SE = tp / (tp + fn) if (tp + fn) > 0 else 0
        SP = tn / (tn + fp) if (tn + fp) > 0 else 0
        ACC = accuracy_score(y_true_bin, y_pred_bin)
        MCC = matthews_corrcoef(y_true_bin, y_pred_bin)
        try:
            AUC = roc_auc_score(y_true_bin, y_pred_bin)
        except:
            AUC = 0.5
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    report_lines = [
        f"\n{'=' * 25} Comprehensive Evaluation ({now}) {'=' * 25}",
        f"[Regression (Average of all rows)]",
        f"  Mean_MAE  : {final_mae:.6f}",
        f"  Mean_MSE  : {final_mse:.6f}",
        f"  Mean_RMSE : {final_rmse:.6f}",
        f"  Mean_R2   : {final_r2:.6f}",
        f"[Classification (Global Flattened)]",
        f"  Sensitivity (SE) : {SE:.6f}",
        f"  Specificity (SP) : {SP:.6f}",
        f"  Accuracy    (ACC): {ACC:.6f}",
        f"  Matthews    (MCC): {MCC:.6f}",
        f"  ROC_AUC     (AUC): {AUC:.6f}",
        f"{'=' * 75}\n"
    ]

    with open(output_file, 'a', encoding='utf-8') as f:
        for line in report_lines:
            print(line)
            f.write(line + "\n")

    return True


if __name__ == "__main__":
    if platform.system() == "Windows":
        base_dir = r"D:\Workspace\python\Stacking-main"
        print(">>> : Windows")
    else:
        base_dir = "/root/autodl-tmp/Stacking-main"
        print(">>> : Linux")

    res_path = os.path.join(base_dir, "Results", "pre", "stacking", "Joint pre", "Joint_pre.txt")
    true_path = os.path.join(base_dir, "data", "Joint_HVMs_4347", "BR_Joint.txt")
    log_path = os.path.join(base_dir, "Results", "metrics", "stacking", "Joint_pre_Log.txt")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(res_path) or not os.path.exists(true_path):
        print(f"Error: Invalid file paths!\nPrediction file: {res_path}\nGround truth file: {true_path}")
    else:
        pre_Stacking = pd.read_csv(res_path, sep='\t', index_col=0).T.astype(np.float32)
        BR = pd.read_csv(true_path, sep='\t', index_col=0).T.astype(np.float32)

        common_loci = BR.columns.intersection(pre_Stacking.columns)
        print(f"Matched common loci count: {len(common_loci)}")

        if len(common_loci) == 0:
            print("Error: No matching loci found. Please check data format!")
        else:
            BR_aligned = BR[common_loci]
            pre_aligned = pre_Stacking[common_loci]

            calculate_comprehensive_metrics(BR_aligned, pre_aligned, log_path)