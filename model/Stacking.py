import datetime
import time
import pandas as pd
import numpy as np
import warnings
import os
import platform
import optuna

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def objective(trial, X_sample, y_sample):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.9)
    }
    model_xgb = XGBRegressor(**xgb_params, n_jobs=1, verbosity=0)
    score = -cross_val_score(model_xgb, X_sample, y_sample, cv=3, scoring='neg_mean_squared_error').mean()
    return score


def process_single_locus(i, x_vals_list, y_vals, best_params):
    y = y_vals.ravel()


    raw_weights = []
    for x_source in x_vals_list:
        corr = np.abs(np.corrcoef(x_source, y)[0, 1])
        raw_weights.append(corr if not np.isnan(corr) else 1e-5)
    norm_weights = np.array(raw_weights) / (np.sum(raw_weights) + 1e-9)


    weighted_feats = [x_vals_list[j] * norm_weights[j] for j in range(len(x_vals_list))]


    final_feats = list(weighted_feats)
    if len(weighted_feats) > 1:

        for j in range(len(weighted_feats)):
            for k in range(j + 1, len(weighted_feats)):
                interaction_term = weighted_feats[j] * weighted_feats[k]
                final_feats.append(interaction_term)


    X = np.column_stack(final_feats)
    # ---------------------------

    outer_kf = KFold(n_splits=5, shuffle=False)
    locus_results = []

    for outer_train_idx, outer_test_idx in outer_kf.split(X):
        X_outer_train, y_outer_train = X[outer_train_idx], y[outer_train_idx]
        X_outer_test, y_outer_test = X[outer_test_idx], y[outer_test_idx]

        base_models = {
            'mlp': MLPRegressor(
                hidden_layer_sizes=best_params['mlp_hidden'],
                alpha=best_params['mlp_alpha'],
                max_iter=3000,
                early_stopping=True,
                tol=1e-3
            ),
            'xgb': XGBRegressor(
                n_estimators=best_params['xgb_n_estimators'],
                max_depth=best_params['xgb_max_depth'],
                learning_rate=best_params['xgb_lr'],
                subsample=best_params['xgb_subsample'],
                n_jobs=1, verbosity=0
            ),
            'ridge': RidgeCV(alphas=np.logspace(-3, 3, 7))
        }

        base_preds = np.zeros((len(X_outer_test), len(base_models)))
        mse_list = []

        for j, (name, model) in enumerate(base_models.items()):
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
            inner_preds = np.zeros_like(y_outer_train)
            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
                model.fit(X_outer_train[inner_train_idx], y_outer_train[inner_train_idx])
                inner_preds[inner_val_idx] = model.predict(X_outer_train[inner_val_idx])

            model.fit(X_outer_train, y_outer_train)
            base_preds[:, j] = model.predict(X_outer_test)
            mse_list.append(mean_squared_error(y_outer_train, inner_preds))

        inv_mse = 1 / (np.array(mse_list) + 1e-9)
        weights = inv_mse / inv_mse.sum()
        meta_input = np.dot(base_preds, weights.reshape(-1, 1))

        meta_model = BayesianRidge()
        meta_model.fit(meta_input, y_outer_test)
        locus_results.extend(meta_model.predict(meta_input).tolist())

    return locus_results


if __name__ == "__main__":
    global_start_time = time.time()

    if platform.system() == "Windows":
        data_dir = r"D:\Workspace\python\Stacking-main\data"
        save_base = r"D:\Workspace\python\Stacking-main\result\pre\stacking"
    else:
        data_dir = "/root/autodl-tmp/Stacking-main/data/"
        save_base = "/root/autodl-tmp/Stacking-main/result/pre/stacking"

    if not os.path.exists(save_base): os.makedirs(save_base)


    X_BL = pd.read_csv(os.path.join(data_dir, "Gau_BL.txt"), sep='\t', index_col=0).T.astype(np.float32)
    X_SA = pd.read_csv(os.path.join(data_dir, "Gau_SA.txt"), sep='\t', index_col=0).T.astype(np.float32)
    X_EP = pd.read_csv(os.path.join(data_dir, "Gau_EP.txt"), sep='\t', index_col=0).T.astype(np.float32)
    y_BR = pd.read_csv(os.path.join(data_dir, "Gau_BR.txt"), sep='\t', index_col=0).T.astype(np.float32)

    tasks = [
        ("BL", [X_BL]), ("SA", [X_SA]), ("EP", [X_EP]),("Full_Triple", [X_BL, X_SA, X_EP])]

    params_log = []
    num_loci = len(y_BR.columns)
    total_tasks = len(tasks)

    for idx, (task_name, x_sources) in enumerate(tasks):
        print(f"\n{'=' * 20}  {idx + 1}/{total_tasks}: {task_name}  {'=' * 20}")
        task_start_time = time.time()

        actual_sample_size = min(num_loci, 10000)
        sample_indices = np.random.choice(range(num_loci), actual_sample_size, replace=False)

        X_sample = np.vstack([np.column_stack([src.values[:, i] for src in x_sources]) for i in sample_indices])
        y_sample = np.hstack([y_BR.values[:, i] for i in sample_indices])

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_sample, y_sample), n_trials=25)

        best_params = study.best_params
        if 'mlp_hidden' not in best_params: best_params['mlp_hidden'] = (100,)
        if 'mlp_alpha' not in best_params: best_params['mlp_alpha'] = 0.0001
        best_params['task'] = task_name
        params_log.append(best_params)

        output_path = os.path.join(save_base, f"{task_name}.txt")
        batch_size = 1000

        for start_idx in range(0, num_loci, batch_size):
            end_idx = min(start_idx + batch_size, num_loci)
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(process_single_locus)(i, [src.iloc[:, i].values for src in x_sources], y_BR.iloc[:, i].values,
                                              best_params)
                for i in range(start_idx, end_idx)
            )
            batch_df = pd.DataFrame(np.array(results).T, index=y_BR.index, columns=y_BR.columns[start_idx:end_idx])
            mode, header = ('a', False) if start_idx > 0 else ('w', True)
            batch_df.T.to_csv(output_path, sep='\t', mode=mode, header=header)

            elapsed_task = time.time() - task_start_time
            speed = end_idx / elapsed_task
            remaining_loci_in_all = (num_loci - end_idx) + (total_tasks - (idx + 1)) * num_loci
            eta_sec = remaining_loci_in_all / speed
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(eta_sec))

    pd.DataFrame(params_log).to_csv(os.path.join(save_base, "Best_Parameters.csv"), index=False)