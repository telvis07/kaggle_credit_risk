"""

author: Telvis Calhoun

Script to build ML models

%matplotlib inline
%load_ext autoreload
%autoreload 2

"""
import os
import shutil
import scipy
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import numpy as np

from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, accuracy_score)

from preprocess import load_train_data_train_test_split, load_train_and_kaggle_submission
from utils import plot_roc_curve_interp
import matplotlib.pyplot as plt


def run_ml_model(model_config:dict, train_fn, results_output_json_fn, test_fn=None,
                 model_dir="./model/credit_risk_random_forest", npzdir="./npzdir",
                 b_show_plot=False, output_png=None,
                 kaggle_submission_output_csv="submission.csv"):
    """
    Train ML model with parameters and generate kaggle submission.

    :param model_config: Model Configuration
    :param train_fn: training file with all features merged into one file
    :param results_output_json_fn: file path to store JSON with experiment parameters and perf results
    :param test_fn: testing file containing Kaggle submission entries
    :param model_dir: Directory path to store the model
    :param npzdir: Directory path to store predict() and predict_proba() output
    :param b_show_plot: Boolean to show the plot (if running from jupyter notebook)
    :param output_png: File path to write ROC curve PNG
    :param kaggle_submission_output_csv: File path to write Kaggle submission output
    :return: Python dictionary containing model experiment performance results
    """

    for _dir in [npzdir, model_dir]:
        # model_dir="./model/credit_risk_dnn"
        if os.path.isdir(_dir):
            shutil.rmtree(_dir)

        os.makedirs(_dir)
        print("mkdir",_dir)

    X_train, X_test, y_train, y_test, scaler = load_train_data_train_test_split(train_fn=train_fn)
    X_ktest, submission_df = load_train_and_kaggle_submission(test_fn=test_fn, scaler=scaler)

    model_type = model_config['model_type']

    if model_type == "rf":
        clf = RandomForestClassifier(**model_config)
        clf.fit(X_train, y_train)
    elif model_type == "xgboost":
        if "scale_pos_weight" in model_config:
            # rescale weight to make it same as test set

            sum_wpos = np.sum(y_train == 1.0)
            sum_wneg = np.sum(y_train == 0.0)
            model_config['scale_pos_weight'] = sum_wneg / sum_wpos

            # print weight statistics
            print('weight statistics: wpos=%g, wneg=%g, ratio=%g' % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))

        clf = xgb.XGBClassifier(**model_config)
        clf.fit(X_train, y_train)
    else:
        raise RuntimeError("Invalid model_type: {}".format(model_type))

    print("Model Config", json.dumps(model_config, indent=2))

    print("Run clf.predict()")
    y_pred = clf.predict(X_test)

    print("Run clf.predict_proba()")
    y_pred_proba = clf.predict_proba(X_test)

    print(y_pred_proba.shape), print(y_test.shape)
    print(y_pred_proba[:, 1].shape)
    y_pred_proba = y_pred_proba[:, 1]

    save_predictions_npz(fn=os.path.join(npzdir, "predict.npz"), predictions=y_pred)
    save_predictions_npz(fn=os.path.join(npzdir, "predict_proba.npz"), predictions=y_pred_proba)

    # show results
    print("roc_auc_score", roc_auc_score(y_test, y_pred_proba))
    print("pauc(.50)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.5))
    print("pauc(.05)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.05))
    print("pauc(.005)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.005))
    print("recall_score", recall_score(y_test, y_pred))
    print("precision_score", precision_score(y_test, y_pred))

    stats_json = {
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "pauc@0.50": roc_auc_score(y_test, y_pred_proba, max_fpr=0.5),
        "pauc@0.05": roc_auc_score(y_test, y_pred_proba, max_fpr=0.05),
        "pauc@0.005": roc_auc_score(y_test, y_pred_proba, max_fpr=0.005),
        "recall_score": recall_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "meta": {
            "train_fn": train_fn,
            "kaggle_test_fn": test_fn,
            "kaggle_submission_output_csv": kaggle_submission_output_csv,
            "results_output_json_fn": results_output_json_fn,
            "model_dir" : model_dir,
            "b_show_plot" : b_show_plot,
            "output_png": output_png,
        },
        "model_config": model_config
    }

    # preds = y_pred_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plot_roc_curve_interp(fpr=fpr, tpr=tpr, label='default', output_png=output_png)

    if b_show_plot:
        plt.show()

    with open(results_output_json_fn, 'w') as fp:
        json.dump(stats_json, fp, indent=2)
        print("Wrote to", results_output_json_fn)

    # make kaggle predictions
    print("Making kaggle submission predictions")
    submission_df["TARGET"] = clf.predict(X_ktest)
    submission_df.to_csv(kaggle_submission_output_csv, index=False)
    print("Wrote to", kaggle_submission_output_csv)
    print("Kaggle Prediction labels:", submission_df["TARGET"].value_counts())

    return stats_json


def save_predictions_npz(fn, predictions):
    """
    Save the numpy array to compressed numpy array file

    :param fn: path to output file
    :param predictions: numpy array containing model predictions
    :return: None
    """
    np.savez_compressed(fn, predictions=predictions)
    print("[save_predictions_npz] Wrote to", fn)
