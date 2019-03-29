"""
author: Telvis Calhoun

Script to build models with various parameters and generate kaggle submission CSV
"""

import os
import datetime as dt
import json
import argparse
import shutil


from merge_feature_files import feature_files
from ml import run_ml_model
from dnn import api_dnn


MODEL_CONFIGS = {
    'rf': {
        'model_type': 'rf',
        'n_estimators': 200,
        'max_depth': 12,
        'random_state': 1234,
        'class_weight': "balanced",
        'n_jobs': 4,
    },
    'xgboost': {
        'model_type': 'xgboost',
        'n_estimators': 200,
        'max_depth': 12,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic',
        'n_jobs': 4,
    },
    "xgboost_kaggle_higgs": {
        'model_type': 'xgboost',
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'auc',
        'silent': 1,
        'objective': 'binary:logistic',
        'n_jobs': 16,
        'scale_pos_weight': None
    },
    "dnn__default": {
        "n_inputs": None,
        "n_hidden1": 500,
        "n_hidden2": 500,
        "n_outputs": 2,
        "learning_rate": 0.01,
        "n_epoch": 15,
        "batch_size": 500,
        "nn_type": "l1_regularizer",
        "do_gradient_clipping": True,
        "do_metric_type": "precision"
        # "activation": "elu",
        # "init_type": "he"
    },
    "dnn__weighted_cross_entropy_with_logits":{
            "__note": "I couldn't get this to work. FIXME",
            "n_inputs": None,
            "n_hidden1": 500,
            "n_hidden2": 500,
            "n_outputs": 2,
            "learning_rate": 0.01,
            "n_epoch": 15,
            "batch_size": 500,
            "nn_type": "weighted_cross_entropy_with_logits",
            "do_gradient_clipping": True,
            "do_metric_type": "precision"
            # "activation": "elu",
            # "init_type": "he"
    }
}
MODEL_TYPES = tuple(MODEL_CONFIGS.keys())


def main(model_type, outputdir=None, feature_set_dir="./merged_datasets", force=False, run_all_feature_sets=False):
    """
    Wrapper code to run ML or DNN modeling functions

    :param model_type: model key defined in MODEL_CONFIGS
    :param outputdir: directory to store output files from the model functions.
    :param feature_set_dir: directory to read train, test features files containing all features merged into 1 file.
    :param force: Overwrite an existing experiment if found, otherwise it will skip already completed experiments.
    :param run_all_feature_sets: Run for all experimental feature sets, otherwise just run the 'best' feature set.

    :return: None
    """
    if outputdir is None:
        outputdir = "ml.experiment.{}".format(dt.date.today().strftime("%Y%m%d-%H%M%S"))

    print("outputdir", outputdir)

    if run_all_feature_sets:
        feat_set_list = list(feature_files.keys())
    else:
        feat_set_list = ['main_plus_all']

    for feat_set in feat_set_list:
        results_json = os.path.join(outputdir, feat_set, "results.json")
        results_png = os.path.join(outputdir, feat_set, "roc.png")
        kaggle_submission_output_csv = os.path.join(outputdir, feat_set, "submissions.csv")

        npzdir = os.path.join(outputdir, feat_set, "predictions")
        output_model_dir = os.path.join(outputdir, feat_set, "model")
        input_csv = os.path.join(feature_set_dir, feat_set, "train.csv.gz")
        test_csv = os.path.join(feature_set_dir, feat_set, "test.csv.gz")

        if os.path.isfile(results_json) and not force:
            print("Skipping", feat_set, "Found", results_json)
            continue

        need_dirs = [npzdir, output_model_dir]
        for _dir in need_dirs:
            if os.path.isdir(_dir):
                shutil.rmtree(_dir)
            os.makedirs(_dir)

        if model_type.startswith('dnn'):
            metrics = api_dnn(model_config=MODEL_CONFIGS[model_type],
                              train_fn=input_csv,
                              test_fn=test_csv,
                              kaggle_submission_output_csv=kaggle_submission_output_csv,
                              results_output_json_fn=results_json,
                              model_dir=output_model_dir,
                              output_png=results_png,
                              npzdir=npzdir)
        else:
            metrics = run_ml_model(model_config=MODEL_CONFIGS[model_type],
                                   train_fn=input_csv,
                                   test_fn=test_csv,
                                   kaggle_submission_output_csv=kaggle_submission_output_csv,
                                   results_output_json_fn=results_json,
                                   model_dir=output_model_dir,
                                   output_png=results_png,
                                   npzdir=npzdir)
        print(json.dumps(metrics, indent=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--feature_set_dir", default="./merged_datasets")
    parser.add_argument("-o", "--outputdir")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument('-m', "--model_type", default="dnn__default")
    parser.add_argument("--run_all_feature_sets", action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(outputdir=args.outputdir, feature_set_dir=args.feature_set_dir, force=args.force,
         model_type=args.model_type, run_all_feature_sets=args.run_all_feature_sets)

