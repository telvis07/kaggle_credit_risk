"""
author: Telvis Calhoun

Script to generate features from subsets of all the feature files.
"""
import pandas as pd
import os
from preprocess import (fix_number_null_values, show_null_value_cols)
import argparse

TRAIN_FILE = "output/application_train_test_encoded/application_train.csv.gz"
TEST_FILE = "output/application_train_test_encoded/application_test.csv.gz"

feature_files = {
    "main_only": [
    ],
    "main_bureau": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
    ],
    "main_bureau_poscash": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_numeric_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_label_features.csv.gz"
        # "output/application_train_test_encoded/application_test.csv.gz",
    ],
    "main_bureau_ccbal": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/CCBAL_features/CCBAL_label_features.csv.gz",
        "./output/CCBAL_features/CCBAL_numeric_features.csv.gz"
        # "output/application_train_test_encoded/application_test.csv.gz",
    ],
    "main_bureau_prevapp": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_label_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_numeric_features.csv.gz"
        # "output/application_train_test_encoded/application_test.csv.gz",
    ],
    "main_bureau_instpay": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/INSTPAY_features/INSTPAY_numeric_features.csv.gz"
        # "output/application_train_test_encoded/application_test.csv.gz",
    ],
    "main_plus_all_numeric_files": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_numeric_features.csv.gz",
        "./output/CCBAL_features/CCBAL_numeric_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_numeric_features.csv.gz",
        "./output/INSTPAY_features/INSTPAY_numeric_features.csv.gz"
    ],
    "main_plus_all_label_files": [
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_label_features.csv.gz",
        "./output/CCBAL_features/CCBAL_label_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_label_features.csv.gz",
    ],
    "main_plus_all":[
        # "output/application_train_test_encoded/application_train.csv.gz",
        "output/bureau_features/bureau_numeric_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_numeric_features.csv.gz",
        "./output/CCBAL_features/CCBAL_numeric_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_numeric_features.csv.gz",
        "./output/INSTPAY_features/INSTPAY_numeric_features.csv.gz",
        #
        "output/bureau_features/bureau_features_label_features.csv.gz",
        "./output/POS_CASH_balance_features/POS_CASH_balance_label_features.csv.gz",
        "./output/CCBAL_features/CCBAL_label_features.csv.gz",
        "./output/PREVAPP_features/PREVAPP_label_features.csv.gz",
    ]

}


def merge_features_files(rawdata_fn, output_base_fn, root_datadir="./", feat_set_key="all"):
    """
    Merge the feature files specified by ``feat_set_key``.
    :param rawdata_fn: path to train.csv.gz or test.csv.gz
    :param output_base_fn: File path to write merged feature set files.
    :param root_datadir: Directory path to write feature set files
    :param feat_set_key: Dictionary key that maps to lists of files to merge.
    :return:
    """
    feature_files_list = feature_files[feat_set_key]
    outputdir = os.path.join(root_datadir, feat_set_key)
    print("outputdir", outputdir)

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    print("Loading", rawdata_fn)
    df = pd.read_csv(rawdata_fn)
    output_fn = os.path.join(outputdir, output_base_fn)

    if os.path.isfile(output_fn):
        os.unlink(output_fn)

    for fn in feature_files_list:
        print("Loading", fn)
        temp_df = pd.read_csv(fn)
        df = df.merge(temp_df, how="left", copy=False)

    print("df.shape", df.shape)
    nullcols = show_null_value_cols(df)
    fix_number_null_values(df=df, null_cols=nullcols, impute=None)

    print("Writing to", output_fn)
    df.to_csv(output_fn, compression="gzip", index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", choices=tuple(feature_files.keys()),
                        default="main_only", help="Which features to use")

    parser.add_argument("-o", "--root_datadir",
                        default="./merged_datasets", help="Which features to use")

    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.do_train:
        rawdata_fn = TRAIN_FILE
        output_base_fn = "train.csv.gz"
    elif args.do_test:
        rawdata_fn = TEST_FILE
        output_base_fn = "test.csv.gz"
    else:
        raise RuntimeError("Please specify --do_train or --do_test")

    merge_features_files(rawdata_fn=rawdata_fn,
                         output_base_fn=output_base_fn,
                         root_datadir=args.root_datadir,
                         feat_set_key=args.dataset)
