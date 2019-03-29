"""
author: Telvis Calhoun

Script to generate features from : application_train.csv and application_test.csv
"""
import pandas as pd
import os
import shutil


from preprocess import (do_data_cleaning, generate_encoders, add_onehot_col, add_label_col)


def clean_train_test_files(outputdir, impute="mean"):
    # read raw datums from net
    input_csv = './data/application_train.csv'
    output_csv = os.path.join(outputdir, 'application_train.clean.csv')
    print("Reading", input_csv, "to", output_csv)
    do_data_cleaning(input_csv=input_csv, output_csv=output_csv, impute=impute)

    # read raw datums from net
    input_csv = './data/application_test.csv'
    output_csv = os.path.join(outputdir, 'application_test.clean.csv')
    print("Reading", input_csv, "to", output_csv)
    do_data_cleaning(input_csv=input_csv, output_csv=output_csv, impute=impute)


def generate_train_test_feature_encodings(cleandir, output_features_dir):
    # TODO: put train and test in separate directories

    # read the processed datums
    input_clean_csv = os.path.join(cleandir, 'application_train.clean.csv')
    print("encoding", input_clean_csv)
    df = pd.read_csv(input_clean_csv)

    # generate one-hot encoders
    one_hot_encoders_di, label_encoders_di = generate_encoders(df=df)

    df = add_onehot_col(df=df, one_hot_encoders_di=one_hot_encoders_di,
                        idxcol="SK_ID_CURR", output_feat_dir=output_features_dir, drop=True,
                        filename_prefix="train_")

    df = add_label_col(df=df, label_encoders_di=label_encoders_di,
                       idxcol="SK_ID_CURR", output_feat_dir=output_features_dir, drop=True,
                       filename_prefix="train_")

    fn = os.path.join(output_features_dir, 'application_train.csv.gz')
    df.to_csv(fn, compression='gzip', index=False)
    print("Wrote to", fn)

    # read the processed datums
    input_clean_csv = os.path.join(cleandir, 'application_test.clean.csv')
    print("encoding", input_clean_csv)
    df = pd.read_csv(input_clean_csv)

    df = add_onehot_col(df=df, one_hot_encoders_di=one_hot_encoders_di,
                        idxcol="SK_ID_CURR", output_feat_dir=output_features_dir, drop=True,
                        filename_prefix="test_")

    df = add_label_col(df=df, label_encoders_di=label_encoders_di,
                       idxcol="SK_ID_CURR", output_feat_dir=output_features_dir, drop=True,
                       filename_prefix="test_")

    fn = os.path.join(output_features_dir, 'application_test.csv.gz')
    df.to_csv(fn, compression='gzip', index=False)
    print("Wrote to", fn)


def main(cleanup=True):
    outputdir = "/Users/telvis/notebooks/kaggle_credit_risk/output/"
    output_clean_dir = os.path.join(outputdir, "application_train_test_cleaned")
    output_encoded_dir = os.path.join(outputdir, "application_train_test_encoded")

    if cleanup and os.path.isdir(outputdir):
        shutil.rmtree(outputdir)

    for _dir in [output_clean_dir, output_encoded_dir]:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

    clean_train_test_files(output_clean_dir)
    generate_train_test_feature_encodings(cleandir=output_clean_dir, output_features_dir=output_encoded_dir)


if __name__ == '__main__':
    main()
