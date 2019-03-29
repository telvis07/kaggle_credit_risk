"""
author: Telvis Calhoun

Script to generate features from : bureau_balance.csv and bureau.csv
"""
import pandas as pd
from preprocess import (fix_null_values, generate_encoders, add_onehot_col, add_label_col,
                        rename_cols_with_feat_name_prefix)
import os
import shutil

FEAT_CODE = "BUR"


def add_numeric_stats_cols(df, column_names, idxcol="SK_ID_CURR"):
    """
    Generate basic numeric stats for columns

    :param df: Pandas dataframe with all data
    :param column_names: Numerical columns to calc stats
    :param idxcol: Primary key columns
    :return: Pandas dataframe with stats columns
    """
    df.set_index(idxcol, drop=True, inplace=True)
    null_cols = {c: "int64" for c in column_names}
    fix_null_values(df, null_cols.items(), impute=None)
    stats_df = None

    for colname in column_names:
        temp_df = df.groupby(idxcol)[colname].agg(['min', 'max', 'mean', 'median'])
        rename_cols = {'min': "{}_min".format(colname),
                       'max': "{}_max".format(colname),
                       'mean': "{}_mean".format(colname),
                       'median': "{}_median".format(colname)}
        temp_df.rename(index=int, inplace=True, columns=rename_cols)
        print(temp_df.shape, df.shape)

        #         display(temp_df)
        if stats_df is None:
            stats_df = temp_df.copy()
        else:
            stats_df = pd.concat([stats_df, temp_df], axis=1, copy=False)

    temp_df = df.groupby(idxcol).size().reset_index(name='bureau_record_counts').set_index(idxcol)
    stats_df = pd.concat([stats_df, temp_df], axis=1, copy=False, join="inner")
    stats_df.reset_index(inplace=True)

    return stats_df


def do_numeric_col_stats(fn_balance='./data/bureau_balance.csv',
                         fn='./data/bureau.csv',
                         output_features_dir="./output/bureau_features"):
    """
    Generate ['min', 'max', 'mean', 'median'] for numerical columns

    :param fn_balance: File path to bureau_balance.csv
    :param fn: File path to bureau.csv
    :param output_features_dir: Directory path to write the features
    :return:
    """
    df_bbalance = pd.read_csv(fn_balance)
    df_bbalance["MONTHS_BALANCE"] = df_bbalance["MONTHS_BALANCE"]

    df = pd.read_csv(fn)
    df = df.merge(df_bbalance, on="SK_ID_BUREAU", how="left")
    print("shape", df.shape)
    print("columns", df.columns)

    numeric_cols = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG',
                    'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                    'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE',
                    'MONTHS_BALANCE']
    df = df[["SK_ID_CURR"] + numeric_cols].copy()

    # handle duplicate colnames across raw datasets.
    numeric_cols = rename_cols_with_feat_name_prefix(df=df, feat_code=FEAT_CODE,
                                                     colnames=numeric_cols, idxcol="SK_ID_CURR")

    temp_df = add_numeric_stats_cols(df=df, column_names=numeric_cols)
    output_fn = os.path.join(output_features_dir, "bureau_numeric_features.csv.gz")
    temp_df.to_csv(output_fn, compression="gzip", index=False)
    print("Wrote to", output_fn)


def do_label_onehot_sums(fn='./data/bureau.csv', label_cols=('CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'),
                         output_features_dir="./output/bureau_features"):
    """
    Generate onehot encoded features per categorical column. (1) SK_ID_CURR maps to many
    SK_ID_BUREAU entries, so aggregate (sum) each category by SK_ID_CURR.

    :param fn: File path to bureau.csv
    :param label_cols: Categorical columns in the dataset
    :param output_features_dir: Directory path to write the features
    :return:
    """
    df = pd.read_csv(fn)
    label_cols = list(label_cols)
    df = df[["SK_ID_CURR", "SK_ID_BUREAU"] + label_cols].copy()
    # handle duplicate colnames across raw datasets.
    label_cols = rename_cols_with_feat_name_prefix(df=df, feat_code=FEAT_CODE,
                                                   colnames=label_cols, idxcol="SK_ID_CURR")
    print("columns", df.columns)

    combined_idxcol = "SK_ID_CURR__SK_ID_BUREAU"

    df[combined_idxcol] = df.apply(lambda x: "{}_{}".format(x['SK_ID_CURR'], x["SK_ID_BUREAU"]), axis=1)
    print("columns", df.columns)

    object_cols = dict(df[label_cols].nunique())
    one_hot_encoders_di, label_encoders_di = generate_encoders(df, object_cols=object_cols)
    temp_df = add_onehot_col(df=df, one_hot_encoders_di=one_hot_encoders_di,
                             idxcol=combined_idxcol, output_feat_dir=output_features_dir, drop=True,
                             filename_prefix="test_", force=True)

    temp_df = add_label_col(df=temp_df, label_encoders_di=label_encoders_di,
                            idxcol=combined_idxcol, output_feat_dir=output_features_dir, drop=True,
                            filename_prefix="test_", force=True)

    agg_columns = set(temp_df.columns) - {'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_CURR__SK_ID_BUREAU'}
    agg_columns = sorted(list(agg_columns))
    print("agg_columns", agg_columns)
    grp_df = temp_df.groupby("SK_ID_CURR")[agg_columns].agg(['sum'])
    grp_df.reset_index(inplace=True)
    grp_df.columns = ["SK_ID_CURR"]+agg_columns

    output_fn = os.path.join(output_features_dir, "bureau_features_label_features.csv.gz")
    grp_df.to_csv(output_fn, compression='gzip', index=False)
    print("Wrote to", output_fn)


def main():
    output_features_dir = "./output/bureau_features"
    if os.path.isdir(output_features_dir):
        shutil.rmtree(output_features_dir)

    os.makedirs(output_features_dir)

    #
    do_numeric_col_stats(output_features_dir=output_features_dir)
    do_label_onehot_sums(output_features_dir=output_features_dir)


if __name__ == '__main__':
    main()
