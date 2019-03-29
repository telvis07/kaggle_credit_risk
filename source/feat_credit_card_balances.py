"""
author: Telvis Calhoun

Script to generate features from : credit_card_balance.csv
"""
import pandas as pd
from preprocess import (fix_null_values, generate_encoders, add_onehot_col, add_label_col,
                        rename_cols_with_feat_name_prefix)
import os
import shutil


FEAT_CODE = "CCBAL"


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
    # print(temp_df.head())
    # print(temp_df.index, stats_df.index)
    stats_df = pd.concat([stats_df, temp_df], axis=1, copy=False, join="inner")
    stats_df.reset_index(inplace=True)

    return stats_df


def do_numeric_col_stats(fn, output_fn, usecols):
    """
    Generate ['min', 'max', 'mean', 'median'] for numerical columns

    :param fn: File path to raw data
    :param output_features_dir: Directory path to write the features
    :return:
    """
    # Calc basic statistics for numerical fields.

    df = pd.read_csv(fn)
    print("shape", df.shape)
    print("columns", df.columns)

    df = df[["SK_ID_CURR"] + usecols].copy()
    # handle duplicate colnames across raw datasets.
    usecols = rename_cols_with_feat_name_prefix(df=df, feat_code=FEAT_CODE,
                                                colnames=usecols, idxcol="SK_ID_CURR")

    temp_df = add_numeric_stats_cols(df=df, column_names=usecols)
    temp_df.to_csv(output_fn, compression="gzip", index=False)
    print("Wrote to", output_fn)


def do_label_onehot_sums(fn, output_fn, output_features_dir, usecols):
    """
    Generate onehot encoded features per categorical column. (1) SK_ID_CURR maps to many
    SK_ID_PREV entries, so aggregate (sum) each category by SK_ID_CURR.

    :param fn: File path to bureau.csv
    :param label_cols: Categorical columns in the dataset
    :param output_features_dir: Directory path to write the features
    :return:
    """

    df = pd.read_csv(fn)
    label_cols = list(usecols)
    df = df[["SK_ID_CURR", "SK_ID_PREV"] + label_cols].copy()

    # make sure the columns are strings (object type)
    for col in label_cols:
        df[col] = df[col].astype(str)

    # handle duplicate colnames across raw datasets.
    usecols = rename_cols_with_feat_name_prefix(df=df, feat_code=FEAT_CODE,
                                                colnames=usecols, idxcol="SK_ID_CURR")

    combined_idxcol = "SK_ID_CURR__SK_ID_PREV"

    df[combined_idxcol] = df.apply(lambda x: "{}_{}".format(x['SK_ID_CURR'], x["SK_ID_PREV"]), axis=1)

    object_cols = dict(df[usecols].nunique())
    one_hot_encoders_di, label_encoders_di = generate_encoders(df, object_cols=object_cols)
    temp_df = add_onehot_col(df=df, one_hot_encoders_di=one_hot_encoders_di,
                             idxcol=combined_idxcol, output_feat_dir=output_features_dir, drop=True,
                             filename_prefix="test_", force=True)

    temp_df = add_label_col(df=temp_df, label_encoders_di=label_encoders_di,
                            idxcol=combined_idxcol, output_feat_dir=output_features_dir, drop=True,
                            filename_prefix="test_", force=True)

    agg_columns = set(temp_df.columns) - {'SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_CURR__SK_ID_PREV'}
    agg_columns = sorted(list(agg_columns))
    print("agg_columns", agg_columns)
    grp_df = temp_df.groupby("SK_ID_CURR")[agg_columns].agg(['sum'])
    grp_df.reset_index(inplace=True)
    grp_df.columns = ["SK_ID_CURR"]+agg_columns

    grp_df.to_csv(output_fn, compression='gzip', index=False)
    print("Wrote to", output_fn)


def main():
    output_features_dir = "./output/{}_features".format(FEAT_CODE)
    if os.path.isdir(output_features_dir):
        shutil.rmtree(output_features_dir)

    os.makedirs(output_features_dir)
    input_csv_fn = './data/credit_card_balance.csv'

    #
    numerical_cols = ['MONTHS_BALANCE', 'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
        'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
        'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT',
        'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
    output_fn = os.path.join(output_features_dir, "{}_numeric_features.csv.gz".format(FEAT_CODE))

    do_numeric_col_stats(fn=input_csv_fn,
                         output_fn=output_fn,
                         usecols=numerical_cols)
    #
    label_cols = ["CNT_DRAWINGS_OTHER_CURRENT"]

    output_fn = os.path.join(output_features_dir, "{}_label_features.csv.gz".format(FEAT_CODE))
    do_label_onehot_sums(fn=input_csv_fn,
                         output_fn=output_fn,
                         usecols=label_cols,
                         output_features_dir=output_features_dir)


if __name__ == '__main__':
    main()
