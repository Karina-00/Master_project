from functools import reduce

import numpy as np
import pandas as pd


def rename_importnace_col(df, prefix):
    df.rename(columns={'importance': f'{prefix}_importance'}, inplace=True)

def merge_feature_importances(dfs):
    return reduce(lambda left, right: pd.merge(left, right, on='feature', how='outer'), dfs)


def rank_importances(feature_importance_df):
    df_without_feature = feature_importance_df.drop('feature', axis=1)

    abs_df = df_without_feature.abs()

    ranked_df = abs_df.rank(ascending=False, axis=0, method='min')

    ranked_df.fillna(ranked_df.max()+1, inplace=True)

    ranked_df['average_rank'] = np.round(ranked_df.mean(axis=1, skipna=True),2)

    ranked_df.insert(0, 'feature', feature_importance_df['feature'])

    return ranked_df
