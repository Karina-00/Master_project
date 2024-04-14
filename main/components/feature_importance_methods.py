from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, RFECV


from main.constants import CATEGORICAL_ATTRIBUTES
from main.components.preprocessing_methods import get_continuous_attributes_except,get_categorical_attributes_except


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


def feature_selection_mutual_info_regression(X_train, y_train, target_attribute, continuous_preprocessor, categorical_preprocessor):
    preprocessor = ColumnTransformer(
    verbose_feature_names_out=False,
    transformers=[
        ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
        ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
    ])

    pipeline = Pipeline([('preprocessor', preprocessor)])

    X_transformed = pipeline.fit_transform(X_train, y_train)

    mutual_info_scores = mutual_info_regression(X_transformed, y_train)
    feature_names = preprocessor.get_feature_names_out()

    feature_importances = pd.DataFrame({'feature': feature_names, 'mutual_info_score': mutual_info_scores})
    feature_importances_sorted = feature_importances.sort_values(by='mutual_info_score', key=abs, ascending=False)

    plt.figure(figsize=(10, 30))
    sns.barplot(feature_importances_sorted, x="mutual_info_score", y="feature", legend=False).set(title=f"Mutual information importance - {target_attribute}")
    plt.show()

    return feature_importances_sorted



def recursive_feature_elimination(X_train, y_train, model, target_attribute, continuous_preprocessing, categorical_preprocessor, scoring_metric="neg_mean_absolute_error"):
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessing, get_continuous_attributes_except(target_attribute)),
            ('cat', categorical_preprocessor, get_categorical_attributes_except(target_attribute))
        ])

    min_features_to_select = 1
    cv = RepeatedKFold(n_repeats=5, n_splits=5, random_state=42)
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring_metric,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )

    pipeline = Pipeline([('preprocessor', preprocessor), ('feature_selection', rfecv)])
    pipeline.fit(X_train, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")

    ranked_featrures_rfecv = pd.DataFrame({'features': preprocessor.get_feature_names_out(), 'ranking': rfecv.ranking_})
    ranked_featrures_rfecv_ranked = ranked_featrures_rfecv.sort_values(by='ranking').reset_index(drop=True)

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    scores = abs(rfecv.cv_results_["mean_test_score"])

    plt.figure(figsize=(20,10))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test score")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        scores,
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title(f"Recursive Feature Elimination \nwith correlated features - {target_attribute}")
    plt.show()
    return ranked_featrures_rfecv_ranked


def get_permutation_importance(X_train, y_train, model, continuous_preprocessor, categorical_preprocessor, target_attribute):
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
            ('cat', categorical_preprocessor, get_categorical_attributes_except(target_attribute))
        ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    pipeline.fit(X_train, y_train)

    result = permutation_importance(pipeline, X_train, y_train, n_repeats=10,  n_jobs=-1)
    importances = result.importances_mean
    importances_std = result.importances_std

    df_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': importances, 'std': importances_std})

    df_importances_sorted = df_importances.sort_values(by='importance', ascending=False)

    permutation_importance_selected_features = df_importances_sorted[df_importances_sorted['importance'] > 0]

    print(f'selected {len(permutation_importance_selected_features)} features')
    print(permutation_importance_selected_features)

    plt.figure(figsize=(10, 30))
    sns.barplot(df_importances_sorted, x="importance", y="feature", legend=False).set(title=f'Permutation importance - {target_attribute}')
    plt.show()
