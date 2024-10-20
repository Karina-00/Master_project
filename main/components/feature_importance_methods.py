from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline

# https://jundongl.github.io/scikit-feature/algorithms.html
from skfeature.function.statistical_based import gini_index

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFECV

from sklearn.tree import plot_tree


import dtreeviz

from main.constants import CATEGORICAL_ATTRIBUTES, CLASS_NAMES, CONTINUOUS_ATTRIBUTES
from main.components.preprocessing_methods import get_continuous_attributes_except,get_categorical_attributes_except



#  Transforms feature importance df with one hot encoded features into a feature importance df with original features
def get_original_feature_importance_df(transformed_feature_importance_df, importance_label='importance', aggregation_method='sum'):
    transformed_feature_importance_df['original_feature'] = transformed_feature_importance_df['feature'].apply(lambda x: x.split('_')[0])
    if aggregation_method == 'sum':
        original_feature_importance_df = transformed_feature_importance_df.groupby('original_feature')[importance_label].sum().reset_index()
    if aggregation_method == 'mean':
        original_feature_importance_df = transformed_feature_importance_df.groupby('original_feature')[importance_label].mean().reset_index()
    original_feature_importance_df[importance_label] = original_feature_importance_df[importance_label].clip(upper=1)
    original_feature_importance_df.rename(columns={'original_feature': 'feature'}, inplace=True)

    return original_feature_importance_df.sort_values(by=importance_label, key=abs, ascending=False)


def get_feature_importance_logistic_regressison(pipeline: Pipeline, target_attribute, significance_threshold=0):
    coef_lists = pipeline['model'].coef_
    column_names = pipeline['preprocessor'].get_feature_names_out()
    all_feature_importances = {}

    for i, coef in enumerate(coef_lists):
        feature_importnces = [*zip(column_names, coef)]
        feature_importances_df = pd.DataFrame(feature_importnces, columns=['feature', 'importance'])
        feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)

        sns.barplot(feature_importances_sorted.head(20), x="importance", y="feature").set(title=CLASS_NAMES[i])
        plt.show()
        all_feature_importances[CLASS_NAMES[i]] = feature_importances_sorted

    combined_df = pd.concat(all_feature_importances.values())
    average_importances = combined_df.groupby('feature').importance.mean().reset_index()
    average_importances = average_importances.sort_values(by='importance', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(average_importances)

    return original_feature_importance



def get_feature_importance_lasso(pipeline: Pipeline, target_attribute, significance_threshold=0):
    lasso_coefs = pipeline['model'].coef_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_lasso = [*zip(column_names, lasso_coefs)]
    feature_importances = pd.DataFrame(feature_importnces_lasso, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances[feature_importances['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(feature_importances_sorted)

    plt.figure(figsize=(10, 10))
    sns.barplot(original_feature_importance, x="importance", y="feature").set(title=target_attribute)
    plt.show()

    return original_feature_importance


def get_feature_importance_tree(pipeline: Pipeline, target_attribute, significance_threshold=0):
    feature_importances = pipeline['model'].feature_importances_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_with_columns = [*zip(column_names, feature_importances)]
    feature_importances_df = pd.DataFrame(feature_importnces_with_columns, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(feature_importances_sorted)

    sns.barplot(original_feature_importance, x="importance", y="feature").set(title=target_attribute)
    plt.show()

    plot_tree(pipeline['model'], filled=True, rounded=True, feature_names=column_names)
    # plt.savefig(f'charts/trees/{target_attribute[:4]}_decision_tree.pdf')
    plt.show()

    return original_feature_importance


def get_feature_importance_knn(pipeline: Pipeline, target_attribute, significance_threshold=0):

    return None


def plot_fancy_tree(pipeline, X_train, y_train, target_name, class_names=None, is_classification=True):
    X_transformed = pipeline['preprocessor'].fit_transform(X_train, y_train)
    column_names = pipeline['preprocessor'].get_feature_names_out()

    if is_classification:
        viz_model = dtreeviz.model(pipeline['model'],
                            X_train=X_transformed,
                            y_train=y_train.astype(int),
                            feature_names=column_names,
                            target_name=target_name,
                            class_names=class_names,
                            )
    else:
        viz_model = dtreeviz.model(pipeline['model'],
                    X_train=X_transformed,
                    y_train=y_train,
                    feature_names=column_names,
                    target_name=target_name,
                    )
        
    #  if I wanted to save the tree
    # v = viz_model.view(scale=1.8)
    # v.show()
    # v.save("charts/trees/pco_multiclass_decision_tree.svg")  

    return viz_model.view(scale=1.8)


def plot_fancy_tree_smote(pipeline, X_train, y_train, target_name, show_artificial_data=False, class_names=None, is_classification=True):
    preprocessing_pipeline = ImblearnPipeline(steps=pipeline.steps[:-1])

    if show_artificial_data:
        smote_pipeline = ImblearnPipeline(steps=pipeline.steps[:-2])
        _, y_transformed = smote_pipeline.fit_resample(X_train, y_train)
        X_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)
    else:
        new_pipeline = Pipeline(steps=[pipeline.steps[0], pipeline.steps[2]])
        X_transformed = new_pipeline.fit_transform(X_train, y_train)
        y_transformed = y_train
    
    column_names = preprocessing_pipeline.get_feature_names_out()

    if is_classification:
        viz_model = dtreeviz.model(pipeline['model'],
                            X_train=X_transformed,
                            y_train=y_transformed.astype(int),
                            feature_names=column_names,
                            target_name=target_name,
                            class_names=class_names,
                            )
    else:
        viz_model = dtreeviz.model(pipeline['model'],
                    X_train=X_transformed,
                    y_train=y_train,
                    feature_names=column_names,
                    target_name=target_name,
                    )
        
    #  if I wanted to save the tree
    # v = viz_model.view(scale=1.8)
    # v.show()
    # v.save("charts/trees/pco_multiclass_decision_tree.svg")  

    return viz_model.view(scale=1.8, ticks_fontsize=6)

def get_feature_importance_rf(pipeline: Pipeline, target_attribute, significance_threshold=0.0):
    feature_importances = pipeline['model'].feature_importances_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_with_columns = [*zip(column_names, feature_importances)]
    feature_importances_df = pd.DataFrame(feature_importnces_with_columns, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(feature_importances_sorted)

    plt.figure(figsize=(10, 30))
    sns.barplot(original_feature_importance, x="importance", y="feature").set(title=target_attribute)
    plt.show()

    return original_feature_importance


# def get_feature_importance_svm(pipeline: Pipeline, target_attribute, significance_threshold=0.0):
#     coefficients = pipeline['model'].coef_[0]
#     column_names = pipeline['preprocessor'].get_feature_names_out()

#     feature_importances_df = pd.DataFrame({'feature': column_names, 'importance': coefficients})
#     feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)

#     sns.barplot(data=feature_importances_sorted, x="importance", y="feature").set(title=target_attribute)
#     plt.show()

#     return feature_importances_sorted


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


def feature_selection_mutual_info_regression(X_train, y_train, target_attribute, continuous_preprocessor, categorical_preprocessor, figsize=(10, 30)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    preprocessor = ColumnTransformer(
    verbose_feature_names_out=False,
    transformers=[
        ('num', continuous_preprocessor, continuous_attributes),
        ('cat', categorical_preprocessor, categorical_attributes)
    ])

    pipeline = Pipeline([('preprocessor', preprocessor)])

    X_transformed = pipeline.fit_transform(X_train, y_train)

    mutual_info_scores = mutual_info_regression(X_transformed, y_train, random_state=42)
    feature_names = preprocessor.get_feature_names_out()

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': mutual_info_scores})
    feature_importances_sorted = feature_importances.sort_values(by='importance', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(feature_importances_sorted)

    plt.figure(figsize=figsize)
    sns.barplot(original_feature_importance[original_feature_importance['importance']>0], x="importance", y="feature", legend=False).set(title=f"Mutual information importance - {target_attribute}")
    plt.show()

    return original_feature_importance


def feature_selection_mutual_info_classification(X_train, y_train, target_attribute, continuous_preprocessor, categorical_preprocessor, figsize=(10, 30)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    preprocessor = ColumnTransformer(
    verbose_feature_names_out=False,
    transformers=[
        ('num', continuous_preprocessor, continuous_attributes),
        ('cat', categorical_preprocessor, categorical_attributes)
    ])

    pipeline = Pipeline([('preprocessor', preprocessor)])

    X_transformed = pipeline.fit_transform(X_train, y_train)

    mutual_info_scores = mutual_info_classif(X_transformed, y_train)
    feature_names = preprocessor.get_feature_names_out()

    feature_importances = pd.DataFrame({'feature': feature_names, 'mutual_info_score': mutual_info_scores})
    feature_importances_sorted = feature_importances.sort_values(by='mutual_info_score', key=abs, ascending=False)
    original_feature_importance = get_original_feature_importance_df(feature_importances_sorted, 'mutual_info_score')

    plt.figure(figsize=figsize)
    sns.barplot(original_feature_importance[original_feature_importance['mutual_info_score']>0], x="mutual_info_score", y="feature", legend=False)
    plt.show()

    return original_feature_importance


def recursive_feature_elimination(X_train, y_train, model, target_attribute, continuous_preprocessing, categorical_preprocessor, scoring_metric="neg_mean_absolute_error", figsize=(20, 10)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessing, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
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

    ranked_featrures_rfecv = pd.DataFrame({'feature': preprocessor.get_feature_names_out(), 'ranking': rfecv.ranking_})

    ranked_featrures_rfecv['original_feature'] = ranked_featrures_rfecv['feature'].apply(lambda x: x.split('_')[0])
    # minimum (highest) place in the ranking of all occurances
    original_feature_importance_df = ranked_featrures_rfecv.groupby('original_feature')['ranking'].min().reset_index()
    original_feature_importance_df.rename(columns={'original_feature': 'feature'}, inplace=True)

    ranked_featrures_rfecv_ranked = original_feature_importance_df.sort_values(by='ranking').reset_index(drop=True)

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    scores = abs(rfecv.cv_results_["mean_test_score"])

    plt.figure(figsize=figsize)
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


def get_permutation_importance(X_train, y_train, model, continuous_preprocessor, categorical_preprocessor, target_attribute, threshold=0, figsize=(10, 30)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
        ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    pipeline.fit(X_train, y_train)

    result = permutation_importance(pipeline, X_train, y_train, n_repeats=10, n_jobs=-1, random_state=42)
    importances = result.importances_mean

    df_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': importances})
    df_importances_sorted = df_importances.sort_values(by='importance', ascending=False)

    permutation_importance_selected_features = df_importances_sorted[df_importances_sorted['importance'] > threshold]

    print(f'selected {len(permutation_importance_selected_features)} features')
    plt.figure(figsize=figsize)
    sns.barplot(df_importances_sorted, x="importance", y="feature", legend=False).set(title=f'Permutation importance - {target_attribute}')
    plt.show()

    return permutation_importance_selected_features


def feature_selection_chi2(feature_selection_model, target_attribute, continuous_preprocessor, categorical_preprocessor,  X_train, y_train, threshold=0.05, figsize=(10, 30)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
        ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('feature_selection', feature_selection_model)])

    X_new = pipeline.fit_transform(X_train, y_train)

    feature_importances = dict(zip(preprocessor.get_feature_names_out(), feature_selection_model.pvalues_))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=False)
    df_importances = pd.DataFrame(sorted_importances, columns=['feature', 'p_value'])

    original_feature_importance = get_original_feature_importance_df(df_importances, importance_label='p_value')
    original_feature_importance.sort_values(by='p_value', ascending=True, inplace=True)


    selected_features = original_feature_importance[original_feature_importance['p_value'] < threshold]
    selected_feature_names = list(selected_features['feature'].values)
    print(f'Selected {len(selected_feature_names)} features')

    # print(selected_feature_names)
    # selected_indices = feature_selection_model.get_support(indices=True)
    # selected_feature_names = [preprocessor.get_feature_names_out()[i] for i in selected_indices]

    plt.figure(figsize=figsize)
    sns.barplot(original_feature_importance, x="p_value", y="feature", legend=False)
    plt.show()

    return selected_feature_names


def feature_selection_gini_index(target_attribute, continuous_preprocessor, categorical_preprocessor, X_train, y_train, threshold=0.5, figsize=(10, 30)):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))
    
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
        ])
    pipeline = Pipeline([('preprocessor', preprocessor)])
    pipeline.set_output(transform='default')

    X_train_transformed = pipeline.fit_transform(X_train, y_train)

    scores = gini_index.gini_index(X_train_transformed, y_train.astype(int))

    gini_index_scores = pd.DataFrame({'feature':  preprocessor.get_feature_names_out(), 'gini_index': scores})
    gini_index_scores.sort_values(by='gini_index', inplace=True)

    original_feature_importance = get_original_feature_importance_df(gini_index_scores, importance_label='gini_index', aggregation_method='mean')
    original_feature_importance.sort_values(by='gini_index', ascending=True, inplace=True)


    selected_features = original_feature_importance[original_feature_importance['gini_index'] < threshold]
    selected_feature_names = list(selected_features['feature'].values)
    print(f'Selected {len(selected_feature_names)} features')

    plt.figure(figsize=figsize)
    sns.barplot(original_feature_importance, x="gini_index", y="feature", legend=False)
    plt.show()

    return selected_feature_names