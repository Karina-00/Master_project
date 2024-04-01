import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, PredictionErrorDisplay
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from main.constants import CATEGORICAL_ATTRIBUTES
from main.components.preprocessing_methods import get_continuous_attributes_except
from sklearn.tree import plot_tree



def get_feature_importance_lasso(pipeline: Pipeline, target_attribute, significance_threshold=0):
    lasso_coefs = pipeline['model'].coef_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_lasso = [*zip(column_names, lasso_coefs)]
    feature_importances = pd.DataFrame(feature_importnces_lasso, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances[feature_importances['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)

    sns.barplot(feature_importances_sorted.head(20), x="importance", y="feature").set(title=target_attribute)
    plt.show()

    return feature_importances_sorted


def get_feature_importance_tree(pipeline: Pipeline, target_attribute, significance_threshold=0):
    feature_importances = pipeline['model'].feature_importances_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_with_columns = [*zip(column_names, feature_importances)]
    feature_importances_df = pd.DataFrame(feature_importnces_with_columns, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)

    sns.barplot(feature_importances_sorted, x="importance", y="feature").set(title=target_attribute)
    plt.show()

    plot_tree(pipeline['model'], feature_names=column_names)
    plt.savefig(f'charts/decision_tree_.pdf')
    plt.show()

    return feature_importances_sorted

def get_feature_importance_rf(pipeline: Pipeline, target_attribute, significance_threshold=0.0):
    feature_importances = pipeline['model'].feature_importances_
    column_names = pipeline['preprocessor'].get_feature_names_out()

    feature_importnces_with_columns = [*zip(column_names, feature_importances)]
    feature_importances_df = pd.DataFrame(feature_importnces_with_columns, columns=['feature', 'importance'])
    feature_importances_sorted = feature_importances_df[feature_importances_df['importance'].abs() > significance_threshold].sort_values(by='importance', key=abs, ascending=False)

    sns.barplot(feature_importances_sorted, x="importance", y="feature").set(title=target_attribute)
    plt.show()

    return feature_importances_sorted


def show_plots(y_true, y_pred, dataset_label):
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=axs[0],
        random_state=42,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=42,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle(f"Predictions on a {dataset_label} set")
    plt.tight_layout()
    plt.show()


def get_regression_metrics(y_true, y_pred):
    mse = round(mean_squared_error(y_true, y_pred), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    rmse = round(root_mean_squared_error(y_true, y_pred), 3)
    r2 = round(r2_score(y_true, y_pred), 3)

    return mse, mae, rmse, r2


def train_model(model, target_attribute, X_train, y_train, X_test, y_test, continuous_preprocessor, categorical_preprocessor, feature_importance_method):
    model_scores_df = pd.DataFrame(columns=['model', 'data_set', 'mse', "mae", "rmse", 'r2'])
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
            ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
        ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_mse, train_mae, train_rmse, train_r2 = get_regression_metrics(y_train, y_pred_train)
    test_mse, test_mae, test_rmse, test_r2 = get_regression_metrics(y_test, y_pred_test)

    show_plots(y_train, y_pred_train, 'training')
    show_plots(y_test, y_pred_test, 'test')

    model_scores_df.loc[len(model_scores_df)] = [str(model), 'training', train_mse, train_mae, train_rmse, train_r2]
    model_scores_df.loc[len(model_scores_df)] = [str(model), 'test', test_mse, test_mae, test_rmse, test_r2]

    feature_importnces = feature_importance_method(pipeline, target_attribute)
    
    return model_scores_df, feature_importnces



def hyperparameter_tuning_linear(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid, main_parameter='alpha'):
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
            ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)

    # grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='r2', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    
    main_param_values = cv_results[f'param_model__{main_parameter}']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({main_parameter: main_param_values, 'mean_train_mae': mean_train_score, 'mean_test_mae': mean_test_score})

    #  plot train error vs test error
    sns.lineplot(x=main_parameter, y='value', hue='variable', data=pd.melt(results_df, [main_parameter]))
    # if main_parameter == 'alpha':
    #     plt.xscale("log")
    plt.show()

    return results_df


def hyperparameter_tuning_general(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid):
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
            ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)

    # grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='r2', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)
    cv_results = grid_search.cv_results_
    
    tuned_param_values = cv_results[f'params']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({'params': tuned_param_values, 'mean_train_mae': mean_train_score, 'mean_test_mae': mean_test_score})
    # sns.lineplot(x='params', y='value', hue='variable', data=pd.melt(results_df, [tuned_param_values]))
    # sns.lineplot(x='params', y='value', hue='variable', data=pd.melt(results_df, tuned_param_values))
    # plt.show()

    return results_df


def compare_random_states(X_train, y_train, model, target_attribute, continuous_preprocessor, categorical_preprocessor):
    random_options = range(1,100)

    tune_df = pd.DataFrame(index=random_options, columns=['cv_mae'])

    for random_o in tqdm(random_options):
        preprocessor = ColumnTransformer(
            verbose_feature_names_out=False,
            transformers=[
                ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
                ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
            ])
        model.set_params(random_state=random_o)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_o)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)

        # cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)

        tune_df.at[random_o,'cv_mae'] = cv_score.mean()

    return tune_df
