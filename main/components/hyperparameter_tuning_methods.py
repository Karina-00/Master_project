import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from main.constants import CATEGORICAL_ATTRIBUTES
from main.components.preprocessing_methods import get_continuous_attributes_except




def hyperparameter_tuning_linear(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid, main_parameter='alpha'):
    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
                            ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
                        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    
    main_param_values = cv_results[f'param_model__{main_parameter}']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({main_parameter: main_param_values, 'mean_train_mae': mean_train_score, 'mean_test_mae': mean_test_score})

    #  plot train error vs test error
    sns.lineplot(x=main_parameter, y='value', hue='variable', data=pd.melt(results_df, [main_parameter]))
    if main_parameter == 'alpha':
        plt.xscale("log")
    plt.show()

    return results_df


def hyperparameter_tuning_general(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid):
    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
                            ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
                        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', return_train_score=True, verbose=3, n_jobs=-1).fit(X_train, y_train)
    cv_results = grid_search.cv_results_
    
    tuned_param_values = cv_results[f'params']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({'params': tuned_param_values, 'mean_train_mae': mean_train_score, 'mean_test_mae': mean_test_score})
    # sns.lineplot(x='params', y='value', hue='variable', data=pd.melt(results_df, [tuned_param_values]))
    # sns.lineplot(x='params', y='value', hue='variable', data=pd.melt(results_df, tuned_param_values))
    # plt.show()

    return results_df