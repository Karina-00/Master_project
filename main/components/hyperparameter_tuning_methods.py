import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, PredictionErrorDisplay
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTENC


from sklearn.preprocessing import OneHotEncoder

from main.constants import CATEGORICAL_ATTRIBUTES, CLASS_NAMES, CONTINUOUS_ATTRIBUTES
from main.components.preprocessing_methods import get_continuous_attributes_except,get_categorical_attributes_except


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

    
    # y_pred_train = pipeline.predict(X_train)
    # y_pred_test = pipeline.predict(X_test)

    # TODO: make sure it is the right thing to do
    # Predict and adjust predictions to be non-negative
    y_pred_train = np.maximum(0, pipeline.predict(X_train))
    y_pred_test = np.maximum(0, pipeline.predict(X_test))

    train_mse, train_mae, train_rmse, train_r2 = get_regression_metrics(y_train, y_pred_train)
    test_mse, test_mae, test_rmse, test_r2 = get_regression_metrics(y_test, y_pred_test)

    show_plots(y_train, y_pred_train, 'training')
    show_plots(y_test, y_pred_test, 'test')

    model_scores_df.loc[len(model_scores_df)] = [str(model), 'training', train_mse, train_mae, train_rmse, train_r2]
    model_scores_df.loc[len(model_scores_df)] = [str(model), 'test', test_mse, test_mae, test_rmse, test_r2]

    feature_importnces = feature_importance_method(pipeline, target_attribute)
    
    return model_scores_df, feature_importnces


def plot_confussion_matrices(model, X_train, y_train, X_test, y_test, class_names):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_train,
        y_train,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=axs[0],
    )
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=axs[1],
    )

    axs[0].set_title('Training Set')
    axs[1].set_title('Test Set')
    plt.tight_layout()
    plt.show()



def validate_model_classification(model, target_attribute, class_names, X_train, y_train, X_test, y_test, continuous_preprocessor, categorical_preprocessor, feature_importance_method):
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

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    plot_confussion_matrices(pipeline, X_train, y_train, X_test, y_test, class_names)

    report_train = classification_report(y_train, y_pred_train, target_names=class_names)
    report_test = classification_report(y_test, y_pred_test,  target_names=class_names)

    print('Training set')
    print(report_train)
    print('Test set')
    print(report_test)

    feature_importnces = feature_importance_method(pipeline, target_attribute)

    return feature_importnces, pipeline


def validate_model_classification_smote(model, target_attribute, class_names, X_train, y_train, X_test, y_test, continous_imputer_pipeline, categorical_imputer_pipeline, feature_importance_method):
    pipeline = get_smote_pipeline(model, target_attribute, continous_imputer_pipeline, categorical_imputer_pipeline)
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    plot_confussion_matrices(pipeline, X_train, y_train, X_test, y_test, class_names)

    report_train = classification_report(y_train, y_pred_train, target_names=class_names)
    report_test = classification_report(y_test, y_pred_test,  target_names=class_names)

    print('Training set')
    print(report_train)
    print('Test set')
    print(report_test)

    feature_importnces = feature_importance_method(pipeline, target_attribute)

    return feature_importnces, pipeline


def hyperparameter_tuning_general(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid, scoring_metric='neg_mean_absolute_error'):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))
    
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring_metric, return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print(f"Best score ({scoring_metric}):", best_score)
    
    tuned_param_values = cv_results[f'params']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({'params': tuned_param_values, f'mean_train_score': mean_train_score, 'mean_test_score': mean_test_score})

    return results_df


def get_smote_pipeline(model, target_attribute, continous_imputer_pipeline, categorical_imputer_pipeline):
    imputer_transformer = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continous_imputer_pipeline, CONTINUOUS_ATTRIBUTES),
            ('cat', categorical_imputer_pipeline, get_categorical_attributes_except(target_attribute))
        ])
    imputer_transformer.set_output(transform='pandas')

    smt = SMOTENC(random_state=42, categorical_features=get_categorical_attributes_except(target_attribute))

    categorical_one_hot_encoder = Pipeline([
        ('one_hot_encoder', OneHotEncoder(handle_unknown='error', drop='if_binary', sparse_output=False)),
        ])

    one_hot_encoding_transformer = ColumnTransformer(
            verbose_feature_names_out=False,
            remainder='passthrough',
            transformers=[
                ('cat', categorical_one_hot_encoder, get_categorical_attributes_except(target_attribute)),
            ]).set_output(transform='pandas')
    

    model_pipeline = ImblearnPipeline([
                ('imputing', imputer_transformer),
                ('smt', smt),
                ('preprocessor', one_hot_encoding_transformer),
                ('model', model),
                ])

    return model_pipeline


def hyperparameter_tuning_clasification_smote(X_train, y_train, target_attribute, model, continous_imputer_pipeline, categorical_imputer_pipeline, param_grid, scoring_metric='f1'):
    pipeline = get_smote_pipeline(model, target_attribute, continous_imputer_pipeline, categorical_imputer_pipeline)
    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring_metric, return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print(f"Best score ({scoring_metric}):", best_score)
    
    tuned_param_values = cv_results[f'params']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({'params': tuned_param_values, f'mean_train_score': mean_train_score, 'mean_test_score': mean_test_score})

    return results_df



def hyperparameter_tuning_linear(X_train, y_train, target_attribute, model, continuous_preprocessor, categorical_preprocessor, param_grid, main_parameter='alpha', scoring_metric='neg_mean_absolute_error'):
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))
    
    preprocessor = ColumnTransformer(
        verbose_feature_names_out=False,
        transformers=[
            ('num', continuous_preprocessor, continuous_attributes),
            ('cat', categorical_preprocessor, categorical_attributes)
        ])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring_metric, return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters:", best_params)
    print(f"Best score ({scoring_metric}):", best_score)
    
    main_param_values = cv_results[f'param_model__{main_parameter}']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({main_parameter: main_param_values, 'mean_train_score': mean_train_score, 'mean_test_score': mean_test_score})
    sns.lineplot(x=main_parameter, y='value', hue='variable', data=pd.melt(results_df, [main_parameter]))
    plt.show()

    return results_df


def hyperparameter_tuning_linear_on_preprocessed_dataset(X_train, y_train, target_attribute, model, param_grid, main_parameter='alpha', scoring_metric='neg_mean_absolute_error'):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring_metric, return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters:", best_params)
    print(f"Best score ({scoring_metric}):", best_score)
    
    main_param_values = cv_results[f'param_model__{main_parameter}']
    mean_test_score = abs(cv_results['mean_test_score'])
    mean_train_score = abs(cv_results['mean_train_score'])

    results_df = pd.DataFrame({main_parameter: main_param_values, 'mean_train_score': mean_train_score, 'mean_test_score': mean_test_score})
    sns.lineplot(x=main_parameter, y='value', hue='variable', data=pd.melt(results_df, [main_parameter]))
    plt.show()

    return results_df


def compare_random_states(X_train, y_train, model, target_attribute, continuous_preprocessor, categorical_preprocessor, scoring_metric='neg_mean_absolute_error'):
    random_options = range(1,100)
    categorical_attributes = list(set(get_categorical_attributes_except(target_attribute)) & set(X_train.columns))
    continuous_attributes = list(set(get_continuous_attributes_except(target_attribute)) & set(X_train.columns))

    tune_df = pd.DataFrame(index=random_options, columns=['cv_score'])

    for random_o in tqdm(random_options):

        preprocessor = ColumnTransformer(
            verbose_feature_names_out=False,
            transformers=[
                ('num', continuous_preprocessor, continuous_attributes),
                ('cat', categorical_preprocessor, categorical_attributes)
            ])
        model.set_params(random_state=random_o)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_o)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=-1)

        tune_df.at[random_o,'cv_score'] = cv_score.mean()

    return tune_df


# def smote_oversampling(X_train, y_train, X_test, should_scale_data=True):
#     continuous_imputer = None

#     if should_scale_data:
#         continuous_imputer = Pipeline([
#             ('scaler', MinMaxScaler()),
#             ('imputer', KNNImputer(n_neighbors=7)),
#             ])
#     else:
#         continuous_imputer = Pipeline([('imputer', KNNImputer(n_neighbors=7))])

#     categorical_imputer = Pipeline([
#         ('imputer', IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=10, n_jobs=-1), max_iter=40, initial_strategy='most_frequent')),
#         ])

#     imputer_transformer = ColumnTransformer(
#         verbose_feature_names_out=False,
#         transformers=[
#             ('num', continuous_imputer, CONTINUOUS_ATTRIBUTES),
#             ('cat', categorical_imputer, get_categorical_attributes_except(PCO))
#         ])
#     imputer_transformer.set_output(transform='pandas')

#     smote_pipeline = ImblearnPipeline(steps=[
#         ('imputation', imputer_transformer),
#         ('smote', SMOTENC(random_state=42, categorical_features=get_categorical_attributes_except(PCO))),
#     ])

#     X_resampled, y_resampled = smote_pipeline.fit_resample(X_train, y_train)

#     categorical_one_hot_encoder = Pipeline([
#         ('one_hot_encoder', OneHotEncoder(handle_unknown='error', drop='if_binary', sparse_output=False)),
#         # ('feature_rename', CustomFeatureRenamer(feature_renaming_map)),
#         ])

#     identity_function = FunctionTransformer(func=lambda x: x)

#     one_hot_encoding_pipeline = Pipeline([
#         ('encoder_scaler', ColumnTransformer(
#                 verbose_feature_names_out=False,
#                 transformers=[
#                     ('num', identity_function, CONTINUOUS_ATTRIBUTES),
#                     ('cat', categorical_one_hot_encoder, get_categorical_attributes_except(PCO)),
#                 ]).set_output(transform='pandas')),
#     ])

#     X_train_final = one_hot_encoding_pipeline.fit_transform(X_resampled)

#     return X_train_final, y_resampled


# X_train_prep, y_train_prep = smote_oversampling(X_train, y_train, X_test, should_scale_data=True)
# X_train_prep.shape