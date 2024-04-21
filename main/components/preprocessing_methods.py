import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from main.constants import CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES


def remove_outliers(df, target_attribute, show_plot=True):
    if show_plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        fig.suptitle(f'{target_attribute} before and after outlier removal', fontsize=16)

        df[target_attribute].hist(ax=axes[0])
        axes[0].set_title('Before')

    mean = df[target_attribute].mean()
    std = df[target_attribute].std()
    zscore_3_upper_threshold = mean + (3*std)
    zscore_3_lower_threshold = mean - (3*std)
    print(f'Left only the values within the interval: [{round(zscore_3_lower_threshold,2)}, {round(zscore_3_upper_threshold, 2)}]')

    values_to_remove = (df[target_attribute] > zscore_3_upper_threshold) | (df[target_attribute] < zscore_3_lower_threshold)

    number_of_rows_to_remove = df[values_to_remove].shape[0]
    number_of_not_na_rows = df[target_attribute].notna().sum()
    percent_to_remove = round((number_of_rows_to_remove / number_of_not_na_rows) * 100, 2)
    
    print(f'Removed {number_of_rows_to_remove} outlier values of {target_attribute} -> {percent_to_remove} % of all the not null examples')

    df.loc[values_to_remove, target_attribute] = np.nan

    if show_plot:
        df[target_attribute].hist(ax=axes[1])
        axes[1].set_title('After')
        plt.show()

    return df


def get_all_attributes_except(all_attributes, attribute_to_remove):
    remaining_attributes = all_attributes.copy()
    if attribute_to_remove in remaining_attributes:
        remaining_attributes.remove(attribute_to_remove)
    return remaining_attributes


def get_continuous_attributes_except(attribute):
    return get_all_attributes_except(CONTINUOUS_ATTRIBUTES, attribute)


def get_categorical_attributes_except(attribute):
    return get_all_attributes_except(CATEGORICAL_ATTRIBUTES, attribute)



def explore_all_variations_of_preprocessing(X_train, y_train, target_attribute, models, continuous_preprocessings, categorical_preprocessings, scoring_metric='neg_mean_absolute_error'):
    scores_df = pd.DataFrame(columns=['continuous_preprocessing', 'categorical_pteprocessing', 'model', scoring_metric])

    i = 1
    total_iterations = len(continuous_preprocessings) * len(categorical_preprocessings) * len(models)

    for continuous_preprocessor_name, continuous_preprocessor in continuous_preprocessings.items():
        for categorical_preprocessor_name, categorical_preprocessor in categorical_preprocessings.items():
            for model in models:
                preprocessor = ColumnTransformer(
                    verbose_feature_names_out=False,
                    transformers=[
                        ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
                        ('cat', categorical_preprocessor, get_categorical_attributes_except(target_attribute))
                    ])

                pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=-1)
                
                scores_df.loc[len(scores_df)] = [continuous_preprocessor_name, categorical_preprocessor_name, str(model), abs(scores.mean())]
                print(f'{i}/{total_iterations}', str(model), continuous_preprocessor_name, categorical_preprocessor_name, scores, abs(scores.mean()))
                i += 1

    return scores_df