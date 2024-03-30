import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from main.constants import CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES



def get_continuous_attributes_except(attribute):
    remaining_attributes = CONTINUOUS_ATTRIBUTES.copy()
    remaining_attributes.remove(attribute)
    return remaining_attributes


def explore_all_variations_of_preprocessing(X_train, y_train, target_attribute, models, continuous_preprocessings, categorical_preprocessings):
    scores_df = pd.DataFrame(columns=['continuous_preprocessing', 'categorical_pteprocessing', 'model', 'MAE'])  # TODO: change to R2

    i = 1
    total_iterations = len(continuous_preprocessings) * len(categorical_preprocessings) * len(models)

    for continuous_preprocessor_name, continuous_preprocessor in continuous_preprocessings.items():
        for categorical_preprocessor_name, categorical_preprocessor in categorical_preprocessings.items():
            for model in models:
                preprocessor = ColumnTransformer(
                    verbose_feature_names_out=False,
                    transformers=[
                        ('num', continuous_preprocessor, get_continuous_attributes_except(target_attribute)),
                        ('cat', categorical_preprocessor, CATEGORICAL_ATTRIBUTES)
                    ])

                pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)

                # scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
                
                scores_df.loc[len(scores_df)] = [continuous_preprocessor_name, categorical_preprocessor_name, str(model), abs(scores.mean())]
                print(f'{i}/{total_iterations}', str(model), continuous_preprocessor_name, categorical_preprocessor_name, scores, abs(scores.mean()))
                i += 1

    return scores_df