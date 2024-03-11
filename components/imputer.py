import pandas as pd


class Imputer:
    def __init__(self, imputing_method_continuous, imputing_methohd_categorical, continuous_attributes, categorical_attributes):
        self.imputing_method_continuous = imputing_method_continuous
        self.imputing_method_categorical = imputing_methohd_categorical
        self.continuous_attributes = continuous_attributes
        self.categorical_attributes = categorical_attributes
        self.imputer_continuous = None
        self.imputer_categorical = None

    def fit(self, X, y=None):
        self.imputer_continuous = self.imputing_method_continuous
        self.imputer_continuous.set_output(transform='pandas')
        self.imputer_continuous.fit(X[self.continuous_attributes])

        self.imputer_categorical = self.imputing_method_categorical
        self.imputer_categorical.set_output(transform='pandas')
        self.imputer_categorical.fit(X[self.categorical_attributes]).astype('Int8')

    def fit_transform(self, X, y=None):
        self.imputer_continuous = self.imputing_method_continuous
        self.imputer_continuous.set_output(transform='pandas')
        X_continuous_imputed = self.imputer_continuous.fit_transform(X[self.continuous_attributes])

        self.imputer_categorical = self.imputing_method_categorical
        self.imputer_categorical.set_output(transform='pandas')
        X_categorical_imputed = self.imputer_categorical.fit_transform(X[self.categorical_attributes]).astype('Int8')

        return pd.concat([X_continuous_imputed, X_categorical_imputed], axis=1)
    
    def transform(self, X, y=None):
        if not self.imputer_continuous or not self.imputer_categorical:
            return RuntimeError('Call fit_tranform() method first')
        X_continuous_imputed = self.imputer_continuous.transform(X[self.continuous_attributes])
        X_categorical_imputed = self.imputer_categorical.transform(X[self.categorical_attributes]).astype('Int8')

        return pd.concat([X_continuous_imputed, X_categorical_imputed], axis=1)