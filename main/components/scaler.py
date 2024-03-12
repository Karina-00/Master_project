from sklearn.compose import ColumnTransformer


class Scaler:
    def __init__(self, scaling_method, attributes_to_scale):
        self.scaling_method = scaling_method
        self.attributes_to_scale = attributes_to_scale
        self.scaler = None

    def fit(self, X, y=None):
        self.scaler = ColumnTransformer(
            transformers=[
                ('scaler', self.scaling_method, self.attributes_to_scale)
            ],
            # so that the categorical columns remain unchanged
            remainder='passthrough',
            # so it doesn't rename the columns
            verbose_feature_names_out=False,
            )
        self.scaler.set_output(transform='pandas')
        self.scaler.fit(X)


    def fit_transform(self, X, y=None):
        self.scaler = ColumnTransformer(
            transformers=[
                ('scaler', self.scaling_method, self.attributes_to_scale)
            ],
            # so that the categorical columns remain unchanged
            remainder='passthrough',
            # so it doesn't rename the columns
            verbose_feature_names_out=False,
            )
        self.scaler.set_output(transform='pandas')
        return self.scaler.fit_transform(X)
    
    def transform(self, X, y=None):
        if not self.scaler:
            return RuntimeError('Call fit_tranform() method first')
        return self.scaler.transform(X)
