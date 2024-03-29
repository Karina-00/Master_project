import pandas as pd
import numpy as np


class CustomOneHotEncoder:
    def __init__(self, columns_for_one_hot_encoding, new_column_names_map, advanced_encoding=False):
        self.columns_for_one_hot_encoding = columns_for_one_hot_encoding
        self.new_column_names_map = new_column_names_map
        self.advanced_encoding = advanced_encoding
        self.all_columns = []
        self.get_feature_names_out = self.get_feature_names_out()
        

    def fit(self, X, y=None):
        # there's no fitting, we apply the same one hot encoding to all data sets
        pass

    def fit_transform(self, X, y=None):
        for col, prefix in self.columns_for_one_hot_encoding.items():
            X = pd.get_dummies(X, columns=[col], prefix=prefix, dtype=np.int8)

        X = X.rename(columns=self.new_column_names_map)

        # more advanced encoding for nodules:
        # if nodules_both_sides = 1, then nodules_right = 1 and nodules_left = 1
        if self.advanced_encoding:
            X['nodules_right'] = X['nodules_right'] | X['nodules_both_sides']
            X['nodules_left'] = X['nodules_left'] | X['nodules_both_sides']
        # 
        self.all_columns = list(X.columns)
        return X
    
    def transform(self, X, y=None):
        for col, prefix in self.columns_for_one_hot_encoding.items():
            X = pd.get_dummies(X, columns=[col], prefix=prefix, dtype=np.int8)

        X = X.rename(columns=self.new_column_names_map)

        # more advanced encoding for nodules:
        # if nodules_both_sides = 1, then nodules_right = 1 and nodules_left = 1
        if self.advanced_encoding:
            X['nodules_right'] = X['nodules_right'] | X['nodules_both_sides']
            X['nodules_left'] = X['nodules_left'] | X['nodules_both_sides']
        # 
        self.all_columns = list(X.columns)
        return X
    
    def update_categorical_attributes_list(self, categorical_attributes):
        #  remove no longer existing columns
        for old_col in self.columns_for_one_hot_encoding.keys():
            if old_col in categorical_attributes:
                categorical_attributes.remove(old_col)

        # add new columns
        for new_col in self.new_column_names_map.values():
            if new_col not in categorical_attributes:
                categorical_attributes.append(new_col)

        return categorical_attributes
    
    def get_name(self):
        if self.advanced_encoding:
            return "advanced"
        else:
            return "basic"
        
    def get_feature_names_out(self):
        return self.all_columns
