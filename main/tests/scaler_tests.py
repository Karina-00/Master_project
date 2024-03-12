import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler

from main.components.scaler import Scaler
from main.constants import IGF, CONTINUOUS_ATTRIBUTES


def round_numeric_values(arr, decimals=5):
    # Create a mask to identify non-null numeric values
    mask = np.isnan(arr)
    # Round only the numeric values using the mask
    arr_rounded = np.where(mask, np.round(arr, decimals), arr)
    
    return arr_rounded


def get_continuous_attributes_except(attribute):
    remaining_attributes = CONTINUOUS_ATTRIBUTES.copy()
    remaining_attributes.remove(attribute)
    return remaining_attributes


def test_scaler__scales_properly(scaling_method=RobustScaler()):
    column_1 = 'thyroid volume'
    column_2 = ' HDL mg/dl'
    column_3 = 'prolactin'
    columns_to_test = [column_1, column_2, column_3]


    for column in columns_to_test:
        df_tmp = df.drop(columns=[IGF])

        # expected
        scaler = scaling_method
        expected_column = scaler.fit_transform(df_tmp[[column]])

        # result
        true_scaler = Scaler(scaling_method, attributes_to_scale=get_continuous_attributes_except(IGF))
        scaled_df = true_scaler.fit_transform(df_tmp)
        result_column = scaled_df[[column]].astype('float64').values

        expected_column_rounded = round_numeric_values(expected_column)
        result_column_rounded = round_numeric_values(result_column)

        # Create masks to exclude NaN values
        expected_mask = ~np.isnan(expected_column_rounded)
        result_mask = ~np.isnan(result_column_rounded)

        try:
            # Compare only non-NaN values
            assert np.array_equal(expected_column_rounded[expected_mask], result_column_rounded[result_mask])
            print("===PASSED===", column)
        except AssertionError:
            diff_indices = np.where(expected_column_rounded != result_column_rounded)[0]
            print("---FAILED---", column)
            print(f"Arrays differ in column {column} at indices: {diff_indices}")
    

def test_scaler__categorical_variables_remain_unchanged(scaling_method=RobustScaler()):
    column_1 = 'PCO 0-healthy control, 1-PCOS, 2-FHA 3-POF, 4-High Andro'
    column_2 = 'LDL>135'
    column_3 = 'Impaired Glucose Tolerance'

    columns_to_test = [column_1, column_2, column_3]

    for column in columns_to_test:
        df_tmp = df.drop(columns=[IGF])

        scaler = Scaler(scaling_method, attributes_to_scale=get_continuous_attributes_except(IGF))
        scaled_df = scaler.fit_transform(df_tmp)
        result_column = scaled_df[[column]].astype("Int8")

        try:
            assert result_column.equals(df_tmp[[column]])
            print("===PASSED===", column)
        except AssertionError:
            print("---FAILED---", column)



if __name__ == "__main__":
    dataset_file_path = 'data/preprocessed_dataset.csv'
    df = pd.read_csv(dataset_file_path)
    
    test_scaler__scales_properly(scaling_method=RobustScaler())
    test_scaler__scales_properly(scaling_method=PowerTransformer())
    test_scaler__categorical_variables_remain_unchanged(scaling_method=RobustScaler())
    test_scaler__categorical_variables_remain_unchanged(scaling_method=PowerTransformer())