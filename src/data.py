import numpy as np
import ray
from ray.data.preprocessor import Preprocessor
import pandas as pd


def get_x_y_split(data, columns, targets, n_steps_in, n_steps_out, gap, include_target_in_X = False):
    """This function converts a dataframe into X and Y sequences for training"""
    # columns = [column for column in data.columns if column not in targets]

    # Include target column
    if include_target_in_X:
        columns = columns + targets

    complete_x_array = data[columns].to_numpy()
    complete_y_array = data[targets].to_numpy()

    X_arrays = []
    y_arrays = []

    upper_bound = len(data) - (n_steps_in + n_steps_out + gap)

    # Loop through the entire dataset
    for index in range(0, upper_bound):
        # Based on parameters construct the training data made up of 
        # smaller arrays
        starting_X_index = index
        ending_X_index = starting_X_index + n_steps_in # number of features

        starting_y_index = ending_X_index + gap
        ending_y_index = starting_y_index + n_steps_out

        individual_x_array = complete_x_array[starting_X_index: ending_X_index, :]
        individual_y_array = complete_y_array[starting_y_index: ending_y_index, :]

        X_arrays.append(individual_x_array)
        y_arrays.append(individual_y_array)

    # Convert a list of arrays into an array
    X_array = np.array(X_arrays)
    y_array = np.array(y_arrays)

    # Flatten the array in case it's 3 dimensional
    if len(y_array.shape) == 3:
        y_array = np.array([individual_array.flatten() for individual_array in y_array])
    
    return X_array, y_array


def preprocess(data, columns, targets, n_steps_in, n_steps_out, gap, include_target_in_X=False, resample_units=None):
    
    X, y = get_x_y_split(
        data, 
        columns=columns, 
        targets=targets, 
        n_steps_in=n_steps_in, 
        n_steps_out=n_steps_out, 
        gap=gap, 
        include_target_in_X=include_target_in_X
    )
    
    return {"X": X, "y": y}


def load_data(dataset_location):
    return pd.read_csv(dataset_location)

def split_dataset(data):
    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

def to_ray_dataset(data):
    return ray.data.from_pandas(data)

class CustomPreprocessor(Preprocessor):
    def _fit(self, ds):
        return self

    def _transform_pandas(self, batch):
        return preprocess(
            batch,
            # resample_units="10T", # Resample values by 60 minutes
            columns=['price'],
            targets=['price'], 
            n_steps_in=5, 
            n_steps_out=10, 
            gap=60, 
            include_target_in_X=True
        )