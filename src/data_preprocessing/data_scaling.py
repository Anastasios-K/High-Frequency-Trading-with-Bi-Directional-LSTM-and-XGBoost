import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker


class DataScaler:

    def __init__(self,
                 config: ConfigLoader,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 info_tracker: InfoTracker):
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.info_tracker = info_tracker

        self.scaled_train: np.array
        self.scaled_test: np.array

        self.__scale_train_test()
        self.__store_train_test_in_tracking()

    def __choose_scaler(self):
        """ Pick the right scaling method based on configuration. """
        scaling_method = self.config.scaling_method.method
        self.info_tracker.scaling_method = scaling_method

        # Pick a scaling method between Robust, MinMax and StandardScaler
        if scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=self.config.scaling_method.minmax_range)
        else:
            scaler = StandardScaler()
        return scaler

    def __fit_scaler(self):
        """ Fit the train data in the selected scaler. """

        # Choose a scaler
        scaler = self.__choose_scaler()

        # Fit the train data ONLY!!!
        scaler.fit(self.train_data)

        # Prepare path to save scaler
        scaler_path = os.path.join(
            self.config.paths.path2save_models,
            self.config.model.name
        )
        os.makedirs(scaler_path, exist_ok=True)

        # Save the fitted scaler
        joblib.dump(scaler, os.path.join(scaler_path, f"{self.config.scaling_method.method}_scaler.pkl"))
        return scaler

    def __scale_train_test(self):
        """ Scale the train and test data. """

        # Get the fitted scaler
        scaler = self.__fit_scaler()

        # Use the fitted scaled to transform the train and test data
        self.scaled_train = scaler.transform(self.train_data)
        self.scaled_test = scaler.transform(self.test_data)

    def __store_train_test_in_tracking(self):
        """ Move the original train and test data into the info tracker. """
        self.info_tracker.train_data = self.train_data
        self.info_tracker.test_data = self.test_data




