import os
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..model_development.BiDirectional_LSTM.sliding_window_for_LSTM import LstmReshaper


class DataScaler:

    def __init__(self,
                 config: ConfigLoader,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 train_labels: pd.Series,
                 test_labels: pd.Series,
                 info_tracker: InfoTracker):
        self.__config = config
        self.__train_data = train_data
        self.__test_data = test_data
        self.__info_tracker = info_tracker

        self.__scaled_train_data: pd.DataFrame = pd.DataFrame()
        self.__scaled_test_data: pd.DataFrame = pd.DataFrame()

        self.__scale_train_test()
        self.__store_train_test_in_tracking()

        self.__scaled_train_data[config.df_features.labels] = train_labels
        self.__scaled_test_data[config.df_features.labels] = test_labels

    @property
    def config(self):
        return self.__config

    @property
    def info_tracker(self):
        return self.__info_tracker

    @property
    def scaled_train_data(self):
        return self.__scaled_train_data

    @property
    def scaled_test_data(self):
        return self.__scaled_test_data

    def __choose_scaler(self):
        """ Pick the right scaling method based on configuration. """
        scaling_method = self.config.scaling_method.method
        # Store scaling method in the info tracker.
        self.info_tracker.scaling_method = scaling_method

        # Pick a scaling method between Robust, MinMax and StandardScaler.
        if scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=self.config.scaling_method.minmax_range)
        else:
            scaler = StandardScaler()
        return scaler

    def __fit_scaler(self):
        """ Fit the train data in the selected scaler. """

        # Choose a scaler.
        scaler = self.__choose_scaler()

        # Fit the train data ONLY!!!
        scaler.fit(self.__train_data)

        # Prepare path to save scaler.
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
        scaled_train_data = scaler.transform(self.__train_data)
        scaled_test_data = scaler.transform(self.__test_data)

        # Convert to Pandas df with timestamps for better control and synchronisation.
        self.__scaled_train_data = pd.DataFrame(data=scaled_train_data, index=self.__train_data.index)
        self.__scaled_test_data = pd.DataFrame(data=scaled_test_data, index=self.__test_data.index)

    def __store_train_test_in_tracking(self):
        """ Move the original train and test data into the info tracker. """
        self.info_tracker.train_data = self.__train_data
        self.info_tracker.test_data = self.__test_data

    def reshape_data_for_modelling(self):
        return LstmReshaper(
            config=self.config,
            info_tracker=self.info_tracker,
            scaled_train_data=self.scaled_train_data,
            scaled_test_data=self.scaled_test_data
        )




