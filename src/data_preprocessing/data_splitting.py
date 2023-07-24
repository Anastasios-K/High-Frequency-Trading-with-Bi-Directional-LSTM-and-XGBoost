import pandas as pd
from sklearn.model_selection import train_test_split
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker

from ..data_preprocessing.data_scaling import DataScaler


class TrainTestSplitter:

    def __init__(self, data: pd.DataFrame, config: ConfigLoader, info_tracker: InfoTracker):
        self.__data = data
        self.config = config
        self.info_tracker = info_tracker

        self.train_data: pd.DataFrame = pd.DataFrame()
        self.test_data: pd.DataFrame = pd.DataFrame()

        self.split_data_into_train_n_test()

    def split_data_into_train_n_test(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.__data,
            self.info_tracker.labels,
            test_size=0.3,
            shuffle=False
        )
        self.train_data = x_train
        self.test_data = x_test
        self.info_tracker.train_labels = y_train
        self.info_tracker.test_labels = y_test

    def scale_data(self):
        return DataScaler(
            config=self.config,
            train_data=self.train_data,
            test_data=self.test_data,
            info_tracker=self.info_tracker
        )
