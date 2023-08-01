import pandas as pd
from sklearn.model_selection import train_test_split
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker

from ..data_preprocessing.s5_data_scaling import DataScaler


class TrainTestSplitter:

    def __init__(self,
                 data: pd.DataFrame,
                 config: ConfigLoader,
                 info_tracker: InfoTracker):
        self.__data = data
        self.__config = config
        self.__info_tracker = info_tracker

        self.__train_data: pd.DataFrame = pd.DataFrame()
        self.__test_data: pd.DataFrame = pd.DataFrame()
        self.__train_labels: pd.Series = pd.Series()
        self.__test_labels: pd.Series = pd.Series()

        self.__split_data_into_train_n_test()

    @property
    def config(self):
        return self.__config

    @property
    def info_tracker(self):
        return self.__info_tracker

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def train_labels(self):
        return self.__train_labels

    @property
    def test_labels(self):
        return self.__test_labels

    def __separate_data_n_labels(self) -> (pd.DataFrame, pd.Series):
        """ Separate data and labels. Also drop labels from the data dataframe. """

        labels = self.__data[self.__config.df_features.labels]
        data = self.__data.copy().drop(columns=[self.__config.df_features.labels])
        return data, labels

    def __split_data_into_train_n_test(self) -> None:
        """ Split data and labels in training and test sets. """

        data, labels = self.__separate_data_n_labels()

        x_train, x_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.3,
            shuffle=False
        )
        self.__train_data = x_train
        self.__test_data = x_test
        self.__train_labels = y_train
        self.__test_labels = y_test

    def scale_data(self) -> DataScaler:
        return DataScaler(
            config=self.config,
            train_data=self.train_data,
            test_data=self.test_data,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            info_tracker=self.info_tracker
        )
