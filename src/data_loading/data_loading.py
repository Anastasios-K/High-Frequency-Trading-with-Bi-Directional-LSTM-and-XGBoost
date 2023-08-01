import pandas as pd
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.s1_data_engineering import DataEngineer


class DataLoader(object):

    def __init__(self, config: ConfigLoader):
        self.__config = config
        self.__data: pd.DataFrame = pd.read_csv(self.__config.data_link.link)
        self.__info_tracker = InfoTracker()

    @property
    def config(self):
        return self.__config

    @property
    def data(self):
        return self.__data

    @property
    def info_tracker(self):
        return self.__info_tracker

    def data_engineering(self):
        return DataEngineer(
            data=self.__data,
            config=self.__config,
            info_tracker=self.__info_tracker
        )
