import pandas as pd
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.data_engineering import DataEngineer


class DataLoader(object):

    def __init__(self, config: ConfigLoader):
        self.__config = config
        self.data: pd.DataFrame = pd.read_csv(self.__config.data_link.link)
        self.info_tracker = InfoTracker()

    def data_engineering(self):
        return DataEngineer(
            data=self.data,
            config=self.__config,
            info_tracker=self.info_tracker
        )
