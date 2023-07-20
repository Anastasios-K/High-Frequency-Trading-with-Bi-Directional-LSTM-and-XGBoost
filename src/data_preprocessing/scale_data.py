import pandas as pd
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