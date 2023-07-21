import pandas as pd
import numpy as np


class InfoTracker:

    def __init__(self):
        self.duplicated_values: int = np.nan
        self.missing_values: dict = dict()
        self.labels: pd.Series = pd.Series()
        self.train_labels: pd.Series = pd.Series()
        self.test_labels: pd.Series = pd.Series()
        self.scaling_method: str = ""
        self.train_data: pd.DataFrame = pd.DataFrame()
        self.test_data: pd.DataFrame = pd.DataFrame()
