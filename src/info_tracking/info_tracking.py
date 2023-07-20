import numpy as np
import pandas as pd


class InfoTracker:

    def __init__(self):
        self.duplicated_values: int = np.nan
        self.missing_values: dict = dict()
        self.labels: pd.Series = pd.Series()
        self.train_labels: pd.Series = pd.Series()
        self.test_labels: pd.Series = pd.Series()
