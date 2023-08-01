import pandas as pd
import numpy as np


class InfoTracker:

    def __init__(self):
        self.duplicated_values: int = None
        self.missing_values: dict = None
        self.train_labels: pd.Series = None
        self.test_labels: pd.Series = None
        self.scaling_method: str = None
        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.lstm_reshaped_test_labels: np.array = None
        self.lstm_reshaped_train_labels: np.array = None
