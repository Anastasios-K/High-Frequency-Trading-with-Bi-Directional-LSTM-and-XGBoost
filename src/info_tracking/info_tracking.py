import numpy as np


class InfoTracker:

    def __init__(self):
        self.duplicated_values: int = np.nan
        self.missing_values: dict = {}
