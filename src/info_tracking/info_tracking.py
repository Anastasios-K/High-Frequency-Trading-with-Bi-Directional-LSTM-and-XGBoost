import pandas as pd
import numpy as np


class InfoTracker:

    def __init__(self):
        self.__duplicated_values: int = None
        self.__missing_values: dict = None
        self.__scaling_method: str = None
        self.__train_data: pd.DataFrame = None
        self.__test_data: pd.DataFrame = None

    @property
    def duplicated_values(self):
        return self.__duplicated_values

    @duplicated_values.setter
    def duplicated_values(self, value: int):
        self.__duplicated_values = value

    @property
    def missing_values(self):
        return self.__missing_values

    @missing_values.setter
    def missing_values(self, value: dict):
        self.__missing_values = value

    @property
    def scaling_method(self):
        return self.__scaling_method

    @scaling_method.setter
    def scaling_method(self, value: str):
        self.__scaling_method = value

    @property
    def train_data(self):
        return self.__train_data

    @train_data.setter
    def train_data(self, value: pd.DataFrame):
        self.__train_data = value

    @property
    def test_data(self):
        return self.__test_data

    @test_data.setter
    def test_data(self, value: pd.DataFrame):
        self.__test_data = value
