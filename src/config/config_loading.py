from dataclasses import dataclass
from typing import List
import typing as t
from ..helper.helper import Helper


@dataclass
class Paths:
    datapath: str
    path2save_exploration: str

    @classmethod
    def read_config(cls: t.Type["Paths"], obj: dict):
        return cls(
            datapath=obj["paths"]["data_path"],
            path2save_exploration=obj["paths"]["paths2save"]["exploration"]
        )


@dataclass
class DataFeatures:
    date: str
    close: str
    open: str
    high: str
    low: str
    volume: str
    labels: str

    # diff_open_close: str
    # diff_high_low: str
    # avg_open_close: str
    # price_direction: str

    @classmethod
    def read_config(cls: t.Type["DataFeatures"], obj: dict):
        return cls(
            date=obj["data_fuatures_in_use"]["date"],
            close=obj["data_fuatures_in_use"]["close"],
            open=obj["data_fuatures_in_use"]["open"],
            high=obj["data_fuatures_in_use"]["high"],
            low=obj["data_fuatures_in_use"]["low"],
            volume=obj["data_fuatures_in_use"]["volume"],
            labels=obj["data_fuatures_in_use"]["labels"]

            # diff_open_close=obj["data_fuatures"]["created_by_candlesticks"]["diff_open_close"],
            # diff_high_low=obj["data_fuatures"]["created_by_candlesticks"]["diff_high_low"],
            # avg_open_close=obj["data_fuatures"]["created_by_candlesticks"]["avg_open_close"],
            # price_direction=obj["data_fuatures"]["created_by_candlesticks"]["price_direction"],
        )


@dataclass
class DataEngineering:
    fill_method: str
    poly_order: int

    @classmethod
    def read_config(cls: t.Type["DataEngineering"], obj: dict):
        return cls(
            fill_method=obj["data_engineering"]["fill_method"],
            poly_order=obj["data_engineering"]["poly_order"]
        )


@dataclass
class LabelTolerance:
    tolerance: int

    @classmethod
    def read_config(cls: t.Type["LabelTolerance"], obj: dict):
        return cls(
            tolerance=obj["label_tolerance"]["tolerance"]
        )


class ConfigLoader(object):

    def __init__(self, config_path):
        config_file = Helper.read_yaml_file(path=config_path)

        self.paths = Paths.read_config(obj=config_file)
        self.df_features = DataFeatures.read_config(obj=config_file)
        self.dataengin = DataEngineering.read_config(obj=config_file)
        self.labeltolerance = LabelTolerance.read_config(obj=config_file)
