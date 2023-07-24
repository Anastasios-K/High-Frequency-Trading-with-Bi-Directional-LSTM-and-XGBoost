from dataclasses import dataclass
from typing import List
import typing as t
from ..helper.helper import Helper


@dataclass
class DataLink:
    link: str

    @classmethod
    def read_config(cls: t.Type["DataLink"], obj: dict):
        return cls(
            link=obj["data_link"]
        )


@dataclass
class Paths:
    path2save_exploration: str
    path2save_data: str
    path2save_models: str

    @classmethod
    def read_config(cls: t.Type["Paths"], obj: dict):
        return cls(
            path2save_exploration=obj["paths2save"]["exploration"],
            path2save_data=obj["paths2save"]["data"],
            path2save_models=obj["paths2save"]["models"]
        )


@dataclass
class Model:
    model: str
    name: str

    @classmethod
    def read_config(cls: t.Type["Model"], obj: dict):
        return cls(
            model=obj["model"]["model"],
            name=obj["model"]["name"]
        )


@dataclass
class DataFeatures:
    date: str
    close: str
    open: str
    high: str
    low: str
    labels: str

    @classmethod
    def read_config(cls: t.Type["DataFeatures"], obj: dict):
        return cls(
            date=obj["data_fuatures_in_use"]["date"],
            close=obj["data_fuatures_in_use"]["close"],
            open=obj["data_fuatures_in_use"]["open"],
            high=obj["data_fuatures_in_use"]["high"],
            low=obj["data_fuatures_in_use"]["low"],
            labels=obj["data_fuatures_in_use"]["labels"]
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
    tollerance: int

    @classmethod
    def read_config(cls: t.Type["LabelTolerance"], obj: dict):
        return cls(
            tollerance=obj["label_tolerance"]["tolerance"]
        )


@dataclass
class ScalingMethod:
    method: str
    minmax_range: int

    @classmethod
    def read_config(cls: t.Type["ScalingMethod"], obj: dict):
        return cls(
            method=obj["scaling_method"],
            minmax_range=obj["min_max_scaler_range"]
        )


class ConfigLoader(object):

    def __init__(self, config_path):
        config_file = Helper.read_yaml_file(path=config_path)

        self.data_link = DataLink.read_config(obj=config_file)
        self.paths = Paths.read_config(obj=config_file)
        self.model = Model.read_config(obj=config_file)
        self.df_features = DataFeatures.read_config(obj=config_file)
        self.dataengin = DataEngineering.read_config(obj=config_file)
        self.labeltolerance = LabelTolerance.read_config(obj=config_file)
        self.scaling_method = ScalingMethod.read_config(obj=config_file)
