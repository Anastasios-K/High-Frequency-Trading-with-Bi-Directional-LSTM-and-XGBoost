from dataclasses import dataclass
from typing import List
import typing as t
from ..helper.helper import Helper


@dataclass
class Paths:
    datapath: str

    @classmethod
    def read_config(cls: t.Type["Paths"], obj: dict):
        return cls(
            datapath=obj["paths"]["data_path"]
        )


@dataclass
class FeaturesInUse:
    date: str
    close: str
    open: str
    high: str
    low: str
    volume: str

    @classmethod
    def read_config(cls: t.Type["FeaturesInUse"], obj: dict):
        return cls(
            date=obj["data_fuatures_in_use"]["date"],
            close=obj["data_fuatures_in_use"]["close"],
            open=obj["data_fuatures_in_use"]["open"],
            high=obj["data_fuatures_in_use"]["high"],
            low=obj["data_fuatures_in_use"]["low"],
            volume=obj["data_fuatures_in_use"]["volume"],
        )


@dataclass
class DataEngin:
    fill_method: str
    poly_order: int

    @classmethod
    def read_config(cls: t.Type["DataEngin"], obj: dict):
        return cls(
            fill_method=obj["data_engineering"]["fill_method"],
            poly_order=obj["data_engineering"]["poly_order"]
        )


class ConfigLoader(object):

    def __init__(self, config_path):
        config_file = Helper.read_yaml_file(path=config_path)

        self.paths = Paths.read_config(obj=config_file)
        self.dfstructure = FeaturesInUse.read_config(obj=config_file)
        self.dataengin = DataEngin.read_config(obj=config_file)
