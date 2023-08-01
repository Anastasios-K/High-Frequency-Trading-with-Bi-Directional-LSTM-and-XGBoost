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


@dataclass
class LstmGeneralParams:
    window_length: int

    @classmethod
    def read_config(cls: t.Type["LstmGeneralParams"], obj: dict):
        return cls(
            window_length=obj["BiLSTM"]["General_params"]["window_length"]
        )


@dataclass
class LstmHyperParams:
    lstm_units_min: int
    lstm_units_max: int
    lstm_units_step: int
    dense_units_min: int
    dense_units_max: int
    dense_units_step: int
    lr_min: float
    lr_max: float
    lr_step: int
    drop_out_min: float
    drop_out_max: float
    drop_out_step: int

    @classmethod
    def read_config(cls: t.Type["LstmHyperParams"], obj: dict):
        return cls(
            lstm_units_min=obj["BiLSTM"]["Hyper_params"]["lstm_units_min"],
            lstm_units_max=obj["BiLSTM"]["Hyper_params"]["lstm_units_max"],
            lstm_units_step=obj["BiLSTM"]["Hyper_params"]["lstm_units_step"],
            dense_units_min=obj["BiLSTM"]["Hyper_params"]["dense_units_min"],
            dense_units_max=obj["BiLSTM"]["Hyper_params"]["dense_units_max"],
            dense_units_step=obj["BiLSTM"]["Hyper_params"]["dense_units_step"],
            lr_min=obj["BiLSTM"]["Hyper_params"]["lr_min"],
            lr_max=obj["BiLSTM"]["Hyper_params"]["lr_max"],
            lr_step=obj["BiLSTM"]["Hyper_params"]["lr_step"],
            drop_out_min=obj["BiLSTM"]["Hyper_params"]["drop_out_min"],
            drop_out_max=obj["BiLSTM"]["Hyper_params"]["drop_out_max"],
            drop_out_step=obj["BiLSTM"]["Hyper_params"]["drop_out_step"]
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
        self.lstm_general_params = LstmGeneralParams.read_config(obj=config_file)
        self.lstm_hyper_params = LstmHyperParams.read_config(obj=config_file)

