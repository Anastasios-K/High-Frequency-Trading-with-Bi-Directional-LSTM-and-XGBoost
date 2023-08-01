import pandas as pd
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.s4_data_splitting import TrainTestSplitter


class LabelCreator:

    def __init__(self,
                 data: pd.DataFrame,
                 config: ConfigLoader,
                 info_tracker: InfoTracker):
        self.__data = data
        self.__config = config
        self.__info_tracker = info_tracker

        self.__shifted_col: str = "Shifted"
        self.__diff_col: str = "Diff"

        self.__create_labels()

    @property
    def config(self):
        return self.__config

    @property
    def data(self):
        return self.__data

    @property
    def info_tracker(self):
        return self.__info_tracker

    def __calc_price_difference(self) -> pd.DataFrame:
        """
        Create a new feature filled with the shifted Close prices by 1.
        Use the shifted and original Close features to calculate the percent difference.
        Add the 2 new features (shifted close and difference %) in the given dataset.
        """
        temp_df = self.data.copy()
        dff = self.config.df_features

        # Shift Close prices by 1 step
        temp_df[self.__shifted_col] = temp_df[dff.close].shift(1)

        # Calculate the precentage difference of Close price between the original and shifted timestamps
        temp_df[self.__diff_col] = (temp_df[dff.close] / temp_df[self.__shifted_col]) - 1
        return temp_df

    def __create_labels(self) -> None:
        """
        Call the _calc_price_difference to get the 2 new features (shifted close and % difference).
        Create 3 classes based on the 3 conditions below and put them into the label feature.
        To create the classes also consider the tollerance factor (set it up in config).
        Drop the unused features (shifted close and difference).
        Drop the row with Nan values generated because of the shifting action.
        """
        config = self.config
        label_col = config.df_features.labels

        # data with shifted prices and price difference features
        data_with_shift_n_diff = self.__calc_price_difference()
        df = data_with_shift_n_diff

        # tollerance factor
        tollerance = config.labeltolerance.tollerance

        # condition 1 - BUY if % diff is higher than the tollerance threshold
        df.loc[df[self.__diff_col] > tollerance, label_col] = 1

        # condition 2 - SELL if % diff is lower than the tollerance threshold
        df.loc[df[self.__diff_col] < -tollerance, label_col] = 0

        # condition 3 - DO NOTHING if % diff is neither lower nor higher than the tollerance threshold
        df.loc[(df[self.__diff_col] < tollerance) &
               (df[self.__diff_col] > -tollerance), label_col] = 2

        # drop Nan values as well as price difference and shifted columns
        df.dropna(inplace=True)

        self.__data = df.drop(columns=[self.__diff_col, self.__shifted_col])

    def split_data_in_train_test(self) -> TrainTestSplitter:
        return TrainTestSplitter(
            config=self.config,
            data=self.data,
            info_tracker=self.info_tracker
        )
