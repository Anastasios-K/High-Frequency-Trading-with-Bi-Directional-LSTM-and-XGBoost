import os.path

import pandas as pd
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.data_splitting import TrainTestSplitter


class LabelCreator(object):

    def __init__(self, data: pd.DataFrame, config: ConfigLoader, info_tracker: InfoTracker):
        self.data = data
        self.config = config
        self.info_tracker = info_tracker

        self.__shifted_col: str = "Shifted"
        self.__diff_col: str = "Diff"

        self.__create_labels()

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

    def __create_labels(self) -> (pd.DataFrame, pd.Series):
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

        # save preprocessed data and labels - To be able to access and use it without executing preprocessing again
        df.to_csv(os.path.join(self.config.paths.path2save_data, "preprocessed_data.csv"),
                  index=False, index_label=True)

        self.info_tracker.labels = df[label_col]
        self.data = df.drop(columns=[self.__diff_col, self.__shifted_col, label_col])

    # @classmethod
    # def create_label_weights(cls, train_labels: np.ndarray) -> dict:
    #     """
    #     Calculate class weights based on each class population.
    #     Crucial step when the training data is imbalanced.
    #     """
    #     label_weights = {}
    #     # capture the 2nd array dimenssion (the number of columns in pd.DataFrame)
    #     lbs_amount = len(train_labels.unique())
    #
    #     for idx in range(lbs_amount):
    #         weight_dict = {
    #             idx: (1 / np.count_nonzero(train_labels == idx)) * (len(train_labels)) / lbs_amount
    #         }
    #         label_weights.update(weight_dict)
    #     return label_weights

    def split_data_in_train_test(self):
        return TrainTestSplitter(
            config=self.config,
            data=self.data,
            info_tracker=self.info_tracker
        )
