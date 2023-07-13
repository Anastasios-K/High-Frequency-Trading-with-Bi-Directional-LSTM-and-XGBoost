import pandas as pd
from ..config.config_loader import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.data_exploration import DataExplorator


class DataEngineer(object):

    def __init__(self, data: pd.DataFrame, config: ConfigLoader, info_tracker: InfoTracker):
        self.__config = config
        self.data = data
        self.info_tracker = info_tracker

        self.__remove_unused_data()
        self.__fix_data_type()
        self.__replace_missing_values()
        self.__index_and_sort_by_timestamps()
        self.__remove_duplicates()

    def __remove_unused_data(self):
        """ Remove unused data features and keep only the data features that are in interest """
        data = self.data
        config = self.__config

        # set up existing and desired data features
        df_features = data.columns
        desired_features = list(config.dfstructure.__dict__.values())

        # Convert the above elements into sets and take the difference
        features2drop = set(df_features).difference(desired_features)

        # Drop the features that are included into the difference set from the given dataset
        data.drop(columns=list(features2drop), inplace=True)

        self.data = data

    def __fix_data_type(self):
        """ Fix the data type of the data features. """
        data = self.data

        # for each feature in data
        for col in data.columns:
            # if feature is date
            if col == "date":
                val = pd.to_datetime(data[col])
                data[col] = val
            # if feature is other than date
            else:
                val = pd.to_numeric(data[col])
                data[col] = val
        print(data.dtypes)

    def __count_missing_values(self) -> dict:
        """ Count the missing values for each data feature and store them in a dictionary. """
        data = self.data

        nan_dict = {}

        for col in data.columns:
            nan_dict[col] = data[col].isna().sum()

        return nan_dict

    def __replace_missing_values(self):
        """ Replace the NaN values based on predetermined method, set in the configurations. """
        config = self.__config
        data = self.data

        # count missing values
        nan_amount = self.__count_missing_values()
        self.info_tracker.missing_values = nan_amount

        # if there is even one missing value
        if sum(nan_amount.values()) > 0:

            # if filling method is polynomial
            if config.dataengin.fill_method == "polynomial":
                [
                    data[col].interpolate(
                        method=config.dataengin.fill_method,
                        order=config.dataengin.poly_order,
                        direction="both",
                        inplace=True
                    )
                    for col
                    in data.columns
                ]

            # if filling method is linear
            elif config.dataengin.fill_method == "linear":
                [
                    data[col].interpolate(
                        method=config.dataengin.fill_method,
                        direction="both",
                        inplace=True
                    )
                    for col
                    in data.columns
                ]

            else:
                raise ValueError("An invalid fill method is given.")

        self.data = data

    def __index_and_sort_by_timestamps(self):
        """ Set timestamps as index and sort the data. """

        # Set date column as index
        # BUT date column is NOT removed - It is required for duplicate detection
        self.data.set_index(keys=self.__config.dfstructure.date, drop=False, inplace=True)
        # sort data
        self.data.sort_index(ascending=True, inplace=True)

    def __count_duplicates(self) -> int:
        """ Count duplicated rows using timestamps. """
        config = self.__config
        data = self.data

        dupli_amount = data[config.dfstructure.date].duplicated(False).sum()
        return dupli_amount

    def __remove_duplicates(self):
        """ Remove duplicates identified based on the Date feature. Drops the Date feature at the end. """
        config = self.__config
        data = self.data

        # Count duplicated rows
        dupl_amount = self.__count_duplicates()
        self.info_tracker.duplicated_values = dupl_amount

        # make sure that timestamp is set as index before removing duplicated rows
        data[config.dfstructure.date] = data.index

        # drop duplicated rows and keep the first valid row
        data.drop_duplicates(
            subset=config.dfstructure.date,
            keep="first",
            inplace=True
        )
        # drop the date column as it is not used anymore
        data.drop(
            columns=[config.dfstructure.date],
            inplace=True
        )
        self.data = data

    def data_exploration(self):
        return DataExplorator(
            data=self.data,
            config=self.__config,
            info_tracker=self.info_tracker
        )
