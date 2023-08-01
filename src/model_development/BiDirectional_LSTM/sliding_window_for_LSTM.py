import numpy as np
import pandas as pd
from ...config.config_loading import ConfigLoader
from ...info_tracking.info_tracking import InfoTracker


class LstmReshaper:

    def __init__(self,
                 config: ConfigLoader,
                 info_tracker: InfoTracker,
                 scaled_train_data: np.array,
                 scaled_test_data: np.array):
        self.__config = config
        self.__info_tracker = info_tracker
        self.__scaled_train_data = scaled_train_data
        self.__scaled_test_data = scaled_test_data

        self.__reshaped_train_data: np.array = np.array([])
        self.__reshaped_test_data: np.array = np.array([])
        self.__reshaped_train_labels: np.array = np.array([])
        self.__reshaped_test_labels: np.array = np.array([])

        self.__apply_sw_to_train_n_test()

    @property
    def config(self):
        return self.__config

    @property
    def info_tracker(self):
        return self.__info_tracker

    @property
    def reshaped_train_data(self):
        return self.__reshaped_train_data

    @property
    def reshaped_test_data(self):
        return self.__reshaped_test_data

    @property
    def reshaped_train_labels(self):
        return self.__reshaped_train_labels

    @property
    def reshaped_test_labels(self):
        return self.__reshaped_test_labels

    def __sliding_window_process(self, data: pd.DataFrame) -> (np.array, np.array):
        """ Apply Sliding Window to the data, creating data batches and reshaping data. """

        # Load the Sliding Window length.
        window_length = self.config.lstm_general_params.window_length
        # Create empty lists to store the temporary array and label that correspond to each data batch.
        data_list = []
        label_list = []

        # In the for loop below:
        # Move down into the data by one row and capture a data sub-part that
        # starts from the current row index (i) and
        # ends at the row that is equal to the sliding window length + the current row index (i)

        # Iterate over the whole ramge minus the window length,
        # because the last arrays have less rows than the window length.
        for i in range(len(data) - window_length):

            # Extract data with labels, based on the sliding window length.
            temp_data = data.iloc[i:window_length + i, :]
            # Convert dataframe to array.
            temp_array = np.array(temp_data)
            # Extract the corresponding label.
            # We have a 2D array here - The corresponding label is at the right bottom corner.
            # Timewise, this is the right label as sychronisation is conducted in "LabelCreator" object.
            label = temp_array[-1, -1]
            # Labels are removed from the data.
            temp_array_no_labels = temp_array[:, :-1]

            data_list.append(temp_array_no_labels)
            label_list.append(label)

        final_data = np.array(data_list)
        final_labels = np.array(label_list)
        return final_data, final_labels

    def __apply_sw_to_train_n_test(self) -> None:
        """ Apply the sliding window to the train and test data. """
        self.__reshaped_train_data, \
            self.__reshaped_train_labels = self.__sliding_window_process(data=self.__scaled_train_data)

        self.__reshaped_test_data, \
            self.__reshaped_test_labels = self.__sliding_window_process(data=self.__scaled_test_data)





