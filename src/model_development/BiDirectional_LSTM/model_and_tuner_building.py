import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Flatten
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from keras_tuner.tuners import RandomSearch
from ...config.config_loading import ConfigLoader
from ...info_tracking.info_tracking import InfoTracker


class BiLstmBuilder:

    def __init__(
            self,
            config: ConfigLoader,
            info_tracker: InfoTracker,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            train_labels: pd.Series,
            test_labels: pd.Series
    ):
        self.__config = config
        self.__info_tracker = info_tracker
        self.__train_data = train_data
        self.__test_data = test_data
        self.__train_labels = train_labels
        self.__test_labels = test_labels

        # self._reshape_data()
        self.keras_hypermodel = self._build_hypermodel()

    @property
    def config(self):
        return self.__config

    @property
    def info_tracker(self):
        return self.__info_tracker

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def train_labels(self):
        return self.__train_labels

    @property
    def test_labels(self):
        return self.__test_labels

    def _build_model(self, hp) -> None:
        """
        Build Tensorfow Bi-Directional LSTM model.
        The hp parameter is used by Keras Tuner ONLY.
            The whole method is passed to Keras Tuner object in the fumction build_hypermodel.
        The mothod initialises all the hyper parameters which are set in the configuration file "config.yaml".
        The method also incorporates the loss fucntion, optimiser and metrics.
        """
        # Load general and hyper parameters.
        gparams = self.__config.lstm_general_params
        hparams = self.__config.lstm_hyper_params

        # Set initialisation.
        kernel_init = GlorotUniform(seed=gparams.seed)
        recurrent_init = Orthogonal(seed=gparams.seed)
        bias_init = Zeros()

        # Initialise the LSTM model.
        model = Sequential()

        # Prepare the Hyper-parameters.
        # LSTM number of units
        lstm_units = hp.Int(
            "lstm_units",
            min_value=hparams.lstm_units_min,
            max_value=hparams.lstm_units_max,
            step=hparams.lstm_units_step
        )
        # Dense number of units
        dense_units = hp.Int(
            "dense_units",
            min_value=hparams.dense_units_min,
            max_value=hparams.dense_units_max,
            step=hparams.dense_units_step
        )
        # Drop-out value.
        drop_out_values = hp.Choice(
            "drop out values",
            values=np.linspace(
                start=hparams.drop_out_min,
                stop=hparams.drop_out_max,
                num=hparams.drop_out_step,
                dtype=float).tolist()
        )
        # Learning rate value.
        lr_values = hp.Choice(
            "learning rate values",
            values=np.linspace(
                start=hparams.lr_min,
                stop=hparams.lr_max,
                num=hparams.lr_step,
                dtype=float).tolist()
        )
        # Add one LSTM layer into the model.
        model.add(
            Bidirectional(
                LSTM(
                    name="LSTM_1",
                    units=lstm_units,
                    return_sequences=False,
                    input_shape=self.train_data.shape[1:],
                    activation=gparams.lstm_activation_function,
                    recurrent_activation=gparams.recurrent_function,
                    kernel_initializer=kernel_init,
                    recurrent_initializer=recurrent_init
                ),
                merge_mode="concat"
            )
        )
        # Add Drop-out in the model.
        model.add(Dropout(
            name="DropOut_1",
            rate=drop_out_values,
            seed=gparams.seed
        ))
        # Add the first Dense layer.
        model.add(Dense(
            name="Dense_1",
            units=dense_units,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            activation=gparams.dense_activation_function
        ))
        # Add the second Dense layer.
        model.add(Dense(
            name="Dense_2",
            units=gparams.number_of_classes,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            activation=gparams.classification_activation_function
        ))
        # Compile the model.
        model.compile(
            optimizer=Adam(learning_rate=lr_values),
            loss=CategoricalCrossentropy(),
            metrics=[
                AUC(name="AUC", curve="PR"),
                Precision(name="precision"),
                Recall(name="recall")
            ]
        )
        return model

    def _count_max_trials(self) -> int:
        """ Count the maximum possible combinations given the hyper parameters."""
        hparams = self.__config.lstm_hyper_params
        count_lstm = np.arange(
            start=hparams.lstm_units_min,
            stop=hparams.lstm_units_max + 1,
            step=hparams.lstm_units_step
        )
        count_dense = np.arange(
            start=hparams.dense_units_min,
            stop=hparams.dense_units_max + 1,
            step=hparams.dense_units_step
        )
        count_drop_out = np.linspace(
            start=hparams.drop_out_min,
            stop=hparams.drop_out_max,
            num=hparams.drop_out_step,
            endpoint=True
        )
        count_lr_values = np.linspace(
            start=hparams.lr_min,
            stop=hparams.lr_max,
            num=hparams.lr_step,
            endpoint=True
        )
        trials = (len(count_lstm) *
                  len(count_dense) *
                  len(count_drop_out) *
                  len(count_lr_values))
        print(f"The maximum trials are: {trials}")
        return trials

    def _build_hypermodel(self) -> RandomSearch:
        """ Initialise Keras Tuner. """
        tuner = RandomSearch(
            hypermodel=self._build_model,
            objective="val_loss",
            max_trials=self._count_max_trials(),
            project_name=self.__config.model.name,
            overwrite=True,
            directory=os.path.join(
                *self.__config.paths.path2save_models
            ),
            seed=self.__config.lstm_general_params.seed
        )
        tuner.search_space_summary(extended=False)
        return tuner

