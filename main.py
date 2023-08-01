from src.config.config_loading import ConfigLoader
from src.data_loading.data_loading import DataLoader


class RunHFTproject:
    def __init__(self, config_path):
        config = ConfigLoader(config_path)
        self.run = DataLoader(config=config)\
            .data_engineering()\
            .data_exploration()\
            .label_creation()\
            .split_data_in_train_test()\
            .scale_data()\
            .reshape_data_for_modelling()


if __name__ == "__main__":
    CONFIG_PATH = "src\\config\\config.yaml"
    run = RunHFTproject(config_path=CONFIG_PATH)
