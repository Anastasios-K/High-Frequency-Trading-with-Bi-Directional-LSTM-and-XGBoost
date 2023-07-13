from src.config.config_loader import ConfigLoader
from src.data_loading.data_loading import DataLoader


class RunHFTproject:
    def __init__(self, config_path):
        config = ConfigLoader(config_path)
        self.run = DataLoader(config=config)\
            .data_engineering()\
            .data_exploration()


if __name__ == "__main__":
    CONFIG_PATH = "src\\config\\config.yaml"
    run = RunHFTproject(config_path=CONFIG_PATH)
