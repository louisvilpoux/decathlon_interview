import pandas as pd
import os
import logger

class Reader():
    def __init__(self, path_folder_data, logger):
        self.path_folder_data = path_folder_data
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.logger = logger
        self.fill_dataframes()
    
    def generate_df_train(self):
        data_loaded = False
        for file_name in os.listdir(self.path_folder_data):
            if file_name == "train.csv":
                full_path_file = os.path.join(self.path_folder_data, file_name)
                self.df_train = pd.read_csv(full_path_file)
                data_loaded = True
        return data_loaded

    def generate_df_test(self):
        data_loaded = False
        for file_name in os.listdir(self.path_folder_data):
            if file_name == "test.csv":
                full_path_file = os.path.join(self.path_folder_data, file_name)
                self.df_test = pd.read_csv(full_path_file)
                data_loaded = True
        return data_loaded

    def fill_dataframes(self):
        if os.path.isdir(self.path_folder_data):
            loaded_train_data = self.generate_df_train()
            if not loaded_train_data:
                self.logger.logger.error(f"Train data cannot be loaded. End of the program.")
            loaded_test_data = self.generate_df_test()
            if not loaded_test_data:
                self.logger.logger.error(f"Test data cannot be loaded. End of the program.")
        else:
            self.logger.logger.error(f"{self.path_folder_data} is not a directory. End of the program.")


if __name__ == "__main__":
    reader = Reader("/home/louis/projects/perso/decathlon/test_data_scientist",logger)