from utils.yaml_reader import Config
from typing import Union
from sklearn.model_selection import train_test_split
import pandas as pd
import os

conf = Config()

class DataReader:
    def __init__(self, filename:str) -> None:
        self.external_data_path = conf.params["data_preprocess"]["source_directory"]
        self.desination_data_path = conf.params["data_preprocess"]["destination_directory"]
        self.filename = filename
        self.seed = conf.params["base"]["seed"]
        self.test_size = conf.params["train"]["test_size"]

    def reader(self) -> pd.DataFrame:
        file_format = self.filename.split(".")[-1]
        try:
            if file_format == "csv":
                df = pd.read_csv(os.path.join(self.external_data_path, self.filename), engine = "pyarrow")
            elif file_format == "parquet":
                df = pd.read_parquet(os.path.join(self.external_data_path, self.filename), engine = "pyarrow")
            else:
                df = pd.read_json(os.path.join(self.external_data_path, self.filename))
            return df
        except Exception as exp:
            print(f"error due to : {exp}")
    
    def split_dataset(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(data, random_state=self.seed, test_size=self.test_size)
        return train, test
    
if __name__ == '__main__':
    df_reader = DataReader("train.csv")
    df = df_reader.reader()
    train, test = df_reader.split_dataset(df)
    train.to_csv(os.path.join(df_reader.desination_data_path, "train_df.csv"))
    test.to_csv(os.path.join(df_reader.desination_data_path, "test_df.csv"))

