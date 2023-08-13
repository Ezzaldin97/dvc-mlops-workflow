from utils.yaml_reader import Config
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

conf = Config()

class DataPreparation:
    def __init__(self, filename:str) -> None:
        self.external_data_path = conf.params["data_preprocess"]["source_directory"]
        self.desination_data_path = conf.params["data_preprocess"]["destination_directory"]
        self.filename = filename
        self.seed = conf.params["base"]["seed"]
        self.mi_threshold = conf.params["train"]["mi_threshold"]
    
    @staticmethod
    def reduce_memory_usage(df:pd.DataFrame) -> pd.DataFrame:
        numerics = ['int8', 'int16', 'int32', 'int64', 'float8','float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in tqdm(df.columns):
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if (c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max) or (c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max):
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if (c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max) or (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        return df
    
    def data_reader(self) -> pd.DataFrame:
        df = DataPreparation.reduce_memory_usage(pd.read_csv(os.path.join(self.desination_data_path, self.filename), engine = "pyarrow"))
        df.drop_duplicates(inplace = True)
        return df
    
    def feature_engineering(self, df:pd.DataFrame) -> pd.DataFrame:
        df["cpu_interaction"] = df["clock_speed"] * df["n_cores"]
        df["camera_interaction"] = df["pc"] + df["fc"]
        df["px_interaction"] = df["px_height"]*df["px_width"]
        df["sc_interaction"] = df["sc_h"]*df["sc_w"]
        df["g_interaction"] = df["three_g"] + df["four_g"]
        df["capabilities_interaction"] = df["wifi"]+df["touch_screen"]+df["blue"]+df["dual_sim"]+df["g_interaction"]
        return df
    
if __name__ == '__main__':
    train_prepare = DataPreparation("train_df.csv")
    test_prepare = DataPreparation("test_df.csv")
    train, test = train_prepare.data_reader(), test_prepare.data_reader()
    train, test = train_prepare.feature_engineering(train), test_prepare.feature_engineering(test)
    train.to_csv(os.path.join(train_prepare.desination_data_path, "pre_train_df.csv"), index = False)
    test.to_csv(os.path.join(test_prepare.desination_data_path, "pre_test_df.csv"), index = False)
    

    
