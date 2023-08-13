from utils.yaml_reader import Config
from typing import List
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import os
import pickle

conf = Config()

class FeatureSelection:
    def __init__(self, filename:str) -> None:
        self.external_data_path = conf.params["data_preprocess"]["source_directory"]
        self.desination_data_path = conf.params["data_preprocess"]["destination_directory"]
        self.filename = filename
        self.seed = conf.params["base"]["seed"]
        self.target = conf.params["train"]["label"]
        self.mi_threshold = conf.params["train"]["mi_threshold"]
    
    def select_best_features(self) -> List[str]:
        df = pd.read_csv(os.path.join(self.desination_data_path, self.filename), engine = "pyarrow")
        X, y = df.drop(self.target, axis = 1), df[self.target]
        mi_scores = mutual_info_classif(X, y, random_state = self.seed)
        mi_scores = pd.Series(mi_scores, name = "MI-Scores", index = X.columns)
        mi_scores.sort_values(ascending = False)
        mi_scores = mi_scores[mi_scores>self.mi_threshold]
        return list(mi_scores.index)
    
if __name__ == '__main__':
    f_selector = FeatureSelection("pre_train_df.csv")
    best_features = f_selector.select_best_features()
    with open("./bin/best_features.pkl", "wb") as pkl:
        pickle.dump(best_features, pkl)