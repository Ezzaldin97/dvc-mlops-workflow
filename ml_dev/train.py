from utils.yaml_reader import Config
from typing import Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pandas as pd
import os
import pickle

conf = Config()

class AutoTrain:
    def __init__(self) -> None:
        self.train = pd.read_csv(os.path.join(conf.params["data_preprocess"]["destination_directory"], "pre_train_df.csv"), engine = "pyarrow")
        self.test = pd.read_csv(os.path.join(conf.params["data_preprocess"]["destination_directory"], "pre_test_df.csv"), engine = "pyarrow")
        with open(os.path.join(conf.params["data_preprocess"]["artifact_directory"], "best_features.pkl"), "rb") as f_pkl:
            self.best_features = pickle.load(f_pkl)
        self.seed = conf.params["base"]["seed"]
        self.target = conf.params["train"]["label"]
        self.trails = conf.params["train"]["trails"]
        self.multiclass_technique = conf.params["train"]["multiclass_technique"]
        self.average = conf.params["train"]["average"]

    def optimize(self, space:Dict) -> Dict: 
        X_train, y_train = self.train.drop(self.target, axis = 1), self.train[self.target]
        X_valid, y_valid = self.test.drop(self.target, axis = 1), self.test[self.target]
        booster = GradientBoostingClassifier(**space)
        booster.fit(X_train[self.best_features], y_train)
        y_pred_probs = booster.predict_proba(X_valid[self.best_features])
        y_pred = booster.predict(X_valid[self.best_features])
        auc_roc = -1*roc_auc_score(y_valid, y_pred_probs, multi_class = self.multiclass_technique)
        f1 = -1*f1_score(y_valid, y_pred, average = self.average)
        print(f"AUC-ROC: {auc_roc}")
        print(f"F1-SCORE: {f1}")
        return {"loss":auc_roc, "status":STATUS_OK}
    
    def fit(self, params) -> None:
        booster = GradientBoostingClassifier(**params)
        X, y = self.train.drop(self.target, axis = 1), self.train[self.target]
        booster.fit(X[self.best_features], y)
        with open("./bin/model.pkl", "wb") as pkl:
            pickle.dump(booster, pkl)

if __name__ == '__main__':
    trainer = AutoTrain()
    SEARCH_SPACE={
        "n_estimators":scope.int(hp.quniform("n_estimators", 20, 700, 5)),
        "max_depth":scope.int(hp.quniform("max_depth", 1, 12, 1)),
        "min_samples_split":scope.int(hp.quniform("min_samples_split", 100, 150, 5)),
        "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf", 30, 500, 20)),
        "learning_rate":scope.float(hp.quniform("learning_rate", 0.01, 0.3, 0.001))
    }
    trails = Trials()
    best_res = fmin(
        fn = trainer.optimize,
        space = SEARCH_SPACE,
        algo = tpe.suggest,
        max_evals=trainer.trails,
        trials=trails
    )
    params = {
        "n_estimators":int(best_res["n_estimators"]),
        "max_depth":int(best_res["max_depth"]),
        "min_samples_split":int(best_res["min_samples_split"]),
        "min_samples_leaf":int(best_res["min_samples_leaf"]),
        "learning_rate":best_res["learning_rate"]
    }
    with open("./bin/hyperparams.pkl", "wb") as pkl:
        pickle.dump(params, pkl)
    trainer.fit(params)
