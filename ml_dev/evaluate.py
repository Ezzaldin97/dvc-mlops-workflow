from utils.yaml_reader import Config
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import yaml

conf = Config()

class PerformanceEvaluation:
    def __init__(self) -> None:
        self.test = pd.read_csv(os.path.join(conf.params["data_preprocess"]["destination_directory"], "pre_test_df.csv"), engine = "pyarrow")
        with open(os.path.join(conf.params["data_preprocess"]["artifact_directory"], "best_features.pkl"), "rb") as f_pkl:
            self.best_features = pickle.load(f_pkl)
        with open(os.path.join(conf.params["data_preprocess"]["artifact_directory"], "model.pkl"), "rb") as m_pkl:
            self.model = pickle.load(m_pkl)
        self.seed = conf.params["base"]["seed"]
        self.target = conf.params["train"]["label"]
        self.multiclass_technique = conf.params["train"]["multiclass_technique"]
        self.average = conf.params["train"]["average"]
    
    def eval_performance(self) -> None:
        metrics = {}
        X, y = self.test.drop(self.target, axis = 1), self.test[self.target]
        preds = self.model.predict(X[self.best_features])
        probs = self.model.predict_proba(X[self.best_features])
        metrics["f1-score"] = f1_score(y, preds, average = self.average)
        metrics["roc_auc_score"] = roc_auc_score(y, probs, multi_class = self.multiclass_technique)
        with open(os.path.join(".", "data", "metrics.yaml"), 'w') as file:
            yaml.dump(metrics, file, default_flow_style=False)
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join(".", "data", "confusion_matrix.png"), dpi=150, bbox_inches='tight', pad_inches=0)
        print(f"Evaluation done!")
        print(metrics)

if __name__ == '__main__':
    evaluate = PerformanceEvaluation()
    evaluate.eval_performance()