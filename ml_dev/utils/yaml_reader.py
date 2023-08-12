import yaml
import os

class Config:
    def __init__(self):
        with open(os.path.join(".", "params.yml"), "r") as f:
            self.params = yaml.safe_load(f)
        with open(os.path.join(".", "dvc.yml"), "r") as f:
            self.dvc_pipeline = yaml.safe_load(f)