#!/usr/bin/python3

import pandas as pd


import numpy as np

import sys
sys.path.append("..")
from train_optuna import objective

from joblib import load





if __name__ == "__main__":
    print ("### Experiment using Standard architecture")
    for task in ["Height", "Weight"]:
        tResults = pd.read_csv(f"../results/optuna.results.{task}.Standard.csv")
        bestModel = tResults[tResults.value == np.min(tResults.value)].iloc[0]
        # TOOD: add this to parameters
        best_params = {k.replace("params_", ''):bestModel[k] for k in bestModel.keys() if "params_" in k}
        best_params["task"] = task

        early_stop_callback = load(f"../results/Standard.{task}.earlystopping.dump")
        n_epochs = early_stop_callback.stopped_epoch
        model = objective(None, **best_params, n_epochs = n_epochs, final = True)


#
