#!/usr/bin/python3
import optuna
from optuna.samplers import TPESampler
import sys

import os
import time
from joblib import dump

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import seed_everything

import numpy as np
from typing import Optional

import sys
sys.path.append("..")
from helpers import *
from parameters import *
from model import *
from dataset import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback



class myEarlyStopping(Callback):
    def __init__(
        self,
        monitor: str,
        patience: int = 3,
        divergence_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        current = logs[self.monitor].squeeze()
        self.wait_count += 1
        if torch.gt(current, self.divergence_threshold) and self.wait_count >= self.patience:
            print (current, self.divergence_threshold)
            should_stop = trainer.strategy.reduce_boolean_decision(True, all=False)
            trainer.should_stop = trainer.should_stop or should_stop
            self.stopped_epoch = trainer.current_epoch




## stupid workaround! cannot assign variables
def objective(trial, task = None,
                freeze = None,
                gamma = None,
                lr = None,
                modelname = None,
                nd1 = None,
                nd2 = None,
                nd3 = None,
                step_size = None,
                final = None,
                n_epochs = None):

    # we want that the trial is not dependend on the others, so seed is fixed here.
    seed_everything(1977)
    result_directory = os.path.join(rootDir, "Standard_no_pretrain", task)

    # retrain or tune?
    if trial is None:
        gamma = gamma/10
        headSize = [2**nd1, 2**nd2, 2**nd3]
    else:
        n_epochs = 100
        modelname = trial.suggest_categorical('modelname', ["resnet18", "resnet34", "effv2s"])
        freeze = trial.suggest_int('freeze', -1, 4)

        step_size = trial.suggest_int('step_size', 15, 30)
        gamma =  trial.suggest_int('gamma', 5, 10)
        gamma = gamma/10

        # number
        nd1 = trial.suggest_int('nd1', 3, 10)
        nd2 = trial.suggest_int('nd2', 3, 10)
        nd3 = trial.suggest_int('nd3', 3, 10)
        headSize = [2**nd1, 2**nd2, 2**nd3]

        lr = trial.suggest_float('lr', 1e-5, 1e-2, log = True)
        print (trial.params)
        recreatePath (result_directory)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_checkpoint = ModelCheckpoint(save_top_k=1, mode='min', monitor="valid_mae")
    early_stop_callback = EarlyStopping(monitor="valid_mae", min_delta=0.05, patience=10, verbose=False, mode="min")
    myEarly_stop_callback = myEarlyStopping(monitor="valid_mae", patience=5, divergence_threshold = 100.0) # MSE -> higher
    progress_bar = TQDMProgressBar(refresh_rate = 10)

    bs = 32
    if modelname == "effv2s":
        bs = 16
    dm = ScoutViewDataModule(task = task, addPretrain = False, returnLabel = False, returnAsDict = False, BATCH_SIZE = bs)
    if trial is None:
        # retrain same model
        if final == False:
            callbacks = [model_checkpoint, myEarly_stop_callback, early_stop_callback, progress_bar]
            default_root_dir = os.path.join("../checkpoints", "Standard_no_pretrain", task, "best")
            logger = TensorBoardLogger(save_dir=os.path.join("../checkpoints", "Standard_no_pretrain", task, "best"), name="logs")
            dm.setup()
        else:
            callbacks = [model_checkpoint, myEarly_stop_callback, early_stop_callback, progress_bar]
            default_root_dir = os.path.join("../checkpoints", "Standard_no_pretrain", task, "final")
            logger = TensorBoardLogger(save_dir=os.path.join("../checkpoints", "Standard_no_pretrain", task, "final"), name="logs")
            dm.setup("final")
    else:
        callbacks = [model_checkpoint, myEarly_stop_callback, early_stop_callback, progress_bar]
        default_root_dir = os.path.join(rootDir, "Standard_no_pretrain/")
        logger = TensorBoardLogger(save_dir=os.path.join(rootDir, "Standard_no_pretrain", task), name="logs")
        dm.setup()

    lightning_model = TopoAge (learning_rate = lr, model = modelname, headSize = headSize,
            freeze = freeze, num_classes = 1, flatten = True, loss = "L2",
            step_size = step_size, gamma = gamma)


    trainer = pl.Trainer(
        max_epochs = n_epochs,
        callbacks = callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices=[0],#"auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        default_root_dir = default_root_dir,
        deterministic=True,
        log_every_n_steps=10)

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=dm)
    error = model_checkpoint.best_model_score
    if final != True:
        best_model = lightning_model.load_from_checkpoint(model_checkpoint.best_model_path, model = modelname)
    else:
        best_model = lightning_model

    # test only the best model-- on train this will be val, else test
    z = trainer.predict(model=best_model, dataloaders=dm.test_dataloader())
    preds = torch.cat(z).cpu().numpy()
    test_df = dm.test_dataset.df.copy()
    gt = test_df["Target"].values
    assert (preds.shape == gt.shape)
    mae = np.mean(np.abs(preds-gt))
    print ("MAE:", mae, "MAE by checkpoint", error)

    # ensure we dont have old models clogging up memory
    if trial is None:
        if final == False:
            # we retrained the model, we save the preds=val preds
            test_df["preds"] = preds
            mae = np.mean(np.abs(test_df["preds"]-test_df["Target"]))
            print ("final MAE on val:", mae)
            # we only need these
            test_df = test_df[["preds", "Target"]]
            test_df.to_csv(f"../results/Standard_no_pretrain.{task}.validation.csv")
            shutil.copyfile (model_checkpoint.best_model_path, f"../results/Standard_no_pretrain.{task}.best_train.model.ckpt")
            dump(early_stop_callback, f"../results/Standard_no_pretrain.{task}.earlystopping.dump")
        else:
            test_df["preds"] = preds
            mae = np.mean(np.abs(test_df["preds"]-test_df["Target"]))
            print ("final MAE on test:", mae)
            test_df = test_df[["preds", "Target"]]
            test_df.to_csv(f"../results/Standard_no_pretrain.{task}.test.csv")
            shutil.copyfile (model_checkpoint.best_model_path, f"../results/Standard_no_pretrain.{task}.final.model.ckpt")
        return None
    else:
        trainer.test(model=best_model, dataloaders=dm.val_dataloader()) # test_loader==val_loader for optuna

    import gc
    best_model.cpu()
    lightning_model.cpu()
    del best_model, trainer, lightning_model
    gc.collect()
    torch.cuda.empty_cache()
    return error



if __name__ == "__main__":
    print ("### Experiment using Standard Architecture")
    os.makedirs ("../results", exist_ok = True)
    for task in ["Height", "Weight"]:
        sampler = TPESampler(seed=571)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(sampler=sampler)
        func = lambda trial: objective(trial, task = task)
        study.optimize(func, n_trials=100)  # Invoke optimization of the objective function.
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        print(df)
        df.to_csv(f"../results/optuna.results.{task}.Standard_no_pretrain.csv")


#
