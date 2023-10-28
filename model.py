#
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchinfo import summary
import torchmetrics

#from datasets import *


class TopoAge(pl.LightningModule):
    def __init__(self, learning_rate = 3e-4, num_classes = 1, model = None, headSize = [512, 256, 64],
                step_size = 1, gamma = 0.9, flatten = True,
                dropoutLevel = 0.1, freeze = None, showStats = True, loss = "L2"):
        super().__init__()
        self.headSize = headSize
        self.dropoutLevel = dropoutLevel
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.lossType = loss
        self.model = model
        self.flatten = flatten
        self.num_classes = num_classes

        self.save_hyperparameters(ignore="model")

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        if model == "effv2s":
            num_filters = 1280
            if freeze == -1:
                backbone = models.efficientnet_v2_s ()
                backbone.classifier = nn.Identity()
            else:
                backbone = models.efficientnet_v2_s (weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
                backbone.classifier = nn.Identity()
                freezeToInt = [-1, 8, 56, 251, 452][freeze]
                for param in list(backbone.parameters())[:freezeToInt+1]:
                    param.requires_grad=False
            self.feature_extractor = backbone


        if "resnet" in model:
            fmodel = "ResNet" + model[6:]
            if freeze == -1:
                backbone = eval("models."+model+"()")
                num_filters = backbone.fc.in_features
                backbone.fc = nn.Identity()
            else:
                backbone = eval("models."+model+"(weights=models."+fmodel+"_Weights.IMAGENET1K_V1)")
                num_filters = backbone.fc.in_features
                backbone.fc = nn.Identity()

                #freezeToInt = [0, 39, 81, 143, 159][freeze]
                if model == "resnet18":
                    freezeToInt = [-1, 14, 29, 44, 64][freeze]
                if model == "resnet34":
                    freezeToInt = [-1, 20, 47, 86, 133][freeze]
                if model == "resnet50":
                    freezeToInt = [-1, 32, 71, 128, 159][freeze]
                for param in list(backbone.parameters())[:freezeToInt+1]:
                    param.requires_grad=False
            self.feature_extractor = backbone


        # create head
        clf = []
        clf.append(nn.Dropout(self.dropoutLevel))
        cSize = num_filters
        for j, z in enumerate(headSize):
            clf.append(nn.Linear(cSize, z))
            clf.append(nn.ReLU())
            cSize = z
        clf.append(nn.Linear(cSize, self.num_classes))
        self.classifier = nn.Sequential(*clf)


    def forward(self, x):
        fv = self.feature_extractor(x)
        fv = fv.flatten(1)
        x = self.classifier(fv)
        if self.flatten == True:
            x = x.flatten()
        return x


    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)

        if self.lossType == "L1":
            loss = F.l1_loss(y_hat, y)

        if self.lossType == "L2":
            loss = F.mse_loss(y_hat.float(), y.float()).float()

        return loss, y, y_hat


    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        self.log("train_loss", loss)
        if self.lossType == "L1":
            self.train_mae(y_hat, y)
        elif self.lossType == "L2":
            self.train_mae(y_hat, y)
        else:
            raise Exception ("Unknown loss")
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss  # this is passed to the optimzer for training


    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        self.log("valid_loss", loss)
        if self.lossType == "L1":
            self.valid_mae(y_hat, y)
        elif self.lossType == "L2":
            self.valid_mae(y_hat, y)
        else:
            raise Exception ("Unknown loss")
        self.log("valid_mae", self.valid_mae, on_epoch=True, on_step=False, prog_bar=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        if self.lossType == "L1":
            y_hat = self(x)
            return y_hat

        if self.lossType == "L2":
            y_hat = self(x)
            return y_hat

        raise Exception ("Unknown loss")


    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step (batch)
        self.test_mae(y_hat, y)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False)


    def configure_optimizers(self):
        print ("Setting learning rate to", self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


if __name__ == "__main__":
    pTbl = {}
    for model in ["resnet18", "resnet34", "effv2s"]:
        print ("\n\n###", model)
        pTbl[model] = {}
        imgSize = (768, 512)
        for freeze in [-1, 0, 1, 2, 3, 4]:
            m = TopoAge(model = model, freeze = freeze, showStats = True, headSize = [1024])
            print(summary(m.feature_extractor, input_size=(1, 3, imgSize[0], imgSize[1]), verbose = 0))
            z = summary(m.feature_extractor, input_size=(1, 3, imgSize[0], imgSize[1]), verbose = 1)
            pTbl[model][(imgSize,freeze, "Trainable")] =  '{:,.0f}'.format(z.trainable_params)
            pTbl[model][(imgSize,freeze, "Non-trainable")] = '{:,.0f}'.format(z.total_params- z.trainable_params)
    pd.DataFrame(pTbl).to_excel("./results/paramCount.xlsx")



#
