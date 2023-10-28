import numpy as np
import os
import pandas as pd
import shutil


def recreatePath (path):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass
    os.makedirs (path)

 #

def getData (split, basePath = "../data", task = None, verbose = False):
    df_train_file = os.path.join(basePath, split+"_final.csv")
    if verbose: print ("Reading from", df_train_file)
    df_train = pd.read_csv(df_train_file)
    df_train["dummy_col"] = 1
    df_train["Target"] = df_train[task]
    #df_train["Age"] = df_train["age"] # copy over calculated age (=correct)
    if verbose: print ("Loaded data with", df_train.shape)
    df_train = df_train[np.isnan(df_train["Target"]) == False]
    if verbose: print ("With target:", df_train.shape)
    df_train.reset_index(drop = True)
    return df_train


def get_image_list(task):
    # global here
    train_df = getData("train", task)
    val_df = getData("val", task)
    test_df = getData("test", task)
    return train_df, val_df, test_df


#
