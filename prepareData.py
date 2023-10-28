
from termcolor import colored
from collections import defaultdict
import json
import datetime
import tempfile
import string
import configparser
import sys
from datetime import datetime
import random
import re
import scipy
import scipy.signal
from collections import OrderedDict
from operator import itemgetter

import cv2
import numpy as np
import SimpleITK as sitk
import png
import pandas as pd
from glob import glob
import time, os
import pydicom
import imageio
import shutil
from sklearn.model_selection import KFold
import progressbar
from random import sample

from pprint import pprint
import collections
import scipy.misc



class Config ():
    def __init__(self):
        self.greyToRGB = "pseudoRGB"
        #self.rescale = "1024_height"
        self.rescale = None #"512_scale"
        self.dest_ckey = 240
        self.slimLine = False
        self.optLine = False

config = Config ()



def cropImage (z, row):
    rstd = [np.std(z[k,:,0]) for k in range(z.shape[0])]
    cstd = [np.std(z[:,k,0]) for k in range(z.shape[1])]
    cutout = z[np.array(rstd) > 5.0,:][:,np.array(cstd) > 5.0]
    if row.PixelSpacing == 2:
        rCutout = cv2.resize(cutout, (cutout.shape[1]*2, cutout.shape[0]*2))
    else:
        rCutout = cutout.copy()
    #shapes.append(rCutout.shape[0:2] )

    # its never larger than 1024x566, so we can take this as blueprint
    f = np.zeros((1024, 566, 3), dtype = np.uint8)

    bx = (566 - rCutout.shape[1])//2
    by = (1024 - rCutout.shape[0])//2
    f[by:by+rCutout.shape[0], bx:bx+rCutout.shape[1], :] = rCutout

    #  now we can crop it
    f = f[128:1024-128,27:27+512,:]
    return (f)



def pseudoRGB (img, method = "clahe", visualize = False):
    if method not in ["clahe"]:
        exit ("Pseudo RGB method " + str(method) + " is unknown.")

    conversionFactor = 256
    if img.dtype == np.uint8:
        conversionFactor  = 1
        method = 'clahe'

    if method == "clahe":
        factor = 0.5
        clipfactor = 2
        baseFactor = 16.0
        spreadFactor = 2.0

        clahe = cv2.createCLAHE(clipLimit=baseFactor*spreadFactor*clipfactor, tileGridSize=(int(2*factor),int(2*factor)))
        red = clahe.apply(img)
        clahe = cv2.createCLAHE(clipLimit=baseFactor*clipfactor, tileGridSize=(int(4*factor),int(4*factor)))
        green = clahe.apply(img)
    img = cv2.merge((0*green, green, red)) # 1st for original image
    return img



def convertToPNG (datasets):
    expPath = "/data/data/ped.bmi/export"
    for split in ["train", "test", "val", "pretrain"]:
        data = datasets[split].copy()

        # recreate
        delpath = os.path.join(expPath, "png", split)
        recreatePath (delpath)

        maxV = 0
        minV = 0
        for i, (idx, row) in enumerate(data.iterrows()):
            delpath = os.path.join(expPath, "png", split)

            f = data.at[idx, "dcm_path"]
            try:
                ds=pydicom.read_file(f)
            except Exception as e:
                raise Exception ("This should not happen. Every DCM should have been read once during prepareDataSets!")

            dcmsrc = sitk.ReadImage(f)
            npimg = sitk.GetArrayFromImage(dcmsrc)

            # just warn
            if np.max(npimg) > 1024:
                print (np.max(npimg))
                print ("There should be no values higher than 1024!")
            if np.min(npimg) < -1024:
                print (np.min(npimg))
                print ("There should be no values lower than -1024!")

            npimg[npimg<-256] = -256
            npimg[npimg>1023] = 1023

            # apply CLAHE
            img = npimg[0,:,:]
            img = (img - img.min())/(img.max() - img.min()) * 255.0
            img = np.uint8(img)

            # apply pseudoRGB
            imgRGB = pseudoRGB (img)
            imgRGB [:,:,0] = img
            imgRGB = cropImage (imgRGB, row)

            pngfname = os.path.basename(f).replace(".dcm", ".png")
            outfname = os.path.join(delpath, pngfname)
            imageio.imwrite(outfname, imgRGB)

            # should be relative to ./export/final
            data.at[idx, "Image"] = os.path.join(os.path.basename(delpath), pngfname)


#
