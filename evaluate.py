#

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

from pprint import pprint
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tbparse import SummaryReader
import scipy
import scipy.stats
import seaborn as sns
import pydicom

from helpers import *
from PIL import Image

sys.path.append('../../../frugalML/')
sys.path.append('../../frugalML/')
sys.path.append('../frugalML/')
from frugalML import StringCounting

modelList = ["Standard", "Standard_no_pretrain"]


def cropImage (k, crop = None, ofs = (0,0)):
    kf = np.asarray(k, dtype = np.float32)
    s = np.asarray((kf[:,:,2] + kf[:,:,1] + kf[:,:,0])/3, dtype = np.float32)
    s = np.asarray( 255*(s - np.min(s))/(np.max(s) - np.min(s)), dtype = np.uint8)
    k[:,:,0] = s
    k[:,:,1] = s
    k[:,:,2] = s
    k = k[ofs[0]:ofs[0]+crop[0], ofs[1]:ofs[1]+crop[1]]
    return k


def addText (finalImage, text = '', org = (0,0), fontFace = 'Arial', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     from PIL import Image
     from PIL import ImageDraw, ImageFont
     tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     return (tmpImg.copy())



def plotHistogramm(allResults):
    # we need this for patients..
    table1 = {}

    allData = {}
    for task in ["Height", "Weight"]:
        data = []
        tmp = allResults["Height"]["train"].copy()
        tmp["Group"] = "Train"
        data.append(tmp)
        tmp = allResults["Height"]["val"].copy()
        tmp["Group"] = "Validation"
        data.append(tmp)
        tmp = allResults["Height"]["test"].copy()
        tmp["Group"] = "Test"
        data.append(tmp)
        allData[task] = pd.concat(data,axis=0)

    plt.figure()
    sns.set(style="white", context="talk")
    f, axs  = plt.subplots(1, 2, figsize = (20,7)) #gridspec_kw={'width_ratios': [1,2]})
    sns.swarmplot(ax = axs[0], y='Height', x='Group', data=allData["Height"])
    axs[0].set_xlabel("",fontsize=23)
    axs[0].set_ylabel("Height [cm]",fontsize=23)
    sns.swarmplot(ax = axs[1], y='Weight', x='Group', data=allData["Weight"])
    axs[1].set_xlabel("",fontsize=23)
    axs[1].set_ylabel("Weight [kg]",fontsize=23)
    #axs[0].tick_params(axis='both', which='major', labelsize=23)
    axs[0].tick_params(axis='x', which='both', labelsize=23)
    axs[1].tick_params(axis='x', which='both', labelsize=23)
    #ax.set_aspect('auto')
    f.savefig("./paper/Figure_3.png", dpi = 300, bbox_inches='tight')
    plt.close('all')



def plotResults (testResults, tunit = '', fign = [], task = None, fname = None, xlim = None, ylim = None):
    fTbl = testResults.copy()
    f, axs  = plt.subplots(1, 3, figsize = (31,9))
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    for j in range(len(axs)):
        axs[j].tick_params(axis='x', labelsize= 23)
        axs[j].tick_params(axis='y', labelsize= 23)
    np.random.seed(42)
    random.seed(42)

    fTbl["error"] = fTbl["Target"] - fTbl["Predictions"]

    # plot them
    colors = {'F':'red', 'M':'blue'}
    axs [0].scatter (fTbl["Target"], fTbl["Predictions"], color = fTbl["Sex"].map(colors))
    vmin = np.min([fTbl["Target"], fTbl["Predictions"]])
    vmax = np.max([fTbl["Target"], fTbl["Predictions"]])

    axs[0].plot([vmin, vmax], [vmin, vmax], '-', color = 'k')
    axs[0].plot([vmin+10, vmax+10], [vmin, vmax], linestyle = 'dashed', color = 'k')
    axs[0].plot([vmin-10, vmax-10], [vmin, vmax], linestyle = 'dashed', color = 'k')
    if xlim is not None:
        axs[0].set_xlim(xlim)


    absdiffs = fTbl["Predictions"]-fTbl["Target"]
    reldiffs = 100*(fTbl["Predictions"]-fTbl["Target"])/(fTbl["Target"])
    diffs = absdiffs
    axs [1].hist (diffs, bins = 15)
    if ylim is not None:
        axs[1].set_ylim(ylim)
    #axs[1].set_xlim([-8,8])

    # make bland-altman plot thing
    fTbl["absdiffs"] = fTbl["Target"] -fTbl["Predictions"]
    fTbl["reldiffs"] = 100*(fTbl["Target"] -fTbl["Predictions"])/(fTbl["Target"])

    axs[2].scatter(fTbl["Target"], fTbl["absdiffs"], color = fTbl["Sex"].map(colors))
    #axs[2].plot([0, 21.0], [0, 0], '-', color = 'k')
    #axs[2].set_ylim([-8, 8])


    axs[0].set_xlabel (f"Ground truth {task} [{tunit}]", fontsize = 30)
    axs[0].set_ylabel (f"Predicted {task} [{tunit}]", fontsize = 30)

    axs[1].set_xlabel (f"Prediction difference [{tunit}]", fontsize = 30)
    axs[1].set_ylabel ("Count", fontsize = 30)

    axs[2].set_xlabel (f"Ground truth {task} [{tunit}]", fontsize = 30)
    axs[2].set_ylabel (f"Prediction error [{tunit}]", fontsize = 30)

    #f.suptitle("Results on the " + ttset, fontsize = 32)
    axs[0].text(-0.17, 1.0, fign[0], horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, fontsize = 36)
    axs[1].text(-0.17, 1.0, fign[1], horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize = 36)
    axs[2].text(-0.17, 1.0, fign[2], horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, fontsize = 36)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.13,
                        right=0.9,
                        top=0.9,
                        wspace=0.26,
                        hspace=0.4)
    f.savefig ("./results/" + fname, dpi = 300)


    fTbl["error"] = fTbl["Target"] - fTbl["Predictions"]
    examples = fTbl[ abs(fTbl["error"]) > 2.5]
    print (len(examples), " are beyond 2.5, from ", fTbl.shape[0], " scans")
    print ("this is ", len(examples)/fTbl.shape[0], " %")
    examples = fTbl[ abs(fTbl["error"]) > 5.0]
    print (len(examples), " are beyond 5.0 , from ", fTbl.shape[0], " scans")
    print ("this is ", len(examples)/fTbl.shape[0], " %")



def plotCombinedResults (testResults, ttset = None, task = None, fname = None, xlim = None, ylim = None):
    plotResults (allResults["Height"]["test"], fign = ['a', 'b', 'c'], tunit = "cm", task = "height", fname = f"Figure_5_Height.png")
    plotResults (allResults["Weight"]["test"],  fign = ['d', 'e', 'f'], tunit = "kg", task = "weight", fname = f"Figure_5_Weight.png")
    # join plots

    fontFace = "Arial"
    imA = cv2.imread("./results/Figure_5_Height.png")
    imB = cv2.imread("./results/Figure_5_Weight.png")
    img = np.vstack([imA, imB])
    cv2.imwrite("./paper/Figure_5.png", img)



def getPreds (model, task, split = "test"):
    # reread val sets and recompute MAE
    vTbl = pd.read_csv("./results/" + model + "." + task + "." + split + ".csv")
    vTbl[split+"_preds"] = vTbl["preds"]
    diffs = np.abs(vTbl[split+"_preds"] - vTbl["Target"])
    MAE = np.mean(diffs)
    Std = np.std(diffs)
    return MAE, vTbl


def getResults (task):
    # reread val sets and recompute MAE
    results = {}
    for model in modelList:
        MAE, vTbl = getPreds (model, task, "validation")
        results[model] = MAE
    return results



def findGoodBadExamples (allResults):
    fTbl = allResults["Height"]["test"].copy()
    fTbl["diffs"] = np.abs (fTbl["Target"] -fTbl["Predictions"])
    fTbl = fTbl.sort_values(["diffs"])

    minEH = fTbl.iloc[0]
    maxEH = fTbl.iloc[-1]
    print ("#### HEIGHT:")
    print ("Min", minEH["Target"], minEH["Predictions"], " D:", minEH["diffs"],"AGE:", minEH["age"])
    print ("Max", maxEH["Target"], maxEH["Predictions"], " D:", maxEH["diffs"],"AGE:", maxEH["age"])

    fTbl = allResults["Weight"]["test"].copy()
    fTbl["diffs"] = np.abs (fTbl["Target"] -fTbl["Predictions"])
    fTbl = fTbl.sort_values(["diffs"])

    minEW = fTbl.iloc[0]
    maxEW = fTbl.iloc[-1]
    print ("Min", minEW["Target"], minEW["Predictions"], " D:", minEW["diffs"], "AGE:", minEW["age"])
    print ("Max", maxEW["Target"], maxEW["Predictions"], " D:", maxEW["diffs"], "AGE:", maxEW["age"])

    finalImg = np.zeros ((768+2*16, 3*512+4*16, 3), dtype = np.uint8)
    finalImg = finalImg*0 + 255

    k = cv2.imread(minEH.Image)
    k = cropImage(k, crop = (768,512), ofs = (0,0))
    k [768-30:768,:,:] = 0
    age = np.round(minEH["Target"], 1)
    pred = np.round(minEH["Predictions"], 1)
    age = np.round(age/100,2)
    pred = np.round(pred/100,2)
    k = addText (k, text=f"True: {age:.2f}m, Prediction: {pred:.2f}m", org = (20, 768-30), fontSize = 24)
    k = addText (k, text="a", org = (470, 768-30), fontSize = 24)
    finalImg [16:16+768, 16:512+16, :] = k#[0:248, :, :]


    k = cv2.imread(maxEH.Image)
    k = cropImage(k, crop = (333,512), ofs = (236,0))
    k [333-30:333,:,:] = 0
    age = np.round(maxEH["Target"], 1)
    pred = np.round(maxEH["Predictions"], 1)
    age = np.round(age/100,2)
    pred = np.round(pred/100,2)
    k = addText (k, text=f"True: {age:.2f}m, Prediction: {pred:.2f}m", org = (20, 333-30), fontSize = 24)
    k = addText (k, text="b", org = (470, 333-30), fontSize = 24)

    finalImg [16:16+333, 16+16+512:16+16+512+512, :] = k#[0:248, :, :]


    k = cv2.imread(minEW.Image)
    k = cropImage(k, crop = (419,512), ofs = (210,0))
    k [419-30:419,:,:] = 0

    age = np.round(minEW["Target"], 1)
    pred = np.round(minEW["Predictions"], 1)
    k = addText (k, text=f"True: {age:.2f}kg, Prediction: {pred:.2f}kg", org = (20, 419-30), fontSize = 24)
    k = addText (k, text="c", org = (470, 419-30), fontSize = 24)
    finalImg [16+333+16:16+333+16+419, 16+16+512:16+16+512+512, :] = k#[0:248, :, :]

    k = cv2.imread(maxEW.Image)
    k = cropImage(k, crop = (768,512), ofs = (0,0))
    k [768-30:768,:,:] = 0
    age = np.round(maxEW["Target"], 1)
    pred = np.round(maxEW["Predictions"], 1)
    k = addText (k, text=f"True: {age:.2f}kg, Prediction: {pred:.2f}kg", org = (20, 768-30), fontSize = 24)
    k = addText (k, text="d", org = (470,768-30), fontSize = 24)
    finalImg [16:16+768, 16+2*512+2*16:16+2*512+2*16+512, :] = k
    #Image.fromarray(finalImg[::2,::2])

    finalImg = cv2.resize(finalImg, (0,0), fx = 2.5, fy = 2.5)
    cv2.imwrite(f"paper/Figure_6.png", finalImg)




def prepareImg (row, borderSize = 16, task = None):
    k = cv2.imread(row.Image)
    lSize = 56
    finalImg = np.zeros ((k.shape[0] + 2*borderSize + lSize, k.shape[1] + 2*borderSize, 3), dtype = np.uint8)
    finalImg = finalImg*0 + 255
    finalImg[borderSize:borderSize+k.shape[0], borderSize:borderSize+k.shape[1], :] = k
    finalImg[borderSize+k.shape[0]:borderSize+k.shape[0]+lSize, borderSize:borderSize+k.shape[1], :] = 128
    tgt = np.round(row["Target"], 1)
    pred = np.round(row["Predictions"], 1)
    suffix = "kg" if task == "Weight" else "cm"
    finalImg = addText (finalImg, text="True: "+str(tgt) + suffix + ", Prediction: " + str(pred) + suffix, org = (20, lSize//8+borderSize+k.shape[0]), fontSize = 24)
    finalImg = addText (finalImg, text="a", org = (470,  lSize//8+borderSize+k.shape[0]), fontSize = 24)
    return finalImg.copy()



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid




def printModelParameters (results):
    for m in results.keys():
        print ("\n###", m)
        tResults = pd.read_csv("./results/optuna.results."+m+".csv")
        bestModel = tResults[tResults.value == np.min(tResults.value)].iloc[0]
        best_params = {k.replace("params_", ''):bestModel[k] for k in bestModel.keys() if "params_" in k}
        pprint (best_params)



def getDemographics (allResults):
    pats = {}
    for split in ["train", "val", "test", "pretrain"]:
        pats[split] = []
    for task in ["Height", "Weight"]:
        for split in ["train", "val", "test", "pretrain"]:
            data = allResults[task][split].copy()
            data["newID"] = data["pat_name"] + data["birth_date"]
            subdata = data.drop_duplicates(subset='newID', keep="first")
            pats[split] = subdata.copy()
            print (task, split, subdata.shape[0])

    pats["all"] = pd.concat([pats["train"], pats["test"], pats["val"]], axis = 0)
    dTable = pd.DataFrame()
    for ds in ["all", "train", "val", "test"]:
        sd = pats[ds].copy()
        sd["Height"] = sd["Height"]/100
        Fn = str(int(sd[sd["Sex"]  == "F"].shape[0]))
        Mn = str(int(sd[sd["Sex"]  == "M"].shape[0]))
        Nall = sd.shape[0]
        Fper = round( (sd[sd["Sex"]  == "F"].shape[0])/sd.shape[0]*100, 1)
        dTable.at["Gender", ds] = f"{Fper}% ({Fn}/{Nall})"
        for t in ["Height", "Weight", "age"]:
            rd = 1
            if t == "Height":
                rd = 2
            meanH = round(np.mean(sd[t]) ,rd)
            stdH = round(np.std(sd[t]),rd)
            dTable.at[t, ds] = f"{meanH} +/- {stdH} (N = {Nall})"
            print (ds, t, np.min(sd[t]), np.max(sd[t]))

    print(dTable)
    dTable.to_excel("./paper/Table_2.xlsx")

    pats["all"] = pd.concat([pats["train"], pats["pretrain"]], axis = 0)
    dTable = pd.DataFrame()
    for ds in ["all", "train", "pretrain"]:
        sd = pats[ds].copy()
        sd["Height"] = sd["Height"]/100
        Fn = str(int(sd[sd["Sex"]  == "F"].shape[0]))
        Mn = str(int(sd[sd["Sex"]  == "M"].shape[0]))
        Nall = sd.shape[0]
        Fper = round( (sd[sd["Sex"]  == "F"].shape[0])/sd.shape[0]*100, 1)
        dTable.at["Gender", ds] = f"{Fper}% ({Fn}/{Nall})"
        for t in ["Height", "Weight", "age"]:
            rd = 1
            if t == "Height":
                rd = 2
            meanH = round(np.mean(sd[t]) ,rd)
            stdH = round(np.std(sd[t]),rd)
            dTable.at[t, ds] = f"{meanH} +/- {stdH} (N = {Nall})"
            print (ds, t, np.min(sd[t]), np.max(sd[t]))

    print(dTable)
    dTable.to_excel("./paper/Table_S1.xlsx")


def getScannerCounts (allResults, th = 50):
    models = StringCounting.StringCounting ()
    submodels = {}
    for task in ["Height", "Weight"]:
        submodels[task] = {}
        for dset in ["train", "val", "test"]:
            data = allResults[task][dset]
            submodels[task][dset] = StringCounting.StringCounting ()
            for i, (idx, row) in enumerate(data.iterrows()):
                f = row["dcm_path"]
                try:
                    ds=pydicom.read_file(f)
                except Exception as e:
                    raise Exception ("This should not happen. Every DCM should have been read once during prepareDataSets!")
                submodels[task][dset].update([ds["ManufacturerModelName"].value])
                models.update([ds["ManufacturerModelName"].value])

    scs = models.counts
    scs = pd.DataFrame.from_dict(scs, orient="index")
    scs = scs.sort_values(0, ascending = False)

    mask = scs[0].ge(th)
    z = scs[~mask].sum()
    z.index = ["Other"]
    scs = pd.concat([scs[mask], z])

    newscs = scs.copy()
    for task in ["Height", "Weight"]:
        for dset in ["train", "val", "test"]:
            curscs = {}
            curscs['Other'] = 0
            for k in submodels[task][dset].counts:
                if k in scs.index:
                    curscs[k] = int(submodels[task][dset].counts[k])
                else:
                    curscs['Other'] = int(curscs['Other'] +submodels[task][dset].counts[k])
            #curscs = pd.DataFrame.from_dict(curscs, orient="index")
            newscs = pd.concat([newscs,pd.DataFrame(curscs, index = [task+"_"+dset]).T], axis = 1)
    newscs = newscs.fillna(0).astype(np.uint32)
    newscs = newscs.drop([0], axis = 1).copy()

    nkeys = {}
    for k in newscs.keys():
        nkeys[k] = k+ " (N = " + str(    sum(newscs[k].values)) +")"
    newscs.rename(columns=nkeys, inplace=True)
    newscs.to_excel("./paper/Table_1.xlsx")



def showModelParams(model, task):
    d = pd.read_csv(f"./results/optuna.results.{task}.{model}.csv")
    d = d.sort_values(["value"])
    bh = d.iloc[0]
    print(f"Best parameters for height: arch {bh.params_modelname}, ",
            f"freeze {bh.params_freeze}, step {bh.params_step_size}, ",
            f"gamma {bh.params_gamma}, lr {np.round(bh.params_lr,5)}, ",
            f"head {bh.params_nd1, bh.params_nd2, bh.params_nd3}")


if __name__ == "__main__":
    for bestModel in modelList:
        print ("###", bestModel)
        for task in ["Height", "Weight"]:
            print ("\ttask", task)
            results = getResults(task)

            data = getData("val", "./data", task = task)
            data = data.reset_index(drop = True)
            MAE, vTbl = getPreds (bestModel, task, "validation")
            assert (len(data) == len(vTbl))
            data["Predictions"] = vTbl["validation_preds"]
            results["val"] = data.copy()
            absdiffs = np.abs( (results["val"]["Target"]-results["val"]["Predictions"]) )
            print ("\t\tOn val, abs. diff:", np.round(np.mean(absdiffs),2), "+/-", np.round(np.std(absdiffs),2))
            reldiffs = np.abs( (results["val"]["Target"]-results["val"]["Predictions"])/results["val"]["Target"] )
            print ("\t\tOn val, rel. diff in %:", np.round(100*np.mean(reldiffs),2), "+/-", np.round(100*np.std(reldiffs),2))
            cor = scipy.stats.pearsonr(results["val"]["Target"], results["val"]["Predictions"])
            print ("Correlation is ", np.round(cor[0], 3))


    allResults = {}
    for task in ["Height", "Weight"]:
        print ("###", task)
        results = getResults(task)

        bestModel = min(results, key=results.get)
        results = {"model": bestModel}
        print (f"Best model for {task} is {bestModel}")
        showModelParams (bestModel, task)
        # base data
        data = getData("train", "./data", task = task)
        results["train"] = data.copy()
        data = getData("pretrain", "./data", task = task)
        results["pretrain"] = data.copy()

        data = getData("val", "./data", task = task)
        data = data.reset_index(drop = True)
        MAE, vTbl = getPreds (bestModel, task, "validation")
        assert (len(data) == len(vTbl))
        data["Predictions"] = vTbl["validation_preds"]
        results["val"] = data.copy()

        data = getData("test", "./data", task = task)
        data = data.reset_index (drop = True)
        MAE, vTbl = getPreds (bestModel, task, "test")
        assert (len(data) == len(vTbl))
        data["Predictions"] = vTbl["test_preds"]
        results["test"] = data.copy()

        absdiffs = np.abs( (results["test"]["Target"]-results["test"]["Predictions"]) )
        print ("On test, abs. diff:", np.round(np.mean(absdiffs),2), "+/-", np.round(np.std(absdiffs),2))

        reldiffs = np.abs( (results["test"]["Target"]-results["test"]["Predictions"])/results["test"]["Target"] )
        print ("On test, rel. diff in %:", np.round(100*np.mean(reldiffs),2), "+/-", np.round(100*np.std(reldiffs),2))

        cor = scipy.stats.pearsonr(results["test"]["Target"], results["test"]["Predictions"])
        print ("Correlation is ", np.round(cor[0], 3))

        allData = pd.concat([results[s] for s in ["train", "test", "val"]], axis = 0)
        results["all"] = allData.reset_index(drop = True).copy()
        results["test"]["error"] = results["test"]["Target"] - results["test"]["Predictions"]
        allResults[task] = results.copy()

    getDemographics (allResults)
    getScannerCounts (allResults)

    plotHistogramm(allResults)

    plotCombinedResults (allResults, ttset = "test set", task = task, fname = f"Figure_6_{task}.png")

    testResults = allResults["Height"]["test"]
    findGoodBadExamples (allResults)

#
