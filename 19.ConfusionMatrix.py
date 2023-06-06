# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:00:15 2023

@author: yoah2447
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import pandas as pd
#import gdal
import rasterio as rio
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

loaded = np.load(r'D:/DATA/comparisondata/MTBF/counties_rast250.npz')
counties_rast = loaded['counties_rast250'].flatten()


years=range(1810,2010,5)
DF = pd.DataFrame()
DF['county'] = counties_rast.flatten()

with rio.open("D:/DATA/HISDAC/HISDAC_FBUY/FBUY/FBUY.tif") as img :

    FBUY_old= img.read()
    DF['HISDAC_v1'] = FBUY_old.flatten()


with rio.open("D:/DATA/Raster/FBUY/FBUY.tif") as img :

    FBUY_new= img.read()
    DF['HISDAC_v2'] = FBUY_new.flatten()

DF['builtyear_v1'] =DF['HISDAC_v1'].apply(lambda x: 1 if x>=1 else 0)
DF['builtyear_v2'] =DF['HISDAC_v2'].apply(lambda x: 1 if x>=1 else 0)


conf_matrix = confusion_matrix(y_true=DF['builtyear_v2'] , y_pred=DF['builtyear_v1'])
