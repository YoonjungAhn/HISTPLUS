# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:39:54 2023

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
%matplotlib inline


#making forloop
from scipy.stats import pearsonr
loaded = np.load(r'/Users/yoonjung/FSU/2022/Research/HISPLUS/counties_rast250.npz')
counties_rast = loaded['counties_rast250'].flatten()

years=range(1975,2021,5)
DF = pd.DataFrame()
DF['county'] = counties_rast.flatten()
#DF['county']= DF['county'].apply(lambda x: '{0:0>5}'.format(x))
#DF['county'].replace('')


#DF = pd.DataFrame(index=range(len(counties_rast)), columns=range(len(years)))
#DF.columns= list(years)

len(counties_rast)
countyDF = pd.DataFrame()
countyDF['county']=DF['county'].unique()[1:]
countyDF['county']= countyDF['county'].apply(lambda x: '{0:0>5}'.format(x))
countyDF['state']  = countyDF['county'].str[0:2]
stateDF = pd.DataFrame()
#stateDF['state'] = countyDF['county'].str[0:2].unique()
stateDF['state']=['01', '03','04', '05', '06', '07','08', '09', '10', '11', '12', '13','14','15', '16',
       '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
       '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
       '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
       '51', '53', '54', '55', '56']

#stateDF = pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/BUI1970statelevelsummary.csv')
stateDF['state']= stateDF ['state'].apply(lambda x: '{0:0>2}'.format(x))

#countyDF= pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/BUI1970countylevelsummary.csv')
countyDF['county']= countyDF['county'].apply(lambda x: '{0:0>5}'.format(x))

Dat = [ 'BUI']#["BUPL" ,"BUPR", 'BUI']
for dat in Dat:
    
    for year in years:
        
        with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/"+dat+"/"+str(year)+"_"+dat+".tif") as img :
        
            new= img.read()
            if dat == 'BUI':
                DF[str(year)+"_new"]= new.flatten()*0.092903
            else:
                DF[str(year)+"_new"]= new.flatten()
        if year ==2020:
            DF[str(year)+"_old"]=0
            
        else:
    
            with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/HISAC/HISDAC_"+dat+"/"+dat+"/"+dat+"_"+str(year)+".tif") as img :
            
                old= img.read()
                DF[str(year)+"_old"]= old.flatten()
                
                del old, new, img
                
                #df = DF.loc[(DF['county']!=0)&(DF['county']!=-1)]
        df = DF.loc[(DF['county']!=0)&(DF['county']!=-1)]
        df = df.groupby(['county'])[str(year)+"_new",str(year)+"_old"].sum().reset_index()
        df['county']= df['county'].apply(lambda x: '{0:0>5}'.format(x))
        df['state'] = df['county'].str[0:2]
        countyDF = pd.merge(countyDF, df, on = ['county','state'])
        
        dfstate = df.groupby(['state'])[str(year)+"_new",str(year)+"_old"].sum().reset_index()
        stateDF = pd.merge(stateDF, dfstate, on = ['state'])
        
        DF = DF.iloc[:,0:1]
        print(year)
        stateDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+str(year)+'statelevelsummary.csv',index=False)
        countyDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+str(year)+'countylevelsummary.csv',index=False)



    #imgmeta=img.meta
#with rio.open("C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/HIDAC_BUI/BUI/data/"+str(year)+"_BUI.tif") as img :

#    BUI_new= img.read()

import glob
import os

Allst = []
Allcounty =  []
Dat = ["BUPL","BUPR","BUI"] #,'BUI',
for dat in Dat:
    Lst = glob.glob(os.path.join('/Users/yoonjung/FSU/2022/Research/HISPLUS/',dat+'*statelevelsummary.csv')) #.remove('censuscomparision')
    Lst = [x for x in Lst if "censuscomparision" not in x]
    Lst.sort()
    stateDF = pd.concat(map(pd.read_csv,Lst),axis=1)
    stateDF['layer'] = dat
    stateDF= stateDF.T.drop_duplicates().T
    stateDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'statelevelsummary.csv',index=False)

    
    Lcnt = glob.glob(os.path.join('/Users/yoonjung/FSU/2022/Research/HISPLUS/',dat+'*countylevelsummary.csv'))
    Lcnt = [x for x in Lcnt if "censuscomparision" not in x]
    Lcnt.sort()
    countyDF = pd.concat(map(pd.read_csv,Lcnt),axis=1)
    #countyDF=pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'*countylevelsummary.csv')
    countyDF['layer'] = dat
    countyDF= countyDF.T.drop_duplicates().T
    countyDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'countylevelsummary.csv',index=False)



Allst = []
Allcounty =  []
Dat = ["BUPL","BUPR","BUI"] #,'BUI',
for dat in Dat:
    stateDF=pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'statelevelsummary.csv')
    Allst.append(stateDF)
    
    countyDF=pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'countylevelsummary.csv')
    Allcounty.append(countyDF)

Allst =pd.concat(Allst, ignore_index=True, axis=0)
Allcounty =pd.concat(Allcounty)    
Allst.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/AllLayers_statelevelsummary.csv',index=False)
Allcounty.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/AllLayers_countylevelsummary.csv',index=False)


stateDF = pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/AllLayers_statelevelsummary.csv')
countyDF = pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/AllLayers_countylevelsummary.csv')
years=range(1810,2021,5)

#correlation
corrdf = pd.DataFrame()
corrdf['year'] =list(years)*3
corrdf['pearson'] = 0
corrdf['pvalue'] = 0
corrdf['layer'] = list(np.repeat(["BUPL","BUPR","BUI"],len(corrdf)/3)) #,'BUI'

corrdfcounty = pd.DataFrame()
corrdfcounty['year'] =  list(years)*3
corrdfcounty['pearson'] = 0
corrdfcounty['pvalue'] = 0
corrdfcounty['layer'] = list(np.repeat(["BUPL","BUPR","BUI"],len(corrdf)/3))


years=range(1810,2016,5)
from scipy.stats import pearsonr
Dat = ["BUPL","BUPR","BUI"] #,,'BUI',"BUPL","BUPR"

for dat in Dat:
    for year in years:
        selectDF = stateDF.loc[stateDF['layer'] == dat]
        coeff = pearsonr(selectDF[str(year)+"_old"], selectDF[str(year)+"_new"])[0]
        corrdf['pearson'].loc[(corrdf['year']==year)&(corrdf['layer']==dat)] = coeff
        pvaleu = pearsonr(selectDF[str(year)+"_old"], selectDF[str(year)+"_new"])[1]
        corrdf['pvalue'].loc[(corrdf['year']==year)&(corrdf['layer']==dat)] = pvaleu
        
        selectDF = countyDF.loc[countyDF['layer'] ==dat]
        coeff = pearsonr(selectDF[str(year)+"_old"], selectDF[str(year)+"_new"])[0]
        corrdfcounty['pearson'].loc[(corrdfcounty['year']==year)&(corrdf['layer']==dat)] = coeff
        pvaleu = pearsonr(selectDF[str(year)+"_old"], selectDF[str(year)+"_new"])[1]
        corrdfcounty['pvalue'].loc[(corrdfcounty['year']==year)&(corrdf['layer']==dat)] = pvaleu


    
    
import matplotlib.pyplot as plt
import numpy as np


# Change the style of plot
plt.figure(figsize=(20,20), dpi=300)
 
# Create a color palette
palette = plt.get_cmap('Set1')
 
# Plot multiple lines
Dat = ["BUI"] #,'BUI',"BUPL","BUPR"
years=range(1810,2011,50)
for dat in Dat:
    corrDF = corrdf.loc[(corrdf['layer'] == dat)&(corrdf['year']!=2020)]
    corrDFcounty =  corrdfcounty.loc[(corrdfcounty['layer'] == dat)&(corrdf['year']!=2020)]
    
    plt.plot(corrDF['year'], corrDF['pearson'], marker='', color=palette(2), linewidth=1.5, alpha=0.9, label='State level')
    plt.plot(corrDFcounty['year'], corrDFcounty['pearson'], marker='', color=palette(1), linewidth=1.5, alpha=0.9, label='County level')
    
    # Add legend
    plt.title(dat, loc='left', fontsize=20, fontweight=0, color='black')
    plt.xticks(list(years))
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Pearson", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylim(0, 1)
    plt.legend(loc='lower left', ncol=1)

    plt.savefig("/Users/yoonjung/Library/CloudStorage/OneDrive-UCB-O365/Research/HISPLUS/plots/"+dat+"_HISDACv1_v2.png",dpi=300)
    plt.show()
        
 
# Add titles

#built year comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.metrics import precision_recall_fscore_support

loaded = np.load(r'/Users/yoonjung/FSU/2022/Research/HISPLUS/counties_rast250.npz')
counties_rast = loaded['counties_rast250'].flatten()


DF = pd.DataFrame()
DF['county'] = counties_rast.flatten()

del counties_rast, loaded

countyDF = pd.DataFrame()
countyDF['county']=DF['county'].unique()[1:]
countyDF['county']= countyDF['county'].apply(lambda x: '{0:0>5}'.format(x))
countyDF['state']  = countyDF['county'].str[0:2]
stateDF = pd.DataFrame()
stateDF['state']=['01', '03','04', '05', '06', '07','08', '09', '10', '11', '12', '13','14','15', '16',
       '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
       '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
       '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
       '51', '53', '54', '55', '56']

#stateDF = pd.read_csv('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/BUIstatelevelsummary.csv')
stateDF ['state']= stateDF ['state'].apply(lambda x: '{0:0>2}'.format(x))

#countyDF= pd.read_csv('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/BUIcountylevelsummary.csv')
countyDF['county']= countyDF['county'].apply(lambda x: '{0:0>5}'.format(x))

Dat = ["FBUY" ] #["BUPL" ,"BUPR", 'BUI']
for dat in Dat:
        
        with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/FBUY.tif") as img :
        
            new= img.read()

            DF["FBUY_new"]= new.flatten()
    
        with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/HISAC/FBUY.tif") as img :
        
            old= img.read()
            DF["FBUY_old"]= old.flatten()
            
            del old, new, img
            df = DF.loc[(DF["FBUY_new"]!=0)&(DF["FBUY_old"]!=0)&(DF['county']!=0)&(DF['county']!=-1)]
            #df = DF.loc[(DF['county']!=0)&(DF['county']!=-1)]
    
            df = df.groupby(['county'])["FBUY_new","FBUY_old"].mean().reset_index()
            df['county']= df['county'].apply(lambda x: '{0:0>5}'.format(x))
            df['state'] = df['county'].str[0:2]
            countyDF = pd.merge(countyDF, df, on = ['county','state'])
            
            dfstate = df.groupby(['state'])["FBUY_new","FBUY_old"].mean().reset_index()
            stateDF = pd.merge(stateDF, dfstate, on = ['state'])
            
            #DF = DF.iloc[:,0:1]
            
            stateDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'statelevelsummary.csv',index=False)
            countyDF.to_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/'+dat+'countylevelsummary.csv',index=False)





from sklearn.metrics import f1_score
df = DF.loc[(DF["FBUY_new"]!=0)&(DF["FBUY_old"]!=0)&(DF['county']!=0)&(DF['county']!=-1)]
f1_score(df["FBUY_old"],  df["FBUY_new"], average=None)

from sklearn.metrics import precision_recall_fscore_support as score


precision, recall, fscore, support = score(df["FBUY_old"],  df["FBUY_new"])
accuracydf = pd.DataFrame()
accuracydf['recall'] = recall
accuracydf['fscore'] = fscore
accuracydf['support'] = support
accuracydf.iloc[:,0:2]= round(accuracydf.iloc[:,0:2],2)*100



print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


#FBUY
from sklearn.metrics import confusion_matrix

with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/FBUY.tif") as img :

    new= img.read()
    DF["FBUY_new"]= new.flatten()
    
with rio.open("/Users/yoonjung/FSU/2022/Research/HISPLUS/HISAC/FBUY.tif") as img :

    new= img.read()
    DF["FBUY_old"]= new.flatten()

selDF = DF[(DF['county']!=0)&(DF['county']!=-1)]
selDF['FBUY_new']=selDF['FBUY_new'].apply(lambda x : 1 if (x >0) else 0)
selDF['FBUY_old']= selDF['FBUY_old'].apply(lambda x : 1 if (x >0) else 0)
cf_matrix = confusion_matrix(selDF["FBUY_new"],selDF["FBUY_old"])

labels = ["True Negtive","False Posistive","False Negtive","True Posistive"]
labels = np.asarray(labels).reshape(2,2)
group_names = ["True Negtive","False Positive","False Negtive","True Positive"]
#group_counts = ["{0:0.0f}".format(value) for value in
#                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)] #group_counts,
labels = np.asarray(labels).reshape(2,2)
Cf_matrix = [value for value in
                    ( cf_matrix.flatten()/np.sum(cf_matrix))*100]
Cf_matrix= np.asarray(Cf_matrix).reshape(2,2)
 
#making plot
sns.set(font_scale=1.5)
ax = sns.heatmap(Cf_matrix, annot=labels, fmt="", cmap='YlGnBu',annot_kws={"size": 16}).set_title('(d)',x=0)
sns.set_style('white')

ax.figure.savefig("/Users/yoonjung/Library/CloudStorage/OneDrive-UCB-O365/Research/HISPLUS/plots/FBUY_HISDACv2_v1_sconfusionmaxtrix.png",dpi=300)





#HISDACv1-v2
#BUI 
import pandas as pd
import geopandas as gpd

countyDF = pd.read_csv('/Users/yoonjung/FSU/2022/Research/HISPLUS/AllLayers_countylevelsummary.csv')

countyb = gpd.read_file('/Users/yoonjung/Library/CloudStorage/OneDrive-UCB-O365/Research/HISPLUS/boundary/US_county_2010/US_county_2010.shp')
countyb.crs
countyb['area']= countyb['geometry'].area

countyDF['county']= countyDF['county'].apply(lambda x: '{0:0>5}'.format(x))
countyDF['state']= countyDF['state'].apply(lambda x: '{0:0>2}'.format(x))
