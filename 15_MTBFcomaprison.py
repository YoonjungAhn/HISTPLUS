# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:37:19 2023

@author: yoah2447
"""

import numpy as np
import pandas as pd
import rasterio as rio
from scipy.stats import pearsonr
### the data stack


### county ids rasterized:
loaded = np.load(r'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/counties_rast250.npz')
counties_rast = loaded['counties_rast250'].flatten()

### fips of the mtbf counties
valcounties_fips_int=[25015,25025,25001,41003,25027,8013,24005,25019,12081,25007,25013,18163,25003,25005,53061,25017,27037,34025,27003,27123,27163,12115,37119,36081,27053,25009,25021,25023,27019,12057,25011]
valcounties_fips_int=[25015,25025,25001,25027,24005,25019,25007,25013,25003,25005,25017,34025,27163,36081,25009,25021,25023,25011]
valcounties_fips_int=[8013]


DF = pd.DataFrame()
DF['county'] = counties_rast
del counties_rast, loaded

#grided MTBF data
ref_bldg_count_arr = np.load(r'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/hist2d_refbldg_counts_noPolygDuplNoSmallPolyg2_2022_incl_nyc.npz')
ref_bldg_count_arr = ref_bldg_count_arr['arr_0']


years=range(1810,2016,5)
years= [1950, 1990,2000]
for year in years:
    ### now load your gridded surface for year x as a numpy array, than flatten it
    ### get the area within relevant counties FIPS:
    refbldgcounts_curryr = np.rot90(ref_bldg_count_arr[:,:,list(years).index(year)]).flatten()
    ### I would create a pandas dataframe with three columns, your counts, the refbldgcounts_curryr, and counties_rast.
    DF[year]=refbldgcounts_curryr 
    ### each row represents a grid cell
    ### then group by counties_rast column for county-level analysis
del ref_bldg_count_arr, refbldgcounts_curryr 

#mtbfdf['totalnum'] = mtbfdf.iloc[:,1:].sum(axis=1, skipna=True)


years= [ 1950, 1990, 2000]
for year in years:
    with rio.open("C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUPL/"+str(year)+"_BUPL.tif") as img :
        BUPR_new= img.read()
        DF['BUPL'+str(year)]= BUPR_new.flatten()
        
years= [ 1950, 1990, 2000]
for year in years:
    with rio.open("C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/MTBF33_wgs84/"+str(year)+"_boulder_counties_surface_BUPL_sum.tif") as img :
        BUPR_new= img.read()
        DF[year]= BUPR_new.flatten().astype(int)

        
import seaborn as sns
corrdf = pd.DataFrame()
corrdf['year'] = [1990, 1950, 2000]
corrdf['pearson'] = 0
corrdf['pvalue'] = 0

mtbdf = DF[DF.county.isin(valcounties_fips_int)&(DF['county']!=0)&(DF['county']!=-1)]
mtbdf.groupby(by=["county"], dropna=False).sum()

DN = "BUPL"
for year in years:
    mtbdfsel  = mtbdf.loc[(mtbdf[year]!=0)&(mtbdf[DN+str(year)]!=0)&(mtbdf['county']!=0)&(mtbdf['county']!=-1)]
    pvaleu = pearsonr(mtbdfsel[year],mtbdfsel[DN+str(year)])[1]
    corrdf['pvalue'].loc[corrdf['year']==year] = pvaleu
    pearson =pearsonr(mtbdfsel[year],mtbdfsel[DN+str(year)])[0]
    corrdf['pearson'].loc[corrdf['year']==year] = pearson
    
    sns.scatterplot(x=mtbdfsel[year], y=mtbdf[DN+str(year)], data=mtbdfsel)
        