

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:32:19 2021
@author: Johannes
"""
import os,sys,subprocess
import pandas as pd

os.chdir(r'C:/Users/yoah2447/Documents/Yoonjung/DATA_USAGE_TUTORIAL/histplus-main/')

### list of counties (to be automated)
incounties=['44001'] #,'44003','44005','44007']

### distance for matching in m
matchdist=25

#select building area (maximum building)
df = pd.read_csv('ztrax_2021_extraction_county_areatypes_44001.csv')
maxdf = df.groupby(['RowID','BuildingOrImprovementNumber'])['BuildingAreaStndCode','BuildingAreaSqFt'].max().reset_index()
DForig = pd.merge(maxdf, df, how='left', on =['RowID','BuildingOrImprovementNumber','BuildingAreaStndCode','BuildingAreaSqFt'])
DForig.to_csv('./ztrax_2021_extraction_county_buildings_44001.csv')


fmw_path='C:/Users/yoah2447/Documents/Yoonjung/DATA_USAGE_TUTORIAL/histplus-main/ztrax_ocm_integration_fme_county.fmw'
# path to fmw file
ztrax_csv_path=r'C:/Users/yoah2447/Documents/Yoonjung/DATA_USAGE_TUTORIAL/histplus-main' # folder where ztrax csvs are stored
ocm_gdb_path=r'C:/Users/yoah2447/Documents/Yoonjung/DATA_USAGE_TUTORIAL/histplus-main/OCM/data' # folder where the ocm geodatabases are stored
output_dir='C:/Users/yoah2447/Documents/Yoonjung/DATA_USAGE_TUTORIAL/histplus-main' # output directory
for county in incounties:
    state=county[:2]
    call = r'"C:\Program Files\FME2019\fme.exe" %s --SourceDataset_FILEGDB "%s\ocm_merged_state_%s.gdb" --SourceDataset_CSV2 "%s\ztrax_2021_extraction_county_buildings_%s.csv" --DestDataset_FILEGDB "%s\ocm_w_ztrax_%s.gdb" --ft "%s" --matchdist "%s"' %(fmw_path,ocm_gdb_path,state,ztrax_csv_path,county,output_dir,county,county,matchdist)
    print(call)
    response=subprocess.check_output(call, shell=True)
    print(response)

