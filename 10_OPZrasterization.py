# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:53:59 2023

@author: yoah2447
"""

import numpy as np
import os
import scipy.stats
import subprocess
from osgeo import gdal
import pandas as pd
import matplotlib.pyplot as plt
import math
import geopandas
import glob

target_variable = ['numofbuilding','BuildingAreaSqFt', 'YearBuilt'] #'BuildingAreaSqFt' #'uncertainty' 
xcoo_col,ycoo_col = 'x','y'    
statistic = np.mean
statistic_str= 'sum'    
template_raster = 'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUPL_2015.tif'#PATH_TO_FBUY_RASTER (.tif)
csv_folder = 'C:/Yoonjung/DATA_USAGE_TUTORIAL/exercises'# path where csv files with input (x,y,target_variable) are stored
surface_folder = 'C:/Yoonjung/HISTPLUS/raster'#path where output tif files are stored
bitdepth = gdal.GDT_Int16 ## or gdal.GDT_Float32, whatever is suitable
crs_coords = '+proj=longlat +ellps=clrk66 +datum=NAD27 +no_defs' #source SRS
crs_grid = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23.0 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs' #target SRS

gdal_edit = r'python C:\Users\yoah2447\Anaconda3\envs\geo\Scripts\gdal_edit.py'

def myround(x, base=5):
    try:
        return base * round(x/base)
    except ValueError :
        return np.nan

def gdalNumpy2floatRaster_compressed(array,outname,template_georef_raster,x_pixels,y_pixels,px_type):

    dst_filename = outname

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, 1, px_type)   
    dataset.GetRasterBand(1).WriteArray(array)                
    mapraster = gdal.Open(template_georef_raster, gdal.GA_ReadOnly) #should check if works
    proj=mapraster.GetProjection() #you can get from a existing tif or import 
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None                

    #set bounding coords
    ulx, xres, xskew, uly, yskew, yres  = mapraster.GetGeoTransform()
    lrx = ulx + (mapraster.RasterXSize * xres)
    lry = uly + (mapraster.RasterYSize * yres)            
    mapraster = None
                    
    gdal_cmd = gdal_edit+' -a_ullr %s %s %s %s "%s"' % (ulx,uly,lrx,lry,outname)
    print(gdal_cmd)
    response=subprocess.check_output(gdal_cmd, shell=True)
    print(response)
    
    outname_lzw=outname.replace('.tif','_lzw.tif')
    gdal_translate = r'gdal_translate %s %s -co COMPRESS=LZW' %(outname,outname_lzw)
    print(gdal_translate)
    response=subprocess.check_output(gdal_translate, shell=True)
    print(response)
    os.remove(outname)
    os.rename(outname_lzw,outname)
    
    
#read raster
raster = gdal.Open(template_raster)
cols = raster.RasterXSize
rows = raster.RasterYSize
geotransform = raster.GetGeoTransform()


counties = ['08013','12057','12081','12115','18163','24005','25001','25003',
            '25005','25007','25009','25011','25013','25015','25017','25019',
            '25021','25023','25025','25027','27003','27019','27037','27053',
            '27123','27163','34025','37119','36005','36047','36061','36081','36085'] 
    
LUcode = ['1000','1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1012','1013','1018','1100','1101','1102',
          '1103','1104','1105','1106','1107','1108','1109','1110','1111','1112','1113','1114','1999']

df = pd.read_csv('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/OPZ/parcelocmztraxmerge_35013.csv')
df['round_year'] = df['YearBuilt'].apply(lambda x: myround(x)).replace(0,np.nan)
df['numofbuilding'] = 1
#df['STD_LAND_U'] = df['STD_LAND_U'].astype(str).str[:-2]
#DFLU = df[df['STD_LAND_U'].isin(LUcode) ]
gpdf1= geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Long, df.Lat))
gpdf1.crs = {'init' :'epsg:4326'}
gpdf1.geometry = gpdf1.geometry.to_crs(crs_grid) #crs_grid , raster.rio.crs
gpdf1=gpdf1[~gpdf1.geometry.is_empty]


ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform() # ulx, uly is the upper left corner, lrx, lry is the lower right corner
lrx = ulx + (raster.RasterXSize * xres)
lry = uly + ( raster.RasterYSize * yres)
pixelWidth = int(abs(geotransform[1]))
pixelHeight = int(abs(geotransform[5]))
x_range = np.arange(ulx,lrx,pixelWidth) #setting grid spacing for x and y gpdf.geometry.x.values.max()
y_range = np.arange(lry,uly,pixelHeight)
topleftX = geotransform[0]
topleftY = geotransform[3]
pixelWidth = int(abs(geotransform[1]))
pixelHeight = int(abs(geotransform[5]))
rasterrange=[[topleftX,topleftX+pixelWidth*cols],[topleftY-pixelHeight*rows,topleftY]]    


y=gpdf1.geometry.y.values.astype(int)
x= gpdf1.geometry.x.values.astype(int)



years=range(1810,2025,5)
parent_dir = 'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/'

for year in years:
    print(year)
    year = float(year)
    gpdf = gpdf1.loc[(gpdf1.round_year>=1810)&(gpdf1.round_year<year+5)]
    
    #making raster
    #BUI
    target_variable = 'BUI'
    statsvals = gpdf['BuildingAreaSqFt'].values.astype(int) 
    statistic = np.mean
    statistic_str= 'sum'  
    xbins, ybins = len(x_range), len(y_range) #number of bins in each dimension
    out_surface =np.zeros((cols,rows)).astype(np.float32)
    out_surface = np.zeros((len(x_range),len(y_range))).astype(np.float32)
    curr_surface = scipy.stats.binned_statistic_2d(gpdf.geometry.x.values,gpdf.geometry.y.values,statsvals,statistic=statistic_str,bins = [xbins, ybins],range=rasterrange)        
    out_surface = np.maximum(out_surface,np.nan_to_num(curr_surface.statistic))       

    #path = os.path.join(parent_dir,target_variable,str(year)[0:-2])
    #os.mkdir(path)
    gdalNumpy2floatRaster_compressed(np.rot90(out_surface),'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUI/'+str(year)[0:-2]+'_35013counties_resident_surface_%s_%s.tif' %(target_variable,statistic_str),template_raster,cols,rows,bitdepth)
    
    
    #BUPR
    target_variable = 'BUPR'
    statsvals = gpdf['numofbuilding'].values.astype(int)
    statistic = np.mean
    statistic_str= 'sum'   
    xbins, ybins = len(x_range), len(y_range) #number of bins in each dimension
    out_surface =np.zeros((cols,rows)).astype(np.float32)
    out_surface = np.zeros((len(x_range),len(y_range))).astype(np.float32)
    curr_surface = scipy.stats.binned_statistic_2d(gpdf.geometry.x.values,gpdf.geometry.y.values,statsvals,statistic=statistic_str,bins = [xbins, ybins],range=rasterrange)        
    out_surface = np.maximum(out_surface,np.nan_to_num(curr_surface.statistic))       
    

    gdalNumpy2floatRaster_compressed(np.rot90(out_surface),'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUPR/'+str(year)[0:-2]+'_35013counties_resident_surface_%s_%s.tif' %(target_variable,statistic_str),template_raster,cols,rows,bitdepth)
    
    #BUPL
    target_variable = 'BUPL'
    statsvals = gpdf['numofbuilding'].values.astype(int) 
    statistic = np.mean
    statistic_str= 'sum'  
    xbins, ybins = len(x_range), len(y_range) #number of bins in each dimension
    out_surface =np.zeros((cols,rows)).astype(np.float32)
    out_surface = np.zeros((len(x_range),len(y_range))).astype(np.float32)
    curr_surface = scipy.stats.binned_statistic_2d(gpdf.geometry.x.values,gpdf.geometry.y.values,statsvals,statistic=statistic_str,bins = [xbins, ybins],range=rasterrange)        
    out_surface = np.maximum(out_surface,np.nan_to_num(curr_surface.statistic))       
    out_surface[out_surface> 1] = 1
    
    gdalNumpy2floatRaster_compressed(np.rot90(out_surface),'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUPL/'+str(year)[0:-2]+'_35013counties_resident_surface_%s_%s.tif' %(target_variable,statistic_str),template_raster,cols,rows,bitdepth)


#FBUY    
years=range(1810,2021,1)
out_surface = np.ones((len(x_range),len(y_range))).astype(np.float32)*2030

for name in glob.glob('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/OPZ/*.csv'):
    df = pd.read_csv(name)
    
    #df['STD_LAND_U'] = df['STD_LAND_U'].astype(str).str[:-2]
    #DFLU = df[df['STD_LAND_U'].isin(LUcode) ]
    gpdf1= geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Long, df.Lat))
    gpdf1.crs = {'init' :'epsg:4326'}
    gpdf1.geometry = gpdf1.geometry.to_crs(crs_grid) #crs_grid , raster.rio.crs
    gpdf1=gpdf1[~gpdf1.geometry.is_empty]
    
    y=gpdf1.geometry.y.values.astype(int)
    x= gpdf1.geometry.x.values.astype(int)
        
    gpdf = gpdf1.copy()
    target_variable = 'FBUY'
    statsvals = gpdf['YearBuilt'].values.astype(int)
    statistic = np.min
    statistic_str= 'min' 
    xbins, ybins = len(x_range), len(y_range) #number of bins in each dimension
    out_surface =np.zeros((cols,rows)).astype(np.float32)
    out_surface = np.zeros((len(x_range),len(y_range))).astype(np.float32)
    curr_surface = scipy.stats.binned_statistic_2d(gpdf.geometry.x.values,gpdf.geometry.y.values,statsvals,statistic=statistic_str,bins = [xbins, ybins],range=rasterrange)        
    out_surface = np.minimum(out_surface,np.nan_to_num(curr_surface.statistic))       
    out_surface = out_surface.replace(2030,0)
    gdalNumpy2floatRaster_compressed(np.rot90(out_surface),'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/FBUY/'+str(year)[0:-2]+'_counties_resident_surface_%s_%s.tif' %(target_variable,statistic_str),template_raster,cols,rows,bitdepth)
    
    
    
