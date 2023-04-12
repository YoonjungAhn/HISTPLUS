# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:30:04 2023

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

template_raster = 'C:/Yoonjung/HISTPLUS/raster/BUPL_2015.tif'#PATH_TO_FBUY_RASTER (.tif)
template_raster = 'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUPL_2015.tif'#PATH_TO_FBUY_RASTER (.tif)

csv_folder = 'C:/Yoonjung/DATA_USAGE_TUTORIAL/exercises'# path where csv files with input (x,y,target_variable) are stored
surface_folder = 'C:/Yoonjung/HISTPLUS/raster'#path where output tif files are stored
bitdepth = gdal.GDT_Float64# or gdal.GDT_Float32, whatever is suitable
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


years= [ 1950, 1990, 2000]
for year in years: 
    out_surface= np.zeros((len(x_range),len(y_range))).astype(np.int64)

    for name in glob.glob('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/MTBF33_wgs84/*.shp'):

        gpdf  = geopandas.read_file(name)
        gpdf.geometry = gpdf.centroid
        gpdf.crs = {'init' :'epsg:4326'}
        gpdf['numofbuilding']=1
        gpdf['round_year'] = gpdf['year_built'].apply(lambda x : myround(x)).replace(0,np.nan)
        gpdf['lat'] = gpdf.geometry.x
        gpdf['long'] = gpdf.geometry.y
        gpdf1 = gpdf.drop_duplicates(subset =['lat','long'])
        
        #make to geopands
        gpdf1.geometry = gpdf1.geometry.to_crs(crs_grid) #crs_grid , raster.rio.crs
        gpdf1=gpdf1[~gpdf1.geometry.is_empty]
        
        y=gpdf1.geometry.y.values.astype(int)
        x= gpdf1.geometry.x.values.astype(int)
            
        gpdf2 = gpdf1.loc[(gpdf1.round_year>=1810)&(gpdf1.round_year<year+5)]
    
        #rasterization
        
        target_variable = 'BUPL'
        statsvals = gpdf2['numofbuilding'].values.astype(np.int64) #.astype(float) 
        statistic = np.sum
        statistic_str= 'sum' 
        xbins, ybins = len(x_range), len(y_range) #number of bins in each dimension
        #curr_surface = scipy.stats.binned_statistic_2d(gpdf2.geometry.x.values,gpdf2.geometry.y.values,statsvals,statistic,bins=[xbins, ybins],range=rasterrange) 
        curr_surface = scipy.stats.binned_statistic_2d(gpdf2.geometry.x.values,gpdf2.geometry.y.values,statsvals,statistic=statistic_str,bins = [xbins, ybins],range=rasterrange)        
        out_surface = np.maximum(out_surface,np.nan_to_num(curr_surface.statistic))   
        out_surface1 = out_surface.astype(np.int64) #.astype(np.int64)
        #path = os.path.join(parent_dir,target_variable,str(year)[0:-2])
    
        gdalNumpy2floatRaster_compressed(np.rot90(out_surface1),'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/MTBF33_wgs84/'+str(year)+'_'+name[72:77]+'_counties_surface_%s_%s.tif' %(target_variable,statistic_str),template_raster,cols,rows,bitdepth)

