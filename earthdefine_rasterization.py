# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:30:38 2023

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

target_variable = ['numofbuilding'] #'BuildingAreaSqFt' #'uncertainty' 
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

out_surface_BUPR = np.zeros((len(x_range),len(y_range))).astype(np.float32)

eddf = geopandas.read_file('C:/Users/yoah2447/Documents/Yoonjung/EarthDefine_US_3D_Building_Footprints_CO_2019/EarthDefine_US_3D_Building_Footprints_CO_2019_points.shp')
eddf.geometry = eddf.geometry.to_crs(crs_grid) #crs_grid , raster.rio.crs
eddf=eddf[~eddf.geometry.is_empty]
edgdf =eddf.groupby(['str_UUID'])['Area_SqFt', 'Stories', 'GrossArea', 'Volume'].mean().reset_index()




