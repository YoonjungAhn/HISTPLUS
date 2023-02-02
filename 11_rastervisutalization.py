# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:02:06 2023

@author: yoah2447
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd

import earthpy as et
import earthpy.plot as ep

# Prettier plotting with seaborn
sns.set(font_scale=1.5)



# user parameters #########################################################################

years=range(1900,2025,5)
for year in years :
    path = 'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUI/'    
    netcdf_file = path + str(year) +'_35013counties_resident_surface_BUI_sum.tif' # input netcdf file
    
    lidar_chm_im = rxr.open_rasterio(netcdf_file,
                                     masked=True).squeeze()
    
    poly = gpd.read_file('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
    crop_extent = poly.loc[(poly.STATEFP== '35')&(poly.COUNTYFP == '013')]
    crop_extent.crs = {'init' :'epsg:4269'}
    crop_extent.geometry = crop_extent.geometry.to_crs(lidar_chm_im.rio.crs)
    
    
    #f, ax = plt.subplots(figsize=(10, 5))
    #lidar_chm_im.plot.imshow()
    #ax.set(title="Lidar Canopy Height Model (CHM)")
    #ax.set_axis_off()
    #plt.show()
    
    
    #fig, ax = plt.subplots(figsize=(6, 6))
    #crop_extent.plot(ax=ax)
    #ax.set_title("Shapefile Crop Extent",fontsize=16)
    #plt.show()
    
    
    #f, ax = plt.subplots(figsize=(10, 5))
    #lidar_chm_im.plot.imshow(ax=ax)
    
    #crop_extent.plot(ax=ax,
    #                 alpha=.8)
    #ax.set(title="Raster Layer with Shapefile Overlayed")
    
    #ax.set_axis_off()
    #plt.show()
    
    
    lidar_clipped = lidar_chm_im.rio.clip(crop_extent.geometry.apply(mapping),
                                          crop_extent.crs)
    
    f, ax = plt.subplots(figsize=(10, 4))
    lidar_clipped.plot(ax=ax, vmin=0,vmax=34000)
    ax.set(title="Settlement development 1900-2020")
    ax.set_axis_off()
    plt.show()
    
    
    f.savefig("C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUI_image/"+str(year)+"BUI.png")


# build gif
import glob
output_gif = r'C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUI_image/tmp.gif'
frame_duration = 0.1
with imageio.get_writer(output_gif, mode='I',duration=frame_duration) as writer:
    for name in glob.glob('C:/Users/yoah2447/Documents/Yoonjung/HISTPLUS/raster/BUI_image/*.png'):
        image = imageio.imread(name)
        writer.append_data(image)
    
with imageio.get_writer(output_gif, mode='I',duration=frame_duration) as writer:
    
for filename in filenames:
    if os.path.exists(filename):
        image = imageio.imread(filename)
        writer.append_data(image)






# Open the file using a context manager ("with rio.open" statement)
with rio.open(netcdf_file) as dem_src:
    dtm_pre_arr = dem_src.read(1)
    
    
ep.plot_bands(dtm_pre_arr,
              title="Lidar Digital Elevation Model (DEM) \n Boulder Flood 2013",
              cmap="Greys",
              )

plt.show()


f, ax = plt.subplots(figsize=(10, 5))
dtm_pre_arr.plot.imshow(ax=ax)

crop_extent.plot(ax=ax,
                 alpha=.8)
ax.set(title="Raster Layer with Shapefile Overlayed")

ax.set_axis_off()
plt.show()