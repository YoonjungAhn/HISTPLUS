# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:58:27 2023

@author: yoah2447
"""

import pandas as pd
import glob
import geopandas

#integrated data completeness
County = []
Complete =[]

for name in glob.glob('D:/DATA/01_ParcelOCMZtraxMerge/*.csv'):
    df = pd.read_csv(name)
    df['YearBuilt'] = df['YearBuilt'].fillna(0)
    DF = df[['FIPS','YearBuilt']]
    DF['FIPS']= DF['FIPS'].apply(lambda x: '{0:0>5}'.format(x))
    
    county = DF['FIPS'].unique()[0]
    County.append(county)
    builyearcomplete = round(len(DF.loc[DF['YearBuilt']!=0])/len(DF),2)
    Complete.append(builyearcomplete)
    
DF = pd.DataFrame()
DF['county'] = County
DF['Complete'] = Complete
DF['county'] = DF['county'].str.replace('38087.0','38087')


DF.to_csv('D:/DATA/Results/Builtyear_completeness.csv',index=False)


US2010 = geopandas.read_file('D:DATA/NHGIS/boundary2010_2008/nhgis0024_shape/nhgis0030_shapefile_tl2010_us_county_2010/US_county_2010.shp')

US2010by = pd.merge(US2010, DF, left_on = 'GEOID10', right_on = 'county', how= 'outer')


#mapping
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Set up the color sheme:
import mapclassify as mc
import geoplot as gplt
scheme = mc.Quantiles(US2010by['Complete'], k=10)

# Map
gplt.choropleth(US2010by, 
    hue='Complete', 
    linewidth=.1,
    scheme=scheme, cmap='inferno_r',
    legend=True,
    edgecolor='black',
    ax=ax
);

ax.set_title('Unemployment rate in US counties', fontsize=13);
plt.style.use('seaborn-poster')

fig, ax = plt.subplots(1, figsize=(10,15))
US2010by.plot(ax=ax, column='Complete', cmap='Blues', alpha=1, legend=True, scheme='User_Defined', 
              classification_kwds={'bins':[0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8, 0.9,1]}) # scheme='quantiles', k=10)
ax.get_legend().set_bbox_to_anchor((1.05, 0.25, 0.2, 0.5))
ax.set_facecolor('white')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.savefig("D:/DATA/Results/builtyear_completeness.png", dpi=400, bbox_inches='tight')


#ax.legend(loc='lower left')
plt.set_title('(a)', fontsize=20)
#plt.set_xlabel('Latitute', fontsize=16)
#plt.set_ylabel('Intensity', fontsize=16)
