# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:46:26 2022

@author: yoah2447
"""

import pandas as pd
import geopandas 
import time
import math
import numpy as np
from shapely import wkt
import fiona 

def normalize_column(df):
    return (df - df.min()) / (df.max() - df.min())

def street_name_fix(StreetName):
    replacements =  {
        'Alley':'aly',
        'Avenue':'ave',
        'Boulevard':'blv',
        'Boulevard':'blvd',
        'Circle':'cir',
        'Court':'ct',
        'Cove':'cv',
        'Canyon':'cyn',
        'Drive':'dr',
        'Expressway':'expy',
        'Highway':'hwy',
        'Lane':'ln',
        'Parkway':'pkwy',
        'Place':'pl',
        'Point':'pt',
        'Road':'rd',
        'Square':'sq',
        'Street':'st',
        'Terrace':'ter',
        'Trail':'tr',
        'Trail':'trl',
        'Way':'wy'
    }
                   
    StreetName = StreetName.upper().strip().rstrip('.')
    replacements = {k.upper():v.upper() for k,v in replacements.items()}

    try:
        return '{} {}'.format(' '.join(StreetName.split()[:-1]), replacements[StreetName.split()[-1]])

    except IndexError:
        return StreetName
   
    except KeyError:
        return StreetName
    
finalcols = ['gml_id', 'state', 'county', 'height', 'Shape_Area',
       'ocm_polygon_geometry', 'ocm_point_geometry', 'APN', 'APN2', 'STATE',
       'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
       'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
       'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
       'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
       'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 'geometry',
       'parcel_polygon_geometry', 'parcel_point_geometry', 'RowID',
       'PropertyAddressLongitude', 'PropertyAddressLatitude',
       'LotSizeSquareFeet', 'ztrax_address',  'BuildingAreaSqFt',
       'YearBuilt', 'PropertyZip', 'PropertyLandUseStndCode',
       'ztrax_point_geometry', 'integrationcode', 'uncertainty']    
    
#read pacel+ztrax
countydf = pd.read_csv('D:/DATA/countyFIPS.csv')
countydf.fips
countyDF = countydf.loc[(countydf.state!="AK")&(countydf.state!="HI")&(countydf.fips!=46113)&(countydf.fips!=51515)]
countyDF.fips= countyDF.fips.apply(lambda x: '{0:0>5}'.format(x))
counties = list(countyDF.fips.apply(lambda x: '{0:0>5}'.format(x)).unique())

statDF = pd.DataFrame({'fips':counties,'totalztrax':np.nan,'totalocm':np.nan,'totalparcel':np.nan,'incompleteztrax':np.nan,'OP_intersect':np.nan, 'OP_distance': np.nan,
                       'P':np.nan,'O':np.nan,'Z':np.nan,'PZ11':np.nan,'OZ11':np.nan,'OP11':np.nan,'OPZ111':np.nan, 'PZ1n':np.nan, 
                       'OZ1n':np.nan,'OPn1':np.nan,'OPZn1n':np.nan,'OPtotalmerge':np.nan,
                       'OPZ_address':np.nan,'OPZ_intersect':np.nan,'OPZ_distance':np.nan,'finalunmatchedZ':np.nan, 
                       'unmatchedOPwithZ':np.nan,'OPZtotalmerge':np.nan})


counties.index( '12101') #conty = '01009'
#integrating OCM and Parcel 
for county in counties[342:]:
    start_time = time.time()
    
    #reading in OCM
    ocmdf =  geopandas.GeoDataFrame()
    layer = 'ocm_'+ county
    layers = geopandas.read_file('D:/DATA/OPENCITYMODEL/ocm_merged_state_'+county[0:2]+".gdb", driver='FileGDB', layer=layer)
    ocmdf = pd.concat([ocmdf, layers], axis=0)
    OCMgdfprj = ocmdf.to_crs( 'epsg:4269')
    ocmgdf = geopandas.GeoDataFrame(OCMgdfprj.drop(['longitude', 'latitude'], axis=1), geometry='geometry')
    #ocmgdf=ocmgdf[ocmgdf.geometry.is_empty].fillna(value=np.nan)
    ocmgdf['ocm_polygon_geometry']= ocmgdf.geometry.apply(lambda x: wkt.dumps(x))
    ocmgdf['ocm_point_geometry'] = ocmgdf['geometry'].centroid.apply(lambda x: wkt.dumps(x))
    OCMgdf = ocmgdf.copy()
    OCMgdf['geometry'] = OCMgdf['geometry'].centroid

    OCM = OCMgdf[['gml_id', 'state','county','height','Shape_Area','ocm_polygon_geometry','ocm_point_geometry','geometry']]
    statDF['totalocm'].loc[statDF['fips']==county] = len(OCM)
    
    #reading in parcel 
    parcel_file_path = 'D:/DATA/ParcelAtlas2022/'+county+'/parcels.shp'  # file path
    try:    
        parcel = geopandas.read_file(parcel_file_path).dropna(subset = ['geometry'])
        
        if 'YEAR_BUILT' not in parcel:
            parcel['YEAR_BUILT'] = np.nan
        if 'APN' not in parcel:
                parcel['APN'] = range(0,len(parcel))
        if 'APN2' not in parcel:
                parcel['APN2'] = np.nan     
        if 'BLDG_AREA' not in parcel:
            parcel['BLDG_AREA'] = np.nan
        if 'STD_LAND_U' not in parcel:
                parcel['STD_LAND_U'] = np.nan
        if 'SIT_FULL_S' not in parcel:
                parcel['SIT_FULL_S'] = np.nan
        if 'STATE' not in parcel:
                parcel['STATE'] = np.nan
        if 'FIPS' not in parcel:
                parcel['FIPS'] = county
        if 'COUNTY' not in parcel:
                parcel['COUNTY'] = np.nan            
        if 'SIT_HSE_NU' not in parcel:
                parcel['SIT_HSE_NU'] = np.nan     
        if 'COUNTY' not in parcel:
                parcel['COUNTY'] = np.nan     
        if 'SIT_DIR' not in parcel:
                parcel['SIT_DIR'] = np.nan  
        if 'SIT_STR_NA' not in parcel:
                parcel['SIT_STR_NA'] = np.nan     
        if 'SIT_STR_SF' not in parcel:
                parcel['SIT_STR_SF'] = np.nan     
        if 'SIT_CITY' not in parcel:
                parcel['SIT_CITY'] = np.nan                     
        if 'SIT_STATE' not in parcel:
                parcel['SIT_STATE'] = np.nan    
        if 'SIT_ZIP' not in parcel:
                parcel['SIT_ZIP'] = np.nan                   
        if 'SIT_ZIP4' not in parcel:
                parcel['SIT_ZIP4'] = np.nan    
        if 'LAND_VALUE' not in parcel:
                parcel['LAND_VALUE'] = np.nan                    
        if 'IMPR_VALUE' not in parcel:
                parcel['IMPR_VALUE'] = np.nan   
        if 'TOT_VALUE' not in parcel:
                parcel['TOT_VALUE'] = np.nan   
        if 'NO_OF_STOR' not in parcel:
                parcel['NO_OF_STOR'] = np.nan   
        if 'NO_OF_UNIT' not in parcel:
                parcel['NO_OF_UNIT'] = np.nan                   
        if 'BEDROOMS' not in parcel:
                parcel['BEDROOMS'] = np.nan                   
        if 'BATHROOMS' not in parcel:
                parcel['BATHROOMS'] = np.nan                   
        if 'SALES_PRIC' not in parcel:
                parcel['SALES_PRIC'] = np.nan                   
        if 'LOT_SIZE' not in parcel:
                parcel['LOT_SIZE'] = np.nan     
                
        parcel.rename(columns={'Xcoord':'XCOORD', 'Ycoord':'YCOORD'}, inplace=True)
        parcel['unique_APN'] = range(0,len(parcel))
        parcel['FIPS'] = parcel['FIPS'].replace(np.nan, county)
        parcel.crs = {'init' :'epsg:4269'}
        #parcelprj = parcel.to_crs( OCM.crs).replace([0], np.nan).fillna(value=np.nan).dropna(subset = ['geometry'])
        
           
        Parcel =  parcel[['unique_APN','APN', 'APN2', 'STATE', 'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR',
               'SIT_STR_NA', 'SIT_STR_SF', 'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE',
               'SIT_ZIP', 'SIT_ZIP4', 'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE',
               'SALES_PRIC', 'YEAR_BUILT', 'STD_LAND_U',
               'LOT_SIZE', 'BLDG_AREA','NO_OF_STOR', 'NO_OF_UNIT',
               'BEDROOMS', 'BATHROOMS','XCOORD', 'YCOORD','geometry']]
        Parcel['parcel_polygon_geometry'] = Parcel.geometry.apply(lambda x: wkt.dumps(x))
        Parcel['parcel_point_geometry'] = Parcel['geometry'].centroid.apply(lambda x: wkt.dumps(x))
        #Parcel['geometry'] = Parcel['geometry'].centroid
         
            
        statDF['totalparcel'].loc[statDF['fips']==county] = len(Parcel)
        
        #intersection spatial join
        instersectdf = geopandas.sjoin(OCM, Parcel, how='right', op='intersects').drop_duplicates(subset ='gml_id', keep='first' ).drop(['index_left'],axis=1).dropna(subset=['gml_id','unique_APN'])
        instersectdf['integrationcode']=np.nan
        instersectdf.loc[(instersectdf.gml_id.isna()), 'integrationcode'] = 'P'
        instersectdf.loc[(instersectdf.unique_APN.isna()), 'integrationcode'] = 'O'
        instersectdf.loc[(instersectdf.gml_id.notna())&(instersectdf.unique_APN.notna()), 'integrationcode'] = 'OP11'  
        instersectdf.loc[(instersectdf.gml_id.notna())&(instersectdf.unique_APN.notna())&(instersectdf.unique_APN.duplicated()==True), 'integrationcode'] ='OPn1'
        instersectdf['integrationcode'].unique()
        statDF['OP_intersect'].loc[statDF['fips']==county] = len(instersectdf.loc[(instersectdf.integrationcode!="O")|(instersectdf.integrationcode!="P")])
   
    except fiona.errors.DriverError:
        statDF['totalparcel'].loc[statDF['fips']==county] = 0
        statDF['OP_intersect'].loc[statDF['fips']==county] = len(ocmgdf)
    
        instersectdf = ocmgdf.copy()
        instersectdf['integrationcode']=np.nan
        instersectdf.loc[(instersectdf.gml_id.isna()), 'integrationcode'] = 'O'
        instersectdf[['unique_APN','APN', 'APN2', 'STATE',
        'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
        'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
        'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
        'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
        'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 
        'parcel_polygon_geometry', 'parcel_point_geometry']] =  pd.DataFrame([np.repeat(np.nan, 31)], index=instersectdf.index)
        instersectdf= instersectdf[['gml_id', 'state', 'county', 'height', 'Shape_Area',
               'ocm_polygon_geometry', 'ocm_point_geometry', 'geometry', 'unique_APN','APN', 'APN2',
               'STATE', 'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA',
               'SIT_STR_SF', 'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP',
               'SIT_ZIP4', 'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC',
               'YEAR_BUILT', 'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR',
               'NO_OF_UNIT', 'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD',
               'parcel_polygon_geometry', 'parcel_point_geometry', 'integrationcode']]

    #unmatched OCM
    unmatchedocm = OCM[~(OCM.gml_id.isin(instersectdf.gml_id))][['gml_id', 'state', 'county', 'height', 'Shape_Area',
           'ocm_polygon_geometry', 'ocm_point_geometry', 'geometry']]
    #unmatchedocm =  instersectdf.loc[instersectdf.integrationcode=="O"][['gml_id', 'state', 'county', 'height', 'Shape_Area',
    #       'ocm_polygon_geometry', 'ocm_point_geometry', 'geometry']]

    try: 
        #unmatched parcel 
        unmatchedparcel=   Parcel[~(Parcel.unique_APN.isin(instersectdf.unique_APN))]
    
        if (len(unmatchedocm)==0)&(len(unmatchedparcel)>0):
            #unmatchedparcel['geometry'] = unmatchedparcel['geometry'].centroid
            unmatchedparcel[['gml_id', 'state', 'county', 'height', 'Shape_Area',
                  'ocm_polygon_geometry', 'ocm_point_geometry']] =  pd.DataFrame([np.repeat(np.nan, 7)], index=unmatchedparcel.index)
            unmatchedparcel['integrationcode'] = np.nan
            unmatchedparcel['integrationcode'] = 'P'
            unmatchedparcel = unmatchedparcel[instersectdf.columns]
            OPdf = pd.concat( [ instersectdf, unmatchedparcel], axis =0) #.drop(['index_left'],axis=1)
            statDF['OPtotalmerge'].loc[statDF['fips']==county] = len(OPdf)
            OPdf = OPdf[['gml_id', 'state', 'county', 'height', 'Shape_Area',
                   'ocm_polygon_geometry', 'ocm_point_geometry', 'unique_APN','APN', 'APN2', 'STATE',
                   'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
                   'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
                   'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
                   'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 'geometry',
                   'parcel_polygon_geometry', 'parcel_point_geometry', 'integrationcode']]
            
            
            OPdf.to_csv('D:/DATA/00_ParcelOCMMerge/parcelocm_'+county+'.csv')
           #test = OPdf[OPdf.duplicated('gml_id')]
        
        else:      #(len(unmatchedocm)>0)&(len(unmatchedparcel)>0):    
            matcheddf = pd.DataFrame()
            X = len(unmatchedocm)
            while (X == X)or(X==0):
               i = 1
               nearest_data = geopandas.sjoin_nearest(unmatchedocm,unmatchedparcel,how='right', distance_col="distances", max_distance = 100)#.dropna(subset=['unique_APN'])        
               nearest_data['AreaSqFt'] = nearest_data.Shape_Area*10.764
               nearest_data.LOT_SIZE = pd.to_numeric(nearest_data.LOT_SIZE.astype(str).str[:-1], errors='coerce').fillna(0)               
               nearest_data.LotSizeSquareFeet=nearest_data.fillna(0).LOT_SIZE.astype(float).astype('int') 
               nearest_data['diagonal_square_meter'] = nearest_data.apply(lambda x: math.sqrt(2)*math.sqrt(x.LOT_SIZE)*0.3048, axis = 1)
               nearest_data['diagonal_square_meter'] = nearest_data['diagonal_square_meter'].astype(float)
               min_value = nearest_data.groupby(['gml_id'])['distances'].min().reset_index()
               nearest_data= nearest_data.merge(min_value, on=['gml_id'],suffixes=('', '_min')) #.drop_duplicates(subset=['uniqueID'])
               nearest_data['uncertainty'] =normalize_column(nearest_data.AreaSqFt.astype(float)/nearest_data.LOT_SIZE.astype(float)) + normalize_column(nearest_data.distances_min.astype(float)/nearest_data.diagonal_square_meter.astype(float))+normalize_column(nearest_data.distances_min)
               nearest_DF = nearest_data.sort_values(['distances_min','uncertainty'], ascending=True).drop_duplicates(subset ='gml_id', keep='first' )
               nearestDAT = nearest_DF.drop(['distances','distances_min','uncertainty'],axis=1)   
               unmatchedocm2 =  unmatchedocm[~(unmatchedocm.gml_id.isin(nearestDAT.gml_id))]
               matcheddf= pd.concat([matcheddf,nearestDAT ])
               unmatchedocm =unmatchedocm2
               X = int(len(unmatchedocm2))
               
               print(X, i)
               if  (X ==X)or(X==0):
                   break
            Matcheddf = matcheddf.drop(['AreaSqFt', 'diagonal_square_meter'],axis=1)
            statDF['OP_distance'].loc[statDF['fips']==county] = len(Matcheddf)
            
            finalunmatchedparcel=  unmatchedparcel[~(unmatchedparcel.unique_APN.isin(Matcheddf.unique_APN))]
            
            finalunmatchedocm=  unmatchedocm[~(unmatchedocm.gml_id.isin(Matcheddf.gml_id))]
            finalunmatchedocm['integrationcode'] = np.nan
            finalunmatchedocm['integrationcode'] = 'O'
            finalunmatchedocm[['unique_APN','APN', 'APN2', 'STATE', 'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR',
                   'SIT_STR_NA', 'SIT_STR_SF', 'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE',
                   'SIT_ZIP', 'SIT_ZIP4', 'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE',
                   'SALES_PRIC', 'YEAR_BUILT', 'STD_LAND_U',
                   'LOT_SIZE', 'BLDG_AREA','NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS','XCOORD', 'YCOORD','parcel_polygon_geometry', 'parcel_point_geometry']] =  pd.DataFrame([np.repeat(np.nan, 31)], index=finalunmatchedocm.index)
            Finalunmatchedocm= finalunmatchedocm[instersectdf.columns]
            OPdf =  pd.concat([ instersectdf, Matcheddf, finalunmatchedparcel, Finalunmatchedocm], axis =0)

            
            OPdf.loc[ (OPdf.gml_id.isna()), 'integrationcode'] = 'P'
            OPdf.loc[( OPdf.unique_APN.isna()), 'integrationcode'] = 'O'
            OPdf.loc[( OPdf.gml_id.notna())&( OPdf.unique_APN.notna()), 'integrationcode'] = 'OP11'  
            OPdf.loc[( OPdf.gml_id.notna())&( OPdf.unique_APN.notna())&( OPdf.unique_APN.duplicated()==True), 'integrationcode'] ='OPn1'
            #OPdf.groupby('integrationcode')['unique_APN',"gml_id"].count()
            
           
            statDF['OPtotalmerge'].loc[statDF['fips']==county] = len(OPdf)
            OPdf.to_csv('D:/DATA/00_ParcelOCMMerge/parcelocm_'+county+'.csv')
            del  nearestDAT, nearest_data
    
    except NameError:
        
        if  (len(unmatchedocm)==0):
            OPdf = instersectdf.copy()
            OPdf.loc[OPdf['integrationcode']=="O", 'geometry'] = geopandas.GeoSeries.from_wkt(OPdf.loc[OPdf['integrationcode']=="O"].ocm_polygon_geometry)
            OPdf.to_csv('D:/DATA/00_ParcelOCMMerge/parcelocm_'+county+'.csv')
            
        else:
            unmatchedocm['integrationcode'] = np.nan
            unmatchedocm['integrationcode'] = 'O'
            #unmatchedocm['geometry'] = unm`atchedocm['ocm_polygon_geometry']
            unmatchedocm[['APN', 'APN2', 'STATE', 'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR',
                   'SIT_STR_NA', 'SIT_STR_SF', 'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE',
                   'SIT_ZIP', 'SIT_ZIP4', 'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE',
                   'SALES_PRIC', 'YEAR_BUILT', 'STD_LAND_U',
                   'LOT_SIZE', 'BLDG_AREA','NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS','XCOORD', 'YCOORD','parcel_polygon_geometry', 'parcel_point_geometry']] =  pd.DataFrame([np.repeat(np.nan, 30)], index=unmatchedocm.index)
            unmatchedocm =unmatchedocm[instersectdf.columns]
            statDF['O'].loc[statDF['fips']==county] = len(unmatchedocm)
            
            OPdf = pd.concat( [instersectdf, unmatchedocm], axis =0) #.drop(['index_left'],axis=1)
            OPdf.loc[OPdf['integrationcode']=="O", 'geometry'] = geopandas.GeoSeries.from_wkt(OPdf.loc[OPdf['integrationcode']=="O"].ocm_polygon_geometry)
            OPdf = OPdf[['gml_id', 'state', 'county', 'height', 'Shape_Area',
                   'ocm_polygon_geometry', 'ocm_point_geometry', 'APN', 'APN2', 'STATE',
                   'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
                   'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
                   'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
                   'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 'geometry',
                   'parcel_polygon_geometry', 'parcel_point_geometry', 'integrationcode']]
            statDF['OPtotalmerge'].loc[statDF['fips']==county] = len(OPdf)
    
            OPdf.to_csv('D:/DATA/00_ParcelOCMMerge/parcelocm_'+county+'.csv')
        
    


#integrating OCM and Parcel and ZTRAX (need to use unique APN)
    


    file_path = 'D:\DATA\ztrax_county_2021_areatype_new/ztrax_2021_extraction_county_areatypes_'+county+'.csv'  # file path
    
    DF = pd.read_csv(file_path)
    if 'BuildingAreaSqFt' not in DF:
        DF['BuildingAreaSqFt'] = np.nan

    maxdf = DF.groupby(['RowID','BuildingOrImprovementNumber'])['BuildingAreaSqFt'].max().reset_index()
    if len(maxdf)==0:
        ztraxdf=DF.drop_duplicates(subset = ['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'])
    else: 
        ztraxdf=DF.merge(maxdf,on=['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'],how='right').drop_duplicates(subset = ['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'])
    sumareas= ztraxdf.groupby(['PropertyAddressLongitude','PropertyAddressLatitude','RowID'])['BuildingAreaSqFt'].sum().reset_index()
    ZtraxDF=sumareas.merge(ztraxdf,on=['PropertyAddressLongitude','PropertyAddressLatitude','BuildingAreaSqFt','RowID'],how='left')
    ZtraxDF['FIPS'] =  ZtraxDF['FIPS'].astype(str).str[:-2].apply(lambda x: '{0:0>5}'.format(x)).replace('0000n',county)
    ZtraxgDF = geopandas.GeoDataFrame(ZtraxDF, geometry=geopandas.points_from_xy(ZtraxDF.PropertyAddressLongitude, ZtraxDF.PropertyAddressLatitude))

    ZtraxgDF.crs = {'init' :'epsg:4326'}
    ZtraxgDFprj = ZtraxgDF.to_crs( OPdf.crs).replace([0], np.nan)
    ZtraxgDFprj['ztrax_point_geometry'] = ZtraxgDFprj.geometry.apply(lambda x: wkt.dumps(x))
    statDF['totalztrax'].loc[statDF['fips']==county] = len(ZtraxgDF)
    
    #merge with addresses
    if( len(ZtraxgDFprj)>0):
        ZtraxgDFprj['ztrax_address'] =ZtraxgDFprj.apply(lambda x : street_name_fix(x.PropertyFullStreetAddress ) if type(x.PropertyFullStreetAddress)==str else  x.PropertyFullStreetAddress , axis = 1)    
        pat = r'^(?P<number>\d+)?(?P<street>.+(?=\bAPT|\bUNIT\bSTE)|.+(?=#)|.+)(?P<apt_unit>(?:\bapt|\bUNIT|#|\bAPT|\bSTE|).+)?'
        
        try : 
            addressdf =ZtraxgDFprj['ztrax_address'].str.extract(pat)
        except AttributeError:
            addressdf = pd.DataFrame(None, index=list(range(0,len(ZtraxgDFprj))), columns=['number','street'])

        ZtraxgDFprj['ztrax_address'] = addressdf['number']+''+ addressdf['street']
        Ztraxselect = ZtraxgDFprj[['RowID','PropertyAddressLongitude', 'PropertyAddressLatitude','LotSizeSquareFeet','ztrax_address','FIPS',
               'BuildingAreaSqFt','YearBuilt', 'PropertyZip','PropertyLandUseStndCode','ztrax_point_geometry','geometry']]
        ZTRAXcomplete =  Ztraxselect[~Ztraxselect[['YearBuilt','BuildingAreaSqFt']].isnull().any(axis=1)]
        ZTRAXincomplete = Ztraxselect[(~ Ztraxselect.RowID.isin(ZTRAXcomplete.RowID))]
        statDF['incompleteztrax'].loc[statDF['fips']==county] = len(ZTRAXincomplete)
        
        if (len(OPdf)==0&len(ZTRAXincomplete)==0):
            Ztraxselect = ZtraxgDFprj[['RowID', 'PropertyAddressLongitude', 'PropertyAddressLatitude','LotSizeSquareFeet', 
             'ztrax_address', 'FIPS', 'BuildingAreaSqFt','YearBuilt','ztrax_point_geometry','geometry']]
            Ztraxselect['integrationcode'] = np.nan
            Ztraxselect['integrationcode'] = 'Z'
            
            Ztraxselect[['gml_id', 'state', 'county', 'height', 'Shape_Area',
                   'ocm_polygon_geometry', 'ocm_point_geometry', 'APN', 'APN2', 'STATE',
                   'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
                   'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
                   'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
                   'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 
                   'parcel_polygon_geometry', 'parcel_point_geometry']]=  pd.DataFrame([np.repeat(np.nan, 37)], index=Ztraxselect.index)
            OPZ = Ztraxselect[finalcols]
            
            OPZ.loc[(OPZ['integrationcode'] =="Z"),'geometry'] =  geopandas.GeoSeries.from_wkt(OPZ.loc[(OPZ['integrationcode'] =="Z")].ztrax_point_geometry)
            OPZ["Long"] = OPZ.geometry.map(lambda p: p.x)
            OPZ["Lat"] =  OPZ.geometry.map(lambda p: p.y)
           
            statDF['OPZtotalmerge'].loc[statDF['fips']==county] = len( OPZ )
            OPZ.to_csv('D:/DATA/01_ParcelOCMZtraxMerge/parcelocmztraxmerge_'+county+'.csv')
            
            
        elif len(OPdf)==0:
            Ztraxselect = ZtraxgDFprj[['RowID', 'PropertyAddressLongitude', 'PropertyAddressLatitude','LotSizeSquareFeet', 
             'ztrax_address', 'FIPS', 'BuildingAreaSqFt','YearBuilt','ztrax_point_geometry','geometry']]

            Ztraxselect['integrationcode'] = np.nan
            Ztraxselect['integrationcode'] = 'Z'
            
            Ztraxselect[['gml_id', 'state', 'county', 'height', 'Shape_Area',
                   'ocm_polygon_geometry', 'ocm_point_geometry', 'unique_APN','APN', 'APN2', 'STATE',
                   'COUNTY', 'FIPS', 'SIT_HSE_NU', 'SIT_DIR', 'SIT_STR_NA', 'SIT_STR_SF',
                   'SIT_FULL_S', 'SIT_CITY', 'SIT_STATE', 'SIT_ZIP', 'SIT_ZIP4',
                   'LAND_VALUE', 'IMPR_VALUE', 'TOT_VALUE', 'SALES_PRIC', 'YEAR_BUILT',
                   'STD_LAND_U', 'LOT_SIZE', 'BLDG_AREA', 'NO_OF_STOR', 'NO_OF_UNIT',
                   'BEDROOMS', 'BATHROOMS', 'XCOORD', 'YCOORD', 
                   'parcel_polygon_geometry', 'parcel_point_geometry']]=  pd.DataFrame([np.repeat(np.nan, 38)], index=Ztraxselect.index)
            OPZ = Ztraxselect[finalcols]

            OPZ.loc[(OPZ['integrationcode'] =="Z"),'geometry'] =  geopandas.GeoSeries.from_wkt(OPZ.loc[(OPZ['integrationcode'] =="Z")].ztrax_point_geometry)
            OPZ["Long"] = OPZ.geometry.map(lambda p: p.x)
            OPZ["Lat"] =  OPZ.geometry.map(lambda p: p.y)
            
            statDF['OPZtotalmerge'].loc[statDF['fips']==county] = len(OPZ )
            OPZ.to_csv('D:/DATA/01_ParcelOCMZtraxMerge/parcelocmztraxmerge_'+county+'.csv')
        
        elif len(Ztraxselect['ztrax_address'].unique()) ==1 :
            Mergedwithadd = OPdf.copy()
            Mergedwithadd[['RowID', 'PropertyAddressLongitude', 'PropertyAddressLatitude',
                   'LotSizeSquareFeet', 'ztrax_address', 'FIPS', 'BuildingAreaSqFt',
                   'YearBuilt', 'PropertyZip', 'PropertyLandUseStndCode',
                   'ztrax_point_geometry', 'geometry']]=  pd.DataFrame([np.repeat(np.nan, 12)], index=Mergedwithadd.index)
            
        else:
            OPdf['FIPS'] = county
            Mergedwithadd = pd.merge( Ztraxselect.dropna(subset = ['ztrax_address','FIPS']), OPdf.dropna(subset = ['SIT_FULL_S',"FIPS"]), left_on=['ztrax_address',"FIPS"], right_on=['SIT_FULL_S',"FIPS"], how = 'inner').drop_duplicates(subset ='RowID', keep='first' ).dropna(subset = ['RowID'])
            Mergedwithadd.loc[(Mergedwithadd.RowID.notna())&(Mergedwithadd.APN.isna())&(Mergedwithadd.gml_id.isna()), 'integrationcode'] = 'Z'              
            Mergedwithadd.YearBuilt.fillna(Mergedwithadd.YEAR_BUILT, inplace=True)
            Mergedwithadd.BuildingAreaSqFt.fillna(Mergedwithadd.BLDG_AREA, inplace=True)
            statDF['OPZ_address'].loc[statDF['fips']==county] = len(Mergedwithadd )
            Mergedwithadd.rename(columns={'geometry_y':'geometry'}, inplace=True)
            Mergedwithadd = Mergedwithadd.drop(['geometry_x'], axis=1)

            
       #select unmatched ztrax with addresses
        #Mergedwithadd['integrationcode']= np.nan
        #unmatchedwithadd = Mergedwithadd.loc[( Mergedwithadd['integrationcode']=='Z')]
        
        #Intersection
        try :
            unmatchedop =   OPdf[~(OPdf.APN.isin(Mergedwithadd.APN))|~(OPdf.gml_id.isin(Mergedwithadd.gml_id))].drop(['index_left'],axis=1)
        
        except KeyError:
            unmatchedop =   OPdf[~(OPdf.APN.isin(Mergedwithadd.APN))|~(OPdf.gml_id.isin(Mergedwithadd.gml_id))]
            
        unmatchedztrax =   Ztraxselect[~(Ztraxselect.RowID.isin((Mergedwithadd.RowID)))|~(Ztraxselect.RowID.isin((Mergedwithadd.RowID)))]
        #instersectopz = geopandas.sjoin(unmatchedztrax, unmatchedop, how='right', op='intersects') #.sort_values("Uncertainty", ascending = True).drop_duplicates(subset ='RowID', keep='first' )
        instersectopz = geopandas.sjoin(unmatchedztrax, unmatchedop, how='inner', op='intersects') #.sort_values("Uncertainty", ascending = True).drop_duplicates(subset ='RowID', keep='first' )
        instersectopz['YEAR_BUILT'] = pd.to_numeric(instersectopz['YEAR_BUILT'], errors='coerce').fillna(0)               
        instersectopz['uncertainty'] =abs(instersectopz.YearBuilt.astype(float).fillna(0)-instersectopz.YEAR_BUILT.astype(float).fillna(0)) + abs(instersectopz.BuildingAreaSqFt.fillna(0)-instersectopz.BLDG_AREA.astype(float).fillna(0))
        
        bestmatchopz =  instersectopz.sort_values("uncertainty", ascending = True).drop_duplicates(subset ='RowID', keep='first').drop(['FIPS_left','FIPS_right'],axis=1).dropna(subset = ['RowID'])
        bestmatchopz['FIPS'] = county
        statDF['OPZ_intersect'].loc[statDF['fips']==county] = len(bestmatchopz)
                    
        
        #Nearest match
        #find unmatched
        try:
            unmatchedop2 =   OPdf[~(OPdf.APN.isin(bestmatchopz.APN))|~(OPdf.gml_id.isin(bestmatchopz.gml_id))].drop(['index_left'],axis=1)
        except KeyError:
            unmatchedop2 =   OPdf[~(OPdf.APN.isin(bestmatchopz.APN))|~(OPdf.gml_id.isin(bestmatchopz.gml_id))]
            
        unmatchedztrax2 =   unmatchedztrax[~(unmatchedztrax.RowID.isin((bestmatchopz.RowID)))|~(unmatchedztrax.RowID.isin((bestmatchopz.RowID)))]
        print(len(unmatchedop2),len(unmatchedztrax2))
        
        if OPdf.APN.isna().sum()==len(OPdf):
            matcheddf = pd.DataFrame()
            X = len(unmatchedztrax2)
            while (X == X)or(X==0):
               i = 1
               nearest_data = geopandas.sjoin_nearest( unmatchedop2,unmatchedztrax2,how='inner', distance_col="distances", max_distance = 100) #.dropna(subset=['gml_id'])        
               nearest_data['AreaSqFt'] = nearest_data.Shape_Area*10.764
               nearest_data.LotSizeSquareFeet=nearest_data.fillna(0).LotSizeSquareFeet.astype('int') 
               nearest_data['diagonal_square_meter'] = nearest_data.apply(lambda x: math.sqrt(2)*math.sqrt(x.LotSizeSquareFeet)*0.3048, axis = 1)
               min_value = nearest_data.groupby(['PropertyAddressLongitude','PropertyAddressLatitude'])['distances'].min().reset_index()
               nearest_data= nearest_data.merge(min_value, on=['PropertyAddressLongitude','PropertyAddressLatitude'],suffixes=('', '_min')) #.drop_duplicates(subset=['uniqueID'])
               nearest_data['uncertainty'] =normalize_column(nearest_data.AreaSqFt/nearest_data.LotSizeSquareFeet) + normalize_column(nearest_data.distances_min/nearest_data.diagonal_square_meter)+normalize_column(nearest_data.distances_min)
               nearest_data.rename(columns={'FIPS_left':'FIPS'}, inplace=True)
               nearest_data = nearest_data.drop(['FIPS_right'],axis=1)
               nearest_DF = nearest_data.sort_values(['distances_min','uncertainty'], ascending=True).drop_duplicates(subset ='RowID', keep='first' )
               nearestDAT = nearest_DF.drop(['distances','distances_min'],axis=1)   
               unmatchedztrax3 =  unmatchedztrax2[~(unmatchedztrax2.RowID.isin(nearestDAT.RowID))]
               matcheddf= pd.concat([matcheddf,nearestDAT ])
               unmatchedztrax2 = unmatchedztrax3
               X = int(len(unmatchedztrax3))
               print(X, i)
               if  (X ==X)or(X==0):
                   break
            Matcheddf = matcheddf.drop(['AreaSqFt', 'diagonal_square_meter'],axis=1)
            statDF['OPZ_distance'].loc[statDF['fips']==county] = len(Matcheddf)
     
        else : 
            matcheddf = pd.DataFrame()
            X = len(unmatchedztrax2)
            while (X == X)or(X==0):
               i = 1
               nearest_data = geopandas.sjoin_nearest( unmatchedop2,unmatchedztrax2,how='inner', distance_col="distances", max_distance = 100) #.dropna(subset=['gml_id','APN'])
               nearest_data.YEAR_BUILT = pd.to_numeric(nearest_data.YEAR_BUILT, errors='coerce').fillna(0)
               nearest_data['uncertainty'] = abs(nearest_data.YearBuilt.astype(float).fillna(0)- nearest_data.YEAR_BUILT.astype(float).fillna(0)) +abs(nearest_data.BuildingAreaSqFt.astype(float).fillna(0)-nearest_data.BLDG_AREA.astype(float).fillna(0))
               nearest_data.rename(columns={'FIPS_left':'FIPS'}, inplace=True)
               nearest_data = nearest_data.drop(['FIPS_right'],axis=1)
               min_value = nearest_data.groupby(['PropertyAddressLongitude','PropertyAddressLatitude'])['distances'].min().reset_index()
               nearest_data= nearest_data.merge(min_value, on=['PropertyAddressLongitude','PropertyAddressLatitude'],suffixes=('', '_min')) #.drop_duplicates(subset=['uniqueID'])
               nearest_DF = nearest_data.sort_values(['distances_min','uncertainty'], ascending=True).drop_duplicates(subset ='RowID', keep='first' )
               nearestDAT = nearest_DF.drop(['distances','distances_min'],axis=1)   
               unmatchedztrax3 =  unmatchedztrax2[~(unmatchedztrax2.RowID.isin(nearestDAT.RowID))]
               matcheddf= pd.concat([matcheddf,nearestDAT ])
               unmatchedztrax2 = unmatchedztrax3
               X = int(len(unmatchedztrax3))
               print(X, i)
               if  (X ==X)or(X==0):
                   break
            Matcheddf = matcheddf #.drop(['index_left'],axis=1)

            statDF['OPZ_distance'].loc[statDF['fips']==county] = len(Matcheddf)



        #merge everything
        unmatchedop3 = unmatchedop2[~(unmatchedop2.unique_APN.isin( matcheddf.unique_APN))|~(unmatchedop2.gml_id.isin(matcheddf.gml_id))] #.dropna(subset=['unique_APN','gml_id'])
        
        Bestmatchopz = bestmatchopz[Matcheddf.columns]
        #Mergedwithadd2 = Mergedwithadd.loc[Mergedwithadd.integrationcode!='Z']
        if X!=0:
            statDF['finalunmatchedZ'].loc[statDF['fips']==county] = len(unmatchedztrax2)
            mergematched = pd.concat([unmatchedop3,Mergedwithadd, Bestmatchopz,matcheddf,unmatchedztrax2])
            
            missingop = OPdf[~(OPdf.unique_APN.isin( mergematched.unique_APN))|~( OPdf.gml_id.isin(mergematched.gml_id))] #.dropna(subset=['unique_APN','gml_id']) #df[df['column name'].isna()]
            finaldf = pd.concat([mergematched, missingop])
            statDF['unmatchedOPwithZ'].loc[statDF['fips']==county] =len(missingop)+len(unmatchedop3)
        
        else :
            mergematched = pd.concat([unmatchedop3,Mergedwithadd, Bestmatchopz,matcheddf])
            missingop = mergematched[~(mergematched.unique_APN.isin( matcheddf.unique_APN))|~(mergematched.gml_id.isin(matcheddf.gml_id))] #.dropna(subset=['unique_APN','gml_id'])
            finaldf = pd.concat([mergematched, missingop])
            statDF['unmatchedOPwithZ'].loc[statDF['fips']==county] =len(missingop)+len(unmatchedop3)

        
        
        finaldf.YearBuilt.fillna(finaldf.YEAR_BUILT, inplace=True)
        finaldf.BuildingAreaSqFt.fillna(finaldf.BLDG_AREA, inplace=True)
        finaldf.FIPS = county
        statDF['OPZtotalmerge'].loc[statDF['fips']==county] = len(finaldf)
        
        #summary stat
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APN.isna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'Z'              
        finaldf.loc[(finaldf.RowID.isna())&(finaldf.unique_APN.isna())&(finaldf.gml_id.notna()), 'integrationcode'] = 'O'              
        finaldf.loc[(finaldf.RowID.isna())&(finaldf.unique_APN.notna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'P'      
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APN.notna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'PZ11'  
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APN.isna())&(finaldf.gml_id.notna()), 'integrationcode'] = 'OZ11'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.notna())&(finaldf.RowID.isna()), 'integrationcode'] = 'OP11' 
        finaldf.loc[(finaldf.unique_APN.notna())&(finaldf.gml_id.notna())&(finaldf.RowID.notna()), 'integrationcode'] = 'OPZ111'    
        finaldf.loc[(finaldf.gml_id.isna())&(finaldf.unique_APN.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.notna()), 'integrationcode'] ='PZ1n'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.gml_id.duplicated()==True)&(finaldf.unique_APN.isna())&(finaldf.RowID.notna()), 'integrationcode'] ='OZ1n'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.isna()), 'integrationcode'] ='OPn1'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.notna()), 'integrationcode'] = 'OPZn1n'  
        finaldf['integrationcode'].unique()
        #finaldf.geometry = finaldf.geometry.centroid
        finaldf =finaldf[finalcols]
        finaldf.loc[finaldf['integrationcode'].str.contains('O'), 'geometry'] =  geopandas.GeoSeries.from_wkt(finaldf.loc[finaldf['integrationcode'].str.contains('O')].ocm_point_geometry)
        finaldf.loc[(finaldf['integrationcode'] =="P")|(finaldf['integrationcode'] =="PZ11")|(finaldf['integrationcode'] =="PZ1n"), 'geometry'] = geopandas.GeoSeries.from_wkt(finaldf.loc[(finaldf['integrationcode'] =="P")|(finaldf['integrationcode'] =="PZ11")|(finaldf['integrationcode'] =="PZ1n")].parcel_point_geometry)
        finaldf.loc[(finaldf['integrationcode'] =="Z"),'geometry'] =  geopandas.GeoSeries.from_wkt(finaldf.loc[(finaldf['integrationcode'] =="Z")].ztrax_point_geometry)
        finaldf["Long"] = finaldf.geometry.map(lambda p: p.x)
        finaldf["Lat"] =  finaldf.geometry.map(lambda p: p.y)
        finaldf.to_csv('D:/DATA/01_ParcelOCMZtraxMerge/parcelocmztraxmerge_'+county+'.csv')
        
        
        statDF['Z'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='Z'])
        statDF['O'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='O'])
        statDF['P'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='P'])
        statDF['PZ11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='PZ11'])
        statDF['OZ11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OZ11'])
        statDF['OP11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OP11'])
        statDF['OPZ111'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPZ111'])
        statDF['PZ1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='PZ1n'])
        statDF['OZ1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OZ1n'])
        statDF['OPn1'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPn1'])
        statDF['OPZn1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPZn1n'])
        round(time.time() - start_time, 2)
        try : 
            del parcel, Parcel, ocmgdf, OCM, nearestDAT, nearest_data, ztraxdf, ZtraxDF, ZtraxgDFprj
        except NameError:
            del   ocmgdf, OCM, nearestDAT, nearest_data, ztraxdf, ZtraxDF, ZtraxgDFprj
        statDF.to_csv('D:\DATA\ZTRAX_completeness_summary/OPZdataprocesssummary.csv')

    else:
        finaldf = OPdf.copy()
        finaldf.geometry = finaldf.geometry.centroid
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APN.isna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'Z'              
        finaldf.loc[(finaldf.RowID.isna())&(finaldf.unique_APN.isna())&(finaldf.gml_id.notna()), 'integrationcode'] = 'O'              
        finaldf.loc[(finaldf.RowID.isna())&(finaldf.unique_APN.notna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'P'      
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APN.notna())&(finaldf.gml_id.isna()), 'integrationcode'] = 'PZ11'  
        finaldf.loc[(finaldf.RowID.notna())&(finaldf.unique_APNN.isna())&(finaldf.gml_id.notna()), 'integrationcode'] = 'OZ11'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.notna())&(finaldf.RowID.isna()), 'integrationcode'] = 'OP11' 
        finaldf.loc[(finaldf.unique_APN.notna())&(finaldf.gml_id.notna())&(finaldf.RowID.notna()), 'integrationcode'] = 'OPZ111'    
        finaldf.loc[(finaldf.gml_id.isna())&(finaldf.unique_APN.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.notna()), 'integrationcode'] ='PZ1n'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.gml_id.duplicated()==True)&(finaldf.unique_APN.isna())&(finaldf.RowID.notna()), 'integrationcode'] ='OZ1n'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.isna()), 'integrationcode'] ='OPn1'
        finaldf.loc[(finaldf.gml_id.notna())&(finaldf.unique_APN.duplicated()==True)&(finaldf.RowID.notna()), 'integrationcode'] = 'OPZn1n'  
        finaldf.YearBuilt.fillna(finaldf.YEAR_BUILT, inplace=True)
        finaldf.BuildingAreaSqFt.fillna(finaldf.BLDG_AREA, inplace=True)
        finaldf.FIPS = county
        finaldf['integrationcode'].unique()
        statDF['OPZtotalmerge'].loc[statDF['fips']==county] = len(finaldf)
        finaldf =finaldf[finalcols]
        finaldf.loc[finaldf['integrationcode'].str.contains('O'), 'geometry'] =  geopandas.GeoSeries.from_wkt(finaldf.loc[finaldf['integrationcode'].str.contains('O')].ocm_point_geometry)
        finaldf.loc[(finaldf['integrationcode'] =="P")|(finaldf['integrationcode'] =="PZ11")|(finaldf['integrationcode'] =="PZ1n"), 'geometry'] = geopandas.GeoSeries.from_wkt(finaldf.loc[(finaldf['integrationcode'] =="P")|(finaldf['integrationcode'] =="PZ11")|(finaldf['integrationcode'] =="PZ1n")].parcel_point_geometry)
        finaldf.loc[(finaldf['integrationcode'] =="Z"),'geometry'] =  geopandas.GeoSeries.from_wkt(finaldf.loc[(finaldf['integrationcode'] =="Z")].ztrax_point_geometry)
        finaldf["Long"] = finaldf.geometry.map(lambda p: p.x)
        finaldf["Lat"] =  finaldf.geometry.map(lambda p: p.y)

        finaldf.to_csv('D:/DATA/01_ParcelOCMZtraxMerge/parcelocmztraxmerge_'+county+'.csv')

        statDF['Z'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='Z'])
        statDF['O'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='O'])
        statDF['P'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='P'])
        statDF['PZ11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='PZ11'])
        statDF['OZ11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OZ11'])
        statDF['OP11'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OP11'])
        statDF['OPZ111'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPZ111'])
        statDF['PZ1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='PZ1n'])
        statDF['OZ1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OZ1n'])
        statDF['OPn1'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPn1'])
        statDF['OPZn1n'].loc[statDF['fips']==county] = len(finaldf.loc[finaldf['integrationcode']=='OPZn1n'])
        round(time.time() - start_time, 2)
        del parcel, Parcel, ocmdf, ocmgdf, OCM, nearestDAT, nearest_data, ztraxdf, ZtraxDF, ZtraxgDFprj
        statDF.to_csv('D:/DATA/ZTRAX_completeness_summary/OPZdataprocesssummary.csv')
#PZmn (not possible)
#OP1n (this should be OPn1)
#OPZ11n(not possible)
#OPZ1n1(this should be OPZn1n)
#OPZ1nn (not possible)



    
#test = pd.concat([Mergedwithadd2, Bestmatchopz,mactheddf])
#test = pd.concat([Mergedwithadd, Bestmatchopz,mactheddf])
#test = finaldf.loc[(finaldf.RowID.notna())&(finaldf.APN.isna())&(finaldf.gml_id.isna())]
#len(ZtraxDF)
#errors=   Mergedwithadd[~(Mergedwithadd.RowID.isin((test.RowID)))]
#errors=   mactheddf[(mactheddf.RowID.isin((test.RowID)))]
#test = finaldf.loc[finaldf.integrationcode=="Z"]
#mactheddf[(mactheddf.RowID.isin(test.RowID))]

#test = finaldf.loc[finaldf.integrationcode=="O"]
#finaldf.loc[finaldf['gml_id']=='ODQ5V1c3N0crOi0xNDc1MTAxOTUw']
#test = finaldf[finaldf.duplicated(['gml_id'])&(finaldf.gml_id.notna())].sort_values('gml_id').head(50)
#mactheddf.integrationcode.unique()
#finaldf = pd.concat([unmatchedop3,Mergedwithadd, Bestmatchopz,mactheddf,unmatchedztrax2])
#missingop.drop_duplicates(subset ='RowID', keep='first' )
#finaldf[~(finaldf.APN.isin( matcheddf.APN))|~(finaldf.gml_id.isin(finaldf.gml_id))].dropna(subset=['APN','gml_id'])
#len(finaldf)

#OPdf.drop_duplicates(subset ='gml_id', keep='first' )
test = finaldf[finaldf.duplicated(['RowID'])]
test.RowID.unique()
#test = pd.concat([Mergedwithadd, Bestmatchopz,matcheddf,unmatchedztrax2])
#test.drop_duplicates(subset ='')

#test = finaldf.loc[(finaldf.unique_APN.notna())&(finaldf.gml_id.notna())]
#test2 = test[test.duplicated(['gml_id'])]
#test2.loc[(test2.gml_id.notna())&(test2.unique_APN.notna())&(test2.RowID.isna()), 'integrationcode'] = 'OPZ111' 
#test2.loc[(test2.gml_id.notna())&(test2.unique_APN.notna())&(test2.RowID.na())]
#OPdf[OPdf.duplicated(['gml_id'])].dropna(subset = ['gml_id'])




# OPdf[~(OPdf.APN.isin(Mergedwithadd.APN))|~(OPdf.gml_id.isin(Mergedwithadd.gml_id))]




#unmatchedop.ocm_polygon_geometry 
#finaldf[~(finaldf.unique_APN.isin( OPdf.unique_APN))|~(finaldf.gml_id.isin(OPdf.gml_id))] #.dropna(subset=['unique_APN','gml_id']) #df[df['column name'].isna()]



