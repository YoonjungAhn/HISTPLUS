# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:06:38 2022

@author: yoah2447
"""

import pandas as pd
import geopandas
import time 
import numpy as np
import fiona 

#address matching
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
    
    

#method 1
LUztrax = pd.read_csv('D:/DATA/ZTRAX_landusecode.csv')
LUztrax =LUztrax[LUztrax["Classification"].str.contains("Residential")]
LUcodeztrax = LUztrax["StndCode"].unique()


countydf = pd.read_csv('D:/DATA/countyFIPS.csv')
countydf.fips
countyDF = countydf.loc[(countydf.state!="AK")&(countydf.state!="HI")]

counties = list(countyDF.fips.apply(lambda x: '{0:0>5}'.format(x)).unique())
counties.index( '18165')
counties.index( '51051')
#files = []
#statdf = pd.DataFrame({'fips':counties,'totalztrax':np.nan,'originalztraxincomplete':np.nan,'numofaddmatch':np.nan, 'numofintersect':np.nan, 
#                       'numofspatialjoin':np.nan, 'finalnumofunmatchedZtrax':np.nan, 'totalparcel':np.nan, 'totalmerge':np.nan})
#finalunmatchedztrax=pd.DataFrame([])
for county in counties[ 2811:]:
    start_time = time.time()
    #file_path = 'D:/DATA/ZTRAX_CSV_COUNTY_RELEVANT_ATTRIBUTES_RL3_imp_indoor_area/ztrax_2021_extraction_county_areatypes_'+str(county)+'_2_area_adj_imputed_bldgarea.csv'  # file path for imputed ZTRAX
    
    #ZTRAX
    file_path = 'D:\DATA\ztrax_county_2021_areatype_new/ztrax_2021_extraction_county_areatypes_'+county+'.csv'  # file path
    
    DF = pd.read_csv(file_path)
    if 'BuildingAreaSqFt' not in DF:
        DF['BuildingAreaSqFt'] = np.nan
    
    #try : 
    maxdf = DF.groupby(['RowID','BuildingOrImprovementNumber'])['BuildingAreaSqFt'].max().reset_index()
    ztraxdf=DF.merge(maxdf,on=['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'],how='right').drop_duplicates(subset = ['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'])
    sumareas= ztraxdf.groupby(['PropertyAddressLongitude','PropertyAddressLatitude','RowID'])['BuildingAreaSqFt'].sum().reset_index()
    ZtraxDF=sumareas.merge(ztraxdf,on=['PropertyAddressLongitude','PropertyAddressLatitude','BuildingAreaSqFt','RowID'],how='left')
    ZtraxDF = ZtraxDF[ZtraxDF['PropertyLandUseStndCode'].isin(LUcodeztrax) ]
    statdf['totalztrax'].loc[statdf['fips']==county] = len(ZtraxDF)
    
    if( len(ZtraxDF)>0):
        ZtraxDF['ztrax_address'] = ZtraxDF.apply(lambda x : street_name_fix(x.PropertyFullStreetAddress ) if type(x.PropertyFullStreetAddress)==str else  x.PropertyFullStreetAddress , axis = 1)    
        pat = r'^(?P<number>\d+)?(?P<street>.+(?=\bAPT|\bUNIT\bSTE)|.+(?=#)|.+)(?P<apt_unit>(?:\bapt|\bUNIT|#|\bAPT|\bSTE|).+)?'
        addressdf = ZtraxDF['ztrax_address'].str.extract(pat)
        ZtraxDF['ztrax_address'] = addressdf['number']+''+ addressdf['street']
        Ztraxselect = ZtraxDF[['RowID','PropertyAddressLongitude', 'PropertyAddressLatitude','LotSizeSquareFeet','ztrax_address','FIPS',
               'BuildingAreaSqFt','YearBuilt', 'PropertyZip','PropertyLandUseStndCode']]
        ZTRAXcomplete =  Ztraxselect[~Ztraxselect[['YearBuilt','BuildingAreaSqFt']].isnull().any(axis=1)]
        ZTRAXincomplete = Ztraxselect[(~ Ztraxselect.RowID.isin(ZTRAXcomplete.RowID))]
        statdf['originalztraxincomplete'].loc[statdf['fips']==county] = len(ZTRAXincomplete)
    

        #parcel
        try:
            parcel_file_path = 'D:/DATA/ParcelAtlas2022/'+county+'/parcels.shp'  # file path
            parcel = geopandas.read_file(parcel_file_path)

            if 'YEAR_BUILT' not in parcel:
                parcel['YEAR_BUILT'] = np.nan
            if 'APN' not in parcel:
                    parcel['APN'] = np.nan
            if 'BLDG_AREA' not in parcel:
                parcel['BLDG_AREA'] = np.nan
            if 'STD_LAND_U' not in parcel:
                    parcel['STD_LAND_U'] = np.nan
            if 'SIT_FULL_S' not in parcel:
                    parcel['SIT_FULL_S'] = np.nan

                
            parcel.rename(columns={'Xcoord':'XCOORD', 'Ycoord':'YCOORD'}, inplace=True)
            
            parcel['YEAR_BUILT']= parcel['YEAR_BUILT'].replace(0, np.nan)
            parcel['BLDG_AREA']= parcel['BLDG_AREA'].replace(0, np.nan)
            parcel['STD_LAND_U']= parcel['STD_LAND_U'].replace(0, np.nan)
            
            #state = pd.read_csv('D:/DATA/us-state-ansi-fips.csv')
            #sts = state['st'].apply(lambda x: '{0:0>2}'.format(x)).unique()
            LUcode = ['1000','1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1012','1013','1018','1100','1101','1102'
                      ,'1103','1104','1105','1106','1107','1108','1109','1110','1111','1112','1113','1114','1999']
            residentparcels = parcel[parcel['STD_LAND_U'].isin(LUcode) ]
            Residentparcels = residentparcels[['APN','SIT_FULL_S','BLDG_AREA','YEAR_BUILT', 'XCOORD', 'YCOORD', 'geometry']]
            statdf['totalparcel'].loc[statdf['fips']==county] =len(Residentparcels)
            
            if (len(Residentparcels)==0&len(ZTRAXincomplete)==0):
                ztraxfinal = ZtraxDF[['RowID', 'ztrax_address', 'BuildingAreaSqFt', 'YearBuilt',
                       'PropertyAddressLongitude', 'PropertyAddressLatitude', 'FIPS']]
                statdf['totalmerge'].loc[statdf['fips']==county] = len(ztraxfinal)
                ztraxfinal.to_csv('D:/DATA/01_ParcelZtraxMerge/parcelztraxmerge_'+county+'.csv')
            elif len(Residentparcels)==0:
                ztraxfinal = ZtraxDF[['RowID', 'ztrax_address', 'BuildingAreaSqFt', 'YearBuilt',
                       'PropertyAddressLongitude', 'PropertyAddressLatitude', 'FIPS']]
                statdf['totalmerge'].loc[statdf['fips']==county] = len(ztraxfinal)
                ztraxfinal.to_csv('D:/DATA/01_ParcelZtraxMerge/parcelztraxmerge_'+county+'.csv')
            else:
                #exclude parcle that ZTRAX is already complete
                parcelselect = Residentparcels[~(Residentparcels.SIT_FULL_S.isin(ZTRAXcomplete.ztrax_address))]
                #address match ZATRAX + parcel 
    
                if(len(ZTRAXincomplete)!=0)&(len(ZTRAXincomplete) <= len(parcelselect)):
                    Mergedwithadd = pd.merge(ZTRAXincomplete, parcelselect, left_on=['ztrax_address'], right_on=['SIT_FULL_S'], how = 'left').drop_duplicates(subset ='RowID', keep='first' ).dropna(subset = ['RowID'])
                elif len(parcelselect)==0:
                    Mergedwithadd=ZTRAXincomplete
                    Mergedwithadd['YEAR_BUILT']=np.nan
                    Mergedwithadd['BLDG_AREA']=np.nan
                    Mergedwithadd['APN']=np.nan
                    Mergedwithadd['SIT_FULL_S']=np.nan
                elif len(ZTRAXincomplete)==0:
                    Mergedwithadd = pd.concat([ZTRAXcomplete,parcelselect],axis =1)

                else : 
                    Mergedwithadd = pd.merge(ZTRAXincomplete, parcelselect, left_on=['ztrax_address'], right_on=['SIT_FULL_S'], how = 'right').drop_duplicates(subset ='RowID', keep='first' ).dropna(subset = ['RowID'])
    
                Mergedwithadd.YearBuilt.fillna(Mergedwithadd.YEAR_BUILT, inplace=True)
                Mergedwithadd.BuildingAreaSqFt.fillna(Mergedwithadd.BLDG_AREA, inplace=True)
                statdf['numofaddmatch'].loc[statdf['fips']==county] = len(Mergedwithadd)
            
                
                # select unmatched parcels
                unmatchedparcel = parcelselect[~(parcelselect.SIT_FULL_S.isin(Mergedwithadd.SIT_FULL_S))]
                unmatchedztrax = ZTRAXincomplete[~(ZTRAXincomplete.RowID.isin(Mergedwithadd.RowID))]
                
                #intersect match
                umatchedztraxgdf = geopandas.GeoDataFrame(unmatchedztrax, geometry=geopandas.points_from_xy(unmatchedztrax.PropertyAddressLongitude, unmatchedztrax.PropertyAddressLatitude))
                umatchedztraxgdf.crs = {'init' :'epsg:4326'}
                
                instersectdf = geopandas.sjoin(umatchedztraxgdf, unmatchedparcel, how='right', op='intersects') #.drop_duplicates(subset ='RowID', keep='first' )
                
                def normalize_column(df):
                    return (df - df.min()) / (df.max() - df.min())
                
                instersectdf['uncertainty'] =abs(instersectdf.BuildingAreaSqFt-instersectdf.BLDG_AREA) + abs(instersectdf.YearBuilt-instersectdf.YEAR_BUILT)
                instersectDF = instersectdf.sort_values(['RowID','uncertainty'], ascending=True).drop_duplicates(subset ='RowID', keep='first' ).dropna(subset = ['RowID'])
            
            
                #replace na in ZTRAX with parcel
                instersectDF.YearBuilt.fillna(instersectDF.YEAR_BUILT, inplace=True)
                instersectDF.BuildingAreaSqFt.fillna(instersectDF.BLDG_AREA, inplace=True)
                #instersectDF.RowID.fillna(instersectDF.APN, inplace=True)
                statdf['numofintersect'].loc[statdf['fips']==county] = len(instersectDF)
            
                #selecting unmatched data
                unmatchedparcel2 = parcelselect[~(parcelselect.SIT_FULL_S.isin(instersectDF.SIT_FULL_S))]
                unmatchedztrax2 = unmatchedztrax[~(unmatchedztrax.RowID.isin(instersectDF.RowID))]
                unmatchedztrax2gdf = geopandas.GeoDataFrame(unmatchedztrax2, geometry=geopandas.points_from_xy(unmatchedztrax2.PropertyAddressLongitude, unmatchedztrax2.PropertyAddressLatitude))
                unmatchedztrax2gdf.crs = {'init' :'epsg:4326'}
                unmatchedztrax2gdf = unmatchedztrax2gdf.to_crs(parcel.crs)
            
            
                #select parcels are not in ZTRAX
                mergedf = pd.concat([Mergedwithadd,ZTRAXcomplete,instersectDF,unmatchedztrax2],axis=0).drop_duplicates(subset ='RowID', keep='first' )
                parcelselect = Residentparcels[~(Residentparcels.SIT_FULL_S.isin(mergedf.ztrax_address))]
                
                #nearest match
                nearest_data = geopandas.sjoin_nearest(unmatchedztrax2gdf, parcelselect,how='right', distance_col="distances", max_distance = 50)
                nearest_data['uncertainty'] =abs(nearest_data.YearBuilt.astype(float).fillna(0)-nearest_data.YEAR_BUILT.astype(float).fillna(0)) #+abs(nearest_data.BuildingAreaSqFt.fillna(0)-nearest_data.BLDG_AREA.fillna(0))
                min_value = nearest_data.groupby(['PropertyAddressLongitude','PropertyAddressLatitude'])['distances'].min().reset_index()
                nearest_data= nearest_data.merge(min_value, on=['PropertyAddressLongitude','PropertyAddressLatitude'],suffixes=('', '_min')) #.drop_duplicates(subset=['RowID'])
                
                Neardf = nearest_data.sort_values(['RowID','uncertainty','distances'], ascending=True).drop_duplicates(subset ='RowID', keep='first' ).dropna(subset = ['RowID'])
                Neardf.YearBuilt.fillna(Neardf.YEAR_BUILT, inplace=True)
                Neardf.BuildingAreaSqFt.fillna(Neardf.BLDG_AREA, inplace=True)
                statdf['numofspatialjoin'].loc[statdf['fips']==county] = len(Neardf)
             
                #complete ZTRAX
                mergedf = pd.concat([Mergedwithadd,ZTRAXcomplete,instersectDF,unmatchedztrax2,Neardf],axis=0).drop_duplicates(subset ='RowID', keep='first' )
                ZTRAXfinal = mergedf[['RowID', 'PropertyAddressLongitude', 'PropertyAddressLatitude','LotSizeSquareFeet', 
                 'ztrax_address', 'FIPS', 'BuildingAreaSqFt','YearBuilt']]
            
                #find parcel not in final ztrax
                parcelselect = Residentparcels[~(Residentparcels.SIT_FULL_S.isin(mergedf.ztrax_address))]
                parcelfinal = parcelselect[['APN', 'SIT_FULL_S', 'BLDG_AREA', 'YEAR_BUILT', 'XCOORD', 'YCOORD']]
            
                parcelfinal.rename(columns={'APN':'RowID', 'SIT_FULL_S':'ztrax_address', 'BLDG_AREA':"BuildingAreaSqFt", 
                                             'YEAR_BUILT':"YearBuilt", 'XCOORD':"PropertyAddressLongitude", 
                                             'YCOORD':'PropertyAddressLatitude'}, inplace=True)
                parcelfinal['FIPS'] = county 
                finalmerge = pd.concat([ZTRAXfinal,parcelfinal])
                
                unmatchedztrax =  ZtraxDF[~(ZtraxDF.RowID.isin(ZTRAXfinal.RowID))]
                statdf['finalnumofunmatchedZtrax'].loc[statdf['fips']==county] = len(unmatchedztrax)
                
                if len(unmatchedztrax)>0:
                    unmatchedztrax.to_csv('D:\DATA\02_numatchedZtraxwithparcel/umatched_parcelztraxmerge_'+county+'.csv')
                else :
                    pass
            
                statdf['totalmerge'].loc[statdf['fips']==county] = len(finalmerge)
            
                finalmerge.to_csv('D:/DATA/01_ParcelZtraxMerge/parcelztraxmerge_'+county+'.csv')
                
                end_time = time.time()
                print(county,"total time taken this loop: ", end_time - start_time)
        except fiona.errors.DriverError:
            ztraxfinal = ZtraxDF[['RowID', 'ztrax_address', 'BuildingAreaSqFt', 'YearBuilt',
                   'PropertyAddressLongitude', 'PropertyAddressLatitude', 'FIPS']]
            statdf['totalmerge'].loc[statdf['fips']==county] = len(ztraxfinal)
            ztraxfinal.to_csv('D:/DATA/01_ParcelZtraxMerge/parcelztraxmerge_'+county+'.csv')
            pass
            
    else : # when ztrax has zero observation
        try:

            parcel_file_path = 'D:/DATA/ParcelAtlas2022/'+county+'/parcels.shp'  # file path
            
            parcel = geopandas.read_file(parcel_file_path)
        
            parcel_file_path = 'D:/DATA/ParcelAtlas2022/'+county+'/parcels.shp'  # file path
            parcel = geopandas.read_file(parcel_file_path)
            if 'YEAR_BUILT' not in parcel:
                parcel['YEAR_BUILT'] = np.nan
            if 'APN' not in parcel:
                    parcel['APN'] = np.nan
            if 'BLDG_AREA' not in parcel:
                parcel['BLDG_AREA'] = np.nan
            if 'STD_LAND_U' not in parcel:
                    parcel['STD_LAND_U'] = np.nan
            if 'SIT_FULL_S' not in parcel:
                    parcel['SIT_FULL_S'] = np.nan  
            parcel.rename(columns={'Xcoord':'XCOORD', 'Ycoord':'YCOORD'}, inplace=True)
            
            parcel['YEAR_BUILT']= parcel['YEAR_BUILT'].replace(0, np.nan)
            parcel['BLDG_AREA']= parcel['BLDG_AREA'].replace(0, np.nan)
            parcel['STD_LAND_U']= parcel['STD_LAND_U'].replace(0, np.nan)
            
            #state = pd.read_csv('D:/DATA/us-state-ansi-fips.csv')
            #sts = state['st'].apply(lambda x: '{0:0>2}'.format(x)).unique()
            LUcode = ['1000','1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1012','1013','1018','1100','1101','1102'
                      ,'1103','1104','1105','1106','1107','1108','1109','1110','1111','1112','1113','1114','1999']
            residentparcels = parcel[parcel['STD_LAND_U'].isin(LUcode) ]
            Residentparcels = residentparcels[['APN','SIT_FULL_S','BLDG_AREA','YEAR_BUILT', 'XCOORD', 'YCOORD', 'geometry']]
            statdf['totalparcel'].loc[statdf['fips']==county] =len(Residentparcels)
            parcelfinal = Residentparcels[['APN', 'SIT_FULL_S', 'BLDG_AREA', 'YEAR_BUILT', 'XCOORD', 'YCOORD']]
            parcelfinal.rename(columns={'APN':'RowID', 'SIT_FULL_S':'ztrax_address', 'BLDG_AREA':"BuildingAreaSqFt", 
                                         'YEAR_BUILT':"YearBuilt", 'XCOORD':"PropertyAddressLongitude", 
                                         'YCOORD':'PropertyAddressLatitude'}, inplace=True)
            statdf['totalmerge'].loc[statdf['fips']==county] = len(parcelfinal)

            parcelfinal.to_csv('D:/DATA/01_ParcelZtraxMerge/parcelztraxmerge_'+county+'.csv')
        except fiona.errors.DriverError:
             pass
                
statdf.to_csv("D:/DATA/ZTRAX_completeness_summary/mergeresult.csv")                
    #except KeyError:
    #    pass
        
        
#check this county : '06015','08057' 
#need to go back to the county which don't have  'BuildingAreaSqFt' data