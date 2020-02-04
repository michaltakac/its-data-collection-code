from pytz import timezone
import pytz
from datetime import datetime, timedelta
from dateutil import tz
from pandas._testing import assert_frame_equal
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from geopandas import sjoin
from shapely.geometry import Point
import time
start_time = time.time()

# Globals
crs_degree = {'init': 'epsg:4326'}  # CGS_WGS_1984 (what the GPS uses)

# Paths
data_collection = "/Users/michaltakac/Dropbox/Data Collection"
data_in_process = data_collection + "/Data Processing/Data in process"
arcgis_project = "/Users/michaltakac/Dropbox/Data Collection/ArcGIS"
int_centroid_zones = data_in_process + \
    "/Demand/SFCTA test/InternalCentroidZones.shp"
int_legs = "/Users/michaltakac/Dropbox/Data Collection/Raw data/SFCTA demand data/internal_fremont_legs.csv"

print('\nLoading data...')

# Load internal centroid zones from shapefile
icz = GeoDataFrame.from_file(int_centroid_zones)

# internal fremont legs
int_data = pd.read_csv(int_legs)

# Process XY coordinates data as Point geometry for start nodes
int_start_nodes = [Point(xy) for xy in zip(
    int_data.start_node_lng, int_data.start_node_lat)]

# Process XY coordinates data as Point geometry for end nodes
int_end_nodes = [Point(xy) for xy in zip(
    int_data.end_node_lng, int_data.end_node_lat)]

# Create GeoDataFrame from start points
int_start_points = GeoDataFrame(pd.read_csv(
    int_legs), crs=crs_degree, geometry=int_start_nodes)

# Create GeoDataFrame from end points
int_end_points = GeoDataFrame(pd.read_csv(
    int_legs), crs=crs_degree, geometry=int_end_nodes)

# Load internal fremont legs from shapefile -- currently not used
# int_legs_shp = GeoDataFrame.from_file(data_in_process + '/Demand/SFCTA test/internal_fremont_legs.shp')

print('Data loaded sucessfully.\n')

# Spatial join (start nodes)
# docs: http://geopandas.org/mergingdata.html
print('Spatial join #1 of internal start nodes with Internal Centroid zones...')

int_int_start_nodes_OD = gpd.sjoin(
    int_start_points, icz, how='left', op='within')
# Some columns had stange names...
int_int_start_nodes_OD.rename(
    columns={
        "start_node": "start_node_lat",
        "start_no00": "start_node_lng",
        "end_node_l": "end_node_lat",
        "end_node00": "end_node_lng",
        "CentroidID": "CentroidID_O"
    },
    inplace=True
)
for column in ['index_left', 'index_right', 'OBJECTID']:
    try:
        int_int_start_nodes_OD.drop(column, axis=1, inplace=True)
    except KeyError:
        # ignore if there are no index columns
        pass

print('Spatial join #1 done.\n')
    
# Spatial join (end nodes)
# docs: http://geopandas.org/mergingdata.html
print('Spatial join #2 of internal end nodes with Internal Centroid zones...')

int_int_end_nodes_OD = gpd.sjoin(int_end_points, icz, how='left', op='within')
int_int_end_nodes_OD.rename(
    columns={
        "CentroidID": "CentroidID_D"
    },
    inplace=True
)

for column in ['index_left', 'index_right', 'OBJECTID']:
    try:
        int_int_end_nodes_OD.drop(column, axis=1, inplace=True)
    except KeyError:
        # ignore if there are no index columns
        pass

print('Spatial join #2 done.\n')

print('Combining data with origin centroids and destnation centroids...')
# Internal-to-internal origin-destinaton (OD) demand
int_int_OD = int_int_start_nodes_OD.combine_first(int_int_end_nodes_OD)

# Make it look like it was generated from ArcGIS (has OBJECTID index starting from 1)
int_int_OD['OBJECTID'] = int_int_OD.index + 1

# Export it to csv
pd.DataFrame.to_csv(int_int_OD,
                    "int_int_OD.csv",
                    encoding='utf8',
                    index=False,
                    columns=["OBJECTID", "leg_id", "start_time", "start_node_lat", "start_node_lng",
                             "end_node_lat", "end_node_lng", "CentroidID_O", "CentroidID_D"]
                    )

# Check if dataset generated with python matches the one from ArcGIS
int_int_OD_from_arcgis = pd.read_csv(
    data_collection + "/Demand Data/OD matrix/int_int_OD.csv")
int_int_OD_generated_with_py = pd.read_csv("int_int_OD.csv")

# Test

print('\nTesting the correctness of generated INT-INT-OD...\n')
print('Shape of tested data:', int_int_OD_generated_with_py.shape)
print('Shape of ref data:', int_int_OD_from_arcgis.shape)

try:
    assert_frame_equal(int_int_OD_from_arcgis, int_int_OD_generated_with_py)
    print("Success! Generated OD from Python (without arcpy) is the same as the one generated with arcpy.")
except AssertionError:
    print("Error! Generated OD from Python (without arcpy) is different than the one generated with arcpy!")

# ---------------------------------

# OD clustering


local_tz = timezone('US/Pacific')


def cluster_col_15min(df, outputname):
    df['dt'] = pd.to_datetime(df['start_time'])
    dt_15 = []
    for dt in df['dt']:
        #local_dt = dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        dt_15.append(dt.replace(minute=int(dt.minute/15)*15,
                                second=0).replace(tzinfo=pytz.utc).astimezone(local_tz))

    df['dt_15'] = dt_15
    start_end = pd.pivot_table(df, index=['CentroidID_O', 'CentroidID_D'], columns=['dt_15'], values=['OBJECTID'],
                               aggfunc={'OBJECTID': 'count'}, fill_value=0).rename(columns={'OBJECTID': 'count'})
    start_end.to_csv(outputname)
    return start_end

print('\nClustering the INT-INT-OD into 15 min. intervals...')
cluster_col_15min(int_int_OD_generated_with_py, 'int_int_OD_col_15.csv')
print('Clustering done. File "int_int_OD_col_15.csv" has been generated.\n')

# Test equality of clustered OD matrix with the older one
cluster_OD_15min_int_int = pd.read_csv('int_int_OD_col_15.csv')
cluster_OD_15min_int_int_ref = pd.read_csv(
    data_collection + "/Demand Data/OD matrix/int_int_OD_col_15.csv")

print('Testing the correctness of clustered INT-INT-OD (15 min. intervals)...\n')
print('Shape of tested data:', cluster_OD_15min_int_int.shape)
print('Shape of ref data:', cluster_OD_15min_int_int_ref.shape)

try:
    assert_frame_equal(cluster_OD_15min_int_int, cluster_OD_15min_int_int_ref)
    print("Success! Clustered OD matrix is the same as the reference.")
except AssertionError:
    print("Error! Clustered OD matrix is different than the reference!")

print("\nExecution time: %s seconds" % (time.time() - start_time))