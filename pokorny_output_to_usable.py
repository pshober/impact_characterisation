import os
import sys
import argparse
import time
from astropy.io.votable import parse_single_table
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from scipy.optimize import curve_fit
from mpi4py import MPI
import numpy as np
import pandas as pd
import datetime
import shutil

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
import seaborn as sns
import matplotlib.pyplot as plt
from random import choices

folder_name = '/media/patrick/my_passport/output_files_pokorny'

# create empty csv with correct columns
empty_df = pd.DataFrame({'pos':[],
                         'heliocentric_distance':[],
                         'longitude':[],
                         'latitude':[],
                         'impact_velocity':[],
                         'impact_probability':[]})

empty_df.to_csv('granvik_impact_output.csv')

# converts the txt files to csvs
for filename in os.listdir(folder_name):
    if filename[-3:]=='txt':
        file_txt = os.path.join(folder_name, filename)
        data = pd.read_csv(file_txt, sep=',')

        file_csv = os.path.join(folder_name, filename[:-4]+'.csv')
        data.to_csv(file_csv)

        os.remove(file_txt)

# # any txt files remaining must be empty, so remove them
# for filename in os.listdir(folder_name):
#     if filename[-3:]=='txt':
#         file_txt = os.path.join(folder_name, filename)
#         os.remove(file_txt)

# create a PDF(lat, lon, vel)
results = np.zeros((100,100,100))
lon_bounds = np.linspace(-180.0, 180.0, 101)
lat_bounds = np.linspace(-90.0, 90.0, 101)
vel_bounds = np.linspace(10.0, 80.0, 101)  # rough estimate

def h_index_find(lat, lon, vel, lat_edges, lon_edges, vel_edges):
    closest_lat_idx = min(range(len(lat_edges)), key=lambda i: abs(lat_edges[i] - lat)) - 1
    closest_lon_idx = min(range(len(lon_edges)), key=lambda i: abs(lon_edges[i] - lon)) - 1
    vel_idx = min(range(len(vel_edges)), key=lambda i: abs(vel_edges[i] - vel)) - 1

    if lat_edges[closest_lat_idx] >  lat:
        lat_idx = closest_lat_idx - 1
    elif lat_edges[closest_lat_idx] <=  lat:
        lat_idx = closest_lat_idx

    if lon_edges[closest_lon_idx] >  lon:
        lon_idx = closest_lon_idx - 1
    elif lon_edges[closest_lon_idx] <=  lon:
        lon_idx = closest_lon_idx

    return lat_idx, lon_idx, vel_idx

for i, filename in enumerate(os.listdir(folder_name)):
     df = pd.read_csv(os.path.join(folder_name, filename))

     print(f"{i} Files Processed", flush=True, end="\r")

     for index, row in df.iterrows():
         lat_idx, lon_idx, vel_idx = h_index_find(row['latitude'], row['longitude'], row['impact_velocity'], lat_bounds, lon_bounds, vel_bounds)

         results[lat_idx, lon_idx, vel_idx] += row['impact_probability']

np.save('results_histogram_newrange.npy', results)

def middle4bins(array):
    return np.array([(array[i]+array[i+1]) / 2.0 for i in range(len(array)-1)])

lon_mids = middle4bins(lon_bounds)
lat_mids = middle4bins(lat_bounds)
vel_mids = middle4bins(vel_bounds)

# plot lon v. lat impact 2D histogram
x,y = np.meshgrid(lon_mids, lat_mids)
plt.hist2d(np.concatenate(x),np.concatenate(y), weights=np.concatenate(np.sum(results,axis=-1).T), bins=100); plt.show()

# plot lon v. vel 2D histogram
x,y = np.meshgrid(lon_mids,vel_mids)
plt.hist2d(np.concatenate(x),np.concatenate(y), weights=np.concatenate(np.sum(results,axis=0)), bins=100); plt.show()
