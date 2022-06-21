import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
import time
import os
import sys

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

# any txt files remaining must be empty, so remove them
for filename in os.listdir(folder_name):
    if filename[-3:]=='txt':
        file_txt = os.path.join(folder_name, filename)
        os.remove(file_txt)

# create a PDF(lat, lon, vel)
for filename in os.listdir(folder_name):
