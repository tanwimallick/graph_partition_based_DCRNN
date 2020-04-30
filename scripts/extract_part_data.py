#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 08:17:00 2019

@author: tanwimallick
"""

import pandas as pd
import numpy as np
import os

    
def save_h5(req_df, metric, folder_name, part):

    folder = folder_name + '/part' + str(part)
    if not os.path.exists(folder):
        os.makedirs(folder)

    req_df.to_hdf(folder +'/' + '%s.h5'%metric, key='df', mode='w')


if __name__ == "__main__":

    loop_df = pd.read_csv('graph_sensor_locations_11k.csv')
    loop_ids = loop_df['sensor_id'].astype('str').tolist()
    
    input_data = 'speed'
    folder_name = 'data_partitions_64'
    h5f = pd.read_hdf(input_data + '.h5')
    partition = np.genfromtxt('tiny_11k_graph_new.txt.part.64', dtype=int, delimiter="\n", unpack=False)
    
    partition_ids = np.unique(partition)
    for p in partition_ids: 
    
        indices = partition==p
        part_df = loop_df[indices]
        loop_ids = part_df['sensor_id'].astype('str').tolist()
        
        h5files = h5f[loop_ids] 
        
        save_h5(h5files, input_data, folder_name, p)

    