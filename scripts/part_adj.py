#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:10:39 2019

@author: tanwimallick
"""

import pandas as pd 
import numpy as np
import pickle
import os

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    
    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx

if __name__ == "__main__":
    sensor_df = pd.read_csv('graph_sensor_locations_11k.csv')
    dist_df = pd.read_csv('distances.csv')
    partition = np.genfromtxt('tiny_11k_graph_new.txt.part.64', dtype=int, delimiter="\n", unpack=False)
    folder_name = 'data_partitions_64'
    
    partition_ids = np.unique(partition)
    for p in partition_ids:
    
        indices = partition==p
        part_df = sensor_df[indices]
        distance_df = dist_df.loc[(dist_df['from'].isin(part_df['sensor_id'])) & (dist_df['to'].isin(part_df['sensor_id']))]
        distance_df = distance_df.reset_index(drop=True)
        distance_df['from'] = distance_df['from'].astype('str')
        distance_df['to'] = distance_df['to'].astype('str')
        
        sensor_ids = part_df['sensor_id'].astype(str).values.tolist()
    
        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
        
        # Save to pickle file and the sensor ids
        folder = folder_name + '/part' + str(p) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + 'adj_mat.pkl', 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
        with open(folder + 'sensor_ids.txt', 'w') as f:
            for items in sensor_ids:
                f.write('%s,' %items)
        
        
    

    
    
    
    
    
    