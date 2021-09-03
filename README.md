# GP-DCRNN: Large scale traffic forecasting using diffusion convolution recurrent neural network


This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network.


## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.13.1
- pyaml


## Data Preparation
Download the traffic data files for entire California ['speed.h5'](https://anl.box.com/s/7hfhtie02iufy75ac1d8g8530majwci0), adjacency matrix  ['adj_mat.pkl'](https://anl.box.com/s/4143x1repqa1u26aiz7o2rvw3vpcu0wp) and distance between sensors ['distances.csv'](https://anl.box.com/s/cfnc6wryh4yrp58qfc5z7tyxbbpj4gek),  
and keep in the `scripts/` folder.


```bash
# Generate adjucency matrix for 64 partitions. It will generate 64 folder containing adj_mat.pkl for each partition
# Input: graph_sensor_locations_11k.csv, distances.csv, and tiny_11k_graph_new.txt.part.64 (graph partition from Metis)

python part_adj.py

# Generate speed.h5 and sensor_ids.txt containing station ids for each partition 
# Input: graph_sensor_locations_11k.csv, and tiny_11k_graph_new.txt.part.64

python extract_part_data.py

#Provide the folder name in the following code to generate configuration files for all partitions 

python copy_yaml.py

# Move the folder Ex. data_partition_64 outside the `scripts/` folder. 

# move the data_partition_64 folder outside of the script folder

#To run DCRNN on the local machine with one partition

python dcrnn_train.py --config_filename=data_partitions_64/part0/dcrnn_config.yaml

```

Script to submit job on cooley is 

```bash
qsub_64.sh
```

The model generates prediction of DCRNN is in `data_partition_64/part{0..63}/results/dcrnn_predictions_[1-12].h5`.


