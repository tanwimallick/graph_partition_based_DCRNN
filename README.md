# GP-DCRNN: Large scale traffic forecasting using graph-partitioning-based diffusion convolution recurrent neural network

Graph-partitioning-based DCRNN approach model the traffic on a large California highway network with 11,160 sensor locations. The general idea is to partition the large highway network into a number of small networks, and trained them with a simultaneously. The training process takes around 3 hours in a moderately sized GPU cluster, and the real-time inference can be run on traditional hardware such as CPUs. This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network.


## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.13.1
- pyaml


## Data Preparation
Download the traffic data files for entire California ['speed.h5'](https://anl.box.com/s/7hfhtie02iufy75ac1d8g8530majwci0) and ['flow.h5'](https://anl.app.box.com/s/q00j7jxbulq8pqkivjzt5ztv0ai1xjds), adjacency matrix  ['adj_mat.pkl'](https://anl.box.com/s/4143x1repqa1u26aiz7o2rvw3vpcu0wp) and distance between sensors ['distances.csv'](https://anl.box.com/s/cfnc6wryh4yrp58qfc5z7tyxbbpj4gek),  
and keep in the `scripts/` folder.


Download the traffic data files for [Los Angeles (LA)](https://anl.box.com/s/r5yc2zie02pbwwkz9hf0q1pfl2ofi8zo) and [San Francisco (SFO)] (https://anl.box.com/s/yw0dgzat4zm4jy8grls2ow7n0xcm56ou).

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

Script to submit job on cooley (GPU cluster at Argonne Leadership Computing Facility) is 

```bash
qsub_64.sh
```

The model generates prediction of DCRNN is in `data_partition_64/part{0..63}/results/dcrnn_predictions_[1-12].h5`.


## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@article{mallick2020graph,
  title={Graph-partitioning-based diffusion convolutional recurrent neural network for large-scale traffic forecasting},
  author={Mallick, Tanwi and Balaprakash, Prasanna and Rask, Eric and Macfarlane, Jane},
  journal={Transportation Research Record},
  volume={2674},
  number={9},
  pages={473--488},
  year={2020},
  publisher={SAGE Publications Sage CA: Los Angeles, CA}
}
```
