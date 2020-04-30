import yaml

folder_name = 'data_partitions_64'
path = folder_name + '/part' #change the path

my_dict = yaml.load(open('dcrnn_config.yaml'))
for i in range(0,64): # change the range
    fpath = path + str(i) + '/'
    my_dict['base_dir'] = fpath + 'model'
    my_dict['data']['dataset_dir'] = fpath
    my_dict['data']['graph_pkl_filename'] = fpath + 'adj_mat.pkl'
    my_dict['data']['hdf_filename'] = fpath + 'speed.h5' # change h5 file name
    
    with open(fpath + 'dcrnn_config.yaml', 'w') as yaml_file:
        yaml.dump(my_dict, yaml_file, default_flow_style=False)
 
