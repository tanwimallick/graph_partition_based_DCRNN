from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob

from lib.utils import load_graph_data
from lib.utils import generate_seq2seq_data
from lib.utils import train_val_test_split
from lib.utils import StandardScaler

from model.dcrnn_supervisor import DCRNNSupervisor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        supervisor_config['model']['num_nodes'] = num_nodes = len(sensor_ids)

        # Data preprocessing 
        traffic_df_filename = supervisor_config['data']['hdf_filename'] 
        df_data = pd.read_hdf(traffic_df_filename)
        #df_data = df_data.iloc[int(df_data.shape[0]/3):,:]
        validation_ratio = supervisor_config.get('data').get('validation_ratio')
        test_ratio = supervisor_config.get('data').get('test_ratio')
        df_train, df_val, df_test = train_val_test_split(df_data, val_ratio=validation_ratio, test_ratio=test_ratio)

        batch_size = supervisor_config.get('data').get('batch_size')
        val_batch_size = supervisor_config.get('data').get('val_batch_size')
        test_batch_size = supervisor_config.get('data').get('test_batch_size')
        horizon = supervisor_config.get('model').get('horizon')
        seq_len = supervisor_config.get('model').get('seq_len')
        scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
        
        data_train = generate_seq2seq_data(df_train, batch_size, seq_len, horizon, num_nodes, 'train', scaler)
        data_val = generate_seq2seq_data(df_val, val_batch_size, seq_len, horizon, num_nodes, 'val', scaler)
        data_train.update(data_val)
        #data_train['scaler'] = scaler
    
        data_test = generate_seq2seq_data(df_test, test_batch_size, seq_len, horizon, num_nodes, 'test', scaler)
        #data_test['scaler'] = scaler


        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx, data_train, supervisor_config)
            
            data_tag = supervisor_config.get('data').get('dataset_dir')
            folder = data_tag + '/model/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            # Train
            supervisor.train(sess=sess)

            # Test
            yaml_files = glob.glob('%s/model/*/*.yaml'%data_tag, recursive=True)
            yaml_files.sort(key=os.path.getmtime)
            config_filename = yaml_files[-1] #'config_%d.yaml' % config_id
            
            with open(config_filename) as f:
                config = yaml.load(f)
            # Load model and evaluate
            supervisor.load(sess, config['train']['model_filename'])
            y_preds = supervisor.evaluate(sess, data_test)
            
            n_test_samples = data_test['y_test'].shape[0]
            folder = data_tag + '/results/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            for horizon_i in range(data_test['y_test'].shape[1]):
                y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
                eval_dfs = df_test[seq_len + horizon_i: seq_len + horizon_i + n_test_samples]
                df = pd.DataFrame(y_pred, index=eval_dfs.index, columns=eval_dfs.columns)
                #df = pd.DataFrame(y_pred, columns=df_test.columns)
                filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_prediction_%s.h5' %str(horizon_i+1))
                df.to_hdf(filename, 'results')
                
            print('Predictions saved as %s/results/dcrnn_prediction_[1-12].h5...' %data_tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
