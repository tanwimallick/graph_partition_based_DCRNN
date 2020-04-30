from mpi4py import MPI
import os
import socket
import subprocess


hostname = socket.gethostname()
rank = MPI.COMM_WORLD.Get_rank()
gpu_device = rank % 2
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

arg = '--config_filename=data_partitions_64/part' +  str(rank) + '/dcrnn_config.yaml &> data_partitions_64/part' + str(rank)+'.out'

cmd = "python dcrnn_train.py " + arg 
subprocess.run(cmd, shell=True)





