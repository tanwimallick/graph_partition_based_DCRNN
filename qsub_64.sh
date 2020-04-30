#!/bin/bash                                                                                                                                     
#COBALT -n 32                                                                                                    
#COBALT -t 12:00:00  
#COBALT -A hpcbdsm                                                                                                                          

source ~/.bashrc


echo "Running Cobalt Job $COBALT_JOBID."

mpirun -np 64 -ppn 2 python script64.py

#wait



