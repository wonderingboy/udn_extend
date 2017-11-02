#!/bin/sh

export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export LD_LIBRARY_PATH=/mnt/lustre/zhouhui/pedestrain_det/pedestrian_detection/udn_extend/external/caffe_new/build/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/mnt/lustre/share/intel64/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/mnt/lustre/share/nccl-8.0/lib:$LD_LIBRARY_PATH
now=$(date +"%Y%m%d_%H%M%S") 

srun -p bj11part --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 matlab -nodisplay -r "script_rpn_rcnn_new;exit" 2>&1 | tee log/matlab-${now}.log &

