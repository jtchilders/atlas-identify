#!/usr/bin/env bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname atlas_yolo

MODELDIR=/home/parton/git/atlas-identify

module load horovod
module load keras
echo PYTHON_VERSION=$(python --version 2>&1 )

PPN=2
NTHDS=64

# source /opt/intel/vtune_amplifier/amplxe-vars.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/vtune_amplifier_2018/lib64
# export PE_RANK=$ALPS_APP_PE
# export PMI_NO_FORK=1

# export OMP_NUM_THREADS=128
# export KMP_BLOCKTIME=1
# export KMP_SETTINGS=1
# export KMP_AFFINITY="granularity=fine,compact,1,0"

# export TF_INTRA_THREADS=$OMP_NUM_THREADS
# export TF_INTER_THREADS=1

# env | sort

# echo [$SECONDS] print tensorflow version
# aprun -n 1 -N 1 python -c "import tensorflow as tf;print('tensorflow version: ',tf.__version__)"
echo [$SECONDS] run job
# aprun -n 1 -N 1 -cc depth -d 128 -j 2 amplxe-cl -collect advanced-hotspots -finalization-mode=none  -r ./tf_output -d 3000 -data-limit=1024  --  python $MODELDIR/atlas_yolo.py -c atlas_yolo.json -n 10 -a
aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NTHDS} -j 2 python $MODELDIR/atlas_yolo.py -c atlas_yolo.json -n 10 --horovod=True --num_intra=$NTHDS --num_inter=1
echo [$SECONDS] exited $?
