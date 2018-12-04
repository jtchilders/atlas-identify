#!/bin/bash
#SBATCH -J atlas_identify
#SBATCH -A ATLAS
#SBATCH -p bdwall
#SBATCH -N 10
#SBATCH -o slurm-%j.log
#SBATCH -e slurm-%j.logerror
#SBATCH --ntasks-per-node=1
#SBATCH -t 360
#SBATCH --export NONE

echo [$SECONDS] Running Atlas Identify ML on $(date)
MINICONDA=/home/jchilder/miniconda3
export PATH=$MINICONDA/bin:$PATH
source activate tf12b
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#export PYTHONPATH=/home/jchilder/miniconda3/envs/tf12a/protobuf/install/lib/python3.6/site-packages:$PYTHONPATH
# export PYTHONPATH=/blues/gpfs/group/3/ATLAS/users/jchilder/tf/r1.12_install:$PYTHONPATH
# source $MINICONDA/etc/profile.d/conda.sh
# conda activate tf
# source $MINICONDA/setup.sh
# source /home/jchilder/envs/tensorflow/bin/activate
echo [$SECONDS] python: $(which python)
echo [$SECONDS] python version $(python --version)

#echo [$SECONDS] test MPI
#srun -N $SLURM_JOB_NUM_NODES $(which python) -c "from mpi4py import MPI;print('rank %s of %s' % (MPI.COMM_WORLD.Get_rank(),MPI.COMM_WORLD.Get_size()))"
echo [$SECONDS] tensorflow: $(python -c "import tensorflow;print(tensorflow.__version__)")

INST=/blues/gpfs/group/3/ATLAS/users/jchilder/atlas-identify
export PYTHONPATH=$INST:$PYTHONPATH
echo [$SECONDS] env:
env | sort
CONFIG=slurm-${SLURM_JOB_ID}.config
cp atlas_identify.json $CONFIG
cp submit.sh slurm-${SLURM_JOB_ID}.sh
cp $INST/pix_only_model.py slurm-${SLURM_JOB_ID}.pix_only_model.py
#echo [$SECONDS] ldd
#ldd /home/jchilder/miniconda3/envs/tf12a/lib/python3.6/site-packages/google/protobuf/pyext/_message.cpython-36m-x86_64-linux-gnu.so
echo [$SECONDS] srun ldd
srun -N 1 -n 1 /usr/bin/ldd /home/jchilder/miniconda3/envs/tf12a/lib/python3.6/site-packages/google/protobuf/pyext/_message.cpython-36m-x86_64-linux-gnu.so
echo [$SECONDS] starting job
srun --export ALL -N $SLURM_JOB_NUM_NODES -e slurm-${SLURM_JOB_ID}.error -o slurm-${SLURM_JOB_ID}.output $(which python) $INST/atlas_identify.py -c $CONFIG --num_intra=36 --num_inter=1 --tb_logdir=slurm-$SLURM_JOB_ID --horovod --adam
echo [$SECONDS] exited with code $?

