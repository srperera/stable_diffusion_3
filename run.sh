# Load necessary modules
module purge

echo "loading necessary modules"
module load python/3.12
module load gcc/12.3.0
module load nccl/2.19.3-1
module load cuda/12.6.2
module load miniconda3/24.1.2-py310

# Set environment variables for distributed training
# Need to update couple things here each time we request a new node/multinode job
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LOGLEVEL=INFO
export MASTER_NODE=a0264.ten.osc.edu && echo "MASTER_NODE=$MASTER_NODE"
export MASTER_ADDR=10.6.16.4 && echo "MASTER_ADDR=$MASTER_ADDR"
export MASTER_PORT=29500 && echo "MASTER_PORT=$MASTER_PORT"
export WORLD_SIZE=8 && echo "WORLD_SIZE=$WORLD_SIZE"
export NCCL_DEBUG=WARN

# This is the core command
# srun will execute this command on each of the nodes you requested.
# Hugging Face Accelerate will use the environment variables above.
#srun bash load_modules.sh
srun --nodes=4 --ntasks-per-node=2 --gpus-per-node=2 bash -c '
  export LOCAL_RANK=$SLURM_LOCALID
  '"$args_serialized"'
  python3 -u train.py
'
