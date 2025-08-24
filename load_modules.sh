module list

echo "purging modules"
module purge

echo "loading necessary modules"
module load python/3.12
module load gcc/12.3.0
module load nccl/2.19.3-1

echo "current modules"
module list

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1