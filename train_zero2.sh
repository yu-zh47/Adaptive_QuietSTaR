NNODES=1
GPUS_PER_NODE=8
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
    quiet-star-train.py \
    
    