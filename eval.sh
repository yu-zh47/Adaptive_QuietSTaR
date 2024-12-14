NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
torchrun --nproc-per-node=$NPROC_PER_NODE \
    eval_dist.py \
    --n_votes 8 \
    --n_ahead 8