#!/bin/bash

# To run the new LM trainer, we can use a command such as:

`python trainer.py --task wikitext --output_dir results --cache_dir /scratch/gpfs/PLI/ydong/hf_cache

# Where the `--cache_dir` should be updated to your local directory, e.g. your `tmp` directory.


# Mixer gets 55 perplexity, while vanilla gets 124 perplexity.
python trainer.py --task wikitext --output_dir results --cache_dir /scratch/gpfs/PLI/ydong/hf_cache --max_seq_len 16 --max_steps=3000 --shaped_attention=mixing --eval_steps 1000 --logging_steps=1000

# The shaped attention option can be controlled with "shaped_attention" parameter being either `shaped`, `mixing`, or `vanilla`.

# `Weight frozen == 0` means the weights are trained, and `frozen == 1` means the weights are frozen at random initialization.

# Run Language model training using the trainer class.
python trainer.py --task wikitext --output_dir results --cache_dir /scratch/gpfs/PLI/ydong/hf_cache

########

# Modular addition
python modadd_exp.py --n_embd=128 --weight_frozen=1 --shaped_attention=mixing --train_steps=50 --n_layer=3

# Memorization
python hdmemorize_exp.py --n_embd=128 --weight_frozen=1 --shaped_attention=mixing --train_steps=50 --n_layer=3
