#!/bin/bash

task=${task:-"additionstream_10"}
shaped_attention=${shaped_attention:-"mixing"}
n_head=${n_head:-128}

job_name=algo_${task}_${shaped_attention}_head${n_head}

task=${task} n_head=${n_head} shaped_attention=${shaped_attention} sbatch -J ${job_name} --mail-user=ydong@princeton.edu --mail-type=end --mail-type=begin depth.sh
