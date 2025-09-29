#!/bin/bash

# Script to reproduce results

for ((i=1;i<4;i+=1))
do 
	CUDA_VISIBLE_DEVICES=0 python cleanrl/ppg_procgen_detached.py \
	--seed $i
done
