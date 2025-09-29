#!/bin/bash

# Script to reproduce results

for ((i=1;i<4;i+=1))
do 
	CUDA_VISIBLE_DEVICES=2 python cleanrl/ppg_procgen.py \
	--seed $i
done
