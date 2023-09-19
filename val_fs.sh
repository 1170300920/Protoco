#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_PATH=output

for ds in scifact fever vc
do 
    for shot in 3 6 12 24 48
    do
        for seed in 0 2 3 4
        do
            python -u my_validate.py -k modelpath=${OUTPUT_PATH}/${ds}_shots${shot}_seed${seed}_fs_stage2/finish.pt
        done
    done
done