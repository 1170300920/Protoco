#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export OUTPUT_PATH=$NICL_ROOT/exp_out
export CONFIG_PATH=$NICL_ROOT/tfew_configs

for ds in scifact fever vc
do
    for shot in 3 6 12 24 48
    do
        for seed in 0 2 3 4
        do
            for st in 1500 
            do
                if [[ ${shot} -eq 3 ]]
                then
                    r=$((${st}/2))
                    g=1
                elif [[ ${shot} -eq 6 ]]
                then
                    r=$((${st}/2))
                    g=2
                else
                    r=$((${st}/(${shot}/4)))
                    g=2
                fi
                python -u pl_train.py -c t03b.json+lora.json -k exp_name=${ds}_shots${shot}_seed${seed}_fs_stage1 few_shot_random_seed=${seed} seed=${seed} dataset=${ds} batch_size=1 grad_accum_factor=1 num_steps=${st} eval_batch_size=4 num_shot=${shot} eval_epoch_interval=${r}
            done
        done
    done
done
