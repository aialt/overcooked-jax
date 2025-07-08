#!/bin/bash
cd /home/luka/repo/JAXOvercooked

seeds=(0)

for seed in "${seeds[@]}"; do
  echo "Running $seed"
  # Run the model with the specified parameters
   python -m baselines.IPPO_CL --seq_length=2  \
                                --seed=$seed \
                                --group="test A-GEM" \
                                --tags $seed \
                                --layouts "easy_levels" \
                                --test_interval 0.05 \
                                --num_envs 64 
done
