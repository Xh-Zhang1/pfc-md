
#!/bin/bash


idx=2
for((loadA_idx=5; loadA_idx<20;loadA_idx++))
do
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_A2B -loadA_idx $loadA_idx -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 0 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 1 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 2 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 3 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 4 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 5 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 6 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 7 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 8 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 9 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 10 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 11 -idx $idx

  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 12 -idx $idx

  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 13 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 14 -idx $idx
  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx $loadA_idx -loadB_idx 15 -idx $idx


done





#
###### gpu1
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_A2B -loadA_idx 12 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_A2B -loadA_idx 15 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_A2B -loadA_idx 17 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_A2B -loadA_idx 18 -idx 4
#
#
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 12 -loadB_idx 12 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 15 -loadB_idx 15 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 17 -loadB_idx 17 -idx 4
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 20 -idx 4
#
#
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 15 -loadB_idx 15 -idx 22
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 17 -loadB_idx 17 -idx 22
#CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 20 -idx 22
#
#CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 10 -loadB_idx 10 -idx 22
#CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 12 -loadB_idx 12 -idx 22
#CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 17 -loadB_idx 17 -idx 22
#
#
#
#CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_A2B -loadA_idx 14 -idx 1
#CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 14 -loadB_idx 14 -idx 1








