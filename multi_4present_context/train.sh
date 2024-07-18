#!/bin/bash


#LOG_LOCATION="/home/xiaohan/pycharm/pfc-md/multi_pfc_md_4present/logs/mylogfile.log"



for((idx=92; idx<=100;idx++))
do
  python cluster_training_unit.py -mask type3  -acf softplus -sr 5.0 -drop 0.0 -std 0.1 -idx $idx

done



#bash train.sh 2>&1 | tee logs/mytype3log30_45.log

