#!/bin/bash


#LOG_LOCATION="/home/xiaohan/pycharm/pfc-md/multi_pfc_md_4present/logs/mylogfile.log"


for((idx=90; idx<=95;idx++))
do
  #python cluster_training_unit.py -switch con_A -idx $idx -lsq1 0.999 -lsq2 0.001

  CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_A -md 30 -c_delay 600 -idx $idx

done








#bash train1.sh 2>&1 | tee logs/mytype3log30_45.log

