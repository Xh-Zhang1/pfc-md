
## Dependencies
python 3.6, pytorch, sklearn, numba

## Contents

The folder 'multi_4present_context' contains the models used to study the cross-modal
context-dependent decision-making task (CAC & RDM subtasks). This is referred to first-type
context-dependent decision-making task (Figures 1-5)

The folder 'context_switch_mix' contains the models used to study the cue-reversal context-dependent decision-making task.
This is referred to as the second-type context dependent decision-making task (Figures 7,8)


## Reproducing the results of the paper
You can produce the main figures of the paper by running the files in folder 'multi_4present_context'
 'fig1.py','fig2.py', 'fig3.py', 'fig4.py' and 'fig5.py' using python. 

You can get subspace angle by running the files in folder 'context_switch_mix'
 'analysis/calculate_subspace_angle_plot.py'

 
## Pre-trained models
We provided pretrained models for analysis. You can download from https://drive.google.com/drive/u/0/folders/1yl1lnm_plnFKKcRMXPSKEGMR0R2Y6dXT?lfhs=2.

## Start to train
To train the models, simply run the files 'multi_4present_context/cluster_training_unit.py',
'context_switch_mix/cluster_training_unit.py' using python. 


## Acknowledgement
This code was adapted from the software from the following papers:

(1) G. R. Yang et al. Task representations in neural networks trained to
perform many cognitive tasks. Nat. Neurosci., 22, 297 (2019).

(2) A. E. Orhan and W. J. Ma. A diverse range of factors affect the nature
of neural representations underlying
short-term memory. Nat. Neurosci., 22, 275 (2019).

