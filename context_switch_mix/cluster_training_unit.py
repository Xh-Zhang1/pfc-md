import os
import sys

sys.path.append(os.getcwd())
import train
import default
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_cuda', action='store_true', default=True, help='whether gpu')
parser.add_argument('--activation', '-acf', type=str, default='softplus', help='recurrent nonlinearity')

parser.add_argument('--w2_reg', '-w2', type=float, default=0.1, help='strength of weight decay on recurrent weights')
parser.add_argument('--r2_reg', '-r2', type=float, default=0.1, help='strength of weight decay on recurrent weights')
parser.add_argument('--batch_size', '-bs', type=int, default=512, help='number of trajectories per batch')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005, help='gradient descent learning rate')
parser.add_argument('--is_EI', type=str, default='EI', help='RNN or LSTM')

parser.add_argument('--model_idx', '-idx', type=int, default=0, help='we train 10 model')
parser.add_argument('--rule_name', '-rule', type=str, default='HL_task', help='context')

parser.add_argument('--use_reset', '-reset', type=str, default='yes', help='context')
parser.add_argument('--mask_type', '-mask', type=str, default='type8', help='mask type for hidden layer')
parser.add_argument('--input_mask', '-input_mask', type=str, default='mask0', help='mask type for hidden layer')

parser.add_argument('--mode_mask', '-mode', type=str, default='train', help='mask for trian or test')
parser.add_argument('--n_md', '-md', type=int, default=30, help='number of place cells')
parser.add_argument('--n_rnn', '-rnn', type=int, default=256, help='number of place cells')
parser.add_argument('--coherence', '-p_coh', type=float, default=1, help='gradient descent learning rate')

parser.add_argument('--cue_duration', '-c_dur', type=int, default=800, help='number of steps in trajectory')
parser.add_argument('--cue_delay', '-c_delay', type=int, default=600, help='number of steps in trajectory')
parser.add_argument('--stim', '-stim', type=int, default=100, help='number of steps in trajectory')
parser.add_argument('--response_time', '-resp', type=int, default=100, help='number of steps in trajectory')

parser.add_argument('--scale_random', '-sr', type=float, default=1.0, help='gradient descent learning rate')
parser.add_argument('--dropout', '-drop', type=float, default=0.0, help='gradient descent learning rate')

parser.add_argument('--stim_scale', '-SS', type=float, default=1.0, help='gradient descent learning rate')
parser.add_argument('--stim_std', '-std', type=float, default=0.1, help='gradient descent learning rate')

parser.add_argument('--mask_start', '-cost', type=str, default='respOn', help='mask for trian or test')
parser.add_argument('--sparsity_pc_md', '-sp', type=float, default=0.25, help='gradient descent learning rate')
parser.add_argument('--switch_context', '-switch', type=str, default='con_A', help='mask for trian or test')
parser.add_argument('--start_switch', action='store_true', default=True, help='whether gpu')
parser.add_argument('--loadA_idx', '-loadA_idx', type=str, default=str(0), help='number of steps in trajectory')
parser.add_argument('--loadB_idx', '-loadB_idx', type=str, default=str(0), help='number of steps in trajectory')


parser.add_argument('--cost_lsq_1', '-lsq1', type=float, default=0.999, help='strength of weight decay on recurrent weights')
parser.add_argument('--cost_lsq_2', '-lsq2', type=float, default=0.000, help='strength of weight decay on recurrent weights')





"""
train A   : loadA = 'no',loadB = 'no'
train A->B: loadA = dir_model_A,loadB='no'
train B->A: loadA = 'no',loadB=dir_model_B
"""

'''
source activate pfc-md
cd /home/xh/pycharm_project/context_switch_12.15/context_switch_mix_coh
'''

'''
CUDA_VISIBLE_DEVICES=3 python cluster_training_unit.py -switch con_A -md 30 -c_delay 600 -idx 8

CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_A2B -loadA_idx 15 -idx 6
CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_B2A -loadA_idx 15 -loadB_idx 15 -idx 4



CUDA_VISIBLE_DEVICES=0 python cluster_training_unit.py -switch con_A2B -loadA_idx 20 -idx 21


CUDA_VISIBLE_DEVICES=1 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 20 -idx 4

CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 10 -idx 21
CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 15 -idx 21
CUDA_VISIBLE_DEVICES=2 python cluster_training_unit.py -switch con_B2A -loadA_idx 20 -loadB_idx 20 -idx 21

'''

def load_A_model(hp):
    """
    firstly, loading A model, then train context_A switch context_B
    """
    loadA_idx = hp['loadA_idx']
    loadB_idx = hp['loadB_idx']

    model_name_A = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) \
                 + '_' + str(hp['lsq1']) + '_' + str(hp['cue_delay']) + '_model_' + str(hp['model_idx'])

    load_folder_from_A = os.path.join('./' + 'model_A_' + str(hp['p_coh']), model_name_A, str(loadA_idx))
    print('load_folder_from_A:', load_folder_from_A)

    return load_folder_from_A


def load_B_model(hp):
    loadA_idx = hp['loadA_idx']
    loadB_idx = hp['loadB_idx']

    model_name_B = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                   + str(hp['model_idx']) + '_load_A' + str(loadA_idx)

    load_folder_from_AB = os.path.join('./' + 'model_A2B_' + str(hp['p_coh']), model_name_B, str(loadB_idx))

    return load_folder_from_AB


def train_model(hp):
    hp = hp
    loadA_idx = hp['loadA_idx']
    loadB_idx = hp['loadB_idx']

    if hp['switch_context'] == 'con_A':
        load_model_dir_A = 'no'
        load_model_dir_B = 'no'

        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md'])\
                   + '_' + str(hp['lsq1'])+ '_' + str(hp['cue_delay']) + '_model_' + str(hp['model_idx'])

        local_folder_current = os.path.join('./' + 'model_A_' + str(hp['p_coh']), model_name)
        os.system('rm -rf ' + local_folder_current)

    elif hp['switch_context'] == 'con_A2B':
        load_model_dir_A = load_A_model(hp)
        load_model_dir_B = 'no'
        #### training save
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(loadA_idx)

        local_folder_current = os.path.join('./' + 'model_A2B_' + str(hp['p_coh']), model_name)
        os.system('rm -rf ' + local_folder_current)


    elif hp['switch_context'] == 'con_B2A':
        load_model_dir_A = load_A_model(hp)
        load_model_dir_B = load_B_model(hp)

        #### training save
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(loadA_idx) + '_B' + str(loadB_idx)

        local_folder_current = os.path.join('./' + 'model_B2A_' + str(hp['p_coh']), model_name)
        os.system('rm -rf ' + local_folder_current)

    hp['model_dir_current'] = local_folder_current
    while True:
        trainerObj = train.Trainer(hp=hp, load_model_dir_A=load_model_dir_A, load_model_dir_B=load_model_dir_B,
                                   model_dir=local_folder_current, is_cuda=True)
        stat = trainerObj.train(max_samples=200000, display_step=400)
        if stat == 'OK':
            break
        # else:
        #     run_cmd = 'rm -r ' + local_folder_name
        #     os.system(run_cmd)


if __name__ == "__main__":
    arg = parser.parse_args()
    rule_name = arg.rule_name
    hp = default.get_default_hp(rule_name=rule_name)

    hp['learning_rate'] = arg.learning_rate
    hp['is_cuda'] = arg.is_cuda
    hp['start_switch'] = arg.start_switch
    hp['batch_size_train'] = arg.batch_size
    hp['l2_firing_rate'] = arg.r2_reg
    hp['l2_weight'] = arg.w2_reg
    hp['activation'] = arg.activation

    hp['model_idx'] = arg.model_idx
    hp['is_EI'] = arg.is_EI
    hp['rule_name'] = arg.rule_name
    hp['use_reset'] = arg.use_reset
    hp['mask_type'] = arg.mask_type
    hp['input_mask'] = arg.input_mask
    hp['mode_mask'] = arg.mode_mask
    hp['n_rnn'] = arg.n_rnn
    hp['n_md'] = arg.n_md
    hp['p_coh'] = arg.coherence

    hp['cue_duration'] = arg.cue_duration
    hp['cue_delay'] = arg.cue_delay
    hp['stim'] = arg.stim
    hp['response_time'] = arg.response_time

    hp['mask_start'] = arg.mask_start
    hp['switch_context'] = arg.switch_context

    hp['stim_scale'] = arg.stim_scale
    hp['stim_std'] = arg.stim_std
    hp['sparsity_pc_md'] = arg.sparsity_pc_md
    hp['dropout'] = arg.dropout
    hp['scale_random'] = arg.scale_random
    hp['loadA_idx'] = arg.loadA_idx
    hp['loadB_idx'] = arg.loadB_idx

    hp['lsq1'] = arg.cost_lsq_1
    hp['lsq2'] = arg.cost_lsq_2


    if rule_name == 'both_RDM_HL_task':
        hp['rule_trains'] = ['RDM_task', 'HL_task']  #
        hp['rule_probs'] = [0.5, 0.5]
    elif rule_name == 'HL_task':
        hp['rule_trains'] = ['HL_task']  #
        hp['rule_probs'] = [1.0]

    train_model(hp=hp)
