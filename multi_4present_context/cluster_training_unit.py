
import os
import sys
sys.path.append(os.getcwd())
import task
import train
import tools
import default
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',    		 type=int,	default=1000000,				help='batches per epoch')
parser.add_argument('--RNN_type',			 type=str,	default='RNN',					help='RNN or LSTM')
parser.add_argument('--is_cuda', 			 action='store_true',default=True,			help='whether gpu')

parser.add_argument('--activation',	'-acf',		 type=str,	default='softplus',					help='recurrent nonlinearity')

parser.add_argument('--w2_reg','-w2', type=float, default=0.1,				help='strength of weight decay on recurrent weights')
parser.add_argument('--r2_reg','-r2', type=float, default=0.1,				help='strength of weight decay on recurrent weights')
parser.add_argument('--sequence_length','-sl',	 type=int,	default=10,						help='number of steps in trajectory')
parser.add_argument('--batch_size', '-bs',	 type=int,	default=256,					help='number of trajectories per batch')
parser.add_argument('--learning_rate','-lr', type=float,	default=0.0005,					help='gradient descent learning rate')
parser.add_argument('--is_EI',			 type=str,	default='EI',					help='RNN or LSTM')

parser.add_argument('--model_idx','-idx', type=int,	default=1,					help='we train 10 model')
parser.add_argument('--rule_name',	'-rule',	type=str,	default='both_RDM_HL_task',					help='context')

parser.add_argument('--use_reset',	'-reset',	type=str,	default='yes',					help='context')
parser.add_argument('--mask_type',	'-mask',	type=str,	default='type3',					help='mask type for hidden layer')
parser.add_argument('--input_mask',	'-input_mask',	type=str,	default='input_mask',					help='mask type for hidden layer')

parser.add_argument('--mode_mask',	'-mode',	type=str,	default='train',					help='mask for trian or test')
parser.add_argument('--n_md',  '-md',				 type=int,	default=200,					help='number of place cells')
parser.add_argument('--n_rnn',  '-rnn',	type=int,	default=256,					help='number of place cells')
parser.add_argument('--coherence','-p_coh',  type=float,	default=0.92,					help='gradient descent learning rate')


parser.add_argument('--cue_duration','-c_dur', type=int,	    default=800,						help='number of steps in trajectory')
parser.add_argument('--cue_delay','-c_delay', type=int,	    default=800,						help='number of steps in trajectory')
parser.add_argument('--stim','-stim', type=int,	    default=200,						help='number of steps in trajectory')
parser.add_argument('--response_time','-resp', type=int,	    default=200,						help='number of steps in trajectory')

parser.add_argument('--scale_random','-sr',  type=float,	default=1.0,					help='gradient descent learning rate')
parser.add_argument('--scale_HL','-SH',  type=float,	default=1.0,					help='gradient descent learning rate')
parser.add_argument('--scale_RDM','-SR',  type=float,	default=1.0,					help='gradient descent learning rate')

parser.add_argument('--dropout','-drop',  type=float,	default=0.0,					help='gradient descent learning rate')

parser.add_argument('--stim_scale','-SS',  type=float,	default=1.0,					help='gradient descent learning rate')
parser.add_argument('--stim_std','-std',  type=float,	default=0.0,					help='gradient descent learning rate')

parser.add_argument('--mask_start',	'-cost',	type=str,	default='respOn',					help='mask for trian or test')
#    python cluster_training_unit.py -mask type3  -acf softplus -drop 0.2 -std 0.1 -idx 1
#    python cluster_training_unit.py -mask type2  -acf softplus -SR 1.0 -SH 1.0  -idx 1
#




def train_model(hp):

    hp = hp
    model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
                 '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
                 '_drop'+str(hp['dropout'])+'_'+str(hp['model_idx'])
    # model_name = str(hp['mask_type'])+'_'+str(hp['activation'])+ '_'+str(hp['cue_duration'])+'_'+\
    #              str(hp['cue_delay'])+'_'+str(hp['stim'])+'_'+ str(hp['mask_start'])+'_'+str(hp['model_idx'])
    local_folder_name = os.path.join('./'+'model_'+str(hp['p_coh']),  model_name)
    os.system('rm -rf ' + local_folder_name)

    while True:
        trainerObj = train.Trainer(model_dir=local_folder_name, rule_trains=hp['rule_trains'], hp=hp, is_cuda=hp['is_cuda'])
        stat = trainerObj.train(max_samples=1e7, display_step=400)
        if stat is 'OK':
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
    hp['batch_size_train'] = arg.batch_size
    hp['l2_firing_rate'] = arg.r2_reg
    hp['l2_weight'] = arg.w2_reg
    hp['activation'] = arg.activation

    hp['model_idx'] = arg.model_idx
    hp['is_EI'] = arg.is_EI
    hp['rule_name'] = arg.rule_name
    hp['use_reset'] = arg.use_reset
    hp['mask_type']=arg.mask_type
    hp['input_mask']=arg.input_mask
    hp['mode_mask']=arg.mode_mask
    hp['n_rnn']=arg.n_rnn
    hp['n_md']=arg.n_md
    hp['coherence']=arg.coherence

    hp['cue_duration']=arg.cue_duration
    hp['cue_delay']=arg.cue_delay
    hp['stim']=arg.stim
    hp['response_time']=arg.response_time

    hp['mask_start']=arg.mask_start

    hp['scale_RDM']=arg.scale_RDM
    hp['scale_HL']=arg.scale_HL
    hp['stim_scale']=arg.stim_scale
    hp['stim_std']=arg.stim_std
    hp['dropout']=arg.dropout
    hp['scale_random']=arg.scale_random


    if rule_name == 'both_RDM_HL_task':
        hp['rule_trains'] = ['RDM_task', 'HL_task']#
        hp['rule_probs'] = [0.5,0.5]
    elif rule_name == 'RDM_task':
        hp['rule_trains'] = ['RDM_task']#
        hp['rule_probs'] = [1.0]
    elif rule_name == 'HL_task':
        hp['rule_trains'] = ['HL_task']#
        hp['rule_probs'] = [1.0]

    train_model(hp)

