"""Main training loop"""

from __future__ import division

import numpy as np
import torch
from torch.utils.data import DataLoader

from collections import defaultdict
import time
import sys
import os

import train_stepper
import network
import tools
import dataset


import pdb
_color_list = ['blue', 'red', 'black', 'yellow', 'pink']


def get_perf_rule_dm_vector(output, target_choice, epochs, fix_strength=0.1, action_threshold=0.5, response_duration=int(300/20)):
    '''
    output (75, 512, 4)
    '''

    cue_on = epochs['cue_stimulus'][0][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]

    fix_start = cue_on
    fix_end = response_on - 1

    batch_size = output.shape[1]  # 512
    action_at_fix = np.array([np.mean(output[fix_start:fix_end, i, :]) > fix_strength for i in range(batch_size)])
    fail_action = action_at_fix
    # print('fail_action',fail_action)
    print('output',output[response_off-3:response_off,0,:])

    choice_correct=[]
    actual_choices=[]
    for i in range(batch_size):
        outp = output[response_on:response_off, i, :]
        mean = np.mean(outp,axis=0)
        actual_choice = np.argmax(mean)+1
        targ_choice = target_choice[i]
        #if target_choice[i] == np.argmax(action[i,:]):# and np.max(action[i,:])>0.8:
        if targ_choice == actual_choice:# and np.max(action[i,:])>0.5:
            choice_corr = 1
        else:
            choice_corr = 0
        actual_choices.append(actual_choice)
        choice_correct.append(choice_corr)
    # print('actual_choices',actual_choices)
    # print('target_choice',target_choice)
    # print('choice_correct',choice_correct)

    success_action_prob = np.sum(np.array([choice_correct]))/batch_size
    nosuccess_action_prob = np.sum(fail_action)/batch_size


    return success_action_prob,  nosuccess_action_prob

class Trainer(object):
    def __init__(self, hp=None, load_model_dir_A=None,load_model_dir_B=None, model_dir=None, is_cuda=True):
        tools.mkdir_p(model_dir)

        self.model_dir = model_dir
        self.is_cuda = is_cuda

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load or create hyper-parameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # hyper-parameters for time scale
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        self.hp = hp

        fh_fname = os.path.join(model_dir, 'hp.json')
        if not os.path.isfile(fh_fname):
            tools.save_hp(hp, model_dir)

        if hp['switch_context']=='con_A':
            self.model = network.Network(hp, is_cuda)

        elif hp['switch_context']=='con_A2B':
            print('switch_context: con_A2B')
            self.model = network.Network(hp,  is_cuda)
            self.model.load(load_model_dir_A)

            print('weight: ',self.model.RNN_layer.h2h.weight[0:5,0])
            A_hh_weight = self.model.RNN_layer.h2h.weight.detach().numpy()
            np.save(model_dir+'/model_A_hh.npy',A_hh_weight)
            print('A_hh_weight',model_dir)
            ## null model
            #self.model = network.Network( hp, is_cuda)

        elif hp['switch_context']=='con_B2A':
            self.model = network.Network( hp, is_cuda)
            self.model.load(load_model_dir_A)

            print('A weight: ',self.model.RNN_layer.h2h.weight[0:5,0])
            A_hh_weight = self.model.RNN_layer.h2h.weight.detach().numpy()
            np.save(model_dir+'/model_A_hh.npy',A_hh_weight)

            self.model.load(load_model_dir_B)
            print('B weight: ',self.model.RNN_layer.h2h.weight[0:5,0])
            B_hh_weight = self.model.RNN_layer.h2h.weight.detach().numpy()
            np.save(model_dir+'/model_B_hh.npy',B_hh_weight)
            print('B_hh_weight',model_dir)
            ## null model
            #self.model = network.Network( hp, is_cuda)


        # load or create log
        self.log = tools.load_log(model_dir)
        self.log = defaultdict(list)
        self.log['model_dir'] = model_dir

        # trainner stepper
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda)

        self.min_cost = np.inf
        self.model_save_idx = 0

    def collate_fn(self,batch):
        return batch[0]

    def do_eval(self):
        '''Do evaluation, and then save the model
        '''
        #print('*',self.hp['rule_trains'])
        success_action_prob_two = []
        mean_choice_error_two = []

        for context_name in self.hp['rule_trains']:
            dataset_test = dataset.TaskDataset(context_name, self.hp, mode='test')
            dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=self.collate_fn)
            sample_batched = next(iter(dataloader_test))

            with torch.no_grad():
                self.train_stepper.cost_fcn(**sample_batched)

            clsq_tmp = self.train_stepper.cost_lsq.detach().cpu().numpy()
            creg_tmp = self.train_stepper.cost_reg.detach().cpu().numpy()

            if context_name=='RDM_task':
                target_choice =   1*((sample_batched['strength_cue']> sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0']))> np.abs(sample_batched['mean_vis_1'])) \
                                 +2*((sample_batched['strength_cue']> sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0'])<=np.abs(sample_batched['mean_vis_1']))) \
                                 +3*((sample_batched['strength_cue']<=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])> np.abs(sample_batched['mean_aud_1']))) \
                                 +4*((sample_batched['strength_cue']<=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])<=np.abs(sample_batched['mean_aud_1'])))

                output = self.train_stepper.outputs.detach().cpu().numpy()
                success_action_prob, mean_choice_error = get_perf_rule_dm_vector(output = output,target_choice = target_choice,
                                                                                         epochs = sample_batched['epochs'])
                print('||||||||||| '+context_name+'| cost {:0.5f}'.format(np.mean(clsq_tmp)) + '| c_reg {:0.5f}'.format(np.mean(creg_tmp))+
                       '| mean_error {:0.3f}'.format(mean_choice_error)+'| perf {:0.3f}'.format(success_action_prob))

            if context_name=='HL_task':
                target_choice_A = 1*((sample_batched['strength_cue']> sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0']))> np.abs(sample_batched['mean_vis_1'])) \
                                 +2*((sample_batched['strength_cue']> sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0'])<=np.abs(sample_batched['mean_vis_1']))) \
                                 +3*((sample_batched['strength_cue']<=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])> np.abs(sample_batched['mean_aud_1']))) \
                                 +4*((sample_batched['strength_cue']<=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])<=np.abs(sample_batched['mean_aud_1'])))

                target_choice_B =    1*((sample_batched['strength_cue'] < sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0']))> np.abs(sample_batched['mean_vis_1'])) \
                                    +2*((sample_batched['strength_cue']< sample_batched['batch_0'])*(np.abs(sample_batched['mean_vis_0'])<=np.abs(sample_batched['mean_vis_1']))) \
                                    +3*((sample_batched['strength_cue']>=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])> np.abs(sample_batched['mean_aud_1']))) \
                                    +4*((sample_batched['strength_cue']>=sample_batched['batch_0'])*(np.abs(sample_batched['mean_aud_0'])<=np.abs(sample_batched['mean_aud_1'])))

                if self.hp['switch_context']=='con_A':
                        target_choice = target_choice_A

                if self.hp['switch_context']=='con_A2B':
                    if not self.hp['start_switch']:
                        target_choice = target_choice_A
                    else:
                        target_choice = target_choice_B

                if self.hp['switch_context']=='con_B2A':
                    if not self.hp['start_switch']:
                        target_choice = target_choice_B
                    else:
                        target_choice = target_choice_A


                output = self.train_stepper.outputs.detach().cpu().numpy()
                success_action_prob, mean_choice_error = get_perf_rule_dm_vector(output = output,target_choice = target_choice,
                                                                                 epochs = sample_batched['epochs'])
                print('||||||||||| '+context_name+'| cost {:0.5f}'.format(np.mean(clsq_tmp)) + '| c_reg {:0.5f}'.format(np.mean(creg_tmp))+
                      '| mean_error {:0.3f}'.format(mean_choice_error)+'| perf {:0.3f}'.format(success_action_prob))
            success_action_prob_two.append(success_action_prob)
            mean_choice_error_two.append(mean_choice_error)


        if clsq_tmp < self.min_cost:
            self.min_cost = clsq_tmp
            self.model.save(self.model_dir)
        print('save model:',self.model_dir,'/',str(self.model_save_idx))

        # save model routinely
        routine_save_path = os.path.join(self.model_dir, str(self.model_save_idx))


        self.log['cost_'].append(np.mean(clsq_tmp, dtype=np.float64))
        self.log['creg_'].append(np.mean(creg_tmp, dtype=np.float64))

        self.info = dict()
        self.info['success_trial'] = self.log['success_trial']
        self.info['cost'] = clsq_tmp.tolist()
        self.info['creg'] = creg_tmp.tolist()
        self.info['success_action_prob'] = self.log['success_action_probs']
        self.info['mean_choice_error'] = self.log['mean_choice_errors']
        self.info['model_dir'] = self.model_dir
        self.info['time'] = self.log['times']
        self.info['cost_'] = self.log['cost_']
        self.info['creg_'] = self.log['creg_']

        # save model routinely

        if success_action_prob>0.85:# and mean_choice_error<0.2:
            self.model_save_idx = self.model_save_idx + 1
            tools.mkdir_p(routine_save_path)
            self.model.save(routine_save_path)
            self.log['save_performance'].append(success_action_prob)
        # if self.hp['switch_context'] == 'con_A2B' or self.hp['switch_context']=='con_B2A':
        #     if success_action_prob < 0.85:
        #         self.model_save_idx = self.model_save_idx + 1
        #         tools.mkdir_p(routine_save_path)
        #         self.model.save(routine_save_path)
        #         np.save(routine_save_path + '/' + str(round(success_action_prob, 2)) + '.npy', success_action_prob)

        #print('self.info',self.info)
        tools.save_log(self.info)

        return clsq_tmp, success_action_prob, mean_choice_error


    def save_final_result(self):
        save_path = os.path.join(self.model_dir, 'finalResult')
        tools.mkdir_p(save_path)
        self.model.save(save_path)
        self.info['model_dir'] = save_path
        tools.save_log(self.info)

    def train(self, max_samples=20000, display_step=200,max_model_save_idx=10):
        """Train the network.
        """
        dataset_train_RDM = dataset.TaskDataset(context_name='RDM_task', hp=self.hp, mode='train')
        dataset_train_HL = dataset.TaskDataset(context_name='HL_task', hp=self.hp, mode='train')
        # Display hp
        for key, val in self.hp.items():
            print('{:20s} = '.format(key) + str(val))

        # Record time
        t_start = time.time()
        success_action_prob_list = []

        for step in range(max_samples):
            rule_train_now = self.hp['rng'].choice(self.hp['rule_trains'], p=self.hp['rule_probs'])
            if rule_train_now=='RDM_task':
                sample_batched = next(iter(dataset_train_RDM))
            elif rule_train_now=='HL_task':
                sample_batched = next(iter(dataset_train_HL))


            # set l2 regularization on firing rate at the first few steps, to discourage the divergence of cost function at the very beginning of training
            if self.model_save_idx < 5:
                self.train_stepper.l2_firing_rate_cpu = torch.tensor(1e-3, device=torch.device("cpu"))
                self.train_stepper.l2_firing_rate = torch.tensor(1e-3, device=self.device)
            else:
                self.train_stepper.l2_firing_rate_cpu = torch.tensor(self.hp['l2_firing_rate'], device=torch.device("cpu"))
                self.train_stepper.l2_firing_rate = torch.tensor(self.hp['l2_firing_rate'], device=self.device)


            if self.hp['switch_context']=='con_A':
                max_model_save_idx=30
                display_step = 100
                self.train_stepper.stepper(**sample_batched)


            if self.hp['switch_context']=='con_A2B':
                max_model_save_idx=20
                display_step = 50


                # if step>201:
                #     self.hp['start_switch']=True
                #     self.train_stepper.stepper(**sample_batched)

                self.hp['start_switch']=True
                self.train_stepper.stepper(**sample_batched)
            elif self.hp['switch_context']=='con_B2A':
                max_model_save_idx=20
                display_step = 50
                # if step>3:
                #     self.hp['start_switch']=True
                #     self.train_stepper.stepper(**sample_batched)
                self.hp['start_switch']=True
                self.train_stepper.stepper(**sample_batched)





            if (step) % display_step == 0:

                cost, success_action_prob, mean_choice_error = self.do_eval()
                self.log['times'].append(time.time() - t_start)
                elapsed_time = tools.elapsed_time(self.log['times'][-1])

                print('** {:2d}'.format(step) +'==========================='+'  | Time ',elapsed_time)
                print('learning_rate',self.hp['learning_rate'])

                if success_action_prob>0.8 and step>600:
                    self.hp['learning_rate']=0.00001
                if success_action_prob > 0.85:
                    self.log['success_trial'].append(step)



                success_action_prob_list.append(success_action_prob)
                self.log['success_action_probs'].append(success_action_prob)


                if not np.isfinite(cost):
                    return 'error'

                # if success_action_prob > 0.95 and mean_choice_error<0.2:
                #     self.save_final_result()
                    #break
                if self.model_save_idx > max_model_save_idx:
                    break

            if step > max_samples:
                self.log['trials'].append(step)
                self.log['times'].append(time.time() - t_start)
                self.do_eval()
                break


        print("Optimization finished!")

        return 'OK'



