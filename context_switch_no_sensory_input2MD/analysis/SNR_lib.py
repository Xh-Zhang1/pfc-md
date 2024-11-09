"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""


from __future__ import division

import os,sys
import seaborn as sns
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import scipy
from matplotlib import pyplot as plt
# import run
import tools
import torch
import pdb

def get_epoch(hp):
    # print('************ perform HL_task')
    dt = hp['dt']
    batch_size = 1
    rng = hp['rng']

    # cue stim
    cue_time = int(hp['cue_duration'])
    cue_delay_time = int(hp['cue_delay'])
    cue_on = (rng.uniform(40, 40, batch_size) / dt).astype(int)
    cue_duration = int(cue_time / dt)  # 60
    cue_off = cue_on + cue_duration
    cue_delay = (rng.uniform(cue_delay_time, cue_delay_time, batch_size) / dt).astype(int)

    # stim epoch
    stim_time = int(hp['stim'])
    stim1_on = cue_off + cue_delay
    stim1_during = (rng.uniform(stim_time, stim_time, batch_size) / dt).astype(int)
    stim1_off = stim1_on + stim1_during
    response_on = stim1_off

    # response end time
    response_duration = int(hp['response_time'] / dt)
    response_off = response_on + response_duration


    cue_on = cue_on[0];cue_off=cue_off[0];stim_on=stim1_on[0];stim_off=stim1_off[0];response_on=response_on[0];response_off=response_off[0]
    print('cue_on,cue_off,stim_on,stim_off,response_on',cue_on,cue_off,stim_on,stim_off,response_on)

    epoch = {'cue_on':cue_on,
             'cue_off':cue_off,
             'stim_on':stim_on,
             'stim_off':stim_off,
             'response_on':response_on,
             'response_off':response_off}

    return epoch

def plot_HL_perf(model_dir,hp,batch_size,p_cohs):

    true_actions = []
    p_attend_both_corrs = []

    c_cue = hp['rng'].choice([-1,1], (batch_size,))
    p_cohs =  p_cohs


    for p_coh in p_cohs:
        #print('===================== p_coh',p_coh)
        runnerObj = run.Runner(rule_name='HL_task', hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test1')
        trial_input, run_result = runnerObj.run(batch_size=batch_size, c_cue=c_cue, p_coh=p_coh)

        response_on = trial_input.epochs['response'][0][0]
        response_off = trial_input.epochs['response'][1][0]

        choice_1 = 0;choice_2 = 0;choice_3 = 0;choice_4 = 0

        actual_choices = []
        target_choices = []

        correct_attend_vis=0
        correct_attend_aud=0

        for i in range(batch_size):#
            output = run_result.outputs[response_on:response_off, i, :].numpy()
            mean = np.mean(output,axis=0)
            actual_choice = np.argmax(mean)
            target_choice = trial_input.target_choice[i]
            target_choices.append(target_choice)
            actual_choices.append(actual_choice+1)

            if actual_choice+1 == 1:
                if c_cue[i] == 1*trial_input.context_HL:
                    correct_attend_vis+=1
                if actual_choice+1 == target_choice:
                    choice_1 += 1

            if actual_choice+1 == 2:
                if c_cue[i] == 1*trial_input.context_HL:
                    correct_attend_vis+=1
                if actual_choice+1 == target_choice:
                    choice_2 += 1

            if actual_choice+1 == 3:
                if c_cue[i] == -1*trial_input.context_HL:
                    correct_attend_aud+=1
                if actual_choice+1 == target_choice:
                    choice_3 += 1

            if actual_choice+1 == 4:
                if c_cue[i] == -1*trial_input.context_HL:
                    correct_attend_aud+=1

                if actual_choice+1 == target_choice:
                    choice_4 += 1

        p_attend_both_corr = (correct_attend_vis+correct_attend_aud)/batch_size
        p_attend_both_corrs.append(p_attend_both_corr)

        true_action=(choice_1+choice_2+choice_3+choice_4)/batch_size
        true_actions.append(true_action)

    print('HLrule perf:',p_attend_both_corrs)
    print('choice perf:',true_actions)
    return p_attend_both_corrs, true_actions



def generate_one_trial(model_dir,hp,context_name,cue,p_cohs,front_show,batch_size):
    batch_size = batch_size
    hp['front_show'] = front_show

    if cue is None:
        c_cue=hp['rng'].choice([1,-1], (batch_size,))
    elif type(cue)==int:
        c_cue = hp['rng'].choice([cue], (batch_size,))
    else:
        #print('*** cue')
        c_cue = cue

    runnerObj = run.Runner(rule_name=context_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True,mode='test1')
    trial_input, run_result = runnerObj.run(batch_size=batch_size, c_cue=c_cue,p_coh=p_cohs)


    #### average value over batch_sizes for hidden state
    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list = list([firing_rate[:, i, :] for i in range(batch_size)])
    firing_rate_mean = np.mean(np.array(firing_rate_list), axis=0)
    firing_rate = firing_rate_mean

    #### MD
    firing_rate_md = run_result.firing_rate_md.detach().cpu().numpy()

    fr_MD_list = list([firing_rate_md[:, i, :] for i in range(batch_size)])
    fr_MD_mean = np.mean(np.array(fr_MD_list), axis=0)
    fr_MD = fr_MD_mean



    return firing_rate_mean,fr_MD


def generate_test_trial(context_name,hp,model_dir,

                        cue=None,
                        p_coh=0.9,
                        batch_size=1,
                        c_vis=None,
                        c_aud=None):

    rng  = hp['rng']

    if cue is None:
        c_cue = hp['rng'].choice([1,-1], (batch_size,))
    else:
        c_cue = hp['rng'].choice([cue], (batch_size,))
    if c_vis is None:
        c_vis = hp['gamma_noise']*hp['rng'].choice([-0.2,0.2], (batch_size,))
    else:
        c_vis = hp['rng'].choice([c_vis], (batch_size,))

    if c_aud is None:
        c_aud = hp['gamma_noise']*hp['rng'].choice([-0.2,0.2], (batch_size,))
    else:
        c_aud = hp['rng'].choice([c_aud], (batch_size,))

    runnerObj = run.Runner(rule_name=context_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            p_coh=p_coh,
                                            c_cue=c_cue,
                                            c_vis=c_vis,
                                            c_aud=c_aud)


    return trial_input, run_result

def get_neurons_activity_mode_test1(context_name,hp,model_dir,cue,p_cohs,batch_size):
    runnerObj = run.Runner(rule_name=context_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    trial_input, run_result = runnerObj.run(batch_size=batch_size, c_cue=cue, p_coh=p_cohs)


    #### average value over batch_sizes for hidden state
    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list = list([firing_rate[:,i,:] for i in range(batch_size)])
    firing_rate_mean = np.mean(np.array(firing_rate_list),axis=0)
    firing_rate = firing_rate_mean
    #### MD
    firing_rate_md = run_result.firing_rate_md.detach().cpu().numpy()
    fr_MD_list = list([firing_rate_md[:,i,:] for i in range(batch_size)])
    fr_MD_mean = np.mean(np.array(fr_MD_list),axis=0)
    fr_MD = fr_MD_mean

    return firing_rate_mean,fr_MD



def scatters_MD_different_context(figure_path,hp,epoch,idx,model_name,period,model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92
    if period == 'delay':
        start = epoch['cue_off']
        end = epoch['response_on']
    elif period == 'cue':
        start = epoch['cue_on']
        end = epoch['cue_off']
    elif period == 'all':
        start = epoch['cue_on']
        end = epoch['response_on']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    fr_rnn_vis_A,fr_md_vis_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A,fr_md_aud_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)

    fr_rnn_vis_A2B,fr_md_vis_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                         context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A2B,fr_md_aud_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                         context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)


    fr_rnn_vis_B2A,fr_md_vis_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                         context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_B2A,fr_md_aud_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                         context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)

    max_idx_md_50 = np.argsort(fr_md_vis_A )
    # max_idx_exc = np.argsort(fr_rnn_vis[50:205])
    print(max_idx_md_50)
    # print(max_idx_exc+50)

    c_alpha = 0.6;s=24
    idx_md =30

    fig, axs = plt.subplots(1, 2,figsize=(5.5,2.7))
    plt.subplots_adjust(top=0.85, bottom=0.18, right=0.95, left=0.1, hspace=0.3, wspace=0.4)
    fig.suptitle('md/ '+model_name+'_'+str(idx)+'; '+period,fontsize=8)
    print('fr_md_aud_A',fr_md_aud_A.shape)
    min = np.min(fr_md_aud_A) - 0.2
    max = 0.5 + np.max(fr_md_aud_A)


    print('max',max)

    axs[0].scatter(fr_md_vis_A[0:18],fr_md_vis_A2B[0:18],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    axs[1].scatter(fr_md_vis_B2A[0:18],fr_md_vis_A2B[0:18],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')

    axs[0].scatter(fr_md_vis_A[18:30],fr_md_vis_A2B[18:30],marker="o",s=s,color='tab:orange',alpha=c_alpha,edgecolors='none')
    axs[1].scatter(fr_md_vis_B2A[18:30],fr_md_vis_A2B[18:30],marker="o",s=s,color='tab:orange',alpha=c_alpha,edgecolors='none')

    # axs[0].scatter(fr_md_aud_A[0:idx_md],fr_md_aud_A2B[0:idx_md],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    # axs[1].scatter(fr_md_aud_B2A[0:idx_md],fr_md_aud_A2B[0:idx_md],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')

    for i in range(2):
        axs[0].set_xlabel('con_A',fontsize=10)
        axs[0].set_ylabel('con_B',fontsize=10)
        axs[1].set_xlabel('con_AA',fontsize=10)
        axs[1].set_ylabel('con_B',fontsize=10)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].plot([min,max*0.9],[min,max*0.9],color='grey')
        axs[0].set_xlim([min,max])
        axs[0].set_ylim([min,max])
        axs[1].set_xlim([min,max])
        axs[1].set_ylim([min,max])


    cell_type = [21, 17, 10,  7, 16, 12]#168
    colors_1 = ['orange','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        axs[0].scatter(fr_md_aud_A[i],  fr_md_aud_A2B[i],marker="x",s=26,c=colors_1[j],label=str(i))
        axs[1].scatter(fr_md_aud_B2A[i],fr_md_aud_A2B[i],marker="x",s=26,c=colors_1[j],label=str(i))


    plt.legend(fontsize=3)
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_'+period+'.png')
    plt.show()

def scatters_sns_MD_different_context(figure_path,data_path,hp,epoch,idx,model_name,period,model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92
    if period == 'delay':
        start = epoch['cue_off']
        end = epoch['response_on']

    elif period == 'cue':
        start = epoch['cue_on']
        end = epoch['cue_off']
    elif period == 'all':
        start = epoch['cue_on']
        end = epoch['response_on']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    #
    # fr_rnn_vis_A,fr_md_vis_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
    #                                                  context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)
    # fr_rnn_aud_A,fr_md_aud_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
    #                                                  context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)
    #
    # fr_rnn_vis_A2B,fr_md_vis_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
    #                                                      context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)
    # fr_rnn_aud_A2B,fr_md_aud_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
    #                                                      context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)
    #
    #
    # fr_rnn_vis_B2A,fr_md_vis_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
    #                                                      context_name='HL_task',cue=1, p_cohs=0.92,batch_size=batch_size)
    # fr_rnn_aud_B2A,fr_md_aud_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
    #                                                      context_name='HL_task',cue=-1, p_cohs=0.92,batch_size=batch_size)
    #
    #
    # np.save(data_path+'fr_md_vis_A.npy',fr_md_vis_A)
    # np.save(data_path + 'fr_md_aud_A.npy', fr_md_aud_A)
    # np.save(data_path + 'fr_md_vis_A2B.npy', fr_md_vis_A2B)
    # np.save(data_path + 'fr_md_aud_A2B.npy', fr_md_aud_A2B)
    # np.save(data_path + 'fr_md_vis_B2A.npy', fr_md_vis_B2A)
    # np.save(data_path + 'fr_md_aud_B2A.npy', fr_md_aud_B2A)

    fr_md_vis_A = np.load(data_path+'fr_md_vis_A.npy')
    fr_md_aud_A = np.load(data_path+'fr_md_aud_A.npy')
    fr_md_vis_A2B = np.load(data_path + 'fr_md_vis_A2B.npy')
    fr_md_aud_A2B = np.load(data_path + 'fr_md_aud_A2B.npy')
    fr_md_vis_B2A = np.load(data_path + 'fr_md_vis_B2A.npy')
    fr_md_aud_B2A = np.load(data_path + 'fr_md_aud_B2A.npy')

    print(np.argsort(fr_md_vis_A))
    min = 0.56
    max = 1.12  # np.max(fr_rnn_vis_A)

    print('max', max)

    cell_type = [10, 16, 24,12,7,8]#[10, 16, 24,12,1,2]np.argsort(fr_md_vis_A)[24:30]#[1,2,10,11,16,26]
    print('cell_type',cell_type)
    s = 50
    idx_md = 30
    #colors_1 = sns.color_palette("Set2")
    #colors_1 = ['#9B30FF', '#FF34B3', '#7CFC00', 'green', '#FF4500', '#FF6A6A']
    colors_1 = ['r', 'orange', 'y', 'green', 'b', 'purple']
    # fig2
    for context in np.array(['switch1', 'switch2']):
        fig1 = plt.figure(figsize=(2.7, 2.7))
        ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
        if context == 'switch1':
            plt.scatter(fr_md_vis_A[0:idx_md],fr_md_vis_A2B[0:idx_md], marker="o", s=s, color='#4682B4',edgecolors='white')
            ax1.set_xlabel('con_A', fontsize=12)
            ax1.set_ylabel('con_B', fontsize=12)
        if context == 'switch2':
            plt.scatter(fr_md_vis_B2A[0:idx_md], fr_md_vis_A2B[0:idx_md], marker="o", s=s, color='#4682B4',edgecolors='white')
            ax1.set_xlabel('con_AA', fontsize=12)
            ax1.set_ylabel('con_B', fontsize=12)
        ax1.set_xlim([min, max])
        ax1.set_ylim([min, max])
        plt.xticks([0.6,0.8,1.0], fontsize=12)
        plt.yticks([0.6,0.8,1.0], fontsize=12)

        j = -1
        for i in np.array(cell_type):
            j += 1
            if context == 'switch1':
                plt.scatter(fr_md_vis_A[i],  fr_md_vis_A2B[i], marker="X", s=40, c=colors_1[j], label=str(i))
                # ax1.set_xlabel('con_A', fontsize=12)
                # ax1.set_ylabel('con_B', fontsize=12)
                #plt.legend()
            if context == 'switch2':
                plt.scatter(fr_md_vis_B2A[i],fr_md_vis_A2B[i], marker="X", s=40, c=colors_1[j], label=str(i))
                # ax1.set_xlabel('con_AA', fontsize=12)
                # ax1.set_ylabel('con_B', fontsize=12)

        plt.plot([0, max], [0, max ], color='silver', )

        # set x-label
        ax1.spines[['right', 'top']].set_visible(False)
        #plt.title( model_name + '_' + str(idx)+':'+period,fontsize=6)
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.png')
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.pdf')
        plt.show()







