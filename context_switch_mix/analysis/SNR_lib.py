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

import run
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


def plot_activity_all_4panel_A(figure_path,model_dir_A,model_dir_pre,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=1
    aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,20,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 2#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=45
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis (E)',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud (E)',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis (E)',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud (E)',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()


    ######## MD1 #####################
    fig2, axs2 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig2.suptitle('MD  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_md1):
        axs2[0,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[0,0].set_ylim([0,y_md1])
        axs2[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[0,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[0,1].set_ylim([0,y_md1])
        axs2[0,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[1,0].set_ylim([0,y_md2])
        axs2[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[1,1].set_ylim([0,y_md2])
        axs2[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs2[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs2[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_md'+'.png')
    plt.show()


def plot_activity_all_4panel_A2B(figure_path,model_dir_A,model_dir_pre,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=1
    aud=-1
    if idx>2:
        vis = -1
        aud = 1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,20,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=45
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()


    ######## MD1 #####################
    fig2, axs2 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig2.suptitle('MD  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_md1):
        axs2[0,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[0,0].set_ylim([0,y_md1])
        axs2[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[0,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[0,1].set_ylim([0,y_md1])
        axs2[0,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[1,0].set_ylim([0,y_md2])
        axs2[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[1,1].set_ylim([0,y_md2])
        axs2[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs2[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs2[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_md'+'.png')
    plt.show()


def plot_activity_all_4panel(figure_path,model_dir_A,model_dir_pre,model_name,model_dir,idx,hp,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    vis=1
    aud=-1
    if idx>2:
        vis = -1
        aud = 1



    pfc_mean_RDM_vis,MD_mean_RDM_vis = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='RDM_task',hp=hp,model_dir=model_dir,cue=1,p_cohs=coh_RDM,batch_size=batch_size)
    pfc_mean_RDM_aud,MD_mean_RDM_aud = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='RDM_task',hp=hp,model_dir=model_dir,cue=-1,p_cohs=coh_RDM,batch_size=batch_size)

    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,20,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,20,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_RDM_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_RDM_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_RDM_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.*np.max(MD_mean_RDM_vis[cue_on:cue_off,n_md1:n_md])



    font=8
    start=45
    end = response_on-2
    ########## exc ###################
    '''

    fig, axs = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('Exc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_RDM_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        #axs[0,0].set_title('RDM: '+'vis (E)',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_RDM_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('RDM: '+'aud (E)',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_exc])
        axs[1,0].set_title('HL: '+'vis (E)',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_exc])
        axs[1,1].set_title('HL: '+'aud (E)',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()


    ######### inh ####################
    fig1, axs1 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)

    fig1.suptitle('Inh  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_inh):
        axs1[0,0].plot(pfc_mean_RDM_vis[cue_on+start:end,i],label=str(i))
        axs1[0,0].set_ylim([0,y_inh])
        axs1[0,0].set_title('RDM: '+'vis',fontsize=font)
        #axs1[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs1[0,1].plot(pfc_mean_RDM_aud[cue_on+start:end,i],label=str(i))
        axs1[0,1].set_ylim([0,y_inh])
        axs1[0,1].set_title('RDM: '+'aud',fontsize=font)
        #axs1[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs1[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs1[1,0].set_ylim([0,y_inh])
        axs1[1,0].set_title('HL: '+'vis',fontsize=font)
        #axs1[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs1[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs1[1,1].set_ylim([0,y_inh])
        axs1[1,1].set_title('HL: '+'aud',fontsize=font)
        #axs1[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs1[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs1[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    #plt.ylim([0,10])
    #plt.xlim([0,10])
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_inh'+'.png')
    plt.show()
    '''


    ######## MD1 #####################
    fig2, axs2 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig2.suptitle('MD1  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_md1):
        axs2[0,0].plot(MD_mean_RDM_vis[cue_on+start:end,i],label=str(i))
        #axs2[0,0].set_ylim([0,y_md1])
        axs2[0,0].set_title('RDM: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[0,1].plot(MD_mean_RDM_aud[cue_on+start:end,i],label=str(i))
        #axs2[0,1].set_ylim([0,y_md1])
        axs2[0,1].set_title('RDM: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs2[1,0].set_ylim([0,y_md1])
        axs2[1,0].set_title('HL: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs2[1,1].set_ylim([0,y_md1])
        axs2[1,1].set_title('HL: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs2[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs2[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_md1'+'.png')
    plt.show()


    ######## MD2 #####################
    fig3, axs3 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig3.suptitle('MD2 :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))

    for i in np.array(cell_idx_md2):
        axs3[0,0].plot(MD_mean_RDM_vis[cue_on+start:end,i],label=str(i))
        #axs3[0,0].set_ylim([0,y_md2])
        axs3[0,0].set_title('RDM: '+'vis',fontsize=font)
        #axs3[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs3[0,1].plot(MD_mean_RDM_aud[cue_on+start:end,i],label=str(i))
        #axs3[0,1].set_ylim([0,y_md2])
        axs3[0,1].set_title('RDM: '+'aud',fontsize=font)
        #axs3[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs3[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs3[1,0].set_ylim([0,y_md2])
        axs3[1,0].set_title('HL: '+'vis',fontsize=font)
        #axs3[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs3[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs3[1,1].set_ylim([0,y_md2])
        axs3[1,1].set_title('HL: '+'aud',fontsize=font)
        #axs3[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs3[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs3[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    #plt.savefig(figure_path+model_name+'_'+str(idx)+'_md2'+'.png')
    plt.show()
    #'''

def plot_activity_all_4panel_B2A(figure_path,model_dir_A,model_dir_pre,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=-1
    aud=1
    if idx>3:
        vis = 1
        aud = -1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(model_dir_A=model_dir_A,model_dir_pre=model_dir_pre,context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,20,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=45
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()


    ######## MD1 #####################
    fig2, axs2 = plt.subplots(2, 2,figsize=(6.5,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig2.suptitle('MD  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_md1):
        axs2[0,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[0,0].set_ylim([0,y_md1])
        axs2[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[0,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[0,1].set_ylim([0,y_md1])
        axs2[0,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs2[1,0].set_ylim([0,y_md2])
        axs2[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs2[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs2[1,1].set_ylim([0,y_md2])
        axs2[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs2[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs2[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    plt.savefig(figure_path+model_name+'_'+str(idx)+'_md'+'.png')
    plt.show()


def plot_activity_all_8panel_A(figure_path,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=1
    aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
                                    context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
                                    context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)


    cell_idx_exc = range(0,10,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+5,1)#max_idx_md2+n_md1
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 2#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))

        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))

        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))

        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))

        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))

        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))

        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))

        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)
    # axs[0, 0].set_ylim([0, y_exc])
    # axs[0, 1].set_ylim([0, y_exc])
    # axs[1, 0].set_ylim([0, y_inh])
    # axs[1, 1].set_ylim([0, y_inh])
    # axs[2, 0].set_ylim([0, y_md1])
    # axs[2, 1].set_ylim([0, y_md1])
    # axs[3, 0].set_ylim([0, y_md2])
    # axs[3, 1].set_ylim([0, y_md2])
    #



    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()

def plot_activity_all_8panel_A2B(figure_path,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=1
    aud=-1
    if idx>2:
        vis = -1
        aud = 1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,10,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+10,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+5,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:response_on,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:response_on,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:response_on,0:n_md1])
    y_md2 = 1.*np.max(MD_mean_HL_vis[cue_on:response_on,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)


    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[2,0].set_ylim([0,y_md1])
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[2,1].set_ylim([0,y_md1])
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[3,0].set_ylim([0,y_md2])
        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[3,1].set_ylim([0,y_md2])
        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()

def plot_activity_all_8panel_B2A(figure_path,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=-1
    aud=1
    if idx>2:
        vis = 1
        aud = -1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)



    cell_idx_exc = range(0,10,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+10,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+5,1)#max_idx_md2+n_md1
    #print(cell_idx_exc,cell_idx_inh,cell_idx_md1,cell_idx_md2)
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:response_on,0:n_exc])
    y_inh = 1.*np.max(pfc_mean_HL_aud[cue_on:response_on,n_exc:n_rnn])
    y_md1 = 1.*np.max(MD_mean_HL_vis[cue_on:response_on,0:n_md1])
    y_md2 = 1.*np.max(MD_mean_HL_vis[cue_on:response_on,n_md1:n_md])



    font=10
    start=2
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)


    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[2,0].set_ylim([0,y_md1])
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[2,1].set_ylim([0,y_md1])
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[3,0].set_ylim([0,y_md2])
        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[3,1].set_ylim([0,y_md2])
        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()




def plot_activity_diff_exc_A(figure_path,model_name,model_dir,idx,hp,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    vis=1
    aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)


    cell_idx_exc_readout = range(0,149,1)#max_idx_exc[0:20]
    cell_idx_exc = range(150,200,1)#max_idx_exc[0:20]


    y_exc_readout = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6,6))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc_readout):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[0,0].set_ylim([0,y_exc_readout])
        axs[0,0].set_title('exc_readout: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc_readout):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[0,1].set_ylim([0,y_exc_readout])
        axs[0,1].set_title('exc_readout: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[1,0].set_ylim([0,y_exc])
        axs[1,0].set_title('exc: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[1,1].set_ylim([0,y_exc])
        axs[1,1].set_title('exc: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)



    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()


def plot_activity_diff_exc(figure_path,model_name,model_dir,idx,hp,epoch,context):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    if context=='con_A':
        vis = 1
        aud = -1
    if context=='con_A2B':
        vis = -1
        aud = 1
    if context=='con_B2A':
        vis = 1
        aud = -1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=aud,p_cohs=coh_HL,batch_size=batch_size)
    print(pfc_mean_HL_vis.shape)

    max_idx_exc_readout = np.argsort(np.mean(pfc_mean_HL_vis[:,0:150],axis=0))
    cell_idx_exc_readout = range(0,7,1)#max_idx_exc_readout[-10:]#max_idx_exc[0:20]
    cell_idx_exc = range(150,157,1)#max_idx_exc[0:20]


    y_exc_readout = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_exc = 1.*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6,6))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc_readout):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[0,0].set_ylim([0,y_exc_readout])
        axs[0,0].set_title('exc_readout: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc_readout):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[0,1].set_ylim([0,y_exc_readout])
        axs[0,1].set_title('exc_readout: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[1,0].set_ylim([0,y_exc])
        axs[1,0].set_title('exc: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[1,1].set_ylim([0,y_exc])
        axs[1,1].set_title('exc: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)



    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'_exc'+'.png')
    plt.show()




def get_weight_A(model_name,hp,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md


    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach().numpy()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)

    effective_weights = np.abs(weight) * mask

    print('effective_weights',effective_weights[0:5,0])

    plt.imshow(effective_weights[0:,0:])
    plt.title(model_name)
    plt.show()

def get_weight_A2B(model_name,hp,idx,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md

    H,C = tools.weight_mask_A(hp, is_cuda=False)

    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)

    C = np.abs(C) * mask
    base_weight = torch.abs(weight) * mask
    effective_weights = base_weight * H + C


    print('effective_weights',effective_weights[0:5,0])

    plt.imshow(effective_weights[0:,0:])#[256:286,0:100]
    plt.title(model_name)
    plt.show()

def get_weight_B2A(model_name,hp,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md

    weight_A = tools.get_weight_A(hp, is_cuda=False)

    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)
    term_weight = weight
    term_weight = term_weight.detach().numpy()

    term_weight[0:256,0:256] = weight_A[0:256,0:256]


    print('C',weight_A[0:5,0])
    print('**** lmodel_name',model_name)

    effective_weights = np.abs(term_weight) * mask

    print('effective_weights',effective_weights[0:5,0])

    plt.imshow(effective_weights[256:286,0:100])
    plt.title(model_name)
    plt.show()


def activity_diff_context(model_dir,hp,start,end,context_name,cue,p_cohs,batch_size):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''

    batch_size=batch_size

    if cue is None:
        c_cue=hp['rng'].choice([1,-1], (batch_size,))
    else:
        c_cue=hp['rng'].choice([cue], (batch_size,))

    runnerObj = run.Runner(rule_name=context_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True, mode='test1')
    trial_input, run_result = runnerObj.run(batch_size=batch_size, c_cue=c_cue, c_vis=0.2,c_aud=0.2,p_coh=p_cohs)

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

    #max_idx = np.argsort(np.mean(fr_MD_mean,axis=0))

    # for i in np.array(max_idx[-10:]):
    #     plt.plot(fr_MD[start:end,i],label=str(i))
    # plt.ylim([1.8,3.45])
    # plt.legend()
    #
    # plt.title(context_name)
    # plt.show()
    #print('fr_MD',fr_MD)

    #average value over times
    firing_rate_cue_stim = np.mean(firing_rate[start:end,:],axis=0)
    fr_MD_cue_stim = np.mean(fr_MD[start:end,:],axis=0)
    #print(fr_MD_cue_stim[:20])

    return firing_rate_cue_stim,fr_MD_cue_stim

def plot_scatters_different_context_A2B(figure_path,start,end,model_dir_A,model_dir_A2B,hp,epoch,cell_type):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    attend_vis=1
    attend_aud=-1

    fr_rnn_A,  fr_md_A   = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                            context_name='HL_task',cue=attend_vis, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_A2B,fr_md_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                            context_name='HL_task',cue=-attend_vis, p_cohs=0.92,batch_size=batch_size)

    max_idx_exc = np.argsort(fr_rnn_A[0:204])
    print(max_idx_exc)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    c_alpha = 0.3
    s=24

    plt.scatter(fr_rnn_A[0:205],fr_rnn_A2B[0:205],marker=">",s=s,color='tab:red',alpha=c_alpha,edgecolors='none',label='Exc')
    plt.scatter(fr_rnn_A[205:256],fr_rnn_A2B[205:256],s=s,color='tab:blue',alpha=c_alpha,edgecolors='none',label='Inh')
    plt.scatter(fr_md_A[0:n_md1],fr_md_A2B[0:n_md1],s=s,marker="*",color='tab:orange',alpha=c_alpha,edgecolors='none',label='MD1')
    plt.scatter(fr_md_A[n_md1:n_md],fr_md_A2B[n_md1:n_md],s=s,marker="*",color='tab:purple',alpha=c_alpha,edgecolors='none',label='MD2')


    colors_1 = ['red','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        plt.scatter(fr_rnn_A[i],fr_rnn_A2B[i],marker="x",s=26,c=colors_1[j],label='Inh')
    max =2.5#np.max(fr_rnn_1)*( np.max(fr_rnn_1)>np.max(fr_rnn_2))+np.max(fr_rnn_2)*( np.max(fr_rnn_2)>np.max(fr_rnn_1))

    plt.plot([0,max*0.9],[0,max*0.9],color='grey')
    # plt.xlim([0,max])
    # plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    # plt.yticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    plt.xlabel('A context',fontsize=12)
    plt.ylabel('B context',fontsize=12)
    plt.title(str(cell_type)+':'+str(start)+'_'+str(end),fontsize=8)
    #plt.legend(fontsize=10)
    # fig.savefig(figure_path+rule+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)
    #
    # plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()


def plot_scatters_different_context_B2A(figure_path,start,end,model_dir_A,model_dir_A2B,hp,epoch,cell_type):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    attend_vis=1
    attend_aud=-1

    fr_rnn_pre,  fr_md_pre   = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                 context_name='HL_task',cue=attend_vis, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_post,fr_md_post = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                 context_name='HL_task',cue=-attend_vis, p_cohs=0.92,batch_size=batch_size)



    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    c_alpha = 0.3
    s=24

    plt.scatter(fr_rnn_post[0:205],fr_rnn_pre[0:205],marker=">",s=s,color='tab:red',alpha = c_alpha,edgecolors='none',label='Exc')
    plt.scatter(fr_rnn_post[205:256],fr_rnn_pre[205:256],s=s,color='tab:blue',alpha = c_alpha,edgecolors='none',label='Inh')
    plt.scatter(fr_md_post[0:n_md1],fr_md_pre[0:n_md1],s=s,marker="*",color='tab:orange',alpha = c_alpha,edgecolors='none',label='MD1')
    plt.scatter(fr_md_post[n_md1:n_md],fr_md_pre[n_md1:n_md],s=s,marker="*",color='tab:purple',alpha = c_alpha,edgecolors='none',label='MD2')


    colors_1 = ['red','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        plt.scatter(fr_rnn_post[i],fr_rnn_pre[i],marker="x",s=26,c=colors_1[j],label='Inh')

    max =2.5#np.max(fr_rnn_1)*( np.max(fr_rnn_1)>np.max(fr_rnn_2))+np.max(fr_rnn_2)*( np.max(fr_rnn_2)>np.max(fr_rnn_1))

    plt.plot([0,max*0.9],[0,max*0.9],color='grey')
    # plt.xlim([0,max])
    # plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    # plt.yticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    plt.xlabel('A_post context',fontsize=12)
    plt.ylabel('B_pre context',fontsize=12)
    plt.title(str(cell_type)+':'+str(start)+'_'+str(end),fontsize=8)
    #plt.legend(fontsize=10)
    # fig.savefig(figure_path+rule+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)
    #
    # plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()



def plot_scatters_different_rule(figure_path,start,end,hp,epoch,cell_type,model_dir,context_name):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92



    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    if context_name =='context_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='context_B':
        attend_vis=-1
        attend_aud=1
    if context_name =='context_AA':
        attend_vis=1
        attend_aud=-1


    fr_rnn_vis,fr_md_vis   = activity_diff_context(model_dir=model_dir,hp=hp,start=start,end=end,
                                                 context_name='HL_task',cue=attend_vis, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud,fr_md_aud = activity_diff_context(model_dir=model_dir,hp=hp,start=start,end=end,
                                                 context_name='HL_task',cue=attend_aud, p_cohs=0.92,batch_size=batch_size)

    max_idx_exc_50 = np.argsort(fr_rnn_vis[0:50])
    max_idx_exc = np.argsort(fr_rnn_vis[50:205])
    print(max_idx_exc_50)
    print(max_idx_exc+50)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    c_alpha = 0.6
    s=24

    plt.scatter(fr_rnn_vis[0:205],fr_rnn_aud[0:205],marker=">",s=s,color='tab:red',alpha=c_alpha,edgecolors='none',label='Exc')
    # plt.scatter(fr_rnn_vis[205:256],fr_rnn_aud[205:256],s=s,color='tab:blue',alpha=c_alpha,edgecolors='none',label='Inh')
    # plt.scatter(fr_md_vis[0:n_md1],fr_md_aud[0:n_md1],s=s,marker="*",color='tab:orange',alpha=c_alpha,edgecolors='none',label='MD1')
    # plt.scatter(fr_md_vis[n_md1:n_md],fr_md_aud[n_md1:n_md],s=s,marker="*",color='tab:purple',alpha=c_alpha,edgecolors='none',label='MD2')


    colors_1 = ['orange','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        plt.scatter(fr_rnn_vis[i],fr_rnn_aud[i],marker="x",s=26,c=colors_1[j],label='Inh')
    max =1.5#np.max(fr_rnn_1)*( np.max(fr_rnn_1)>np.max(fr_rnn_2))+np.max(fr_rnn_2)*( np.max(fr_rnn_2)>np.max(fr_rnn_1))

    plt.plot([0,max*0.9],[0,max*0.9],color='grey')
    # plt.xlim([0,max])
    # plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    # plt.yticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=12)
    plt.xlabel('attend to vis',fontsize=12)
    plt.ylabel('attend to aud',fontsize=12)
    plt.title(context_name+':'+str(start)+'_'+str(end),fontsize=8)
    #plt.legend(fontsize=10)
    # fig.savefig(figure_path+rule+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)
    #
    # plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()


def scatters_Exc_different_context(figure_path,start,end,hp,epoch,idx,model_name,model_dir_A,model_dir_A2B,model_dir_B2A):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    coh_RDM=0.92;coh_HL=0.92



    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    fr_rnn_vis_A,fr_md_vis_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                   context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A,fr_md_aud_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                   context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)

    fr_rnn_vis_A2B,fr_md_vis_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A2B,fr_md_aud_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)


    fr_rnn_vis_B2A,fr_md_vis_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_B2A,fr_md_aud_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)

    max_idx_exc_50 = np.argsort(fr_rnn_vis_A[0:150])
    print(max_idx_exc_50)
    # print(max_idx_exc+50)

    c_alpha = 0.6;s=24
    idx_exc =204

    fig, axs = plt.subplots(1, 3,figsize=(9,3))
    plt.subplots_adjust(top=0.88, bottom=0.15, right=0.95, left=0.08, hspace=0.3, wspace=0.3)
    fig.suptitle('Exc_readout  :'+model_name+'_'+str(idx))

    max =np.max(fr_rnn_vis_A)

    axs[0].scatter(fr_rnn_vis_A[0:idx_exc],fr_rnn_aud_A[0:idx_exc],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    axs[1].scatter(fr_rnn_vis_A2B[0:idx_exc],fr_rnn_aud_A2B[0:idx_exc],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    axs[2].scatter(fr_rnn_vis_B2A[0:idx_exc],fr_rnn_aud_B2A[0:idx_exc],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')


    axs[0].set_xlabel('vis',fontsize=12)
    axs[0].set_ylabel('aud',fontsize=12)

    for i in range(3):
        axs[0].set_xlabel('vis',fontsize=12)
        axs[0].set_ylabel('aud',fontsize=12)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].plot([0,max*0.9],[0,max*0.9],color='grey')
        axs[0].set_xlim([-0.05,max])
        axs[0].set_ylim([-0.05,max])

        axs[1].set_xlim([-0.05,max])
        axs[1].set_ylim([-0.05,max])

        axs[2].set_xlim([-0.05,max])
        axs[2].set_ylim([-0.05,max])

    cell_type = max_idx_exc_50[144:150]
    print(cell_type)
    colors_1 = ['orange','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        axs[0].scatter(fr_rnn_vis_A[i],fr_rnn_aud_A[i],marker="x",s=26,c=colors_1[j],label=str(i))
        axs[1].scatter(fr_rnn_vis_A2B[i],fr_rnn_aud_A2B[i],marker="x",s=26,c=colors_1[j],label=str(i))
        axs[2].scatter(fr_rnn_vis_B2A[i],fr_rnn_aud_B2A[i],marker="x",s=26,c=colors_1[j],label=str(i))

    cell_type = [0,1,2,3,4,5]
    colors_1 = ['orange','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        axs[0].scatter(fr_rnn_vis_A[i],fr_rnn_aud_A[i],marker="*",s=26,c=colors_1[j],label=str(i))
        axs[1].scatter(fr_rnn_vis_A2B[i],fr_rnn_aud_A2B[i],marker="*",s=26,c=colors_1[j],label=str(i))
        axs[2].scatter(fr_rnn_vis_B2A[i],fr_rnn_aud_B2A[i],marker="*",s=26,c=colors_1[j],label=str(i))




    plt.title(':'+str(start)+'_'+str(end),fontsize=5)
    plt.legend(fontsize=5)
    plt.savefig(figure_path+model_name+'_'+str(idx)+'.png')
    #
    # plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()


def scatters_Exc_sns_different_context(figure_path,start,end,hp,epoch,idx,model_name,period,model_dir_A,model_dir_A2B,model_dir_B2A):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
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
    p_md1=0.8
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=2


    fr_rnn_vis_A,fr_md_vis_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                   context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A,fr_md_aud_A = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
                                                   context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)

    fr_rnn_vis_A2B,fr_md_vis_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_A2B,fr_md_aud_A2B = activity_diff_context(model_dir=model_dir_A2B,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)


    fr_rnn_vis_B2A,fr_md_vis_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= 1, p_cohs=0.92,batch_size=batch_size)
    fr_rnn_aud_B2A,fr_md_aud_B2A = activity_diff_context(model_dir=model_dir_B2A,hp=hp,start=start,end=end,
                                                     context_name='HL_task',cue= -1, p_cohs=0.92,batch_size=batch_size)



    max_idx_exc_50 = np.argsort(fr_rnn_vis_A[0:205])
    print('max_idx_exc_50',max_idx_exc_50)
    max =1.5#2.5#np.max(fr_rnn_vis_A)
    print('max',max)

    # for cue
    cell_type = cell_type = [171,57,86,40,8]#max_idx_exc_50[199:205]#171,57,86,40,8,


    # #for delay
    # cell_type = [85,132,3,171,133,36]
    s=50
    idx_exc =204
    colors_1 = ['#00FFFF', '#9B30FF', '#7CFC00', 'green', 'blue', '#33A1C9']
    #colors_1 = ['r', 'orange', 'y', 'green', 'blue', 'purple']
    # fig2
    for context in np.array(['A','A2B','B2A']):
        fig1 = plt.figure(figsize=(2.7, 2.7))
        ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
        if context =='A':
            plt.scatter(fr_rnn_vis_A[0:idx_exc], fr_rnn_aud_A[0:idx_exc], marker="o", s=s, color='#EE6363',edgecolors='white')
        if context == 'A2B':
            plt.scatter(fr_rnn_vis_A2B[0:idx_exc], fr_rnn_aud_A2B[0:idx_exc], marker="o", s=s, color='#EE6363',
                        edgecolors='white')
        if context == 'B2A':
            plt.scatter(fr_rnn_vis_B2A[0:idx_exc], fr_rnn_aud_B2A[0:idx_exc], marker="o", s=s, color='#EE6363',
                        edgecolors='white')

        ax1.set_xlim([-0.1, max])
        ax1.set_ylim([-0.1, max])
        plt.xticks([0,0.6,1.2,1.8],fontsize=12)
        plt.yticks([0,0.6,1.2,1.8],fontsize=12)


        j = -1
        for i in np.array(cell_type):
            j+=1
            if context == 'A':
                plt.scatter(fr_rnn_vis_A[i],fr_rnn_aud_A[i],marker="X",s=40,c=colors_1[j],label=str(i))
            if context == 'A2B':
                plt.scatter(fr_rnn_vis_A2B[i],fr_rnn_aud_A2B[i],marker="X",s=40,c=colors_1[j],label=str(i))
            if context == 'B2A':
                plt.scatter(fr_rnn_vis_B2A[i],fr_rnn_aud_B2A[i],marker="X",s=40,c=colors_1[j],label=str(i))


        plt.plot([-0.05, max * 0.9], [-0.05, max * 0.9], color='silver',)

        # set x-label
        ax1.set_xlabel('vis', fontsize=12)
        ax1.set_ylabel('aud', fontsize=12)
        ax1.spines[['right', 'top']].set_visible(False)
        plt.title( model_name + '_' + str(idx)+':'+period,fontsize=6)

        plt.savefig(figure_path+model_name+'_'+str(idx)+'_'+context+'_cue.png')
        plt.savefig(figure_path+model_name+'_'+str(idx)+'_'+context+'_cue.eps', format='eps', dpi=1000)

        plt.show()


def Exc_diff_context_select_example(fig_path, model_name, idx,hp, period,epoch,
                    model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    if period == 'delay':
        start_cal = epoch['cue_off']
        end_cal = epoch['response_on']
    elif period == 'cue':
        start_cal = epoch['cue_on']
        end_cal = epoch['cue_off']
    elif period == 'all':
        start_cal = epoch['cue_on']
        end_cal = epoch['response_on']


    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200




    # pfc_HL_vis_A,MD_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
    #                             model_dir=model_dir_A,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A, MD_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                             model_dir=model_dir_A, cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, MD_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                             model_dir=model_dir_A2B, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A2B, MD_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                             model_dir=model_dir_A2B, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, MD_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                             model_dir=model_dir_B2A, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_B2A, MD_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                             model_dir=model_dir_B2A, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_A = [pfc_HL_vis_A,pfc_HL_aud_A]
    # pfc_A2B = [pfc_HL_vis_A2B,pfc_HL_aud_A2B]
    # pfc_B2A = [pfc_HL_vis_B2A, pfc_HL_aud_B2A]
    #
    #
    # np.save(fig_path+'pfc_A.npy',pfc_A)
    # np.save(fig_path+'pfc_A2B.npy',pfc_A2B)
    # np.save(fig_path + 'pfc_B2A.npy', pfc_B2A)

    pfc_A=np.load(fig_path+'pfc_A.npy')
    pfc_A2B=np.load(fig_path+'pfc_A2B.npy')
    pfc_B2A=np.load(fig_path+'pfc_B2A.npy')


    #'''
    font=12
    start=0
    end = response_on-2
    plot_context = 'con_A'
    ########## exc ###################

    for j in np.array([40]):
        fig, axs = plt.subplots(1, 3,figsize=(10,3))
        plt.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
        #fig.suptitle('pfc  :'+model_name+'_'+str(idx)+':'+str(j))

        # y_max1 = np.max(pfc_A[0][cue_on:, j])
        # y_max2 = np.max(pfc_A[1][cue_on:, j])
        # y_max3 = np.max(pfc_A[2][cue_on:, j])
        #print('y_max', y_max1, y_max3, y_max2)

        y_max = 1.7

        axs[0].plot(pfc_A[0][cue_on + start:end, j],color='tab:red',label='vis')
        axs[0].plot(pfc_A[1][cue_on + start:end, j], color='tab:blue', label='aud')
        axs[0].set_title('con_A(#cell_40)', fontsize=font)

        axs[1].plot(pfc_A2B[0][cue_on + start:end, j],color='tab:red',label='vis')
        axs[1].plot(pfc_A2B[1][cue_on + start:end, j], color='tab:blue',label='aud')
        axs[1].set_title('con_A2B', fontsize=font)

        axs[2].plot(pfc_B2A[0][cue_on + start:end, j], color='tab:red', label='vis')
        axs[2].plot(pfc_B2A[1][cue_on + start:end, j], color='tab:blue', label='aud')
        axs[2].set_title('con_B2A', fontsize=font)

        for i in range(3):
            axs[i].set_ylim([0,y_max])
            axs[i].set_xticks([], fontsize=10)
            axs[i].set_yticks([0,0.5,1,1.5], fontsize=10)
            axs[i].spines[[ 'right', 'top', 'bottom']].set_visible(False)
            axs[0].legend(fontsize=8)

        for k in range(3):
            axs[k].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[k].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    #
        plt.savefig(fig_path +'pfc_tuning_diff_context'+str(j)+'.png')
        plt.savefig(fig_path + 'pfc_tuning_diff_context' + str(j) + '.eps',format='eps', dpi=1000)
        plt.show()





def Exc_activity_diff_context_select(fig_path, model_name, idx,hp, period,epoch,
                    model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    if period == 'delay':
        start_cal = epoch['cue_off']
        end_cal = epoch['response_on']
    elif period == 'cue':
        start_cal = epoch['cue_on']
        end_cal = epoch['cue_off']
    elif period == 'all':
        start_cal = epoch['cue_on']
        end_cal = epoch['response_on']


    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200




    pfc_HL_vis_A,MD_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=model_dir_A,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_A, MD_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_A, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_A2B, MD_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_A2B, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_A2B, MD_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_A2B, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_B2A, MD_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_B2A, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_B2A, MD_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_B2A, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_A = [pfc_HL_vis_A,pfc_HL_aud_A]
    pfc_A2B = [pfc_HL_vis_A2B,pfc_HL_aud_A2B]
    pfc_B2A = [pfc_HL_vis_B2A, pfc_HL_aud_B2A]


    np.save(fig_path+'pfc_A.npy',pfc_A)
    np.save(fig_path+'pfc_A2B.npy',pfc_A2B)
    np.save(fig_path + 'pfc_B2A.npy', pfc_B2A)

    pfc_A=np.load(fig_path+'pfc_A.npy')
    pfc_A2B=np.load(fig_path+'pfc_A2B.npy')
    pfc_B2A=np.load(fig_path+'pfc_B2A.npy')


    #'''
    font=12
    start=0
    end = response_on-2
    plot_context = 'con_A'
    ########## exc ###################

    for j in range(200):
        fig, axs = plt.subplots(1, 3,figsize=(10,3))
        plt.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
        fig.suptitle('pfc  :'+model_name+'_'+str(idx)+str(j))

        # y_max1 = np.max(pfc_A[0][cue_on:, j])
        # y_max2 = np.max(pfc_A[1][cue_on:, j])
        # y_max3 = np.max(pfc_A[2][cue_on:, j])
        #print('y_max', y_max1, y_max3, y_max2)

        #y_max = np.max([y_max1,y_max2,y_max3])

        axs[0].plot(pfc_A[0][cue_on + start:end, j],color='r',label='vis')
        axs[0].plot(pfc_A[1][cue_on + start:end, j], color='g', label='aud')
        axs[0].set_title('con_A:'+str(j), fontsize=font)

        axs[1].plot(pfc_A2B[0][cue_on + start:end, j],color='r',label='vis')
        axs[1].plot(pfc_A2B[1][cue_on + start:end, j], color='g',label='aud')
        axs[1].set_title('con_A2B', fontsize=font)

        axs[2].plot(pfc_B2A[0][cue_on + start:end, j], color='r', label='vis')
        axs[2].plot(pfc_B2A[1][cue_on + start:end, j], color='g', label='aud')
        axs[2].set_title('con_B2A', fontsize=font)


        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)
        # axs[0].set_ylim([0,y_max])
        # axs[1].set_ylim([0, y_max])



        for k in range(2):
            axs[k].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[k].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    #
        plt.savefig(fig_path +  plot_context+'_tuning'+str(j)+'.png')
        plt.show()





def plot_scatters_different_rule_md(figure_path,start,end,hp,epoch,model_name,model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92



    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.8
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

    # max_idx_exc_50 = np.argsort(fr_rnn_vis[0:50])
    # max_idx_exc = np.argsort(fr_rnn_vis[50:205])
    # print(max_idx_exc_50)
    # print(max_idx_exc+50)

    c_alpha = 0.6;s=24
    idx_md =30

    fig, axs = plt.subplots(1, 3,figsize=(9,3))
    plt.subplots_adjust(top=0.88, bottom=0.15, right=0.95, left=0.08, hspace=0.3, wspace=0.3)
    fig.suptitle('Exc_no_readout  :'+model_name)

    max =6

    axs[0].scatter(fr_md_vis_A[0:idx_md],fr_md_aud_A[0:idx_md],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    axs[1].scatter(fr_md_vis_A2B[0:idx_md],fr_md_aud_A2B[0:idx_md],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')
    axs[2].scatter(fr_md_vis_B2A[0:idx_md],fr_md_aud_B2A[0:idx_md],marker="o",s=s,color='tab:red',alpha=c_alpha,edgecolors='none')

    for i in range(3):
        axs[0].set_xlabel('vis',fontsize=12)
        axs[0].set_ylabel('aud',fontsize=12)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].plot([0,max*0.9],[0,max*0.9],color='grey')
        axs[0].set_xlim([-0.05,2])
        axs[0].set_ylim([-0.05,2])

        axs[1].set_xlim([-0.05,2])
        axs[1].set_ylim([-0.05,2])

        axs[2].set_xlim([-0.05,2])
        axs[2].set_ylim([-0.05,2])

    cell_type = [1,2,3,4,5,6]#168
    colors_1 = ['orange','black','lime','green','blue','purple']
    j=-1
    for i in np.array(cell_type):
        j+=1
        axs[0].scatter(fr_md_vis_A[i],  fr_md_aud_A[i],marker="x",s=26,c=colors_1[j],label=str(i))
        axs[1].scatter(fr_md_vis_A2B[i],fr_md_aud_A2B[i],marker="x",s=26,c=colors_1[j],label=str(i))
        axs[2].scatter(fr_md_vis_B2A[i],fr_md_aud_B2A[i],marker="x",s=26,c=colors_1[j],label=str(i))




    plt.title(':'+str(start)+'_'+str(end),fontsize=8)
    plt.legend(fontsize=5)
    # fig.savefig(figure_path+rule+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)
    #
    # plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()

def plot_activity_all_8panel_diff_context(figure_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    hp['switch_context'] = context_name
    print('******* switch_context *******',hp['switch_context'])
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=50


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1
    if context_name =='con_B2A':
        attend_vis=1
        attend_aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)
    #print(pfc_mean_HL_vis.shape)

    max_idx_exc = np.argsort(np.mean(pfc_mean_HL_vis[cue_on:response_on, 0:205], axis=0))
    cell_idx_exc = range(200)#max_idx_exc[195:205]#[ 68, 152, 108, 140,  20,  32,  72, 158, 168, 133]#max_idx_exc[195:205]#range(0,20,1)
    #print('cell_idx_exc',cell_idx_exc)
    cell_idx_inh = range(n_exc,n_exc+10,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+n_md2,1)#max_idx_md2+n_md1
    y_exc = 3#1.2*np.max(pfc_mean_HL_vis[cue_on:stim_on-2,0:n_exc])
    y_inh = 1.2*np.max(pfc_mean_HL_vis[cue_on:stim_on-2,n_exc:n_rnn])
    y_md1 = 2##1.2*np.max(MD_mean_HL_vis[cue_on:stim_on-2,0:n_md1])
    y_md2 = 1.2*np.max(MD_mean_HL_vis[cue_on:stim_on-2,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    axs[0, 0].set_ylim([0, y_exc])
    axs[0, 1].set_ylim([0, y_exc])
    axs[1, 0].set_ylim([0, y_inh])
    axs[1, 1].set_ylim([0, y_inh])
    axs[2, 0].set_ylim([0, y_md1])
    axs[2, 1].set_ylim([0, y_md1])
    axs[3, 0].set_ylim([0, y_md2])
    axs[3, 1].set_ylim([0, y_md2])
    #

    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()

def plot_MD_activity_diff_context(fig_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    figure_path = os.path.join(fig_path, 'plot_MD_activity_diff_context/')
    # figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
    tools.mkdir_p(fig_path)

    hp['switch_context'] = context_name
    print('******* switch_context *******',hp['switch_context'])
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=50


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1
    if context_name =='con_B2A':
        attend_vis=1
        attend_aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)
    #print(pfc_mean_HL_vis.shape)

    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+n_md2,1)#max_idx_md2+n_md1

    y_md1 = 2##1.2*np.max(MD_mean_HL_vis[cue_on:stim_on-2,0:n_md1])
    y_md2 = 1.2*np.max(MD_mean_HL_vis[cue_on:stim_on-2,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################

    fig, axs = plt.subplots(2, 2,figsize=(6,6))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+context_name)


    for i in np.array(cell_idx_md1):
        axs[0,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)


    axs[0, 0].set_ylim([0, y_md1])
    axs[0, 1].set_ylim([0, y_md1])
    axs[1, 0].set_ylim([0, y_md2])
    axs[1, 1].set_ylim([0, y_md2])
    #

    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()





def plot_activity_Exc_diff_context_all(figure_path, model_name, idx, hp,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_vis = [pfc_HL_vis_1,pfc_HL_vis_2,pfc_HL_vis_3]
    pfc_aud = [pfc_HL_aud_1, pfc_HL_aud_2, pfc_HL_aud_3]
    np.save(figure_path+'pfc_HL_vis.npy',pfc_vis)
    np.save(figure_path + 'pfc_HL_aud.npy', pfc_aud)

    pfc_vis=np.load(figure_path+'pfc_HL_vis.npy')
    pfc_aud = np.load(figure_path + 'pfc_HL_aud.npy')




    font=12
    start=0
    end = response_on-2
    ########## exc ###################

    for panel in range(int(205/4)):
        fig, axs = plt.subplots(3, 4,figsize=(11,7))
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0.5)
        fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))

        i_dx=0
        for j in range(4):
            axs[0, j].plot(pfc_vis[0][cue_on + start:end, j+4*panel],color='r',label='vis')
            axs[0, j].plot(pfc_aud[0][cue_on + start:end, j+4*panel], color='g', label='aud')
            axs[0, j].set_title('con_A:'+str(j+4*panel), fontsize=font)

            axs[1, j].plot(pfc_vis[1][cue_on + start:end, j+4*panel],color='r',label='con_B')
            axs[1, j].plot(pfc_aud[1][cue_on + start:end, j+4*panel], color='g')
            axs[1, 0].set_title('con_B', fontsize=font)

            axs[2, j].plot(pfc_vis[2][cue_on + start:end, j+4*panel], color='r',label='con_AA')
            axs[2, j].plot(pfc_aud[2][cue_on + start:end, j+4*panel], color='g')
            axs[2, 0].set_title('con_AA', fontsize=font)
            axs[0, j].legend(fontsize=8)

                # axs[i,j].set_ylim([0,y_exc])
                # axs[i,j].set_title('exc: '+'vis',fontsize=font)
                #axs[0,0].legend(fontsize=5)

        for i in range(2):
            for j in range(4):
                #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
                axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
                axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
        #
        plt.savefig(figure_path+str(panel)+'.png')
        plt.show()

def plot_activity_sns_Exc_diff_context(figure_path, model_name, idx, hp,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    # pfc_vis = [pfc_HL_vis_1,pfc_HL_vis_2,pfc_HL_vis_3]
    # pfc_aud = [pfc_HL_aud_1, pfc_HL_aud_2, pfc_HL_aud_3]
    # np.save(figure_path+'pfc_HL_vis.npy',pfc_vis)
    # np.save(figure_path + 'pfc_HL_aud.npy', pfc_aud)

    pfc_vis=np.load(figure_path+'pfc_HL_vis.npy')
    pfc_aud = np.load(figure_path + 'pfc_HL_aud.npy')




    font=8
    start = cue_on + 7
    end = response_on
    y_exc = 2
    ########## exc ###################
    cell_idx_exc = np.array([40])  # 78

    for context in np.array(['A_vis', 'A_aud','A2B_vis','A2B_aud']):
        fig1 = plt.figure(figsize=(2, 1.5))
        ax1 = fig1.add_axes([0.2, 0.2, 0.7, 0.7])

        for i in np.array(cell_idx_exc):
            if context == 'A_vis':
                ax1.plot(pfc_vis[0][start:end, i], c='black',label=str(i))
                ax1.set_title('A_vis', fontsize=font)
            if context == 'A_aud':
                ax1.plot(pfc_aud[0][start:end, i], c='black',label=str(i))
                ax1.set_title('A_aud', fontsize=font)

            if context == 'A2B_vis':
                ax1.plot(pfc_vis[1][start:end, i], c='black',label=str(i))
                ax1.set_title('A2B_vis', fontsize=font)
            if context == 'A2B_aud':
                ax1.plot(pfc_aud[1][start:end, i], c='black',label=str(i))
                ax1.set_title('A2B_aud', fontsize=font)
        ax1.set_ylim([0, y_exc])


        # plt.xticks([0,60], fontsize=10)
        plt.xticks([], fontsize=10)
        plt.yticks([], fontsize=10)
        # plt.yticks([0.5, 0.8, 1.1, 1.4, 1.7], fontsize=12)

        ax1.spines[['left', 'right', 'top']].set_visible(False)

        ax1.axvspan(cue_off - start+1, cue_off - start+1, color='grey', label='cue_off')

        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.png')
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.eps',format='eps', dpi=1000)
        plt.show()

def plot_activity_MD_diff_context_all(figure_path, model_name, idx, hp,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=hp['p_coh']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    MD_vis = [MD_HL_vis_1,MD_HL_vis_2,MD_HL_vis_3]
    MD_aud = [MD_HL_aud_1, MD_HL_aud_2, MD_HL_aud_3]
    np.save(figure_path+'MD_HL_vis.npy',MD_vis)
    np.save(figure_path + 'MD_HL_aud.npy', MD_aud)

    MD_vis=np.load(figure_path+'MD_HL_vis.npy')
    MD_aud = np.load(figure_path + 'MD_HL_aud.npy')




    font=12
    start=0
    end = response_on-2
    y_exc = 1.5#np.max(MD_vis[0][cue_on:cue_off, :])
    ########## exc ###################

    for panel in range(int(20/4)):
        fig, axs = plt.subplots(3, 4,figsize=(11,7))
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0.5)
        fig.suptitle('MD  :'+model_name+'_'+str(idx)+'_'+str(coh_HL))

        i_dx=0
        for j in range(4):
            axs[0, j].plot(MD_vis[0][cue_on + start:end, j+4*panel],color='r',label='vis')
            axs[0, j].plot(MD_aud[0][cue_on + start:end, j+4*panel], color='g', label='aud')

            axs[1, j].plot(MD_vis[1][cue_on + start:end, j+4*panel],color='r',label='con_B')
            axs[1, j].plot(MD_aud[1][cue_on + start:end, j+4*panel], color='g')


            axs[2, j].plot(MD_vis[2][cue_on + start:end, j+4*panel], color='r',label='con_AA')
            axs[2, j].plot(MD_aud[2][cue_on + start:end, j+4*panel], color='g')


        for j in range(4):
            axs[0, j].set_title('con_A:' + str(j + 4 * panel), fontsize=font)
            axs[0, j].set_title('con_A:' + str(j + 4 * panel), fontsize=font)
            axs[1, 0].set_title('con_B', fontsize=font)
            axs[2, 0].set_title('con_AA', fontsize=font)
            axs[0, j].legend(fontsize=8)



        for i in range(3):
            for j in range(4):
                #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
                axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
                axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
                axs[i, j].set_ylim([0, y_exc])
        #
        plt.savefig(figure_path+'_'+str(idx)+'_'+str(panel)+'_'+str(coh_HL)+'.png')
        plt.show()
def plot_activity_MD_diff_uncertainty(figure_path, model_name, idx, hp,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=hp['p_coh']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_A_low,MD_HL_A_low = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=0.55,batch_size=batch_size)
    pfc_HL_A_high, MD_HL_A_high = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=1, p_cohs=0.99, batch_size=batch_size)

    pfc_HL_A2B_low, MD_HL_A2B_low = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=0.55,batch_size=batch_size)
    pfc_HL_A2B_high, MD_HL_A2B_high = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=0.99,batch_size=batch_size)




    font=12
    start=0
    end = response_on-2
    y_exc = np.max(MD_HL_A_low[cue_on:cue_off, :])
    ########## exc ###################

    for idx in range(30):
        fig, axs = plt.subplots(2, 2, figsize=(5, 4))
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0.5)
        fig.suptitle('MD  :' + model_name + '_' + str(idx) + '_' + str(coh_HL))

        axs[0, 0].plot(MD_HL_A_low[cue_on + start:end, idx],color='black',label='A_low')
        axs[0, 1].plot(MD_HL_A_high[cue_on + start:end, idx], color='black', label='A_high')

        axs[1, 0].plot(MD_HL_A2B_low[cue_on + start:end, idx], color='black', label='A2B_low')
        axs[1, 1].plot(MD_HL_A2B_high[cue_on + start:end, idx], color='black', label='A2B_high')

        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        for i in range(2):
            for j in range(2):
                #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
                axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
                axs[i, j].set_ylim([0, y_exc])

        plt.savefig(figure_path+str(idx)+'vis.png')
        plt.show()


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

def scatters_sns_MD_different_context(figure_path,hp,epoch,idx,model_name,period,model_dir_A,model_dir_A2B,model_dir_B2A):
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


    np.save(figure_path+'fr_md_vis_A.npy',fr_md_vis_A)
    np.save(figure_path + 'fr_md_aud_A.npy', fr_md_aud_A)
    np.save(figure_path + 'fr_md_vis_A2B.npy', fr_md_vis_A2B)
    np.save(figure_path + 'fr_md_aud_A2B.npy', fr_md_aud_A2B)
    np.save(figure_path + 'fr_md_vis_B2A.npy', fr_md_vis_B2A)
    np.save(figure_path + 'fr_md_aud_B2A.npy', fr_md_aud_B2A)

    fr_md_vis_A = np.load(figure_path+'fr_md_vis_A.npy')
    fr_md_aud_A = np.load(figure_path+'fr_md_aud_A.npy')
    fr_md_vis_A2B = np.load(figure_path + 'fr_md_vis_A2B.npy')
    fr_md_aud_A2B = np.load(figure_path + 'fr_md_aud_A2B.npy')
    fr_md_vis_B2A = np.load(figure_path + 'fr_md_vis_B2A.npy')
    fr_md_aud_B2A = np.load(figure_path + 'fr_md_aud_B2A.npy')

    print(np.argsort(fr_md_vis_A))
    max = 1.7  # np.max(fr_rnn_vis_A)
    min = 0.35
    print('max', max)

    cell_type = [10, 23, 27,  16, 24,  11]#np.argsort(fr_md_vis_A)[24:30]#[1,2,10,11,16,26]
    print('cell_type',cell_type)
    s = 50
    idx_md = 30
    #colors_1 = sns.color_palette("Set2")
    colors_1 = ['#9B30FF', '#FF34B3', '#7CFC00', 'green', '#FF4500', '#FF6A6A']
    #colors_1 = ['r', 'orange', 'y', 'green', 'b', 'purple']
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
        plt.xticks([0.5, 0.8,1.1,1.4,1.7], fontsize=12)
        plt.yticks([0.5, 0.8,1.1,1.4,1.7], fontsize=12)

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

        plt.plot([0, max * 0.9], [0, max * 0.9], color='silver', )

        # set x-label
        ax1.spines[['right', 'top']].set_visible(False)
        #plt.title( model_name + '_' + str(idx)+':'+period,fontsize=6)
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.png')
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.eps', format='eps', dpi=1000)

        plt.show()

def MD_diff_context_select_example(figure_path, model_name, idx, hp,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=hp['p_coh']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    MD_vis = [MD_HL_vis_1,MD_HL_vis_2,MD_HL_vis_3]
    MD_aud = [MD_HL_aud_1, MD_HL_aud_2, MD_HL_aud_3]
    np.save(figure_path+'MD_HL_vis.npy',MD_vis)
    np.save(figure_path + 'MD_HL_aud.npy', MD_aud)

    MD_vis=np.load(figure_path+'MD_HL_vis.npy')
    MD_aud = np.load(figure_path + 'MD_HL_aud.npy')




    font=12
    start=0
    end = response_on-2
    y_exc = 1.5#np.max(MD_vis[0][cue_on:cue_off, :])
    colors = sns.color_palette("Paired")
    ########## exc ###################

    for j in np.array([24]):
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])


        ax.plot(MD_vis[0][cue_on + start:end, j],color=colors[1],label='1')
        ax.plot(MD_vis[1][cue_on + start:end, j],color=colors[0],label='2')
        ax.plot(MD_vis[2][cue_on + start:end, j], '--',color=colors[1],label='1')


        ax.axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey')
        #ax.axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey')
        ax.set_ylim([0, y_exc])

        plt.xticks([], fontsize=10)
        # plt.yticks([], fontsize=10)
        #plt.legend(fontsize=8)
        # plt.yticks([0.5, 0.8, 1.1, 1.4, 1.7], fontsize=12)

        ax.spines[['left', 'right', 'top','bottom']].set_visible(False)

        #
        plt.savefig(figure_path+'MD_'+str(idx)+'_'+str(coh_HL)+'.png')
        plt.savefig(figure_path+'MD_'+str(idx)+'_'+str(coh_HL)+ '.eps',format = 'eps', dpi = 1000)

        plt.show()



def MD_activity_diff_context_select(fig_path, model_name, idx,hp, period,epoch,
                    context_model_1, context_model_2,context_model_3):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    if period == 'delay':
        start_cal = epoch['cue_off']
        end_cal = epoch['response_on']
    elif period == 'cue':
        start_cal = epoch['cue_on']
        end_cal = epoch['cue_off']
    elif period == 'all':
        start_cal = epoch['cue_on']
        end_cal = epoch['response_on']


    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200




    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_3, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    MD_vis = [MD_HL_vis_1,MD_HL_vis_2,MD_HL_vis_3]
    MD_aud = [MD_HL_aud_1, MD_HL_aud_2, MD_HL_aud_3]
    np.save(fig_path+'zMD_HL_vis.npy',MD_vis)
    np.save(fig_path + 'zMD_HL_aud.npy', MD_aud)

    MD_vis=np.load(fig_path+'zMD_HL_vis.npy')
    MD_aud = np.load(fig_path + 'zMD_HL_aud.npy')

    context_A_tuning=[]
    context_B_tuning = []

    for i in range(n_md):
        A1 = np.mean(MD_vis[0][start_cal:end_cal, i])
        A2 = np.mean(MD_vis[1][start_cal:end_cal, i])
        A3 = np.mean(MD_vis[2][start_cal:end_cal, i])
        #print(A1,A2,A3)

        if A1>A2 and A3>A2:
            print(A1,A2,A3)
            context_A_tuning.append(i)
        if A1<A2 and A3<A2:
            print(A1,A2,A3)
            context_B_tuning.append(i)


    print('context_A_tuning',context_A_tuning)




    #'''
    font=12
    start=0
    end = response_on-2
    plot_context = 'con_A'
    ########## exc ###################
    if plot_context=='con_A':
        idx_context = context_A_tuning
    if plot_context=='con_B':
        idx_context = context_B_tuning
    for j in np.array(idx_context):
        fig, axs = plt.subplots(1, 2,figsize=(10,4))
        plt.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
        fig.suptitle('MD  :'+model_name+'_'+str(idx))

        y_max1 = np.max(MD_vis[0][cue_on:, j])
        y_max2 = np.max(MD_vis[1][cue_on:, j])
        y_max3 = np.max(MD_vis[2][cue_on:, j])
        #print('y_max', y_max1, y_max3, y_max2)

        y_max = np.max([y_max1,y_max2,y_max3])

        axs[0].plot(MD_vis[0][cue_on + start:end, j],color='r',label='A')
        axs[0].plot(MD_vis[1][cue_on + start:end, j], color='g', label='B')
        axs[0].set_title('con_A2B:'+str(j), fontsize=font)

        axs[1].plot(MD_vis[2][cue_on + start:end, j],color='r',label='AA')
        axs[1].plot(MD_vis[1][cue_on + start:end, j], color='g',label='B')
        axs[1].set_title('con_B2A', fontsize=font)


        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)
        axs[0].set_ylim([0,y_max])
        axs[1].set_ylim([0, y_max])



        for k in range(2):
            axs[k].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[k].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
    #
        plt.savefig(fig_path +  plot_context+'_tuning'+str(j)+'.png')
        plt.show()


def plot_activity_one_trial_cue_minus(figure_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_HL=hp['p_coh']
    print('coh_HL',coh_HL)

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']


    pfc_HL_vis,MD_HL_vis = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=1,
                                              p_cohs=coh_HL,front='1_minus')
    pfc_HL_aud,MD_HL_aud = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=-1,
                                              p_cohs=coh_HL,front='1_minus')


    print('pfc_HL_vis,MD_HL_vis',pfc_HL_vis.shape,MD_HL_vis.shape)
    cell_idx_exc = range(0,20,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+5,1)#max_idx_md2+n_md1

    font=10
    start=0
    end = response_on+5
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_HL)+'_1_minus')


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[0,0].set_ylim([0,y_exc])
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[0,1].set_ylim([0,y_exc])
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[1,0].set_ylim([0,y_inh])
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[1,1].set_ylim([0,y_inh])
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[2,0].set_ylim([0,y_md1])
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[2,1].set_ylim([0,y_md1])
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_HL_vis[cue_on+start:end,i],label=str(i))
        #axs[3,0].set_ylim([0,y_md2])
        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_HL_aud[cue_on+start:end,i],label=str(i))
        #axs[3,1].set_ylim([0,y_md2])
        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    print('cue_off,stim_on',cue_off,stim_on)


    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off, cue_off, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on, stim_on, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()


def plot_all_activity_cue_plus(figure_path,model_name,model_dir,idx,hp,context_name,epoch,front_show):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 1
    coh_HL=hp['p_coh']
    print('coh_HL',coh_HL)

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    if context_name =='con_A':
        cue_vis = 1
        cue_aud = -1
    if context_name =='con_A2B':
        cue_vis = -1
        cue_aud = 1


    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on']
    response_on=epoch['response_on'];response_off=epoch['response_off']

    pfc_HL_vis,MD_HL_vis = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=cue_vis,
                                              p_cohs=coh_HL,front_show=front_show,batch_size=batch_size)
    pfc_HL_aud,MD_HL_aud = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=cue_aud,
                                              p_cohs=coh_HL,front_show=front_show,batch_size=batch_size)

    print('pfc_HL_vis,MD_HL_vis',pfc_HL_vis.shape,MD_HL_vis.shape)

    max_idx_exc = np.argsort(np.mean(pfc_HL_vis[cue_on:response_off,0:205],axis=0))
    max_idx_inh = np.argsort(np.mean(pfc_HL_vis[cue_on:response_off, 205:256], axis=0))
    print('max_idx_exc',max_idx_exc.shape)
    cell_idx_exc =range(0,10,1)#max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc,n_exc+15,1)#max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,5,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+5,1)#max_idx_md2+n_md1

    font=10
    start=0
    end = response_off
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_HL)+'_1_plus')


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_vis[:,i],label=str(i))
        #
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_aud[:,i],label=str(i))

        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_HL_vis[:,i],label=str(i))

        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_HL_aud[:,i],label=str(i))

        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_HL_vis[:,i],label=str(i))
        #axs[2,0].set_ylim([0,y_md1])
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_HL_aud[:,i],label=str(i))

        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_HL_vis[:,i],label=str(i))

        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_HL_aud[:,i],label=str(i))

        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)
    y_exc = np.max(pfc_HL_vis[cue_on:cue_off, cell_idx_exc])
    y_inh = np.max(pfc_HL_vis[cue_on:cue_off, cell_idx_inh])
    y_md1 = np.max(pfc_HL_vis[cue_on:response_off, cell_idx_md1])
    y_md2 = np.max(pfc_HL_vis[cue_on:response_off, cell_idx_md2])

    axs[0, 0].set_ylim([0,y_exc])
    axs[0,1].set_ylim([0,y_exc])
    axs[1,0].set_ylim([0,y_inh])
    axs[1,1].set_ylim([0,y_inh])
    axs[2,0].set_ylim([0,y_md1])
    axs[2,1].set_ylim([0,y_md1])
    axs[3,0].set_ylim([0,y_md2])
    axs[3,1].set_ylim([0,y_md2])
    print('cue_off,stim_on', cue_off, stim_on)


    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i, j].axvspan(cue_off, cue_off, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on, stim_on, color='grey', label='stim_on')


    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()

def plot_exc_activity_cue_plus(figure_path,model_name,model_dir,idx,hp,context_name,epoch,front_show):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 100
    coh_HL=hp['p_coh']
    print('coh_HL',coh_HL)

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    if context_name =='con_A':
        cue_vis = 1
        cue_aud = -1
    if context_name =='con_A2B':
        cue_vis = -1
        cue_aud = 1


    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on']
    response_on=epoch['response_on'];response_off=epoch['response_off']

    pfc_HL_vis,MD_HL_vis = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=cue_vis,
                                              p_cohs=coh_HL,front_show=front_show,batch_size=batch_size)
    pfc_HL_aud,MD_HL_aud = generate_one_trial(model_dir=model_dir,hp=hp,context_name='HL_task',cue=cue_aud,
                                              p_cohs=coh_HL,front_show=front_show,batch_size=batch_size)

    print('pfc_HL_vis,MD_HL_vis',pfc_HL_vis.shape,MD_HL_vis.shape)

    max_idx_exc = np.argsort(np.mean(pfc_HL_vis[cue_on:response_off,0:205],axis=0))
    max_idx_inh = np.argsort(np.mean(pfc_HL_vis[cue_on:response_off, 205:256], axis=0))
    print('max_idx_exc',max_idx_exc.shape)
    cell_idx_exc =range(20,30,1)#max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc,n_exc+15,1)#max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font=10
    start=0
    end = response_off
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(8,6))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_HL)+'_1_plus')


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_vis[:,i],label=str(i))
        #
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_aud[:,i],label=str(i))

        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_HL_vis[:,i],label=str(i))

        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_HL_aud[:,i],label=str(i))

        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    y_exc = np.max(pfc_HL_vis[cue_on:cue_off, cell_idx_exc])
    y_inh = np.max(pfc_HL_vis[cue_on:cue_off, cell_idx_inh])

    axs[0, 0].set_ylim([0,y_exc])
    axs[0,1].set_ylim([0,y_exc])
    axs[1,0].set_ylim([0,y_inh])
    axs[1,1].set_ylim([0,y_inh])

    print('cue_off,stim_on', cue_off, stim_on)


    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i, j].axvspan(cue_off, cue_off, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on, stim_on, color='grey', label='stim_on')


    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()


def plot_exc_activity_cue_plus_diff_context(figure_path, model_name, model_dir_A,
                                            model_dir_A2B,idx, hp, epoch, front_show,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 1
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_vis_A, MD_HL_vis_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=1,
                                               p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)
    pfc_HL_aud_A, MD_HL_aud_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=-1,
                                               p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    print('A2B #################', pfc_HL_vis_A.shape, MD_HL_vis_A.shape)

    pfc_HL_vis_A2B, MD_HL_vis_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=-1,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)
    pfc_HL_aud_A2B, MD_HL_aud_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=1,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)


    max_idx_exc = np.argsort(np.mean(pfc_HL_vis_A[cue_on:response_off, 0:205], axis=0))
    cell_idx_exc = range(from_idx, from_idx+10, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc_exc  :' + model_name + '_' + str(idx)+'_'+str(from_idx))

    for i in np.array(cell_idx_exc):
        axs[0, 0].plot(pfc_HL_vis_A[start:, i], label=str(i))
        #
        axs[0, 0].set_title('A: ' + 'vis', fontsize=font)
        # axs[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_aud_A[start:, i], label=str(i))

        axs[0, 1].set_title('A: ' + 'aud', fontsize=font)
        # axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1, 0].plot(pfc_HL_vis_A2B[start:, i], label=str(i))

        axs[1, 0].set_title('A2B: ' + 'vis', fontsize=font)
        axs[1, 0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[1, 1].plot(pfc_HL_aud_A2B[start:, i], label=str(i))

        axs[1, 1].set_title('A2B: ' + 'aud', fontsize=font)
        # axs[1,1].legend(fontsize=5)

    y_exc = 0.2+np.max(pfc_HL_vis_A[cue_on:cue_off, cell_idx_exc])

    axs[0, 0].set_ylim([0, y_exc])
    axs[0, 1].set_ylim([0, y_exc])
    axs[1, 0].set_ylim([0, y_exc])
    axs[1, 1].set_ylim([0, y_exc])

    print('cue_off,stim_on', cue_off, stim_on)

    for i in range(2):
        for j in range(2):
            # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i, j].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + '_' + str(idx)+'_'+str(from_idx)  + '.png')
    plt.show()




def plot_exc_mean_vis_aud_diff_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch, front_show,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 100
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_vis_aud_A, MD_HL_vis_aud_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=c_cue,
                                               p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)


    print('pfc_HL_vis_A,MD_HL_vis_A', pfc_HL_vis_aud_A.shape, MD_HL_vis_aud_A.shape)

    pfc_HL_vis_aud_A2B, MD_HL_vis_aud_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=c_cue,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)



    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = range(from_idx, from_idx+10, 1)#range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 1, figsize=(5, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc_exc  :' + model_name + '_' + str(idx))

    for i in np.array(cell_idx_exc):
        axs[0].plot(pfc_HL_vis_aud_A[start:, i], label=str(i))
        #
        axs[0].set_title('A: ' + 'mean', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1].plot(pfc_HL_vis_aud_A2B[start:, i], label=str(i))

        axs[1].set_title('A2B: ' + 'mean', fontsize=font)
        axs[1].legend(fontsize=5)

    y_exc = 0.2+np.max(pfc_HL_vis_aud_A[cue_on:cue_off, cell_idx_exc])

    axs[0].set_ylim([0, y_exc])
    axs[1].set_ylim([0, y_exc])

    print('cue_off,stim_on', cue_off, stim_on)

    for i in range(2):

        # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
        axs[i].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
        axs[i].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + '_' + str(idx)+'_'+str(from_idx)  + 'mean.png')
    plt.show()





def plot_exc_vis_diff_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch, front_show,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 100
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_vis_A, MD_HL_vis_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=1,
                                               p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_vis_A2B, MD_HL_vis_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=-1,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)




    cell_idx_exc = range(from_idx, from_idx+10, 1)#range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 1, figsize=(5, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc_exc  :' + model_name + '_' + str(idx))

    for i in np.array(cell_idx_exc):
        axs[0].plot(pfc_HL_vis_A[start:, i], label=str(i))
        #
        axs[0].set_title('A: ' + 'vis', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1].plot(pfc_HL_vis_A2B[start:, i], label=str(i))

        axs[1].set_title('A2B: ' + 'vis', fontsize=font)
        axs[1].legend(fontsize=5)

    y_exc = 0.2+np.max(pfc_HL_vis_A[cue_on:cue_off, cell_idx_exc])

    axs[0].set_ylim([0, y_exc])
    axs[1].set_ylim([0, y_exc])

    print('cue_off,stim_on', cue_off, stim_on)

    for i in range(2):

        # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
        axs[i].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
        axs[i].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + '_' + str(idx)+'_'+str(from_idx)  + 'vis.png')
    plt.show()


def plot_exc_aud_diff_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                              idx, hp, epoch, front_show, from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 100
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))

    cue_on = epoch['cue_on'];
    cue_off = epoch['cue_off'];
    stim_on = epoch['stim_on']
    response_on = epoch['response_on'];
    response_off = epoch['response_off']

    pfc_HL_aud_A, MD_HL_aud_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=-1,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_aud_A2B, MD_HL_aud_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=1,
                                                       p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    cell_idx_exc = range(from_idx, from_idx + 10, 1)  # range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 1, figsize=(5, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('exc:' + model_name + '_' + str(idx))

    for i in np.array(cell_idx_exc):
        axs[0].plot(pfc_HL_aud_A[start:, i], label=str(i))
        #
        axs[0].set_title('A: ' + 'aud', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1].plot(pfc_HL_aud_A2B[start:, i], label=str(i))

        axs[1].set_title('A2B: ' + 'aud', fontsize=font)
        axs[1].legend(fontsize=5)

    y_exc = 0.2 + np.max(pfc_HL_aud_A[cue_on:cue_off, cell_idx_exc])

    axs[0].set_ylim([0, y_exc])
    axs[1].set_ylim([0, y_exc])

    for i in range(2):
        # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
        axs[i].axvspan(cue_off - start, cue_off - start, color='grey', label='cue_off')
        axs[i].axvspan(stim_on - start, stim_on - start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + str(from_idx) + 'aud.png')
    plt.show()


def plot_exc_three_mean_vis_aud_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch, front_show,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 1
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_mean_A, MD_HL_mean_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=1,
                                               p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_mean_A2B, MD_HL_mean_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=-1,
                                                   p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_vis_A, MD_HL_vis_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                                                    cue=1,p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_vis_A2B, MD_HL_vis_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                    cue=-1,p_cohs=coh_HL, front_show=front_show,batch_size=batch_size)

    pfc_HL_aud_A, MD_HL_aud_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                                                    cue=-1,p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_aud_A2B, MD_HL_aud_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                    cue=1,p_cohs=coh_HL, front_show=front_show,batch_size=batch_size)

    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = range(from_idx, from_idx+5, 1)#range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc_exc  :' + model_name + '_' + str(idx))

    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_mean_A[start:, i], label=str(i))
        axs[0,0].set_title('A: ' + 'mean', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_HL_mean_A2B[start:, i], label=str(i))
        axs[1,0].set_title('A2B: ' + 'mean', fontsize=font)
        axs[1,0].legend(fontsize=5)


    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_vis_A[start:, i], label=str(i))
        axs[0,1].set_title('A: ' + 'vis', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,1].plot(pfc_HL_vis_A2B[start:, i], label=str(i))
        axs[1,1].set_title('A2B: ' + 'vis', fontsize=font)
        axs[1,1].legend(fontsize=5)


    for i in np.array(cell_idx_exc):
        axs[0,2].plot(pfc_HL_aud_A[start:, i], label=str(i))
        axs[0,2].set_title('A: ' + 'aud', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,2].plot(pfc_HL_aud_A2B[start:, i], label=str(i))
        axs[1,2].set_title('A2B: ' + 'aud', fontsize=font)
        axs[1,2].legend(fontsize=5)

    y_exc = 0.2+np.max(pfc_HL_mean_A[cue_on:cue_off, cell_idx_exc])



    print('cue_off,stim_on', cue_off, stim_on)

    for i in range(2):
        for j in range(3):
            axs[i,j].set_ylim([0, y_exc])

            # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
            axs[i,j].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + 'mix_'+model_name + '_' + str(idx)+'_'+str(from_idx)  + '.png')
    plt.show()


def plot_Exc_vis_aud_diff_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch, front_show,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 200
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']


    pfc_HL_vis_A, MD_HL_vis_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                                                    cue=1,p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_vis_A2B, MD_HL_vis_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                    cue=-1,p_cohs=coh_HL, front_show=front_show,batch_size=batch_size)

    pfc_HL_aud_A, MD_HL_aud_A = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                                                    cue=-1,p_cohs=coh_HL, front_show=front_show, batch_size=batch_size)

    pfc_HL_aud_A2B, MD_HL_aud_A2B = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                    cue=1,p_cohs=coh_HL, front_show=front_show,batch_size=batch_size)

    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = range(from_idx, from_idx+5, 1)#range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 2, figsize=(8, 5.5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle(front_show+  ':' +model_name + '_' + str(idx))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_vis_A[start:, i], label=str(i))
        axs[0,0].set_title('A: ' + 'vis', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_HL_vis_A2B[start:, i], label=str(i))
        axs[1,0].set_title('A2B: ' + 'vis', fontsize=font)
        axs[1,0].legend(fontsize=5)


    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_aud_A[start:, i], label=str(i))
        axs[0,1].set_title('A: ' + 'aud', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,1].plot(pfc_HL_aud_A2B[start:, i], label=str(i))
        axs[1,1].set_title('A2B: ' + 'aud', fontsize=font)
        axs[1,1].legend(fontsize=5)

    y_exc = 0.2+np.max(pfc_HL_vis_A[cue_on:cue_off, cell_idx_exc])



    print('cue_off,stim_on', cue_off, stim_on)

    for i in range(2):
        for j in range(2):
            axs[i,j].set_ylim([0, y_exc])

            # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
            axs[i,j].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + front_show + model_name + '_' + str(idx)+'_'+str(from_idx)  + '.png')
    plt.show()


def plot_Exc_mix_diff_context(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch,from_idx):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 100
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_mean_A_plus, MD_HL_mean_A_plus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=c_cue,
                                               p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_plus, MD_HL_mean_A2B_plus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=c_cue,
                                                   p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A_minus, MD_HL_mean_A_minus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=c_cue,
                                                     p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_minus, MD_HL_mean_A2B_minus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                         cue=c_cue,
                                                         p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)



    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = range(from_idx, from_idx+5, 1)#range(0, 200, 1)  # max_idx_exc[-10:]#range(0,20,1)
    cell_idx_inh = range(n_exc, n_exc + 15, 1)  # max_idx_inh[-10:]#range(n_exc,n_exc+20,1)#max_idx_inh+n_exc

    font = 10
    start = cue_on
    end = response_off
    ########## exc ###################
    # '''

    fig, axs = plt.subplots(2, 2, figsize=(6, 5.5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc_exc  :' + model_name + '_' + str(idx))

    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_HL_mean_A_plus[start:, i], label=str(i))
        axs[0,0].set_title('A: ' + 'plus', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1,0].plot(pfc_HL_mean_A2B_plus[start:, i], label=str(i))
        axs[1,0].set_title('A2B: ' + 'plus', fontsize=font)

    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_mean_A_minus[start:, i], label=str(i))
        axs[0, 1].set_title('A: ' + 'minus', fontsize=font)
        # axs[0, 0].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[1, 1].plot(pfc_HL_mean_A2B_minus[start:, i], label=str(i))
        axs[1, 1].set_title('A2B: ' + 'minus', fontsize=font)

    y_exc = 0.2+np.max(pfc_HL_mean_A_plus[cue_on:cue_off, cell_idx_exc])



    print('cue_off,stim_on', cue_off, stim_on)
    axs[1, 0].legend(fontsize=5)
    for i in range(2):
        for j in range(2):
            axs[i,j].set_ylim([0, y_exc])

            # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-start, cue_off-start, color='grey', label='cue_off')
            axs[i,j].axvspan(stim_on-start, stim_on-start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + '_' + str(idx)+'_'+str(from_idx) + 'mean.png')
    plt.show()


def plot_sns_exc_Exc_cue_momentary(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 200
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_mean_A_plus, MD_HL_mean_A_plus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=c_cue,
                                               p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_plus, MD_HL_mean_A2B_plus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task', cue=c_cue,
                                               p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A_minus, MD_HL_mean_A_minus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task', cue=c_cue,
                                                p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_minus, MD_HL_mean_A2B_minus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                                                cue=c_cue,p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)


    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = np.array([3])#86

    font = 7
    start = cue_on+7
    end = response_on
    ########## exc ###################
    # '''
    y_exc = 0.1 + np.max(pfc_HL_mean_A_plus[cue_on:cue_off, cell_idx_exc])
    for context in np.array(['A', 'A2B']):

        fig1 = plt.figure(figsize=(2, 1.5))
        ax1 = fig1.add_axes([0.2, 0.2, 0.7, 0.7])
        for i in np.array(cell_idx_exc):

            if context == 'A':
                ax1.plot(pfc_HL_mean_A_plus[start:end, i], c='black', label=str(i))
                ax1.plot(pfc_HL_mean_A_minus[start:end, i], c='grey', label=str(i))
                ax1.set_title('A_momentary', fontsize=font)
            if context == 'A2B':
                ax1.plot(pfc_HL_mean_A2B_plus[start:end, i], c='black', label=str(i))
                ax1.plot(pfc_HL_mean_A2B_minus[start:end, i], c='grey', label=str(i))
                ax1.set_title('A2B_momentary', fontsize=font)

        ax1.set_ylim([0, y_exc])

        # plt.xticks([0,60], fontsize=10)
        plt.xticks([], fontsize=10)
        plt.yticks([], fontsize=10)
        # plt.yticks([0.5, 0.8, 1.1, 1.4, 1.7], fontsize=12)

        ax1.spines[['left','right', 'top']].set_visible(False)
        # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
        ax1.axvspan(cue_off - start + 1, cue_off - start + 1, color='grey', label='cue_off')
        # ax1.axvspan(stim_on - start + 1, stim_on - start + 1, color='grey', label='stim_on')

        plt.savefig(figure_path + 'cue_momentary_' + model_name + '_' + str(idx) + '_' + context + '.png')
        plt.savefig(figure_path + 'cue_momentary_' + model_name + '_' + str(idx) + '_' + context + '.eps',
                    format='eps', dpi=1000)
        plt.show()


def plot_sns_exc_Exc_cue_integrated1(figure_path, model_name, model_dir_A, model_dir_A2B,
                                       idx, hp, epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = 200
    coh_HL = hp['p_coh']
    print('coh_HL', coh_HL)
    n_exc = int(hp['n_rnn'] * 0.8)
    n_md1 = int(hp['n_md'] * hp['p_md1'])
    c_cue = hp['rng'].choice([1, -1], (batch_size,))


    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];stim_on = epoch['stim_on']
    response_on = epoch['response_on'];response_off = epoch['response_off']

    pfc_HL_mean_A_plus, MD_HL_mean_A_plus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                        cue=c_cue,p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_plus, MD_HL_mean_A2B_plus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                        cue=c_cue,p_cohs=coh_HL, front_show='plus_type1', batch_size=batch_size)

    pfc_HL_mean_A_minus, MD_HL_mean_A_minus = generate_one_trial(model_dir=model_dir_A, hp=hp, context_name='HL_task',
                        cue=c_cue,p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)

    pfc_HL_mean_A2B_minus, MD_HL_mean_A2B_minus = generate_one_trial(model_dir=model_dir_A2B, hp=hp, context_name='HL_task',
                        cue=c_cue,p_cohs=coh_HL, front_show='minus_type1', batch_size=batch_size)


    #max_idx_exc = np.argsort(np.mean(pfc_HL_vis_aud_A[cue_on:cue_on+20, 0:205], axis=0))
    cell_idx_exc = np.array([29])#36,116,29

    font = 7
    start = cue_on+7
    end = response_on
    ########## exc ###################
    # '''
    y_exc = 0.1 + np.max(pfc_HL_mean_A_plus[cue_on:cue_off, cell_idx_exc])
    for context in np.array(['A', 'A2B']):

        fig1 = plt.figure(figsize=(2, 1.5))
        ax1 = fig1.add_axes([0.2, 0.2, 0.7, 0.7])
        for i in np.array(cell_idx_exc):

            if context == 'A':
                ax1.plot(pfc_HL_mean_A_plus[start:end, i], c='black',label=str(i))
                ax1.plot(pfc_HL_mean_A_minus[start:end, i], c='grey',label=str(i))
                ax1.set_title('A_integrated', fontsize=font)
            if context == 'A2B':
                ax1.plot(pfc_HL_mean_A2B_plus[start:end, i], c='black',label=str(i))
                ax1.plot(pfc_HL_mean_A2B_minus[start:end, i], c='grey',label=str(i))
                ax1.set_title('A2B_integrated', fontsize=font)

        ax1.set_ylim([0, y_exc])

        #plt.xticks([0,60], fontsize=10)
        plt.xticks([], fontsize=10)
        plt.yticks([], fontsize=10)
        #plt.yticks([0.5, 0.8, 1.1, 1.4, 1.7], fontsize=12)

        ax1.spines[['left','right', 'top']].set_visible(False)



        ax1.axvspan(cue_off - start+1, cue_off - start+1, color='grey', label='cue_off')
        #ax1.axvspan(stim_on - start+1, stim_on - start+1, color='grey', label='stim_on')


        plt.savefig(figure_path + 'cue_integrated_'+model_name + '_' + str(idx) +'_' + context + '.png')
        plt.savefig(figure_path + 'cue_integrated_' + model_name + '_' + str(idx) + '_' + context + '.eps',format = 'eps', dpi = 1000)
        plt.show()






def encode_error_MD_all_panel(figure_path, model_name, idx, hp,epoch,context_model_1, context_model_2_error,context_model_2_right):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    coh_RDM=0.92;coh_HL=hp['p_coh']

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200



    pfc_HL_vis_1,MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=context_model_1,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_1, cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2_error, cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2_error, cue=1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_3, MD_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2_right, cue=1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_3, MD_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=context_model_2_right, cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    MD_vis = [MD_HL_vis_1,MD_HL_vis_2,MD_HL_vis_3]
    MD_aud = [MD_HL_aud_1, MD_HL_aud_2, MD_HL_aud_3]
    np.save(figure_path+'MD_HL_vis.npy',MD_vis)
    np.save(figure_path + 'MD_HL_aud.npy', MD_aud)

    MD_vis=np.load(figure_path+'MD_HL_vis.npy')
    MD_aud = np.load(figure_path + 'MD_HL_aud.npy')




    font=12
    start=0
    end = response_on-2
    y_exc = 1.5#np.max(MD_vis[0][cue_on:cue_off, :])
    ########## exc ###################

    for panel in range(int(20/4)):
        fig, axs = plt.subplots(3, 4,figsize=(11,7))
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0.5)
        fig.suptitle('MD  :'+model_name+'_'+str(idx)+'_'+str(coh_HL))

        i_dx=0
        for j in range(4):
            axs[0, j].plot(MD_vis[0][cue_on + start:end, j+4*panel],color='r',label='vis')
            axs[0, j].plot(MD_aud[0][cue_on + start:end, j+4*panel], color='g', label='aud')

            axs[1, j].plot(MD_vis[1][cue_on + start:end, j+4*panel],color='r',label='con_B')
            axs[1, j].plot(MD_aud[1][cue_on + start:end, j+4*panel], color='g')


            axs[2, j].plot(MD_vis[2][cue_on + start:end, j+4*panel], color='r',label='con_AA')
            axs[2, j].plot(MD_aud[2][cue_on + start:end, j+4*panel], color='g')


        for j in range(4):
            axs[0, j].set_title('con_A:' + str(j + 4 * panel), fontsize=font)
            axs[0, j].set_title('con_A:' + str(j + 4 * panel), fontsize=font)
            axs[1, 0].set_title('con_B', fontsize=font)
            axs[2, 0].set_title('con_AA', fontsize=font)
            axs[0, j].legend(fontsize=8)



        for i in range(3):
            for j in range(4):
                #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
                axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
                axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')
                axs[i, j].set_ylim([0, y_exc])
        #
        plt.savefig(figure_path+'_'+str(idx)+'_'+str(panel)+'_'+str(coh_HL)+'.png')
        plt.show()



def plot_activity_pfc_encode_error(figure_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=100


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)


    cell_idx_exc = range(30,100,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,20,1)#max_idx_md1[0:20]
    cell_idx_md2 = range(n_md1,n_md1+5,1)#max_idx_md2+n_md1
    y_exc = 1.#*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 2.#*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 2.#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_RDM)+'_'+str(coh_HL))



    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('md1: '+'aud',fontsize=font)
        axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    axs[0, 0].set_ylim([0, y_md1])
    axs[0, 1].set_ylim([0, y_md1])
    axs[1, 0].set_ylim([0, y_md2])
    axs[1, 1].set_ylim([0, y_md2])


    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()




def MD_encode_error_all(figure_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1



    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)


    cell_idx_exc = range(30,100,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 = range(n_md1,n_md1+5,1)#max_idx_md2+n_md1
    y_exc = 1.#*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.#*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 2.#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6,5))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx))



    for i in np.array(cell_idx_md1):
        axs[0,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('md1: '+'aud',fontsize=font)
        axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    axs[0, 0].set_ylim([0, y_md1])
    axs[0, 1].set_ylim([0, y_md1])
    axs[1, 0].set_ylim([0, y_md2])
    axs[1, 1].set_ylim([0, y_md2])


    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+'.png')
    plt.show()
def MD_encode_error_example(figure_path,model_name,model_dir,idx,hp,context_name,epoch):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_RDM=0.92;coh_HL=0.92

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1


    model_dir_0 = model_dir+str(idx[0])
    model_dir_1 = model_dir + str(idx[1])
    model_dir_2 = model_dir + str(idx[2])


    pfc_mean_HL_vis_0,MD_mean_HL_vis_0 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir_0,
                                                        cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud_0,MD_mean_HL_aud_0 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir_0,
                                                        cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)

    pfc_mean_HL_vis_1, MD_mean_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,model_dir=model_dir_1,
                                                        cue=attend_vis, p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud_1, MD_mean_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,model_dir=model_dir_1,
                                                        cue=attend_aud, p_cohs=coh_HL,batch_size=batch_size)

    pfc_mean_HL_vis_2, MD_mean_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,model_dir=model_dir_2,
                                                        cue=attend_vis, p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud_2, MD_mean_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,model_dir=model_dir_2,
                                                        cue=attend_aud, p_cohs=coh_HL,batch_size=batch_size)

    np.save(figure_path + 'MD_mean_HL_vis_0.npy', MD_mean_HL_vis_0)
    np.save(figure_path + 'MD_mean_HL_vis_1.npy', MD_mean_HL_vis_1)
    np.save(figure_path + 'MD_mean_HL_vis_2.npy', MD_mean_HL_vis_2)

    np.save(figure_path + 'MD_mean_HL_aud_0.npy', MD_mean_HL_aud_0)
    np.save(figure_path + 'MD_mean_HL_aud_1.npy', MD_mean_HL_aud_1)
    np.save(figure_path + 'MD_mean_HL_aud_2.npy', MD_mean_HL_aud_2)

    MD_mean_HL_vis_0 = np.load(figure_path + 'MD_mean_HL_vis_0.npy')
    MD_mean_HL_vis_1 = np.load(figure_path + 'MD_mean_HL_vis_1.npy')
    MD_mean_HL_vis_2 = np.load(figure_path + 'MD_mean_HL_vis_2.npy')

    MD_mean_HL_aud_0 = np.load(figure_path + 'MD_mean_HL_aud_0.npy')
    MD_mean_HL_aud_1 = np.load(figure_path + 'MD_mean_HL_aud_1.npy')
    MD_mean_HL_aud_2 = np.load(figure_path + 'MD_mean_HL_aud_2.npy')



    y_md1 = 2.1#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])
    y_low=0.8; y_high=2.2



    font=10
    start=1
    end = stim_on-2

    cell=9
    for context in np.array(['context1', 'context2','context11']):
        fig1 = plt.figure(figsize=(2, 1.5))
        ax1 = fig1.add_axes([0.2, 0.2, 0.7, 0.7])

        if context == 'context1':
            plt.plot(MD_mean_HL_aud_0[cue_on + start:end, cell], label='1')
        if context == 'context2':
            plt.plot(MD_mean_HL_aud_1[cue_on + start:end, cell], label='2')
        if context == 'context11':
            plt.plot(MD_mean_HL_aud_2[cue_on + start:end, cell], label='11')

        plt.xticks([], fontsize=10)
        plt.yticks([], fontsize=10)


        ax1.set_ylim([y_low, y_high])
        ax1.axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey')
        #ax1.axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey')
        ax1.spines[['left','right', 'top']].set_visible(False)
        plt.savefig(figure_path + model_name + '_MD_encode_error_' + context + '.png')
        plt.savefig(figure_path + model_name + '_MD_encode_error_' + context + '.eps', format='eps', dpi=1000)
        plt.show()

    # ########## exc ###################
    # cell_idx_md1 = range(0, 10, 1)  # max_idx_md1[0:20]
    # cell_idx_md2 = range(n_md1, n_md1 + 5, 1)  # max_idx_md2+n_md1
    # #'''
    #
    # for rule in np.array(['vis','aud']):
    #     fig, axs = plt.subplots(1, 3,figsize=(10,3))
    #     plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    #     fig.suptitle('MD  :'+model_name+'_'+str(idx))
    #
    #     if rule== 'vis':
    #
    #         for i in np.array(cell_idx_md1):
    #             axs[0].plot(MD_mean_HL_vis_0[cue_on+start:end,i],label=str(i))
    #             axs[1].plot(MD_mean_HL_vis_1[cue_on + start:end, i], label=str(i))
    #             axs[2].plot(MD_mean_HL_vis_2[cue_on + start:end, i], label=str(i))
    #
    #     elif rule == 'aud':
    #         for i in np.array(cell_idx_md1):
    #             axs[0].plot(MD_mean_HL_aud_0[cue_on + start:end, i], label=str(i))
    #             axs[1].plot(MD_mean_HL_aud_1[cue_on + start:end, i], label=str(i))
    #             axs[2].plot(MD_mean_HL_aud_2[cue_on + start:end, i], label=str(i))
    #
    #     for j in range(3):
    #         axs[j].set_ylim([0, y_md1])
    #         axs[j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey')
    #         axs[j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey')
    #
    #     plt.legend(fontsize=5)
    #     plt.savefig(figure_path+model_name+'_'+str(idx)+'.png')
    #     plt.show()
    #
    #














def plot_activity_all_8panel_diff_uncertainty(figure_path,model_name,model_dir,idx,hp,context_name,epoch,coh_HL):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_HL=coh_HL

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200


    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1
    if context_name =='con_B2A':
        attend_vis=1
        attend_aud=-1


    pfc_mean_HL_vis,MD_mean_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud,MD_mean_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)


    cell_idx_exc = range(30,100,1)#max_idx_exc[0:20]
    cell_idx_inh = range(n_exc,n_exc+20,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,18,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+10,1)#max_idx_md2+n_md1
    y_exc = 1.6#*np.max(pfc_mean_HL_aud[cue_on:cue_off,0:n_exc])
    y_inh = 1.6#*np.max(pfc_mean_HL_aud[cue_on:cue_off,n_exc:n_rnn])
    y_md1 = 2.#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.5#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])



    font=10
    start=0
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(4, 2,figsize=(6,10))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_HL))


    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('exc: '+'vis',fontsize=font)
        #axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('exc: '+'aud',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,0].plot(MD_mean_HL_vis[cue_on+start:end,i],label=str(i))
        axs[3,0].set_title('md2: '+'vis',fontsize=font)
        #axs2[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[3,1].plot(MD_mean_HL_aud[cue_on+start:end,i],label=str(i))
        axs[3,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    axs[0, 0].set_ylim([0, y_exc])
    axs[0, 1].set_ylim([0, y_exc])
    axs[1, 0].set_ylim([0, y_inh])
    axs[1, 1].set_ylim([0, y_inh])
    axs[2, 0].set_ylim([0, y_md1])
    axs[2, 1].set_ylim([0, y_md1])
    axs[3, 0].set_ylim([0, y_md2])
    axs[3, 1].set_ylim([0, y_md2])


    for i in range(4):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()
def MD_encode_uncertainty(figure_path,model_name,model_dir,idx,hp,context_name,epoch,coh_HL):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_HL=coh_HL

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1
    if context_name =='con_B2A':
        attend_vis=1
        attend_aud=-1

    pfc_HL_vis,MD_HL_vis = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud,MD_HL_aud = get_neurons_activity_mode_test1(
        context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL,batch_size=batch_size)

    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    cell_idx_md2 =  range(n_md1,n_md1+10,1)#max_idx_md2+n_md1

    y_md1 = 2.#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.5#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])


    font=10
    start=cue_on-1
    end = response_on-2
    ########## exc ###################
    #'''

    fig, axs = plt.subplots(2, 2,figsize=(6,4))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('pfc  :'+model_name+'_'+str(idx)+'; coh_'+str(coh_HL))


    for i in np.array(cell_idx_md1):
        axs[0,0].plot(MD_HL_vis[start:end,i],label=str(i))
        axs[0,0].set_title('md1: '+'vis',fontsize=font)
        # axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0,1].plot(MD_HL_aud[start:end,i],label=str(i))
        axs[0,1].set_title('md1: '+'aud',fontsize=font)
        axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,0].plot(MD_HL_vis[start:end,i],label=str(i))
        axs[1,0].set_title('md2: '+'vis',fontsize=font)
        # axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_md2):
        axs[1,1].plot(MD_HL_aud[start:end,i],label=str(i))
        axs[1,1].set_title('md2: '+'aud',fontsize=font)
        #axs2[1,1].legend(fontsize=5)

    axs[0, 0].set_ylim([0, y_md1])
    axs[0, 1].set_ylim([0, y_md1])
    axs[1, 0].set_ylim([0, y_md2])
    axs[1, 1].set_ylim([0, y_md2])


    for i in range(2):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-1-start, cue_off-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-1-start, stim_on-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'_'+str(idx)+context_name+'.png')
    plt.show()
def MD_encode_uncertainty_example(figure_path,model_name,model_dir,idx,hp,context_name,epoch,coh_HL):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    coh_HL=coh_HL

    n_rnn = hp['n_rnn']
    n_exc = int(n_rnn*0.8)
    n_inh = n_rnn-n_exc
    n_md = hp['n_md']
    p_md1=0.6
    n_md1=int(n_md*p_md1)
    n_md2=int(n_md)-n_md1

    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=200

    if context_name =='con_A':
        attend_vis=1
        attend_aud=-1
    if context_name =='con_A2B':
        attend_vis=-1
        attend_aud=1
    if context_name =='con_B2A':
        attend_vis=1
        attend_aud=-1

    # pfc_HL_vis_0,MD_HL_vis_0 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_vis,p_cohs=coh_HL[0],batch_size=batch_size)
    # pfc_HL_aud_0,MD_HL_aud_0 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,model_dir=model_dir,cue=attend_aud,p_cohs=coh_HL[0],batch_size=batch_size)
    #
    # pfc_HL_vis_1, MD_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, model_dir=model_dir,cue=attend_vis, p_cohs=coh_HL[1], batch_size=batch_size)
    # pfc_HL_aud_1, MD_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, model_dir=model_dir,cue=attend_aud, p_cohs=coh_HL[1], batch_size=batch_size)
    #
    # pfc_HL_vis_2, MD_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, model_dir=model_dir,cue=attend_vis, p_cohs=coh_HL[2], batch_size=batch_size)
    # pfc_HL_aud_2, MD_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, model_dir=model_dir,cue=attend_aud, p_cohs=coh_HL[2], batch_size=batch_size)
    #
    # MD_vis = [MD_HL_vis_0, MD_HL_vis_1, MD_HL_vis_2]
    # MD_aud = [MD_HL_aud_0, MD_HL_aud_1, MD_HL_aud_2]
    # np.save(figure_path + 'MD_HL_vis.npy', MD_vis)
    # np.save(figure_path + 'MD_HL_aud.npy', MD_aud)

    MD_vis = np.load(figure_path + 'MD_HL_vis.npy')
    MD_aud = np.load(figure_path + 'MD_HL_aud.npy')


    y_md1 = 1.1#*np.max(MD_mean_HL_vis[cue_on:cue_off,0:n_md1])
    y_md2 = 1.5#1.*np.max(MD_mean_HL_vis[cue_on:cue_off,n_md1:n_md])


    font=10
    start=cue_on-1
    end = response_on-2

    ########## exc ###################
    colors = sns.color_palette('muted')
    for j in np.array([2]):
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])


        ax.plot(MD_vis[0][start:end, j],color=colors[0],label='0.5')
        ax.plot(MD_vis[1][start:end, j],color=colors[1],label='0.6')
        ax.plot(MD_vis[2][start:end, j],color=colors[2],label='0.9')
        plt.legend(fontsize=8)


        ax.axvspan(cue_off-start-1, cue_off-start-1, color='grey')
        #ax.axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey')
        ax.set_ylim([0.3, y_md1])

        plt.xticks([], fontsize=10)
        plt.yticks([], fontsize=10)

        # plt.yticks([0.5, 0.8, 1.1, 1.4, 1.7], fontsize=12)

        ax.spines[['left', 'right', 'top']].set_visible(False)

        #
        plt.savefig(figure_path+'MD_'+str(idx)+'_'+str(coh_HL)+'_example.png')
        plt.savefig(figure_path+'MD_'+str(idx)+'_'+str(coh_HL)+ '_example.eps',format = 'eps', dpi = 1000)

        plt.show()


def scatters_MD_encode_uncertainty(figure_path,hp,epoch,idx,period,model_dir_A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

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


    # rnn_vis_high,md_vis_high = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
    #                                                  context_name='HL_task',cue=1, p_cohs=0.5,batch_size=batch_size)
    # fr_rnn_aud_high,md_aud_high = activity_diff_context(model_dir=model_dir_A,hp=hp,start=start,end=end,
    #                                                  context_name='HL_task',cue=-1, p_cohs=0.5,batch_size=batch_size)
    #
    # rnn_vis_low, md_vis_low = activity_diff_context(model_dir=model_dir_A, hp=hp, start=start, end=end,
    #                                                         context_name='HL_task', cue=1, p_cohs=0.99,
    #                                                         batch_size=batch_size)
    # rnn_aud_low, md_aud_low = activity_diff_context(model_dir=model_dir_A, hp=hp, start=start, end=end,
    #                                                         context_name='HL_task', cue=-1, p_cohs=0.99,
    #                                                         batch_size=batch_size)





    # np.save(figure_path+'md_vis_high.npy',md_vis_high)
    # np.save(figure_path + 'md_aud_high.npy', md_aud_high)
    # np.save(figure_path + 'md_vis_low.npy', md_vis_low)
    # np.save(figure_path + 'md_aud_low.npy', md_aud_low)

    md_vis_high = np.load(figure_path+'md_vis_high.npy')
    md_aud_high = np.load(figure_path + 'md_aud_high.npy')
    md_vis_low = np.load(figure_path+'md_vis_low.npy')
    md_aud_low = np.load(figure_path + 'md_aud_low.npy')


    print(np.argsort(md_vis_high))
    max = 1.5  # np.max(fr_rnn_vis_A)
    min = 0.35
    print('max', max)

    cell_type = [10, 23, 27,  16, 24,  11]#np.argsort(fr_md_vis_A)[24:30]#[1,2,10,11,16,26]
    print('cell_type',cell_type)
    s = 50
    idx_md = 30
    #colors_1 = sns.color_palette("Set2")
    colors_1 = ['#9B30FF', '#FF34B3', '#7CFC00', 'green', '#FF4500', '#FF6A6A']
    #colors_1 = ['r', 'orange', 'y', 'green', 'b', 'purple']
    # fig2

    fig1 = plt.figure(figsize=(2.7, 2.7))
    ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])

    plt.scatter(md_vis_high[0:20],md_vis_low[0:20], marker="o", s=s, color='#4682B4',edgecolors='white',label='MD1')
    plt.scatter(md_vis_high[20:30], md_vis_low[20:30], marker="o", s=s, color='orange', edgecolors='white',label='MD2')
    ax1.set_xlabel('0.5', fontsize=12)
    ax1.set_ylabel('0.9', fontsize=12)
    plt.legend()

    ax1.set_xlim([min, max])
    ax1.set_ylim([min, max])
    plt.xticks([0.5, 0.8,1.1,1.4], fontsize=12)
    plt.yticks([0.5, 0.8,1.1,1.4], fontsize=12)

    j = -1
    # for i in np.array(cell_type):
    #     j += 1
    #     plt.scatter(md_vis_high[i],md_vis_low[i], marker="X", s=40, c=colors_1[j], label=str(i))
    #     # ax1.set_xlabel('con_AA', fontsize=12)
    #     # ax1.set_ylabel('con_B', fontsize=12)

    plt.plot([0, max * 0.9], [0, max * 0.9], color='silver')

    # set x-label
    ax1.spines[['right', 'top']].set_visible(False)
    #plt.title( model_name + '_' + str(idx)+':'+period,fontsize=6)
    plt.savefig(figure_path + 'scatter_MD_' + str(idx)  + '.png')
    plt.savefig(figure_path + 'scatter_MD_' + str(idx) + '.eps', format='eps', dpi=1000)

    plt.show()






























