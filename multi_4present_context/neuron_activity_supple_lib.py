"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""


from __future__ import division

import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from matplotlib import pyplot as plt
import run
import tools
import pdb
from matplotlib import cm
import seaborn as sns
c_perf = sns.color_palette("hls", 8)#muted


def get_epoch(model_dir,hp):
    model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
                 '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
                 '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])

    local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(1))
    model_dir = hp['root_path']+local_folder_name+'/'

    runnerObj = run.Runner(rule_name='HL_task', hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')
    trial_input, run_result = runnerObj.run(batch_size=1, c_cue=0, c_vis=0,c_aud=0,p_coh=0)
    cue_ons, cue_offs = trial_input.epochs['cue_stimulus']
    stim1_ons, stim1_offs = trial_input.epochs['stimulus']
    response_ons, response_offs = trial_input.epochs['response']
    cue_on = cue_ons[0];cue_off=cue_offs[0];stim_on=stim1_ons[0];stim_off=stim1_offs[0];response_on=response_ons[0];response_off=response_offs[0]
    print('cue_on,cue_off,stim_on,stim_off,response_on',cue_on,cue_off,stim_on,stim_off,response_on)

    epoch = {'cue_on':cue_on,
             'cue_off':cue_off,
             'stim_on':stim_on,
             'stim_off':stim_off,
             'response_on':response_on,
             'response_off':response_off}

    return epoch





def plot_rule(figure_path,data_path,model_dir, hp, start,end,cue,c_vis,c_aud,coh_RDM,coh_HL):

    fr_rnn_1 = np.load(data_path+'cue'+str(cue)+'_fr_rnn_1.npy')
    fr_md_1  = np.load(data_path+'cue'+str(cue)+'_fr_md_1.npy')
    fr_rnn_2 = np.load(data_path+'cue'+str(cue)+'_fr_rnn_2.npy')
    fr_md_2  = np.load(data_path+'cue'+str(cue)+'_fr_md_2.npy')


    if cue==1:
        rule='VIS'
    elif cue==-1:
        rule='AUD'

    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    plt.scatter(fr_rnn_1[0:205],fr_rnn_2[0:205],marker=">",s=20,color='tab:red',label='PFC-Exc')
    plt.scatter(fr_rnn_1[205:256],fr_rnn_2[205:256],s=20,color='tab:blue',label='PFC-Inh')
    plt.scatter(fr_md_1[100:],fr_md_2[100:],s=20,marker="*",color='tab:orange',label='MD1')
    plt.scatter(fr_md_1[0:100],fr_md_2[0:100],s=20,marker="*",color='tab:purple',label='MD2')
    max =1.7

    plt.plot([0,max*0.9],[0,max*0.9],color='grey')
    plt.xlim([0,max])
    plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=15)
    plt.yticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=15)
    # plt.xlabel('HL context',fontsize=12)
    # plt.ylabel('RDM context',fontsize=12)
    #plt.title(rule,fontsize=8)
    #plt.legend(fontsize=10)
    fig.savefig(figure_path+rule+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)

    plt.savefig(figure_path+rule+'_'+str(start)+'_'+str(end))
    plt.show()
def plot_context(figure_path,data_path,model_dir, hp, start,end,context_name,p_cohs):


    fr_rnn_1 = np.load(data_path+context_name+'_fr_rnn_1.npy')
    fr_md_1  = np.load(data_path+context_name+'_fr_md_1.npy')
    fr_rnn_2 = np.load(data_path+context_name+'_fr_rnn_2.npy')
    fr_md_2  = np.load(data_path+context_name+'_fr_md_2.npy')



    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    plt.scatter(fr_rnn_1[0:205],fr_rnn_2[0:205],marker=">",s=20,color='tab:red',label='PFC-Exc')
    plt.scatter(fr_rnn_1[205:256],fr_rnn_2[205:256],s=20,color='tab:blue',label='PFC-Inh')
    plt.scatter(fr_md_1[100:],fr_md_2[100:],s=20,marker="*",color='tab:orange',label='MD1')
    plt.scatter(fr_md_1[0:100],fr_md_2[0:100],s=20,marker="*",color='tab:purple',label='MD2')
    max =1.7#np.max(fr_rnn_1)*( np.max(fr_rnn_1)>np.max(fr_rnn_2))+np.max(fr_rnn_2)*( np.max(fr_rnn_2)>np.max(fr_rnn_1))

    plt.plot([0,max*0.9],[0,max*0.9],color='grey')
    plt.xlim([0,max])
    plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=15)
    plt.yticks([0.0, 0.4, 0.8, 1.2,1.6], fontsize=15)
    # plt.xlabel('Attend to vision',fontsize=12)
    # plt.ylabel('Attend to audition',fontsize=12)
    #plt.title(context_name,fontsize=8)
    plt.legend(fontsize=10)
    fig.savefig(figure_path+context_name+'_'+str(start)+'_'+str(end)+'.eps', format='eps', dpi=1000)
    plt.savefig(figure_path+context_name+'_'+str(start)+'_'+str(end))
    plt.show()
def plot_exc_units_panel4(data_path,figure_path,model_name,model_dir,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=10
    coh_RDM = 0.9
    coh_HL = 0.85

    start = cue_on
    end = stim_on - 2


    RDM_pfc_viss = np.load(data_path+'perfVis_RDM_pfc_viss.npy')
    RDM_pfc_auds = np.load(data_path+'perfVis_RDM_pfc_auds.npy')
    HL_pfc_viss = np.load(data_path+'perfVis_HL_pfc_viss.npy')
    HL_pfc_auds = np.load(data_path+'perfVis_HL_pfc_auds.npy')


    RDM_pfc_vis_mean = np.mean(RDM_pfc_viss, axis=0)
    RDM_pfc_vis_sem = np.std(RDM_pfc_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_pfc_aud_mean = np.mean(RDM_pfc_auds, axis=0)
    RDM_pfc_aud_sem = np.std(RDM_pfc_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_pfc_vis_mean = np.mean(HL_pfc_viss, axis=0)
    HL_pfc_vis_sem = np.std(HL_pfc_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_pfc_aud_mean = np.mean(HL_pfc_auds, axis=0)
    HL_pfc_aud_sem = np.std(HL_pfc_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))



    ############ prefer vis ############
    #'''
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_pfc_vis_mean.shape[0]
    lw=1.7
    i_RDM = 176#55

    axs[0, 0].plot(HL_pfc_vis_mean[:, i_RDM], '-', linewidth=lw, color=c_perf[0], label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_vis_mean[:, i_RDM] - HL_pfc_vis_sem[:, i_RDM],
                           HL_pfc_vis_mean[:, i_RDM] + HL_pfc_vis_sem[:, i_RDM], color='pink')

    axs[1, 0].plot(HL_pfc_aud_mean[:, i_RDM], '-', linewidth=lw, color='grey', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_aud_mean[:, i_RDM] - HL_pfc_aud_sem[:, i_RDM],
                           HL_pfc_aud_mean[:, i_RDM] + HL_pfc_aud_sem[:, i_RDM], color='lightgrey')

    axs[0, 1].plot(RDM_pfc_vis_mean[:, i_RDM], '-', linewidth=lw, color=c_perf[0], label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_pfc_vis_mean[:, i_RDM] - RDM_pfc_vis_sem[:, i_RDM],
                           RDM_pfc_vis_mean[:, i_RDM] + RDM_pfc_vis_sem[:, i_RDM], color='pink')

    axs[1, 1].plot(RDM_pfc_aud_mean[:, i_RDM], '-', linewidth=lw, color='grey', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_pfc_aud_mean[:, i_RDM] - RDM_pfc_aud_sem[:, i_RDM],
                           RDM_pfc_aud_mean[:, i_RDM] + RDM_pfc_aud_sem[:, i_RDM], color='lightgrey')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 0.9
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.06, max])
            axs[1, 0].set_ylim([0.06, 1.1])
            axs[1, 1].set_ylim([0.06, 1.1])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'pfc_select_vis_context.eps', format='eps', dpi=1000)
    plt.show()
    #'''


    ############ prefer aud ############

    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_pfc_vis_mean.shape[0]
    i_RDM = 66

    axs[0, 0].plot(HL_pfc_vis_mean[:, i_RDM], '-', linewidth=lw, color='grey', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_vis_mean[:, i_RDM] - HL_pfc_vis_sem[:, i_RDM],
                           HL_pfc_vis_mean[:, i_RDM] + HL_pfc_vis_sem[:, i_RDM], color='lightgrey')

    axs[1, 0].plot(HL_pfc_aud_mean[:, i_RDM], '-', linewidth=lw, color=c_perf[5], label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_aud_mean[:, i_RDM] - HL_pfc_aud_sem[:, i_RDM],
                           HL_pfc_aud_mean[:, i_RDM] + HL_pfc_aud_sem[:, i_RDM], color='lightsteelblue')

    axs[0, 1].plot(RDM_pfc_vis_mean[:, i_RDM], '-', linewidth=lw, color='grey', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_pfc_vis_mean[:, i_RDM] - RDM_pfc_vis_sem[:, i_RDM],
                           RDM_pfc_vis_mean[:, i_RDM] + RDM_pfc_vis_sem[:, i_RDM], color='lightgrey')

    axs[1, 1].plot(RDM_pfc_aud_mean[:, i_RDM], '-', linewidth=lw, color=c_perf[5], label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),
                           RDM_pfc_aud_mean[:, i_RDM] - RDM_pfc_aud_sem[:, i_RDM],
                           RDM_pfc_aud_mean[:, i_RDM] + RDM_pfc_aud_sem[:, i_RDM], color='lightsteelblue')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 1.15
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.15, max])
            axs[0, 0].set_ylim([0.15, 1.6])
            axs[0, 1].set_ylim([0.15, 1.6])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'pfc_select_aud_context.eps', format='eps', dpi=1000)
    plt.show()
def plot_exc_units_panel4_supple(data_path,figure_path,model_name,model_dir,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=10
    coh_RDM = 0.9
    coh_HL = 0.85

    start = cue_on
    end = stim_on - 2


    RDM_pfc_viss = np.load(data_path+'perfVis_RDM_pfc_viss.npy')
    RDM_pfc_auds = np.load(data_path+'perfVis_RDM_pfc_auds.npy')
    HL_pfc_viss = np.load(data_path+'perfVis_HL_pfc_viss.npy')
    HL_pfc_auds = np.load(data_path+'perfVis_HL_pfc_auds.npy')


    RDM_pfc_vis_mean = np.mean(RDM_pfc_viss, axis=0)
    RDM_pfc_vis_sem = np.std(RDM_pfc_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_pfc_aud_mean = np.mean(RDM_pfc_auds, axis=0)
    RDM_pfc_aud_sem = np.std(RDM_pfc_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_pfc_vis_mean = np.mean(HL_pfc_viss, axis=0)
    HL_pfc_vis_sem = np.std(HL_pfc_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_pfc_aud_mean = np.mean(HL_pfc_auds, axis=0)
    HL_pfc_aud_sem = np.std(HL_pfc_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))



    ############ prefer vis ############
    #'''
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_pfc_vis_mean.shape[0]
    lw=1.7
    i_vis = 194#169

    axs[0, 0].plot(HL_pfc_vis_mean[:, i_vis], '-', linewidth=lw, color=c_perf[0], label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_vis_mean[:, i_vis] - HL_pfc_vis_sem[:, i_vis],
                           HL_pfc_vis_mean[:, i_vis] + HL_pfc_vis_sem[:, i_vis], color='pink')

    axs[1, 0].plot(HL_pfc_aud_mean[:, i_vis], '-', linewidth=lw, color='grey', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_aud_mean[:, i_vis] - HL_pfc_aud_sem[:, i_vis],
                           HL_pfc_aud_mean[:, i_vis] + HL_pfc_aud_sem[:, i_vis], color='lightgrey')

    axs[0, 1].plot(RDM_pfc_vis_mean[:, i_vis], '-', linewidth=lw, color=c_perf[0], label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_pfc_vis_mean[:, i_vis] - RDM_pfc_vis_sem[:, i_vis],
                           RDM_pfc_vis_mean[:, i_vis] + RDM_pfc_vis_sem[:, i_vis], color='pink')

    axs[1, 1].plot(RDM_pfc_aud_mean[:, i_vis], '-', linewidth=lw, color='grey', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_pfc_aud_mean[:, i_vis] - RDM_pfc_aud_sem[:, i_vis],
                           RDM_pfc_aud_mean[:, i_vis] + RDM_pfc_aud_sem[:, i_vis], color='lightgrey')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 0.5
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.06, max])
            axs[1, 0].set_ylim([0.06, 0.8])
            axs[1, 1].set_ylim([0.06, 0.8])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'pfc_select_vis_supple.eps', format='eps', dpi=1000)
    plt.show()
    #'''


    ############ prefer aud ############

    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_pfc_vis_mean.shape[0]
    i_aud = 105#105,142

    axs[0, 0].plot(HL_pfc_vis_mean[:, i_aud], '-', linewidth=lw, color='grey', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_vis_mean[:, i_aud] - HL_pfc_vis_sem[:, i_aud],
                           HL_pfc_vis_mean[:, i_aud] + HL_pfc_vis_sem[:, i_aud], color='lightgrey')

    axs[1, 0].plot(HL_pfc_aud_mean[:, i_aud], '-', linewidth=lw, color=c_perf[5], label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_pfc_aud_mean[:, i_aud] - HL_pfc_aud_sem[:, i_aud],
                           HL_pfc_aud_mean[:, i_aud] + HL_pfc_aud_sem[:, i_aud], color='lightsteelblue')

    axs[0, 1].plot(RDM_pfc_vis_mean[:, i_aud], '-', linewidth=lw, color='grey', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_pfc_vis_mean[:, i_aud] - RDM_pfc_vis_sem[:, i_aud],
                           RDM_pfc_vis_mean[:, i_aud] + RDM_pfc_vis_sem[:, i_aud], color='lightgrey')

    axs[1, 1].plot(RDM_pfc_aud_mean[:, i_aud], '-', linewidth=lw, color=c_perf[5], label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),
                           RDM_pfc_aud_mean[:, i_aud] - RDM_pfc_aud_sem[:, i_aud],
                           RDM_pfc_aud_mean[:, i_aud] + RDM_pfc_aud_sem[:, i_aud], color='lightsteelblue')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 0.3
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.05, max])
            axs[0, 0].set_ylim([0.05, 0.5])
            axs[0, 1].set_ylim([0.05, 0.5])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'pfc_select_aud_supple.eps', format='eps', dpi=1000)
    plt.show()

def plot_MD2_units_pref_panel4(data_path,figure_path,model_name,model_dir,idx,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=100
    start = cue_on
    end = stim_on - 2

    coh_RDM = coh_RDM
    coh_HL = coh_HL


    RDM_MD_viss = np.load(data_path+'perfRDM_RDM_MD_viss.npy')
    RDM_MD_auds = np.load(data_path+'perfRDM_RDM_MD_auds.npy')
    HL_MD_viss = np.load(data_path+'perfRDM_HL_MD_viss.npy')
    HL_MD_auds = np.load(data_path+'perfRDM_HL_MD_auds.npy')


    RDM_MD_vis_mean = np.mean(RDM_MD_viss, axis=0)
    RDM_MD_vis_sem = np.std(RDM_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_MD_aud_mean = np.mean(RDM_MD_auds, axis=0)
    RDM_MD_aud_sem = np.std(RDM_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_MD_vis_mean = np.mean(HL_MD_viss, axis=0)
    HL_MD_vis_sem = np.std(HL_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_MD_aud_mean = np.mean(HL_MD_auds, axis=0)
    HL_MD_aud_sem = np.std(HL_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    lw = 1.7
    # ============================ prefer to RDM ============================
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 5

    axs[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot),HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')


    axs[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 1.5
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.3, max])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'MD2_select_RDM_context.eps', format='eps', dpi=1000)
    plt.show()

    # ============================ prefer to HL ============================
    fig1, axs1 = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 98

    axs1[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs1[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs1[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs1[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs1[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs1[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')

    axs1[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs1[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    max = 1.5
    for i in range(2):
        for j in range(2):
            axs1[i, j].set_ylim([0.3, max])
            axs1[i, j].set_xticks([])
            axs1[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs1[i, j].spines['top'].set_visible(False)
            axs1[i, j].spines['right'].set_visible(False)
            axs1[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs1[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig1.savefig(figure_path + 'MD2_select_HL_context.eps', format='eps', dpi=1000)
    plt.show()
def plot_MD2_units_pref_panel4_supple(data_path,figure_path,model_name,model_dir,idx,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=100
    start = cue_on
    end = stim_on - 2

    coh_RDM = coh_RDM
    coh_HL = coh_HL

    RDM_MD_viss = np.load(data_path+'perfRDM_RDM_MD_viss.npy')
    RDM_MD_auds = np.load(data_path+'perfRDM_RDM_MD_auds.npy')
    HL_MD_viss = np.load(data_path+'perfRDM_HL_MD_viss.npy')
    HL_MD_auds = np.load(data_path+'perfRDM_HL_MD_auds.npy')


    RDM_MD_vis_mean = np.mean(RDM_MD_viss, axis=0)
    RDM_MD_vis_sem = np.std(RDM_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_MD_aud_mean = np.mean(RDM_MD_auds, axis=0)
    RDM_MD_aud_sem = np.std(RDM_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_MD_vis_mean = np.mean(HL_MD_viss, axis=0)
    HL_MD_vis_sem = np.std(HL_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_MD_aud_mean = np.mean(HL_MD_auds, axis=0)
    HL_MD_aud_sem = np.std(HL_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    lw = 1.7
    # ============================ prefer to RDM ============================
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 83

    axs[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot),HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')


    axs[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 1.2
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.3, max])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'MD2_select_RDM_supple.eps', format='eps', dpi=1000)
    plt.show()

    # ============================ prefer to HL ============================
    fig1, axs1 = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 87

    axs1[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs1[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs1[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs1[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs1[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs1[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')

    axs1[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs1[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    max = 1.2
    for i in range(2):
        for j in range(2):
            axs1[i, j].set_ylim([0.3, max])
            axs1[i, j].set_xticks([])
            axs1[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs1[i, j].spines['top'].set_visible(False)
            axs1[i, j].spines['right'].set_visible(False)
            axs1[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs1[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig1.savefig(figure_path + 'MD2_select_HL_supple.eps', format='eps', dpi=1000)
    plt.show()

def plot_MD1_units_pref_panel4(data_path,figure_path,model_name,model_dir,idx,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=100
    start = cue_on
    end = stim_on - 2

    coh_RDM = coh_RDM
    coh_HL = coh_HL



    RDM_MD_viss = np.load(data_path+'perfRDM_RDM_MD_viss.npy')
    RDM_MD_auds = np.load(data_path+'perfRDM_RDM_MD_auds.npy')
    HL_MD_viss = np.load(data_path+'perfRDM_HL_MD_viss.npy')
    HL_MD_auds = np.load(data_path+'perfRDM_HL_MD_auds.npy')


    RDM_MD_vis_mean = np.mean(RDM_MD_viss, axis=0)
    RDM_MD_vis_sem = np.std(RDM_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_MD_aud_mean = np.mean(RDM_MD_auds, axis=0)
    RDM_MD_aud_sem = np.std(RDM_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_MD_vis_mean = np.mean(HL_MD_viss, axis=0)
    HL_MD_vis_sem = np.std(HL_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_MD_aud_mean = np.mean(HL_MD_auds, axis=0)
    HL_MD_aud_sem = np.std(HL_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    lw = 1.7
    # ============================ prefer to RDM ============================
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_RDM = 153

    axs[0, 0].plot(HL_MD_vis_mean[:, i_RDM], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot),HL_MD_vis_mean[:, i_RDM] - HL_MD_vis_sem[:, i_RDM],
                           HL_MD_vis_mean[:, i_RDM] + HL_MD_vis_sem[:, i_RDM], color='#FFD39B')

    axs[1, 0].plot(HL_MD_aud_mean[:, i_RDM], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_RDM] - HL_MD_aud_sem[:, i_RDM],
                           HL_MD_aud_mean[:, i_RDM] + HL_MD_aud_sem[:, i_RDM], color='#FFD39B')

    axs[0, 1].plot(RDM_MD_vis_mean[:, i_RDM], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_vis_mean[:, i_RDM] - RDM_MD_vis_sem[:, i_RDM],
                           RDM_MD_vis_mean[:, i_RDM] + RDM_MD_vis_sem[:, i_RDM], color='#CDBA96')


    axs[1, 1].plot(RDM_MD_aud_mean[:, i_RDM], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_aud_mean[:, i_RDM] - RDM_MD_aud_sem[:, i_RDM],
                           RDM_MD_aud_mean[:, i_RDM] + RDM_MD_aud_sem[:, i_RDM], color='#CDBA96')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 2
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.3, max])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'MD1_select_RDM_context.eps', format='eps', dpi=1000)
    plt.show()

    # ============================ prefer to HL ============================
    fig1, axs1 = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 163

    axs1[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs1[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs1[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs1[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs1[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs1[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')

    axs1[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs1[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    max = 1.5
    for i in range(2):
        for j in range(2):
            axs1[i, j].set_ylim([0.1, max])
            axs1[i, j].set_xticks([])
            axs1[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs1[i, j].spines['top'].set_visible(False)
            axs1[i, j].spines['right'].set_visible(False)
            axs1[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs1[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig1.savefig(figure_path + 'MD1_select_HL_context.eps', format='eps', dpi=1000)
    plt.show()
def plot_MD1_units_pref_panel4_supple(data_path,figure_path,model_name,model_dir,idx,hp,c_vis,c_aud,coh_RDM,coh_HL,epoch):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''
    # figure_path = os.path.join(figure_path, 'pfc/')
    # tools.mkdir_p(figure_path)
    cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']
    batch_size=100
    start = cue_on
    end = stim_on - 2

    coh_RDM = coh_RDM
    coh_HL = coh_HL


    RDM_MD_viss = np.load(data_path+'perfRDM_RDM_MD_viss.npy')
    RDM_MD_auds = np.load(data_path+'perfRDM_RDM_MD_auds.npy')
    HL_MD_viss = np.load(data_path+'perfRDM_HL_MD_viss.npy')
    HL_MD_auds = np.load(data_path+'perfRDM_HL_MD_auds.npy')


    RDM_MD_vis_mean = np.mean(RDM_MD_viss, axis=0)
    RDM_MD_vis_sem = np.std(RDM_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    RDM_MD_aud_mean = np.mean(RDM_MD_auds, axis=0)
    RDM_MD_aud_sem = np.std(RDM_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    HL_MD_vis_mean = np.mean(HL_MD_viss, axis=0)
    HL_MD_vis_sem = np.std(HL_MD_viss, axis=0)#/np.sqrt(len(RDM_pfc_viss))
    HL_MD_aud_mean = np.mean(HL_MD_auds, axis=0)
    HL_MD_aud_sem = np.std(HL_MD_auds, axis=0)#/np.sqrt(len(RDM_pfc_viss))

    lw = 1.7
    # ============================ prefer to RDM ============================
    fig, axs = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_RDM = 107#169,107,188,158,190

    axs[0, 0].plot(HL_MD_vis_mean[:, i_RDM], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs[0, 0].fill_between(np.linspace(0, number_dot, number_dot),HL_MD_vis_mean[:, i_RDM] - HL_MD_vis_sem[:, i_RDM],
                           HL_MD_vis_mean[:, i_RDM] + HL_MD_vis_sem[:, i_RDM], color='#FFD39B')

    axs[1, 0].plot(HL_MD_aud_mean[:, i_RDM], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_RDM] - HL_MD_aud_sem[:, i_RDM],
                           HL_MD_aud_mean[:, i_RDM] + HL_MD_aud_sem[:, i_RDM], color='#FFD39B')

    axs[0, 1].plot(RDM_MD_vis_mean[:, i_RDM], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs[0, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_vis_mean[:, i_RDM] - RDM_MD_vis_sem[:, i_RDM],
                           RDM_MD_vis_mean[:, i_RDM] + RDM_MD_vis_sem[:, i_RDM], color='#CDBA96')


    axs[1, 1].plot(RDM_MD_aud_mean[:, i_RDM], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs[1, 1].fill_between(np.linspace(0, number_dot, number_dot),RDM_MD_aud_mean[:, i_RDM] - RDM_MD_aud_sem[:, i_RDM],
                           RDM_MD_aud_mean[:, i_RDM] + RDM_MD_aud_sem[:, i_RDM], color='#CDBA96')

    title_name = 'AUD'
    # axs[1].set_title(title_name,fontsize=8)
    max = 2.5
    for i in range(2):
        for j in range(2):
            axs[i, j].set_ylim([0.3, max])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig.savefig(figure_path + 'MD1_select_RDM_supple.eps', format='eps', dpi=1000)
    plt.show()

    # ============================ prefer to HL ============================
    fig1, axs1 = plt.subplots(2, 2, figsize=(2, 2.5))
    # fig.suptitle('PFC-Inh perfer to HL context',fontsize=8
    number_dot = RDM_MD_vis_mean.shape[0]
    i_HL = 120#171,120

    axs1[0, 0].plot(HL_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-HL')
    axs1[0, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_vis_mean[:, i_HL] - HL_MD_vis_sem[:, i_HL],
                           HL_MD_vis_mean[:, i_HL] + HL_MD_vis_sem[:, i_HL], color='#FFD39B')

    axs1[1, 0].plot(HL_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:orange', label='perfer-vis')
    axs1[1, 0].fill_between(np.linspace(0, number_dot, number_dot), HL_MD_aud_mean[:, i_HL] - HL_MD_aud_sem[:, i_HL],
                           HL_MD_aud_mean[:, i_HL] + HL_MD_aud_sem[:, i_HL], color='#FFD39B')

    axs1[0, 1].plot(RDM_MD_vis_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-RDM')
    axs1[0, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_vis_mean[:, i_HL] - RDM_MD_vis_sem[:, i_HL],
                           RDM_MD_vis_mean[:, i_HL] + RDM_MD_vis_sem[:, i_HL], color='#CDBA96')

    axs1[1, 1].plot(RDM_MD_aud_mean[:, i_HL], '-', linewidth=lw, color='tab:brown', label='perfer-vis')
    axs1[1, 1].fill_between(np.linspace(0, number_dot, number_dot), RDM_MD_aud_mean[:, i_HL] - RDM_MD_aud_sem[:, i_HL],
                           RDM_MD_aud_mean[:, i_HL] + RDM_MD_aud_sem[:, i_HL], color='#CDBA96')

    max = 1.3
    for i in range(2):
        for j in range(2):
            axs1[i, j].set_ylim([0.1, max])
            axs1[i, j].set_xticks([])
            axs1[i, j].set_yticks([])
            # axs[1].set_title(title_name,fontsize=8)
            # plt.legend()
            axs1[i, j].spines['top'].set_visible(False)
            axs1[i, j].spines['right'].set_visible(False)
            axs1[i, j].spines['left'].set_visible(False)
            # axs[1].spines['bottom'].set_visible(False)
            axs1[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')
    fig1.savefig(figure_path + 'MD1_select_HL_supple.eps', format='eps', dpi=1000)
    plt.show()


