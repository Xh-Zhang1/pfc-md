import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import torch
import seaborn as sns
import matplotlib.colors as colors
import default
import tools
import run

clrs = sns.color_palette("muted")#muted
clrs_fill = sns.color_palette('pastel')
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

def get_neurons_activity_mode_test1(context_name,hp,context,model_dir,cue,p_cohs,batch_size):

    hp['switch_context']=context

    if cue is None:
        cue = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue = hp['rng'].choice([cue], (batch_size,))

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


def activity_all_8panel_diff_context(figure_path,hp,epoch,model_name,model_dir_A,model_dir_A2B,model_dir_B2A):
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
    batch_size=50


    pfc_mean_HL_vis_A,MD_mean_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=model_dir_A,cue=1,p_cohs=coh_HL,batch_size=batch_size)
    pfc_mean_HL_aud_A,MD_mean_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp,
                                model_dir=model_dir_A,cue=-1,p_cohs=coh_HL,batch_size=batch_size)

    pfc_mean_HL_vis_A2B, MD_mean_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_A2B, cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_mean_HL_aud_A2B, MD_mean_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_A2B, cue=1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_mean_HL_vis_B2A, MD_mean_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_B2A, cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_mean_HL_aud_B2A, MD_mean_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
                                model_dir=model_dir_B2A, cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #print(pfc_mean_HL_vis.shape)

    max_idx_exc = np.argsort(np.mean(pfc_mean_HL_vis_A[cue_on:response_on, 0:205], axis=0))
    cell_idx_exc = range(0,10,1)#max_idx_exc[197:205]#max_idx_exc[195:205]#[11,13,15,17,19,20]#range(10,20)#[ 68, 152, 108, 140,  20,  32,  72, 158, 168, 133]#max_idx_exc[195:205]#range(0,20,1)
    #print('cell_idx_exc',cell_idx_exc)
    cell_idx_inh = range(n_exc+17*2,256,1)#max_idx_inh+n_exc
    cell_idx_md1 = range(0,10,1)#max_idx_md1[0:20]
    y_exc = 1.*np.max(pfc_mean_HL_vis_A[cue_on:stim_on-2,0:n_exc])
    y_inh = 1.2*np.max(pfc_mean_HL_vis_A[cue_on:stim_on-2,n_exc:n_rnn])
    y_md1 = 1.2*np.max(MD_mean_HL_vis_A[cue_on:stim_on-2,0:n_md1])



    font=10
    start=0
    end = response_on-2


    ############ context A ########################
    ############ context A ########################
    ############ context A ########################
    fig, axs = plt.subplots(3, 2,figsize=(6,7))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig.suptitle('con_A  :'+model_name+';'+hp['add_mask'])

    for i in np.array(cell_idx_exc):
        axs[0,0].plot(pfc_mean_HL_vis_A[cue_on+start:end,i],label=str(i))
        axs[0,0].set_title('exc: '+'HP>LP (vis)',fontsize=font)
        axs[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_mean_HL_aud_A[cue_on+start:end,i],label=str(i))
        axs[0,1].set_title('exc: '+'HP<LP (aud)',fontsize=font)
        #axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,0].plot(pfc_mean_HL_vis_A[cue_on+start:end,i],label=str(i))
        axs[1,0].set_title('inh: '+'vis',fontsize=font)
        #axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs[1,1].plot(pfc_mean_HL_aud_A[cue_on+start:end,i],label=str(i))
        axs[1,1].set_title('inh: '+'aud',fontsize=font)
        #axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs[2,0].plot(MD_mean_HL_vis_A[cue_on+start:end,i],label=str(i))
        axs[2,0].set_title('md1: '+'vis',fontsize=font)
        #axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[2,1].plot(MD_mean_HL_aud_A[cue_on+start:end,i],label=str(i))
        axs[2,1].set_title('md1: '+'aud',fontsize=font)
        #axs2[0,1].legend(fontsize=5)


    axs[0, 0].set_ylim([0, y_exc])
    axs[0, 1].set_ylim([0, y_exc])
    axs[1, 0].set_ylim([0, y_inh])
    axs[1, 1].set_ylim([0, y_inh])
    axs[2, 0].set_ylim([0, y_md1])
    axs[2, 1].set_ylim([0, y_md1])


    for i in range(3):
        for j in range(2):
            #axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs[i,j].axvspan(cue_off-2-1-start, cue_off-2-1-start, color='grey',label='cue_off')
            axs[i,j].axvspan(stim_on-2-1-start, stim_on-2-1-start, color='grey',label='stim_on')

    plt.savefig(figure_path+model_name+'A.png')
    plt.show()

    ############ context A ########################
    ############ context A ########################
    ############ context A ########################
    fig1, axs1 = plt.subplots(3, 2,figsize=(6,7))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig1.suptitle('con_A2B  :' + model_name + ';' + hp['add_mask'])

    for i in np.array(cell_idx_exc):
        axs1[0, 0].plot(pfc_mean_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs1[0, 0].set_title('exc: ' + 'HP<LP (vis)', fontsize=font)
        #axs1[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs1[0, 1].plot(pfc_mean_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs1[0, 1].set_title('exc: ' + 'HP>LP (aud)', fontsize=font)
        # axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs1[1, 0].plot(pfc_mean_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs1[1, 0].set_title('inh: ' + 'vis', fontsize=font)
        # axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs1[1, 1].plot(pfc_mean_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs1[1, 1].set_title('inh: ' + 'aud', fontsize=font)
        # axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs1[2, 0].plot(MD_mean_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs1[2, 0].set_title('md1: ' + 'vis', fontsize=font)
        # axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs1[2, 1].plot(MD_mean_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs1[2, 1].set_title('md1: ' + 'aud', fontsize=font)
        # axs2[0,1].legend(fontsize=5)

    axs1[0, 0].set_ylim([0, y_exc])
    axs1[0, 1].set_ylim([0, y_exc])
    axs1[1, 0].set_ylim([0, y_inh])
    axs1[1, 1].set_ylim([0, y_inh])
    axs1[2, 0].set_ylim([0, y_md1])
    axs1[2, 1].set_ylim([0, y_md1])


    for i in range(3):
        for j in range(2):
            axs1[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs1[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + 'A2B.png')
    plt.show()

    ############ context A ########################
    fig2, axs2 = plt.subplots(3, 2,figsize=(6,7))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    fig2.suptitle('con_B2A  :' + model_name + ';' + hp['add_mask'])

    for i in np.array(cell_idx_exc):
        axs2[0, 0].plot(pfc_mean_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs2[0, 0].set_title('exc: ' + 'HP>LP (vis)', fontsize=font)
        #axs2[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_exc):
        axs2[0, 1].plot(pfc_mean_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs2[0, 1].set_title('exc: ' + 'HP<LP (aud)', fontsize=font)
        # axs[0,1].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs2[1, 0].plot(pfc_mean_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs2[1, 0].set_title('inh: ' + 'vis', fontsize=font)
        # axs[1,0].legend(fontsize=5)
    for i in np.array(cell_idx_inh):
        axs2[1, 1].plot(pfc_mean_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs2[1, 1].set_title('inh: ' + 'aud', fontsize=font)
        # axs[1,1].legend(fontsize=5)

    for i in np.array(cell_idx_md1):
        axs2[2, 0].plot(MD_mean_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs2[2, 0].set_title('md1: ' + 'vis', fontsize=font)
        # axs2[0,0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs2[2, 1].plot(MD_mean_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs2[2, 1].set_title('md1: ' + 'aud', fontsize=font)
        # axs2[0,1].legend(fontsize=5)

    axs2[0, 0].set_ylim([0, y_exc])
    axs2[0, 1].set_ylim([0, y_exc])
    axs2[1, 0].set_ylim([0, y_inh])
    axs2[1, 1].set_ylim([0, y_inh])
    axs2[2, 0].set_ylim([0, y_md1])
    axs2[2, 1].set_ylim([0, y_md1])


    for i in range(3):
        for j in range(2):
            # axs[i,j].axvspan(cue_on-2-start, cue_on-2-start, color='grey',label='cue_on')
            axs2[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs2[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name + 'B2A.png')
    plt.show()






def get_weight_A(hp,model_dir):
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
    print(effective_weights.shape)

    mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # axs[0].imshow(mat_md1_pc.T,aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()

    return effective_weights


def get_weight_A2B(hp,model_dir):
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

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    # mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]
    #
    # axs[0].imshow(mat_md1_pc.T, aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()
    return effective_weights.numpy()



def get_weight_B2A(hp,model_dir):
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

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    # mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]
    #
    # axs[0].imshow(mat_md1_pc.T, aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()

    return effective_weights.numpy()



def all_weight_hotmap(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B( hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)

    mat_MD1_PC_A = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]

    mat_MD1_PV_A = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]

    mat_PC_MD1_A = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2

    mat_MD2_PC_A = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]

    mat_MD2_VIP_A = effective_weights_A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A = effective_weights_A[256 + 18:, 0:205]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, 0:205]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, 0:205]



    start_exc = 0
    end_exc = 200

    # fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
    # fig.suptitle('md1_PC', fontsize=14)
    # # fig.suptitle('md1_PC', fontsize=14)
    #
    # axs[0].imshow(mat_MD1_PC_A[start_exc:end_exc, :].T, aspect='auto')
    # axs[1].imshow(mat_MD1_PC_A2B[start_exc:end_exc, :].T, aspect='auto')
    # axs[2].imshow(mat_MD1_PC_B2A[start_exc:end_exc, :].T, aspect='auto')
    #
    # axs[0].set_title('A')
    # axs[1].set_title('A2B')
    # axs[2].set_title('B2A')
    # plt.show()
    # print(mat_MD1_PC_A[0, 0], mat_MD1_PC_A2B[0, 0], mat_MD1_PC_B2A[0, 0])
    #
    # fig3, axs3 = plt.subplots(3, 1, figsize=(6, 8))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
    # fig3.suptitle('md2_PC', fontsize=14)
    # # fig.suptitle('md1_PC', fontsize=14)
    #
    # axs3[0].imshow(mat_MD2_PC_A[start_exc:end_exc, :].T, aspect='auto')
    # axs3[1].imshow(mat_MD2_PC_A2B[start_exc:end_exc, :].T, aspect='auto')
    # axs3[2].imshow(mat_MD2_PC_B2A[start_exc:end_exc, :].T, aspect='auto')
    #
    # axs3[0].set_title('A')
    # axs3[1].set_title('A2B')
    # axs3[2].set_title('B2A')
    # plt.show()

    fig1, axs1 = plt.subplots(3, 1, figsize=(4, 10))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    #fig1.suptitle(model_name_B2A+';'+'md1_pv', fontsize=12)
    fig1.suptitle('MD1_PV', fontsize=14)

    axs1[0].imshow(mat_MD1_PV_A.T, aspect='auto')
    axs1[1].imshow(mat_MD1_PV_A2B.T, aspect='auto')
    axs1[2].imshow(mat_MD1_PV_B2A.T, aspect='auto')

    axs1[0].set_title('A')
    axs1[1].set_title('A2B')
    axs1[2].set_title('B2A')
    plt.show()


    fig2, axs2 = plt.subplots(1, 3, figsize=(4, 4))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig1.suptitle(model_name_B2A+';'+'pc_md1', fontsize=12)
    fig2.suptitle('PC_MD2', fontsize=14)

    axs2[0].imshow(mat_PC_MD2_A.T, aspect='auto')
    axs2[1].imshow(mat_PC_MD2_A2B.T, aspect='auto')
    axs2[2].imshow(mat_PC_MD2_B2A.T, aspect='auto')

    axs2[0].set_title('A')
    axs2[1].set_title('A2B')
    axs2[2].set_title('B2A')
    plt.show()

    fig3, axs3 = plt.subplots(3, 1, figsize=(4, 10))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig1.suptitle(model_name_B2A+';'+'pc_md1', fontsize=12)
    fig3.suptitle('MD2_VIP', fontsize=14)

    axs3[0].imshow(mat_MD2_VIP_A.T, aspect='auto')
    axs3[1].imshow(mat_MD2_VIP_A2B.T, aspect='auto')
    axs3[2].imshow(mat_MD2_VIP_B2A.T, aspect='auto')

    axs3[0].set_title('A')
    axs3[1].set_title('A2B')
    axs3[2].set_title('B2A')
    plt.show()

def activity_three_context_md1(fig_path,hp,model_idx,i_dx,model_dir_A, model_dir_A2B,
                                model_dir_B2A, model_name_B2A,
                               start_cell):
    figure_path = os.path.join(fig_path, 'activity_three_context/')
    tools.mkdir_p(figure_path)

    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']



    coh_HL=0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx
    batch_size = 50

    #######rule1

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
    #                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B =get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',model_dir=model_dir_A2B,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',model_dir=model_dir_A2B,
    #                                                 cue=1, p_cohs=coh_HL,batch_size=batch_size)
    # #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',model_dir=model_dir_B2A,
    #                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',model_dir=model_dir_B2A,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # np.save(figure_path+'pfc_HL_vis_A.npy',pfc_HL_vis_A)
    # np.save(figure_path + 'pfc_HL_aud_A.npy', pfc_HL_aud_A)
    # np.save(figure_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(figure_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(figure_path + 'pfc_HL_vis_B2A.npy', pfc_HL_vis_B2A)
    # np.save(figure_path + 'pfc_HL_aud_B2A.npy', pfc_HL_aud_B2A)

    pfc_HL_vis_A = np.load(figure_path+'pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(figure_path+'pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(figure_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(figure_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_vis_B2A = np.load(figure_path + 'pfc_HL_vis_B2A.npy')
    pfc_HL_aud_B2A = np.load(figure_path + 'pfc_HL_aud_B2A.npy')

    ###### get the weight and sort the weight
    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)
    mat_md1_pc_A = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[0:205, 256:256 + 18]

    mat_md2_pc_A = effective_weights_A[0:205, 256 + 18:]
    mat_md2_pc_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_md2_pc_B2A = effective_weights_B2A[0:205, 256 + 18:]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B - mat_md1_pc_A, axis=1))

    mat_md1_pc_A = mat_md1_pc_A[diff_conA_conA2B_index_md1_pc, :]
    mat_md1_pc_A2B = mat_md1_pc_A2B[diff_conA_conA2B_index_md1_pc, :]
    mat_md1_pc_B2A = mat_md1_pc_B2A[diff_conA_conA2B_index_md1_pc, :]

    mat_md2_pc_A = mat_md2_pc_A[diff_conA_conA2B_index_md1_pc, :]
    mat_md2_pc_A2B = mat_md2_pc_A2B[diff_conA_conA2B_index_md1_pc, :]
    mat_md2_pc_B2A = mat_md2_pc_B2A[diff_conA_conA2B_index_md1_pc, :]

    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[0, :]))
    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[1, :]))
    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[2, :]))

    ##### plot activity #################################
    start = 0
    end = response_on - 2

    cell = range(start_cell, start_cell + 3)  # [158,197,199,203,204]#
    # for i in np.array(cell):
    #     print('**********',i)
    #     print(mat_md1_pc_A[i, :])
    #     print(mat_md1_pc_A2B[i,:])
    #     print(mat_md1_pc_B2A[i,:])
    #
    #     fig = plt.figure(figsize=(3.0, 2.6))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    #     plt.plot(mat_md1_pc_A[i, :],'o-',label='A')
    #     plt.plot(mat_md1_pc_A2B[i, :],'o-',label='B')
    #     plt.plot(mat_md1_pc_B2A[i, :],'o-',label='AA')
    #     plt.legend()
    #     plt.title(str(i))
    #     plt.show()

    cell_idx_exc = diff_conA_conA2B_index_md1_pc[
        cell]  # max_idx_exc[0:205]#np.array([148, 188, 147, 24])#range(0,20,1)#max_idx_exc[0:205]#range(0,20,1)
    print('cell_idx_exc', cell_idx_exc)
    cell_idx_md1 = range(0, 9, 1)  # np.array([1,3,9,11])#range(0,18,1)  # max_idx_md1[0:20]
    cell_idx_inh = range(205 + 17 * 2, 256, 1)  # max_idx_md1[0:20]

    y_exc = 1.2  # 1.5* np.max(pfc_HL_vis_A[cue_on:stim_on, cell_idx_exc])
    y_inh = 1.2 * np.max(pfc_HL_vis_A[cue_on:stim_on - 2, 205 + 17 * 2:256])

    fig, axs = plt.subplots(3, 4, figsize=(8, 5))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.97, left=0.05, hspace=0.2, wspace=0.2)
    fig.suptitle(model_name_B2A, fontsize=8)

    k = -1
    for i in np.array(cell):
        k += 1
        axs[k, 2].plot(mat_md1_pc_A[i, :], 'o-', label='1', c='tab:blue')
        axs[k, 2].plot(mat_md1_pc_A2B[i, :], 'o-', label='2', c='tab:orange')
        axs[k, 2].plot(mat_md1_pc_B2A[i, :], 'o-', label='11', c='tab:green', alpha=0.3)
        axs[k, 2].set_title(cell[k])
        axs[k, 2].set_ylim([0, 0.07])
        axs[0, 2].legend(fontsize=5)

        axs[k, 3].plot(mat_md2_pc_A[i, :], 'o-', label='1', c='tab:blue')
        axs[k, 3].plot(mat_md2_pc_A2B[i, :], 'o-', label='2', c='tab:orange')
        axs[k, 3].plot(mat_md2_pc_B2A[i, :], 'o-', label='11', c='tab:green', alpha=0.3)
        axs[k, 3].set_title(cell[k])
        axs[k, 3].set_ylim([0, 0.07])
        axs[0, 3].legend(fontsize=5)

    j = -1
    for i in np.array(cell_idx_exc):
        j += 1
        axs[0, 0].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(cell[j]))
        axs[1, 0].plot(pfc_HL_aud_A2B[cue_on + start:end, i], label=str(cell[j]))
        axs[2, 0].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(cell[j]))
        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('LP>HP')

    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1, 1].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 1].plot(pfc_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 1].legend(fontsize=5)
        axs[0, 1].set_title('LP<HP')

    # for i in np.array(cell_idx_inh):
    #     axs[0,2].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(i))
    #     axs[1,2].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
    #     axs[2,2].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(i))
    #     #axs[0,1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, 0].set_ylim([0, y_exc])
            axs[i, 1].set_ylim([0, y_exc])
            axs[i, 1].set_yticks([])
            axs[i, j].set_xticks([])

            axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name_B2A + '_' + str(start_cell) + '.png')
    plt.show()
    #"""


def MD_activity_three_model(figure_path, hp, model_dir_A, model_dir_A2B,
                             model_dir_B2A):
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.5
    batch_size = 100

    context1 = 'con_A'
    context2 = 'con_A2B'
    context3 = 'con_B2A'

    pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
                                                                model_dir=model_dir_A,
                                                                cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
                                                                model_dir=model_dir_A,
                                                                cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_A2B,
                                                                    cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_A2B,
                                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',
                                                                    model_dir=model_dir_B2A,
                                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task',
                                                                    hp=hp, context='con_B2A', model_dir=model_dir_B2A,
                                                                    cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    ##### plot activity #################################
    start = 0
    end = response_on - 2
    cell_idx_md1 = range(18)#[0, 1, 2, 4, 6, 7]  # max_idx_md1[0:20]

    y_md1 = 2.1  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name_A2B,fontsize=8)

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + '.png')
    plt.show()


def MD_activity_failed_model(figure_path, hp, model_dir_A, model_dir_A2B,
                             model_dir_B2A):
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    context1 = 'con_A2B'
    context2 = 'con_A2B'
    context3 = 'con_B2A'

    pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
                                                                model_dir=model_dir_A,
                                                                cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
                                                                model_dir=model_dir_A,
                                                                cue=1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
                                                                model_dir=model_dir_A2B,
                                                                cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
                                                                model_dir=model_dir_A2B,
                                                                cue=1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
                                                                model_dir=model_dir_B2A,
                                                                cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp, context=context1,
                                                                model_dir=model_dir_B2A,
                                                                cue=1, p_cohs=coh_HL, batch_size=batch_size)

    ##### plot activity #################################
    start = 0
    end = response_on - 2
    cell_idx_md1 = range(18)  # [0, 1, 2, 4, 6, 7]  # max_idx_md1[0:20]

    y_md1 = 2.1  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:30])

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name_A2B,fontsize=8)

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + '.png')
    plt.show()



def MD_mean_fr_failed_model(figure_path, hp, model_dir_A, model_dir_A2B,
                             model_dir_B2A):
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.5
    batch_size = 100

    context1 = 'con_A2B'
    MD_mean_fr_failed_1s = []
    MD_mean_fr_failed_2s = []
    MD_mean_fr_failed_3s = []



    # for i in range(10):
    #
    #     pfc_HL_vis_1, md_HL_vis_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                                 model_dir=model_dir_A,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #     pfc_HL_aud_1, md_HL_aud_1 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                                 model_dir=model_dir_A,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    #     pfc_HL_vis_2, md_HL_vis_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #     pfc_HL_aud_2, md_HL_aud_2 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    #     pfc_HL_vis_3, md_HL_vis_3 = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                                 model_dir=model_dir_B2A,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #     pfc_HL_aud_3, md_HL_aud_3 = get_neurons_activity_mode_test1(context_name='HL_task',hp=hp, context=context1,
    #                                                                 model_dir=model_dir_B2A,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    #     MD_mean_fr_failed_1s.append(md_HL_aud_1)
    #     MD_mean_fr_failed_2s.append(md_HL_aud_2)
    #     MD_mean_fr_failed_3s.append(md_HL_aud_3)
    #
    # np.save(figure_path+'MD_mean_fr_failed_1s.npy',MD_mean_fr_failed_1s)
    # np.save(figure_path+'MD_mean_fr_failed_2s.npy',MD_mean_fr_failed_2s)
    # np.save(figure_path+'MD_mean_fr_failed_3s.npy',MD_mean_fr_failed_3s)

    MD_fr_failed_1s = np.load(figure_path+'MD_mean_fr_failed_1s.npy')
    MD_fr_failed_2s = np.load(figure_path + 'MD_mean_fr_failed_2s.npy')
    MD_fr_failed_3s = np.load(figure_path + 'MD_mean_fr_failed_3s.npy')

    start = cue_on
    end = cue_off + 3
    cell_idx_md1 = range(18)
    MD_mean_fr_failed_1s =np.mean(np.array(MD_fr_failed_1s[:,start:end, cell_idx_md1]),axis=2)
    MD_mean_fr_failed_2s = np.mean(np.array(MD_fr_failed_2s[:,start:end, cell_idx_md1]), axis=2)
    MD_mean_fr_failed_3s = np.mean(np.array(MD_fr_failed_3s[:,start:end, cell_idx_md1]), axis=2)
    print(MD_mean_fr_failed_1s.shape)




    MD_fr_failed_1s_mean = np.mean(MD_mean_fr_failed_1s, axis=0)
    MD_fr_failed_1s_sem = np.std(MD_mean_fr_failed_1s, axis=0)  # /np.sqrt(len(RDM_pfc_viss))

    MD_fr_failed_2s_mean = np.mean(MD_mean_fr_failed_2s, axis=0)
    MD_fr_failed_2s_sem = np.std(MD_mean_fr_failed_2s, axis=0)  # /np.sqrt(len(RDM_pfc_viss))

    MD_fr_failed_3s_mean = np.mean(MD_mean_fr_failed_3s, axis=0)
    MD_fr_failed_3s_sem = np.std(MD_mean_fr_failed_3s, axis=0)  # /np.sqrt(len(RDM_pfc_viss))
    print(MD_fr_failed_1s_mean.shape)




    ##### plot activity #################################


    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.3, 0.15, 0.6, 0.7])

    number_dot = MD_fr_failed_1s_mean.shape[0]
    print(number_dot)

    ax.plot(MD_fr_failed_1s_mean, '-', linewidth=1, color=clrs[0], label='after 100 trials')
    ax.fill_between(np.linspace(0, number_dot, number_dot), MD_fr_failed_1s_mean - MD_fr_failed_1s_sem,
                    MD_fr_failed_1s_mean + MD_fr_failed_1s_sem, facecolor=clrs_fill[0])

    ax.plot(MD_fr_failed_2s_mean, '-', linewidth=1, color=clrs[1], label='after 600 trials')
    ax.fill_between(np.linspace(0, number_dot, number_dot), MD_fr_failed_2s_mean - MD_fr_failed_2s_sem,
                    MD_fr_failed_2s_mean + MD_fr_failed_2s_sem, facecolor=clrs_fill[1])

    ax.plot(MD_fr_failed_3s_mean, '-', linewidth=1, color=clrs[2], label='after 800 trials')
    ax.fill_between(np.linspace(0, number_dot, number_dot), MD_fr_failed_3s_mean - MD_fr_failed_3s_sem,
                    MD_fr_failed_3s_mean + MD_fr_failed_3s_sem, facecolor=clrs_fill[2])



    plt.ylabel('Faring rate', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('correct VS error', fontsize=8)
    plt.yticks([0.8,1.0,1.2,1.4], fontsize=11)
    plt.xticks([], fontsize=11)

    plt.legend(fontsize=5)
    ##################

    ax.axvspan(cue_off - 1 - start, cue_off - 1 - start, color='grey', label='cue_off')
    #ax.axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')
    plt.savefig(figure_path+'MD_mean_fr_failed_model.png')
    plt.savefig(figure_path + 'MD_mean_fr_failed_model.eps', format='eps', dpi=1000)
    plt.show()




def MD_activity_failed_model_example_test(figure_path, hp, model_dir_A, model_dir_A2B,
                                     model_dir_B2A):
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.5
    batch_size = 100

    pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
                                                                model_dir=model_dir_A,
                                                                cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
                                                                model_dir=model_dir_A,
                                                                cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_A2B,
                                                                    cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_A2B,
                                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',
                                                                    model_dir=model_dir_B2A,
                                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task',
                                                                    hp=hp, context='con_B2A', model_dir=model_dir_B2A,
                                                                    cue=-1, p_cohs=coh_HL, batch_size=batch_size)

    ##### plot activity #################################
    start = 0
    end = response_on - 2
    cell_idx_md1 = range(18)#[0, 1, 2, 4, 6, 7]  # max_idx_md1[0:20]

    y_md1 = 2.1  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name_A2B,fontsize=8)

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 0].legend(fontsize=5)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + '.png')
    plt.show()





def MD_switch_fr_A_B_B(figure_path, data_path,hp, model_dir_1, model_dir_2,
                                     model_dir_3):
    data_path = os.path.join(data_path, 'MD_switch_fr_A_B_B/')
    tools.mkdir_p(data_path)
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    pfc_HL_both_A, md_HL_both_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
                                                                model_dir=model_dir_1,
                                                                cue=None, p_cohs=coh_HL, batch_size=batch_size)


    pfc_HL_both_A2B, md_HL_both_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_2,
                                                                    cue=None, p_cohs=coh_HL, batch_size=batch_size)

    pfc_HL_both_B2A, md_HL_both_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
                                                                    model_dir=model_dir_3,
                                                                    cue=None, p_cohs=coh_HL, batch_size=batch_size)


    np.save(figure_path+'md_HL_both_A.npy',md_HL_both_A)
    np.save(figure_path + 'md_HL_both_A2B.npy', md_HL_both_A2B)
    np.save(figure_path + 'md_HL_both_B2A.npy', md_HL_both_B2A)

    md_HL_both_A = np.load(figure_path+'md_HL_both_A.npy')
    md_HL_both_A2B = np.load(figure_path + 'md_HL_both_A2B.npy')
    md_HL_both_B2A = np.load(figure_path + 'md_HL_both_B2A.npy')

    md_HL_both_A_cell = md_HL_both_A
    md_HL_both_A2B_cell = md_HL_both_A2B
    md_HL_both_B2A_cell = md_HL_both_B2A



    cell_idx_md1 = [3,5,8,9,10,16]
    start = 0
    end = response_on - 2
    md_HL_both_A = np.mean(md_HL_both_A[cue_on + start:end, cell_idx_md1],axis=0)
    md_HL_both_A2B = np.mean(md_HL_both_A2B[cue_on + start:end, cell_idx_md1], axis=0)
    md_HL_both_B2A = np.mean(md_HL_both_B2A[cue_on + start:end, cell_idx_md1], axis=0)

    md_HL_both_A_mean = np.mean(md_HL_both_A)
    md_HL_both_A2B_mean = np.mean(md_HL_both_A2B)
    md_HL_both_B2A_mean = np.mean(md_HL_both_B2A)
    print( md_HL_both_A_mean)

    md_HL_both_A_std = np.std(md_HL_both_A)/md_HL_both_A.shape[0]
    md_HL_both_A2B_std = np.std(md_HL_both_A2B)/md_HL_both_A.shape[0]
    md_HL_both_B2A_std = np.std(md_HL_both_B2A)/md_HL_both_A.shape[0]

    #'''
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.7])

    x= np.arange(0,3,1)
    y = [md_HL_both_A_mean,md_HL_both_A2B_mean,md_HL_both_B2A_mean]
    print(x,y)
    plt.xticks([0,1,2], fontsize=11)

    yerr = [md_HL_both_A_std,md_HL_both_A2B_std,md_HL_both_B2A_std]


    plt.errorbar(x, y, yerr=yerr,fmt='-o',c ='tab:orange')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel('Faring rate', fontsize=12)


    plt.savefig(figure_path + 'MD_switch_fr_A_B_B.png')
    plt.savefig(figure_path + 'MD_switch_fr_A_B_B.eps', format='eps', dpi=1000)

    plt.show()
    #'''

def MD_switch_fr_A_B_B_cell_example(figure_path, data_path,hp):
    data_path = os.path.join(data_path, 'MD_switch_fr_A_B_B_cell_example/')
    tools.mkdir_p(data_path)
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    context1 = 'con_A'
    context2 = 'con_A2B'
    context3 = 'con_A2B'

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(data_path+'md_HL_vis_A.npy',md_HL_vis_A)
    # np.save(data_path+'md_HL_vis_A2B.npy',md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_vis_B2A.npy', md_HL_vis_B2A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_aud_B2A.npy', md_HL_aud_B2A)

    md_HL_vis_A   = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_vis_B2A = np.load(data_path + 'md_HL_vis_B2A.npy')
    md_HL_aud_A   = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_aud_B2A = np.load(data_path + 'md_HL_aud_B2A.npy')

    ##### plot activity #################################
    start = 0
    end = response_on - 2
    cell_idx_md1 = [3]#[3,5,8,9,10,16]

    fig, axs = plt.subplots(3, 2, figsize=(3, 4))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    y_md1 = 2.7  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.4  # 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('vis', fontsize=7)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[0, 1].set_title('aud', fontsize=7)
        # axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, stim_on - 2 - 1 - start, facecolor='#EBEBEB')

            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.show()



def scatters_sns_MD_different_context_example(figure_path,hp,epoch,idx,model_name,period,model_dir_A,model_dir_A2B,model_dir_B2A):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    print('seed',hp['seed'])


    # pfc_HL_both_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
    #                                                             model_dir=model_dir_A,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
    #                                                             model_dir=model_dir_A,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',
    #                                                                 model_dir=model_dir_B2A,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                 hp=hp, context='con_B2A', model_dir=model_dir_B2A,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(figure_path+'1md_HL_vis_A.npy',md_HL_vis_A)
    # np.save(figure_path + '1md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(figure_path + '1md_HL_vis_A2B.npy', md_HL_vis_A2B)
    # np.save(figure_path + '1md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(figure_path + '1md_HL_vis_B2A.npy', md_HL_vis_B2A)
    # np.save(figure_path + '1md_HL_aud_B2A.npy', md_HL_aud_B2A)

    md_HL_vis_A = np.load(figure_path+'1md_HL_vis_A.npy')
    md_HL_aud_A = np.load(figure_path+'1md_HL_aud_A.npy')
    md_HL_vis_A2B = np.load(figure_path + '1md_HL_vis_A2B.npy')
    md_HL_aud_A2B = np.load(figure_path + '1md_HL_aud_A2B.npy')
    md_HL_vis_B2A = np.load(figure_path + '1md_HL_vis_B2A.npy')
    md_HL_aud_B2A = np.load(figure_path + '1md_HL_aud_B2A.npy')


    start = cue_on
    end = stim_on-1

    md_HL_vis_A_mean = np.mean(md_HL_vis_A[start:end,],axis=0)
    md_HL_aud_A_mean = np.mean(md_HL_aud_A[start:end, ], axis=0)

    md_HL_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, ], axis=0)
    md_HL_aud_A2B_mean = np.mean(md_HL_aud_A2B[start:end, ], axis=0)

    md_HL_vis_B2A_mean = np.mean(md_HL_vis_B2A[start:end, ], axis=0)
    md_HL_aud_B2A_mean = np.mean(md_HL_aud_B2A[start:end, ], axis=0)

    print('*************',md_HL_vis_A_mean.shape)
    print(md_HL_vis_A_mean)
    print(md_HL_vis_A2B_mean)



    print(np.argsort(md_HL_vis_A_mean))
    max = 2#np.max(md_HL_vis_A)
    min = 0.35
    print('max', max)

    cell_type = [0,1,4,6]#[0,1,2,4,6,7,]
    print('cell_type',cell_type)
    s = 50
    idx_md = 30
    #colors_1 = sns.color_palette("Set2")
    colors_1 = ['#9B30FF', '#FF34B3', '#7CFC00', 'green', '#FF4500', '#FF6A6A']
    #colors_1 = ['r', 'orange', 'y', 'green', 'b', 'purple']
    # fig2
    #'''
    for context in np.array(['switch1', 'switch2']):
        fig1 = plt.figure(figsize=(2.7, 2.7))
        ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
        if context == 'switch1':
            plt.scatter(md_HL_vis_A_mean[0:idx_md],md_HL_vis_A2B_mean[0:idx_md], marker="o", s=s, color='#4682B4',edgecolors='white')
            ax1.set_xlabel('con_A', fontsize=12)
            ax1.set_ylabel('con_B', fontsize=12)
        if context == 'switch2':
            plt.scatter(md_HL_vis_B2A_mean[0:idx_md], md_HL_vis_A2B_mean[0:idx_md], marker="o", s=s, color='#4682B4',edgecolors='white')
            ax1.set_xlabel('con_AA', fontsize=12)
            ax1.set_ylabel('con_B', fontsize=12)
        ax1.set_xlim([min, max])
        ax1.set_ylim([min, max])
        # plt.xticks([0.5, 0.8,1.1,1.4,1.7], fontsize=12)
        # plt.yticks([0.5, 0.8,1.1,1.4,1.7], fontsize=12)

        j = -1
        for i in np.array(cell_type):
            j += 1
            if context == 'switch1':
                plt.scatter(md_HL_vis_A_mean[i],  md_HL_vis_A2B_mean[i], marker="X", s=40, c=colors_1[j], label=str(i))
                # ax1.set_xlabel('con_A', fontsize=12)
                # ax1.set_ylabel('con_B', fontsize=12)
                #plt.legend()
            if context == 'switch2':
                plt.scatter(md_HL_vis_B2A_mean[i],md_HL_vis_A2B_mean[i], marker="X", s=40, c=colors_1[j], label=str(i))
                # ax1.set_xlabel('con_AA', fontsize=12)
                # ax1.set_ylabel('con_B', fontsize=12)

        plt.plot([0, max * 0.9], [0, max * 0.9], color='silver', )

        # set x-label
        ax1.spines[['right', 'top']].set_visible(False)
        #plt.title( model_name + '_' + str(idx)+':'+period,fontsize=6)
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.png')
        plt.savefig(figure_path + model_name + '_' + str(idx) + '_' + context + '.eps', format='eps', dpi=1000)

        plt.show()
    #'''

    ############################ plot activity #############################
    ############################ plot activity #############################
    ############################ plot activity #############################

    fig, axs = plt.subplots(3, 2, figsize=(3, 4))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    cell_idx_md1 = [6]
    y_md1 = 2.4 # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.6#0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('vis',fontsize=7)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], c='tab:blue',label=str(i))
        axs[0, 1].set_title('aud', fontsize=7)
        #axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, stim_on - 2 - 1 - start, facecolor='#EBEBEB')


            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + 'MD_example.eps', format='eps', dpi=1000)
    plt.show()

def sns_PFC_shift_different_context(figure_path,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    print('seed',hp['seed'])


    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
    #                                                             model_dir=model_dir_A,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
    #                                                             model_dir=model_dir_A,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_A2B,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',
    #                                                                 model_dir=model_dir_B2A,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                 hp=hp, context='con_B2A', model_dir=model_dir_B2A,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(figure_path+'1pfc_HL_vis_A.npy',pfc_HL_vis_A)
    # np.save(figure_path + '1pfc_HL_aud_A.npy', pfc_HL_aud_A)
    # np.save(figure_path + '1pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(figure_path + '1pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(figure_path + '1pfc_HL_vis_B2A.npy', pfc_HL_vis_B2A)
    # np.save(figure_path + '1pfc_HL_aud_B2A.npy', pfc_HL_aud_B2A)

    pfc_HL_vis_A = np.load(figure_path+'1pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(figure_path+'1pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(figure_path + '1pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(figure_path + '1pfc_HL_aud_A2B.npy')
    pfc_HL_vis_B2A = np.load(figure_path + '1pfc_HL_vis_B2A.npy')
    pfc_HL_aud_B2A = np.load(figure_path + '1pfc_HL_aud_B2A.npy')


    start = cue_on+1
    end = stim_on

    pfc_HL_vis_A_mean = np.mean(pfc_HL_vis_A[start:end,],axis=0)
    pfc_HL_aud_A_mean = np.mean(pfc_HL_aud_A[start:end, ], axis=0)

    pfc_HL_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, ], axis=0)
    pfc_HL_aud_A2B_mean = np.mean(pfc_HL_aud_A2B[start:end, ], axis=0)

    pfc_HL_vis_B2A_mean = np.mean(pfc_HL_vis_B2A[start:end, ], axis=0)
    pfc_HL_aud_B2A_mean = np.mean(pfc_HL_aud_B2A[start:end, ], axis=0)

    print('*************',pfc_HL_vis_A_mean.shape)
    print(pfc_HL_vis_A_mean)
    print(pfc_HL_vis_A2B_mean)



    print(np.argsort(pfc_HL_vis_A_mean))
    max = 2#np.max(pfc_HL_vis_A)
    min = 0.35
    print('max', max)

    cell_type = [0,1,4,6]#[0,1,2,4,6,7,]
    print('cell_type',cell_type)
    s = 50
    idx_pfc = 30
    #colors_1 = sns.color_palette("Set2")
    colors_1 = ['#9B30FF', '#FF34B3', '#7CFC00', 'green', '#FF4500', '#FF6A6A']


    ############################ plot activity #############################
    ############################ plot activity #############################
    ############################ plot activity #############################

    fig, axs = plt.subplots(3, 2, figsize=(3, 4))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    cell_idx_pfc1 = [3]#[190,15,3]
    y_pfc1 = 1.2 # * np.max(pfc_HL_vis_A2B[cue_on:stim_on, 0:18])

    j=-1
    for i in np.array(cell_idx_pfc1):
        j+=1
        axs[0, 0].plot(pfc_HL_vis_A[start:end, i], c=clrs[j],label=str(i))
        axs[1, 0].plot(pfc_HL_vis_A2B[start:end, i], c=clrs[j],label=str(i))
        axs[2, 0].plot(pfc_HL_vis_B2A[start:end, i], c=clrs[j],label=str(i))



        axs[0, 1].plot(pfc_HL_aud_A[start:end, i], c=clrs[j],label=str(i))
        axs[1, 1].plot(pfc_HL_aud_A2B[start:end, i], c=clrs[j], label=str(i))
        axs[2, 1].plot(pfc_HL_aud_B2A[start:end, i], c=clrs[j],label=str(i))

        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('vis', fontsize=7)
        axs[0, 1].set_title('aud', fontsize=7)
        #axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axvspan(cue_off - start, stim_on - start, facecolor='#EBEBEB')


            axs[i, j].set_ylim([0, y_pfc1])
            axs[i, j].set_ylim([0, y_pfc1])
            # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + 'pfc_example.eps', format='eps', dpi=1000)
    plt.show()



def histgram_delta_weight_all(data_path,hp):
    clrs = sns.color_palette("Set2")  # sns.color_palette("muted")#muted

    # effective_weights_A = get_weight_A(hp, model_dir_A)
    # effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    # effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)
    #
    # np.save(data_path + 'effective_weights_A.npy',effective_weights_A)
    # np.save(data_path + 'effective_weights_A2B.npy',effective_weights_A2B)
    # np.save(data_path + 'effective_weights_B2A.npy',effective_weights_B2A)



    effective_weights_A = np.load(data_path + 'effective_weights_A.npy')
    effective_weights_A2B = np.load(data_path + 'effective_weights_A2B.npy')
    effective_weights_B2A = np.load(data_path + 'effective_weights_B2A.npy')




    mat_MD1_PC_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]


    mat_MD1_PV_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_PC_MD1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2

    mat_MD2_PC_A     = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]


    mat_MD2_VIP_A     = effective_weights_A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A = effective_weights_A[256 + 18:, int(205 * 0.75)+1:int(205)]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, int(205 * 0.75)+1:int(205)]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, int(205 * 0.75)+1:int(205)]

    # mat_PC_MD2_A = effective_weights_A[256 + 18:, :int(205)]
    # mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, :int(205)]
    # mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, :int(205)]


    delta_MD1_PC_1 = np.abs(mat_MD1_PC_A2B - mat_MD1_PC_A)
    delta_MD1_PC_2 = np.abs(mat_MD1_PC_B2A - mat_MD1_PC_A2B)

    delta_MD1_PV_1 = np.abs(mat_MD1_PV_A2B - mat_MD1_PV_A)
    delta_MD1_PV_2 = np.abs(mat_MD1_PV_B2A - mat_MD1_PV_A2B)

    delta_PC_MD1_1 = np.abs(mat_PC_MD1_A2B - mat_PC_MD1_A)
    delta_PC_MD1_2 = np.abs(mat_PC_MD1_B2A - mat_PC_MD1_A2B)

    delta_MD2_PC_1 = np.abs(mat_MD2_PC_A2B - mat_MD2_PC_A)
    delta_MD2_PC_2 = np.abs(mat_MD2_PC_B2A - mat_MD2_PC_A2B)

    delta_MD2_VIP_1 = np.abs(mat_MD2_VIP_A2B - mat_MD2_VIP_A)
    delta_MD2_VIP_2 = np.abs(mat_MD2_VIP_B2A - mat_MD2_VIP_A2B)

    delta_PC_MD2_1 = np.abs(mat_PC_MD2_A2B - mat_PC_MD2_A)
    delta_PC_MD2_2 = np.abs(mat_PC_MD2_B2A - mat_PC_MD2_A2B)
    print('delta_MD1_PC_1',delta_PC_MD2_1.shape,delta_PC_MD2_1)





    delta_MD1_PC_1_mean = np.mean(np.mean(delta_MD1_PC_1))
    delta_MD1_PC_2_mean = np.mean(np.mean(delta_MD1_PC_2))
    delta_MD1_PV_1_mean = np.mean(np.mean(delta_MD1_PV_1))
    delta_MD1_PV_2_mean = np.mean(np.mean(delta_MD1_PV_2))
    delta_PC_MD1_1_mean = np.mean(np.mean(delta_PC_MD1_1))
    delta_PC_MD1_2_mean = np.mean(np.mean(delta_PC_MD1_2))

    delta_MD2_PC_1_mean = np.mean(np.mean(delta_MD2_PC_1))
    delta_MD2_PC_2_mean = np.mean(np.mean(delta_MD2_PC_2))
    delta_MD2_VIP_1_mean = np.mean(np.mean(delta_MD2_VIP_1))
    delta_MD2_VIP_2_mean = np.mean(np.mean(delta_MD2_VIP_2))
    delta_PC_MD2_1_mean = np.mean(np.mean(delta_PC_MD2_1))
    delta_PC_MD2_2_mean = np.mean(np.mean(delta_PC_MD2_2))





    IT_1 = [delta_MD1_PC_1_mean, delta_MD1_PV_1_mean, delta_PC_MD1_1_mean, delta_MD2_PC_1_mean,delta_MD2_VIP_1_mean, delta_PC_MD2_1_mean]
    IT_2 = [delta_MD1_PC_2_mean, delta_MD1_PV_2_mean, delta_PC_MD1_2_mean, delta_MD2_PC_2_mean,delta_MD2_VIP_2_mean, delta_PC_MD2_2_mean]

    barWidth = 0.2
    fig = plt.figure(figsize=(3.8, 3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, IT_1, yerr=0, color=clrs[0], alpha=1, width=barWidth, label='delta_A2B')
    plt.bar(br2, IT_2, yerr=0, color=clrs[1], alpha=1, width=barWidth, label='delta_B2A')

    plt.ylabel('delta connection strength', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(IT_1))],
               ['MD1-PC', 'MD1-PV', 'PC-MD1', 'MD2-PC', 'MD2-VIP', 'PC-MD2'], rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('weight_plot_all', fontsize=8)
    plt.yticks([0.0, 0.004, 0.008, 0.012], fontsize=12)

    plt.legend(fontsize=8)
    plt.show()
def histgram_delta_weight_all0(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    clrs = sns.color_palette("Set2")  # sns.color_palette("muted")#muted

    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)



    mat_MD1_PC_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]


    mat_MD1_PV_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_PC_MD1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2

    mat_MD2_PC_A     = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]


    mat_MD2_VIP_A     = effective_weights_A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A     = effective_weights_A[256 + 18:, 0:205]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, 0:205]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, 0:205]


    delta_MD1_PC_1 = np.abs(mat_MD1_PC_A2B - mat_MD1_PC_A)
    delta_MD1_PC_2 = np.abs(mat_MD1_PC_B2A - mat_MD1_PC_A2B)

    delta_MD1_PV_1 = np.abs(mat_MD1_PV_A2B - mat_MD1_PV_A)
    delta_MD1_PV_2 = np.abs(mat_MD1_PV_B2A - mat_MD1_PV_A2B)

    delta_PC_MD1_1 = np.abs(mat_PC_MD1_A2B - mat_PC_MD1_A)
    delta_PC_MD1_2 = np.abs(mat_PC_MD1_B2A - mat_PC_MD1_A2B)

    delta_MD2_PC_1 = np.abs(mat_MD2_PC_A2B - mat_MD2_PC_A)
    delta_MD2_PC_2 = np.abs(mat_MD2_PC_B2A - mat_MD2_PC_A2B)

    delta_MD2_VIP_1 = np.abs(mat_MD2_VIP_A2B - mat_MD2_VIP_A)
    delta_MD2_VIP_2 = np.abs(mat_MD2_VIP_B2A - mat_MD2_VIP_A2B)

    delta_PC_MD2_1 = np.abs(mat_PC_MD2_A2B - mat_PC_MD2_A)
    delta_PC_MD2_2 = np.abs(mat_PC_MD2_B2A - mat_PC_MD2_A2B)
    print('delta_MD1_PC_1',delta_MD1_PC_1.shape)



    delta_MD1_PC_1_mean = np.mean(np.mean(delta_MD1_PC_1))
    delta_MD1_PC_2_mean = np.mean(np.mean(delta_MD1_PC_2))
    delta_MD1_PV_1_mean = np.mean(np.mean(delta_MD1_PV_1))
    delta_MD1_PV_2_mean = np.mean(np.mean(delta_MD1_PV_2))
    delta_PC_MD1_1_mean = np.mean(np.mean(delta_PC_MD1_1))
    delta_PC_MD1_2_mean = np.mean(np.mean(delta_PC_MD1_2))

    delta_MD2_PC_1_mean = np.mean(np.mean(delta_MD2_PC_1))
    delta_MD2_PC_2_mean = np.mean(np.mean(delta_MD2_PC_2))
    delta_MD2_VIP_1_mean = np.mean(np.mean(delta_MD2_VIP_1))
    delta_MD2_VIP_2_mean = np.mean(np.mean(delta_MD2_VIP_2))
    delta_PC_MD2_1_mean = np.mean(np.mean(delta_PC_MD2_1))
    delta_PC_MD2_2_mean = np.mean(np.mean(delta_PC_MD2_2))


    MD_PC_1 = (delta_MD1_PC_1_mean + delta_MD2_PC_1_mean)/2
    MD_PC_2 = (delta_MD1_PC_2_mean + delta_MD2_PC_2_mean) / 2

    MD_inh_1 = (delta_MD1_PV_1_mean+delta_MD2_VIP_1_mean)/2
    MD_inh_2 = (delta_MD1_PV_2_mean + delta_MD2_VIP_2_mean) / 2

    PC_MD_1 = (delta_PC_MD1_1_mean+delta_PC_MD2_1_mean)/2
    PC_MD_2 = (delta_PC_MD1_2_mean + delta_PC_MD2_2_mean) / 2





    IT_1 = [MD_PC_1,MD_inh_1,PC_MD_1]
    IT_2 = [MD_PC_2,MD_inh_2,PC_MD_2]

    barWidth = 0.2
    fig = plt.figure(figsize=(3.8, 3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, IT_1, yerr=0, color=clrs[0], alpha=1, width=barWidth, label='delta_A2B')
    plt.bar(br2, IT_2, yerr=0, color=clrs[1], alpha=1, width=barWidth, label='delta_B2A')

    plt.ylabel('delta connection strength', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(IT_1))],
               ['MD-PC', 'MD-inh', 'PC-MD'], rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('weight_plot_all', fontsize=8)
    plt.yticks([0.0, 0.004, 0.008, 0.012], fontsize=12)

    plt.legend(fontsize=8)
    fig.savefig(figure_path + 'hist_weight.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'hist_weight.png')
    plt.show()


def histgram_delta_weight_all1(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    clrs = sns.color_palette("Set2")  # sns.color_palette("muted")#muted

    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)



    mat_MD1_PC_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]


    mat_MD1_PV_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_PC_MD1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2

    mat_MD2_PC_A     = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]


    mat_MD2_VIP_A     = effective_weights_A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A     = effective_weights_A[256 + 18:, 0:205]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, 0:205]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, 0:205]


    delta_MD1_PC_1 = np.abs(mat_MD1_PC_A2B - mat_MD1_PC_A)/np.mean(np.mean(mat_MD1_PC_A2B))
    delta_MD1_PC_2 = np.abs(mat_MD1_PC_B2A - mat_MD1_PC_A2B)/np.mean(np.mean(mat_MD1_PC_A2B))

    delta_MD1_PV_1 = np.abs(mat_MD1_PV_A2B - mat_MD1_PV_A)/np.mean(np.mean(mat_MD1_PV_A2B))
    delta_MD1_PV_2 = np.abs(mat_MD1_PV_B2A - mat_MD1_PV_A2B)/np.mean(np.mean(mat_MD1_PV_A2B))

    delta_PC_MD1_1 = np.abs(mat_PC_MD1_A2B - mat_PC_MD1_A)/np.mean(np.mean(mat_PC_MD1_A2B))
    delta_PC_MD1_2 = np.abs(mat_PC_MD1_B2A - mat_PC_MD1_A2B)/np.mean(np.mean(mat_PC_MD1_A2B))

    delta_MD2_PC_1 = np.abs(mat_MD2_PC_A2B - mat_MD2_PC_A)/np.mean(np.mean(mat_MD2_PC_A2B))
    delta_MD2_PC_2 = np.abs(mat_MD2_PC_B2A - mat_MD2_PC_A2B)/np.mean(np.mean(mat_MD2_PC_A2B))

    delta_MD2_VIP_1 = np.abs(mat_MD2_VIP_A2B - mat_MD2_VIP_A)/np.mean(np.mean(mat_MD2_VIP_A2B))
    delta_MD2_VIP_2 = np.abs(mat_MD2_VIP_B2A - mat_MD2_VIP_A2B)/np.mean(np.mean(mat_MD2_VIP_A2B))

    delta_PC_MD2_1 = np.abs(mat_PC_MD2_A2B - mat_PC_MD2_A)/np.mean(np.mean(mat_PC_MD2_A2B))
    delta_PC_MD2_2 = np.abs(mat_PC_MD2_B2A - mat_PC_MD2_A2B)/np.mean(np.mean(mat_PC_MD2_A2B))
    print('delta_MD1_PC_1',delta_MD1_PC_1.shape)



    delta_MD1_PC_1_mean = np.mean(np.mean(delta_MD1_PC_1))
    delta_MD1_PC_2_mean = np.mean(np.mean(delta_MD1_PC_2))
    delta_MD1_PV_1_mean = np.mean(np.mean(delta_MD1_PV_1))
    delta_MD1_PV_2_mean = np.mean(np.mean(delta_MD1_PV_2))
    delta_PC_MD1_1_mean = np.mean(np.mean(delta_PC_MD1_1))
    delta_PC_MD1_2_mean = np.mean(np.mean(delta_PC_MD1_2))

    delta_MD2_PC_1_mean = np.mean(np.mean(delta_MD2_PC_1))
    delta_MD2_PC_2_mean = np.mean(np.mean(delta_MD2_PC_2))
    delta_MD2_VIP_1_mean = np.mean(np.mean(delta_MD2_VIP_1))
    delta_MD2_VIP_2_mean = np.mean(np.mean(delta_MD2_VIP_2))
    delta_PC_MD2_1_mean = np.mean(np.mean(delta_PC_MD2_1))
    delta_PC_MD2_2_mean = np.mean(np.mean(delta_PC_MD2_2))





    IT_1 = [delta_MD1_PC_1_mean, delta_MD1_PV_1_mean, delta_PC_MD1_1_mean, delta_MD2_PC_1_mean,delta_MD2_VIP_1_mean, delta_PC_MD2_1_mean]
    IT_2 = [delta_MD1_PC_2_mean, delta_MD1_PV_2_mean, delta_PC_MD1_2_mean, delta_MD2_PC_2_mean,delta_MD2_VIP_2_mean, delta_PC_MD2_2_mean]

    barWidth = 0.2
    fig = plt.figure(figsize=(3.8, 3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, IT_1, yerr=0, color=clrs[0], alpha=1, width=barWidth, label='A')
    plt.bar(br2, IT_2, yerr=0, color=clrs[1], alpha=1, width=barWidth, label='B')

    plt.ylabel('delta strength', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(IT_1))],
               ['MD1-PC', 'MD1-PV', 'PC-MD1', 'MD2-PC', 'MD2-VIP', 'PC-MD2'], rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('weight_plot_all', fontsize=8)
    #plt.yticks([0.0, 0.01, 0.02, 0.03], fontsize=12)

    plt.legend(fontsize=8)
    fig.savefig(figure_path + 'hist_weight.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'hist_weight.png')
    plt.show()


def histgram_delta_weight_all3(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    clrs = sns.color_palette("Set2")  # sns.color_palette("muted")#muted

    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)



    mat_MD1_PC_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]


    mat_MD1_PV_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_PC_MD1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2

    mat_MD2_PC_A     = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]

    mat_MD2_VIP_A = effective_weights_A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:205 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:205 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A = effective_weights_A[256 + 18:, int(205 * 0.75)+1:int(205)]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, int(205 * 0.75)+1:int(205)]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, int(205 * 0.75)+1:int(205)]





    delta_MD1_PC_1 = np.abs((mat_MD1_PC_A2B - mat_MD1_PC_A)/mat_MD1_PC_A)
    delta_MD1_PC_2 = np.abs((mat_MD1_PC_B2A - mat_MD1_PC_A2B)/mat_MD1_PC_A)

    delta_MD1_PV_1 = np.abs((mat_MD1_PV_A2B - mat_MD1_PV_A)/mat_MD1_PV_A)
    delta_MD1_PV_2 = np.abs((mat_MD1_PV_B2A - mat_MD1_PV_A2B)/mat_MD1_PV_A)

    delta_PC_MD1_1 = np.abs((mat_PC_MD1_A2B - mat_PC_MD1_A)/mat_PC_MD1_A)
    delta_PC_MD1_2 = np.abs((mat_PC_MD1_B2A - mat_PC_MD1_A2B)/mat_PC_MD1_A)

    delta_MD2_PC_1 = np.abs((mat_MD2_PC_A2B - mat_MD2_PC_A)/mat_MD2_PC_A)
    delta_MD2_PC_2 = np.abs((mat_MD2_PC_B2A - mat_MD2_PC_A2B)/mat_MD2_PC_A)

    delta_MD2_VIP_1 = np.abs((mat_MD2_VIP_A2B - mat_MD2_VIP_A)/mat_MD2_VIP_A)
    delta_MD2_VIP_2 = np.abs((mat_MD2_VIP_B2A - mat_MD2_VIP_A2B)/mat_MD2_VIP_A)

    delta_PC_MD2_1 = np.abs((mat_PC_MD2_A2B - mat_PC_MD2_A)/mat_PC_MD2_A)
    delta_PC_MD2_2 = np.abs((mat_PC_MD2_B2A - mat_PC_MD2_A2B)/mat_PC_MD2_A)



    delta_MD1_PC_1_mean = np.mean(np.mean(delta_MD1_PC_1))
    delta_MD1_PC_2_mean = np.mean(np.mean(delta_MD1_PC_2))
    delta_MD1_PV_1_mean = np.mean(np.mean(delta_MD1_PV_1))
    delta_MD1_PV_2_mean = np.mean(np.mean(delta_MD1_PV_2))
    delta_PC_MD1_1_mean = np.mean(np.mean(delta_PC_MD1_1))
    delta_PC_MD1_2_mean = np.mean(np.mean(delta_PC_MD1_2))

    delta_MD2_PC_1_mean = np.mean(np.mean(delta_MD2_PC_1))
    delta_MD2_PC_2_mean = np.mean(np.mean(delta_MD2_PC_2))
    delta_MD2_VIP_1_mean = np.mean(np.mean(delta_MD2_VIP_1))
    delta_MD2_VIP_2_mean = np.mean(np.mean(delta_MD2_VIP_2))
    delta_PC_MD2_1_mean = np.mean(np.mean(delta_PC_MD2_1))
    delta_PC_MD2_2_mean = np.mean(np.mean(delta_PC_MD2_2))


    print('mat_PC_MD2_A',mat_PC_MD2_A)
    print(' delta_PC_MD2_1_mean ', delta_PC_MD2_1_mean )





    IT_1 = [delta_MD1_PC_1_mean, delta_MD1_PV_1_mean, delta_PC_MD1_1_mean, delta_MD2_PC_1_mean,delta_MD2_VIP_1_mean, delta_PC_MD2_1_mean]
    IT_2 = [delta_MD1_PC_2_mean, delta_MD1_PV_2_mean, delta_PC_MD1_2_mean, delta_MD2_PC_2_mean,delta_MD2_VIP_2_mean, delta_PC_MD2_2_mean]

    barWidth = 0.2
    fig = plt.figure(figsize=(3.8, 3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, IT_1, yerr=0, color=clrs[0], alpha=1, width=barWidth, label='A')
    plt.bar(br2, IT_2, yerr=0, color=clrs[1], alpha=1, width=barWidth, label='B')

    plt.ylabel('delta strength', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(IT_1))],
               ['MD1-PC', 'MD1-PV', 'PC-MD1', 'MD2-PC', 'MD2-VIP', 'PC-MD2'], rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('weight_plot_all', fontsize=8)
    #plt.yticks([0.0, 0.01, 0.02, 0.03], fontsize=12)
    #plt.ylim([0,10])

    plt.legend(fontsize=8)
    fig.savefig(figure_path + 'hist_weight.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'hist_weight.png')
    plt.show()


def histgram_delta_weight_all2(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    clrs = sns.color_palette("Set2")  # sns.color_palette("muted")#muted

    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)



    mat_MD1_PC_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_MD1_PC_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_MD1_PC_B2A = effective_weights_B2A[0:205, 256:256 + 18]


    mat_MD1_PV_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_MD1_PV_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_PC_MD1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_PC_MD1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_PC_MD1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    ##### MD2
    mat_MD2_PC_A     = effective_weights_A[0:205, 256 + 18:]
    mat_MD2_PC_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_MD2_PC_B2A = effective_weights_B2A[0:205, 256 + 18:]


    mat_MD2_VIP_A     = effective_weights_A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_A2B = effective_weights_A2B[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_MD2_VIP_B2A = effective_weights_B2A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]

    mat_PC_MD2_A     = effective_weights_A[256 + 18:, 0:205]
    mat_PC_MD2_A2B = effective_weights_A2B[256 + 18:, 0:205]
    mat_PC_MD2_B2A = effective_weights_B2A[256 + 18:, 0:205]

    mat_MD1_PC_A_mean = np.mean(np.mean(mat_MD1_PC_A))
    mat_MD1_PC_A2B_mean = np.mean(np.mean(mat_MD1_PC_A2B))
    mat_MD1_PC_B2A_mean = np.mean(np.mean(mat_MD1_PC_B2A))

    mat_MD1_PV_A_mean = np.mean(np.mean(mat_MD1_PV_A))
    mat_MD1_PV_A2B_mean = np.mean(np.mean(mat_MD1_PV_A2B))
    mat_MD1_PV_B2A_mean = np.mean(np.mean(mat_MD1_PV_B2A))

    mat_PC_MD1_A_mean = np.mean(np.mean(mat_PC_MD1_A))
    mat_PC_MD1_A2B_mean = np.mean(np.mean(mat_PC_MD1_A2B))
    mat_PC_MD1_B2A_mean = np.mean(np.mean(mat_PC_MD1_B2A))

    mat_MD2_PC_A_mean = np.mean(np.mean(mat_MD2_PC_A))
    mat_MD2_PC_A2B_mean = np.mean(np.mean(mat_MD2_PC_A2B))
    mat_MD2_PC_B2A_mean = np.mean(np.mean(mat_MD2_PC_B2A))

    mat_MD2_VIP_A_mean = np.mean(np.mean(mat_MD2_VIP_A))
    mat_MD2_VIP_A2B_mean = np.mean(np.mean(mat_MD2_VIP_A2B))
    mat_MD2_VIP_B2A_mean = np.mean(np.mean(mat_MD2_VIP_B2A))

    mat_PC_MD2_A_mean = np.mean(np.mean(mat_PC_MD2_A))
    mat_PC_MD2_A2B_mean = np.mean(np.mean(mat_PC_MD2_A2B))
    mat_PC_MD2_B2A_mean = np.mean(np.mean(mat_PC_MD2_B2A))


    delta_MD1_PC_1_mean = np.abs(mat_MD1_PC_A_mean-mat_MD1_PC_A2B_mean)/mat_MD1_PC_A2B_mean
    delta_MD1_PC_2_mean = np.abs(mat_MD1_PC_B2A_mean-mat_MD1_PC_A2B_mean)/mat_MD1_PC_A2B_mean

    delta_MD1_PV_1_mean = np.abs(mat_MD1_PV_A_mean - mat_MD1_PV_A2B_mean)/mat_MD1_PV_A2B_mean
    delta_MD1_PV_2_mean = np.abs(mat_MD1_PV_B2A_mean - mat_MD1_PV_A2B_mean)/mat_MD1_PV_A2B_mean

    delta_PC_MD1_1_mean = np.abs(mat_PC_MD1_A_mean - mat_PC_MD1_A2B_mean)/mat_PC_MD1_A2B_mean
    delta_PC_MD1_2_mean = np.abs(mat_PC_MD1_B2A_mean - mat_PC_MD1_A2B_mean)/mat_PC_MD1_A2B_mean

    delta_MD2_PC_1_mean = np.abs(mat_MD2_PC_A_mean - mat_MD2_PC_A2B_mean)/mat_MD2_PC_A2B_mean
    delta_MD2_PC_2_mean = np.abs(mat_MD2_PC_B2A_mean - mat_MD2_PC_A2B_mean)/mat_MD2_PC_A2B_mean

    delta_MD2_VIP_1_mean = np.abs(mat_MD2_VIP_A_mean - mat_MD2_VIP_A2B_mean)/mat_MD2_VIP_A2B_mean
    delta_MD2_VIP_2_mean = np.abs(mat_MD2_VIP_B2A_mean - mat_MD2_VIP_A2B_mean)/mat_MD2_VIP_A2B_mean

    delta_PC_MD2_1_mean = np.abs(mat_PC_MD2_A_mean - mat_PC_MD2_A2B_mean)/mat_PC_MD2_A2B_mean
    delta_PC_MD2_2_mean = np.abs(mat_PC_MD2_B2A_mean - mat_PC_MD2_A2B_mean)/mat_PC_MD2_A2B_mean





    IT_1 = [delta_MD1_PC_1_mean, delta_MD1_PV_1_mean, delta_PC_MD1_1_mean, delta_MD2_PC_1_mean,delta_MD2_VIP_1_mean, delta_PC_MD2_1_mean]
    IT_2 = [delta_MD1_PC_2_mean, delta_MD1_PV_2_mean, delta_PC_MD1_2_mean, delta_MD2_PC_2_mean,delta_MD2_VIP_2_mean, delta_PC_MD2_2_mean]

    barWidth = 0.2
    fig = plt.figure(figsize=(3.8, 3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, IT_1, yerr=0, color=clrs[0], alpha=1, width=barWidth, label='delta_A2B')
    plt.bar(br2, IT_2, yerr=0, color=clrs[1], alpha=1, width=barWidth, label='delta_B2A')

    plt.ylabel('delta strength', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(IT_1))],
               ['MD1-PC', 'MD1-PV', 'PC-MD1', 'MD2-PC', 'MD2-VIP', 'PC-MD2'], rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('weight_plot_all', fontsize=8)
    #plt.yticks([0.0, 0.01, 0.02, 0.03], fontsize=12)

    plt.legend(fontsize=8)
    fig.savefig(figure_path + 'hist_weight.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'hist_weight.png')
    plt.show()

def weight_sorted_md1_to_pc(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)

    mat_md1_pc_A     = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[0:205, 256:256 + 18]

    mat_md1_pv_A     = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_md1_pv_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_md1_pv_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]

    mat_pc_md1_A     = effective_weights_A[256:256 + 18, 0:205]
    mat_pc_md1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_pc_md1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B - mat_md1_pc_A, axis=1))
    # np.save(data_path + 'diff_conA_conA2B_index_md1_pc.npy', diff_conA_conA2B_index_md1_pc)
    # print('****** diff_conA_conA2B_index_md1_pc', diff_conA_conA2B_index_md1_pc)
    mat_md1_pc_A     = effective_weights_A[diff_conA_conA2B_index_md1_pc, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[diff_conA_conA2B_index_md1_pc, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[diff_conA_conA2B_index_md1_pc, 256:256 + 18]

    start_exc = 0
    end_exc = 206

    cell1 = [158, 197, 199, 203, 204]  # [150+4,150+25,150+37,150+39]
    cell2 = [155, 177, 186, 192]

    # for i in np.array([10,18,17]):
    #     print('**********',i)
    #     print(mat_md1_pc_A[i, :])
    #     print(mat_md1_pc_A2B[i,:])
    #     print(mat_md1_pc_B2A[i,:])
    #
    #     fig = plt.figure(figsize=(3.0, 2.6))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    #     plt.plot(mat_md1_pc_A[i, :],'o-',label='A')
    #     plt.plot(mat_md1_pc_A2B[i, :],'o-',label='B')
    #     plt.plot(mat_md1_pc_B2A[i, :],'o-',label='AA',alpha=0.1)
    #     plt.legend()
    #     plt.title(str(i))
    #     plt.show()

    data = mat_md1_pc_A2B[0:205, :].T
    median_val = np.median(data)
    cmap = plt.cm.get_cmap('summer')
    norm = colors.Normalize(vmin=-median_val, vmax=median_val)
    norm.autoscale_None(data)
    # fig = plt.figure(figsize=(2.0, 2))
    # ax = fig.add_axes([0.2, 0.2, 0.5, 0.57])
    # plt.imshow(mat_md1_pc_A2B[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    # plt.colorbar()
    # plt.show()
    # fig.savefig(figure_path + 'colorbar.eps', format='eps', dpi=1000)

    fig, axs = plt.subplots(3, 1, figsize=(6 * 0.8, 4 * 0.8))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.1, wspace=0.1)
    # fig.suptitle(model_name_B2A+';'+'md1_PC', fontsize=14)

    # fig.suptitle('md1_PC', fontsize=14)

    axs[0].imshow(mat_md1_pc_A[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[1].imshow(mat_md1_pc_A2B[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[2].imshow(mat_md1_pc_B2A[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    # axs[0].set_title('A')
    # axs[1].set_title('A2B')
    # axs[2].set_title('B2A')
    # for i in range(3):
    #     axs[0].set_xticks([])

    fig.savefig(figure_path + 'weight_three_context.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'weight_three_context.png', format='png')

    plt.show()


def weight_sorted_md1_to_pv(figure_path,hp,model_dir_A,model_dir_A2B,model_dir_B2A):
    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)

    mat_md1_pc_A   =   effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[0:205, 256:256 + 18]



    mat_md1_pv_A   =   effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_md1_pv_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:256 + 18]
    mat_md1_pv_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]


    mat_pc_md1_A   =   effective_weights_A[256:256 + 18, 0:205]
    mat_pc_md1_A2B = effective_weights_A2B[256:256 + 18, 0:205]
    mat_pc_md1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    diff_conA_conA2B_index_md1_pv = np.argsort(np.mean(mat_md1_pv_A2B - mat_md1_pv_A, axis=1))
    # np.save(data_path + 'diff_conA_conA2B_index_md1_pc.npy', diff_conA_conA2B_index_md1_pc)
    # print('****** diff_conA_conA2B_index_md1_pc', diff_conA_conA2B_index_md1_pc)
    mat_md1_pc_A = effective_weights_A[diff_conA_conA2B_index_md1_pv, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[diff_conA_conA2B_index_md1_pv, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[diff_conA_conA2B_index_md1_pv, 256:256 + 18]

    start_exc = 0
    end_exc = 206

    cell1 = [158, 197, 199, 203, 204]  # [150+4,150+25,150+37,150+39]
    cell2 = [155, 177, 186, 192]

    # for i in np.array([10,18,17]):
    #     print('**********',i)
    #     print(mat_md1_pc_A[i, :])
    #     print(mat_md1_pc_A2B[i,:])
    #     print(mat_md1_pc_B2A[i,:])
    #
    #     fig = plt.figure(figsize=(3.0, 2.6))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    #     plt.plot(mat_md1_pc_A[i, :],'o-',label='A')
    #     plt.plot(mat_md1_pc_A2B[i, :],'o-',label='B')
    #     plt.plot(mat_md1_pc_B2A[i, :],'o-',label='AA',alpha=0.1)
    #     plt.legend()
    #     plt.title(str(i))
    #     plt.show()

    data = mat_md1_pc_A2B[0:205, :].T
    median_val = np.median(data)
    cmap = plt.cm.get_cmap('summer')
    norm = colors.Normalize(vmin=-median_val, vmax=median_val)
    norm.autoscale_None(data)
    # fig = plt.figure(figsize=(2.0, 2))
    # ax = fig.add_axes([0.2, 0.2, 0.5, 0.57])
    # plt.imshow(mat_md1_pc_A2B[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    # plt.colorbar()
    # plt.show()
    # fig.savefig(figure_path + 'colorbar.eps', format='eps', dpi=1000)

    fig, axs = plt.subplots(3, 1, figsize=(6 * 0.8, 4 * 0.8))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.1, wspace=0.1)
    # fig.suptitle(model_name_B2A+';'+'md1_PC', fontsize=14)

    # fig.suptitle('md1_PC', fontsize=14)

    axs[0].imshow(mat_md1_pc_A[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[1].imshow(mat_md1_pc_A2B[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[2].imshow(mat_md1_pc_B2A[start_exc:end_exc, :].T, aspect='auto', cmap=cmap, norm=norm, alpha=1)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    # axs[0].set_title('A')
    # axs[1].set_title('A2B')
    # axs[2].set_title('B2A')
    # for i in range(3):
    #     axs[0].set_xticks([])

    fig.savefig(figure_path + 'weight_three_context.eps', format='eps', dpi=1000)
    fig.savefig(figure_path + 'weight_three_context.png', format='png')

    plt.show()









