

from __future__ import division
import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import math
import run
import pdb
import seaborn as sns
from scipy.linalg import hadamard, subspace_angles
from matplotlib.collections import LineCollection
fs=10

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


def generate_test_trial(model_dir,hp,task_name,batch_size=1,
                        p_coh=0.9,
                        cue=None,
                        c_vis=None,
                        c_aud=None):
    #hp['switch_context'] = context


    # print('c_vis',c_vis)

    rng  = hp['rng']
    cue_scale=hp['cue_scale']

    if cue is None:
        c_cue = hp['rng'].choice([1*cue_scale,-1*cue_scale], (batch_size,))
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




    runnerObj = run.Runner(model_dir=model_dir, rule_name=task_name, hp=hp,is_cuda=False, noise_on=False,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            p_coh=p_coh,
                                            c_cue=c_cue,
                                            c_vis=c_vis,
                                            c_aud=c_aud)


    return trial_input, run_result


def generate_test_trial_context(model_dir,hp,context,task_name,batch_size=1,
                        p_coh=0.9,
                        cue=None,
                        c_vis=None,
                        c_aud=None):
    hp['switch_context'] = context




    rng  = hp['rng']
    cue_scale=hp['cue_scale']

    if cue is None:
        c_cue = hp['rng'].choice([1*cue_scale,-1*cue_scale], (batch_size,))
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




    runnerObj = run.Runner(model_dir=model_dir, rule_name=task_name, hp=hp,is_cuda=False, noise_on=False,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            p_coh=p_coh,
                                            c_cue=c_cue,
                                            c_vis=c_vis,
                                            c_aud=c_aud)


    return trial_input, run_result



def PCA_plot_3D_cue(model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):

    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1



    batch_size=50

    trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir,hp=hp,task_name=task_name,batch_size=batch_size,
                                                      cue=cue,p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)


    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = trial_input_0.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    print("concate_firing_rate_0", concate_firing_rate_0.shape)


    trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir,hp=hp,task_name=task_name,batch_size=batch_size,
                                                      cue=-cue,p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)
    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']
    firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)
    print("concate_firing_rate_1", concate_firing_rate_1.shape)


    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)
    print("*** concate_firing_rate", concate_firing_rate.shape)

    # plot
    _alpha_list = [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8]

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)

    explained_variance_ratio=pca.explained_variance_ratio_
    print('explained_variance_ratio',explained_variance_ratio)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time-start_time
    delim = np.cumsum(time_size)
    print('delim',delim)

    print("##concate_firing_rate_transform",concate_firing_rate_transform.shape)
    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)
    print('concate_transform_split',len(concate_transform_split))


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(0, len(concate_transform_split)):
        if i<batch_size:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], linewidth=0.5,color='tab:blue')
            #ax.plot(concate_transform_split[i][0:6, 0], concate_transform_split[i][0:6, 1], concate_transform_split[i][0:6, 2], linewidth=0.5,color='lime')
            #ax.plot(concate_transform_split[i][5:21, 0], concate_transform_split[i][5:21, 1], concate_transform_split[i][5:21, 2], linewidth=0.5,color='blue')
            #ax.plot(concate_transform_split[i][20:40, 0], concate_transform_split[i][20:40, 1], concate_transform_split[i][20:40, 2], linewidth=0.5,color='orange')
            # ax.plot(concate_transform_split[i][29:48, 0], concate_transform_split[i][29:48, 1], concate_transform_split[i][29:48, 2], linewidth=0.5,color='tab:blue')
            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  linewidth=0.2,marker='*', color='blue')
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  linewidth=0.2,marker='o', color='r')


        else:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], linewidth=0.5,color='tab:purple')
            #ax.plot(concate_transform_split[i][5:10, 0], concate_transform_split[i][5:10, 1], concate_transform_split[i][5:10, 2], linewidth=0.5,color='lime')
            #ax.plot(concate_transform_split[i][9:30, 0], concate_transform_split[i][9:30, 1], concate_transform_split[i][9:30, 2], linewidth=0.5,color='orange')

            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  linewidth=0.2,marker='*', color='blue')
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  linewidth=0.2,marker='o', color='g')


    ax.set_xlabel('integ-PC3', fontsize=fs,labelpad=-5)
    ax.set_ylabel('integ-PC2', fontsize=fs,labelpad=-5)
    ax.set_zlabel('integ-PC1', fontsize=fs,labelpad=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(True)
    #(30, 80);(45, 120)-3;(60, 90)-1or3;(60, 120)-1or3,
    #(80, 120)-3,5;(80, 150)
    #ax.view_init(90, 0)#PC1-PC2=(90, 0);PC2-PC3=(0, 0);PC1-PC3=(0, 90);
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(model_name+'/'+str(idx)+':'+str(start_time[0])+'_'+str(end_time[0]))

    # # # color bar
    # fig2 = plt.figure(figsize=(3, 2.1))
    # ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    # cmap = plt.get_cmap('rainbow', batch_size)
    # mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    # ax2.set_yticklabels(['0','', '1'], size=fs)
    # ax2.set_title('cue', fontsize=fs)

    plt.show()
    return fig


def PCA_plot_2D_cue(figure_path,data_path,model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):
    epochs = get_epoch(hp)

    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1



    batch_size=50

    trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir,hp=hp,task_name=task_name,batch_size=batch_size,
                                                      cue=cue,p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)



    trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
                                                      batch_size=batch_size,
                                                      cue=-cue, p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)

    # np.save(data_path + 'run_result_0.npy', run_result_0)
    # run_result_0 = np.load(data_path + 'run_result_0.npy')
    #
    # np.save(data_path + 'run_result_1.npy', run_result_1)
    # run_result_1 = np.load(data_path + 'run_result_1.npy')






    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = trial_input_0.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    print("concate_firing_rate_0", concate_firing_rate_0.shape)




    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']
    firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)
    print("concate_firing_rate_1", concate_firing_rate_1.shape)


    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)
    print("*** concate_firing_rate", concate_firing_rate.shape)

    # plot
    _alpha_list = [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8]

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)

    explained_variance_ratio=pca.explained_variance_ratio_
    print('explained_variance_ratio',explained_variance_ratio)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time-start_time
    delim = np.cumsum(time_size)
    print('delim',delim)

    print("##concate_firing_rate_transform",concate_firing_rate_transform.shape)
    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)
    print('concate_transform_split',len(concate_transform_split))


    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    colors_0 = sns.color_palette("Paired")
    colors_1 =sns.color_palette("husl", 8)


    for i in range(0, len(concate_transform_split)):

        if i<batch_size:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color=colors_1[0],zorder=0)
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1],linewidth=0.2,marker='o', color='red')


        else:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color=colors_1[5],zorder=1)
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.2,marker='o', color='blue')

        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='green')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='green')

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top','left','bottom']].set_visible(False)


    ax.grid(True)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(model_name+'/'+str(idx)+':'+str(start_time[0])+'_'+str(end_time[0]),fontsize=5)
    plt.savefig(figure_path + 'PCA_conA2B_fail'+str(start_time[0])+'_'+str(end_time[0])+'.eps', format='eps', dpi=1000)

    # # # color bar
    # fig2 = plt.figure(figsize=(3, 2.1))
    # ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    # cmap = plt.get_cmap('rainbow', batch_size)
    # mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    # ax2.set_yticklabels(['0','', '1'], size=fs)
    # ax2.set_title('cue', fontsize=fs)

    plt.show()
    return fig



def cal_explained_variance_pfc(figure_path,data_path,model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):


    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1

    batch_size=50


    ########################## generate the data ##############################

    trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
                                                      batch_size=batch_size,
                                                      cue=cue, p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)

    trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
                                                      batch_size=batch_size,
                                                      cue=-cue, p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)
    firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()


    with open(data_path + 'epochs.pickle', 'wb') as f:
        pickle.dump(trial_input_0.epochs, f)

    with open(data_path + 'epochs.pickle', 'rb') as f:
        epochs = pickle.load(f)

    ############################################################################
    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)


    ############################################################################
    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']

    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)

    # plot

    pca = PCA(n_components=15)
    pca.fit(concate_firing_rate)

    explained_variance_ratio=pca.explained_variance_ratio_

    add_value = []
    j = 0
    for i in explained_variance_ratio:
        j = j + i
        add_value.append(j)

    print('** explained_variance_ratio_pfc', explained_variance_ratio)
    print('** add_value_pfc', add_value)

    return add_value


def cal_explained_variance_md(figure_path,data_path,model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):


    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1

    batch_size=50


    ########################## generate the data ##############################

    trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
                                                      batch_size=batch_size,
                                                      cue=cue, p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)

    trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
                                                      batch_size=batch_size,
                                                      cue=-cue, p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)
    firing_rate_cue_0 = run_result_0.firing_rate_md.detach().cpu().numpy()
    firing_rate_cue_1 = run_result_1.firing_rate_md.detach().cpu().numpy()


    with open(data_path + 'epochs.pickle', 'wb') as f:
        pickle.dump(trial_input_0.epochs, f)

    with open(data_path + 'epochs.pickle', 'rb') as f:
        epochs = pickle.load(f)

    ############################################################################
    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)


    ############################################################################
    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']

    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)

    # plot

    pca = PCA(n_components=15)
    pca.fit(concate_firing_rate)

    explained_variance_ratio=pca.explained_variance_ratio_




    add_value = []
    j=0
    for i in explained_variance_ratio:
        j = j + i
        add_value.append(j)


    print('explained_variance_ratio_md',explained_variance_ratio)
    print('add_value_md', add_value)
    return add_value




def subspace_angle(subspace1, subspace2):
    """
    Calculate the angle between two 2D subspaces.

    Parameters:
    subspace1: 2D array representing the basis of the first subspace.
    subspace2: 2D array representing the basis of the second subspace.

    Returns:
    angle: Angle between the subspaces in radians.
    """
    # Compute the Gram-Schmidt orthogonalization for each subspace
    q1, _ = np.linalg.qr(subspace1)
    q2, _ = np.linalg.qr(subspace2)

    # Compute the dot product of the first basis vector of each subspace
    dot_product = np.dot(q1[:, 0], q2[:, 0])

    # Compute the angle in radians
    angle = np.arccos(np.abs(dot_product))

    return angle



def subspace_angle1(subspace_a, subspace_b):
    min_angle = np.inf

    # Iterate over each vector in subspace_a
    for vector_a in subspace_a.T:  # Transpose to iterate over columns
        # Iterate over each vector in subspace_b
        for vector_b in subspace_b.T:  # Transpose to iterate over columns
            # Compute the angle between the two vectors
            angle = np.arccos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
            # Update min_angle if necessary
            min_angle = min(min_angle, angle)

    # Convert the angle from radians to degrees
    min_angle_degrees = min_angle#np.degrees(min_angle)
    return min_angle_degrees




def angle_between_md_pfc_conA(figure_path,data_path,model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):
    epochs = get_epoch(hp)

    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1



    batch_size=200

    trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir,hp=hp,task_name=task_name,batch_size=batch_size,
                                                      cue=cue,p_coh=p_coh,
                                                      c_vis=None,
                                                      c_aud=None)


    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = trial_input_0.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    # firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    # concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    # print("concate_firing_rate_0", concate_firing_rate_0.shape)

    pfc_fr_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    fr_list_0 = list(pfc_fr_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_0 = np.concatenate(fr_list_0, axis=0)

    md_fr_1 = run_result_0.firing_rate_md.detach().cpu().numpy()
    fr_list_1 = list(md_fr_1[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_1 = np.concatenate(fr_list_1, axis=0)

    # plot
    pca_0 = PCA(n_components=2)
    pca_0.fit(concate_fr_0)
    concate_fr_transform_0 = pca_0.transform(concate_fr_0)

    # plot
    pca_1 = PCA(n_components=2)
    pca_1.fit(concate_fr_1)
    concate_fr_transform_1 = pca_1.transform(concate_fr_1)

    subspace1 = concate_fr_transform_0
    subspace2 = concate_fr_transform_1
    angle_radians = subspace_angle(subspace1, subspace2)
    angle_degrees = np.degrees(angle_radians)
    print("Angle between pfc-MD subspaces:", angle_degrees, "degrees")



    return angle_degrees





def angle_between_conA_conB(figure_path,data_path,model_dir_A,model_dir_A2B,hp,
                    start_time=0,
                    end_time=0,neuron=None,):
    p_coh = 0.99
    batch_size=200


    ########################## generate the data ##############################

    trial_input_A_0, run_result_A_0 = generate_test_trial_context(model_dir=model_dir_A, context='con_A',hp=hp, task_name='HL_task',
                                                      batch_size=batch_size,cue=-1, p_coh=p_coh)

    trial_input_B_0, run_result_B_0 = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                          task_name='HL_task',
                                                          batch_size=batch_size, cue=1, p_coh=p_coh)



    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A_0.epochs['stimulus']


    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    pfc_fr_A_0 = run_result_A_0.firing_rate_binder.detach().cpu().numpy()

    if neuron == 'md':
        pfc_fr_A_0 = run_result_A_0.firing_rate_md.detach().cpu().numpy()

    fr_list_A_0 = list(pfc_fr_A_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_0 = np.concatenate(fr_list_A_0, axis=0)


    pfc_fr_B_0 = run_result_B_0.firing_rate_binder.detach().cpu().numpy()
    if neuron == 'md':
        pfc_fr_B_0 = run_result_B_0.firing_rate_md.detach().cpu().numpy()


    fr_list_B_0 = list(pfc_fr_B_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_0 = np.concatenate(fr_list_B_0, axis=0)

    # plot
    pca_0 = PCA(n_components=3)
    pca_0.fit(concate_fr_A_0)
    explained_var_A_0 = pca_0.explained_variance_ratio_
    print('********** explained_var_A_0',explained_var_A_0)
    concate_fr_transform_A_0 = pca_0.transform(concate_fr_A_0)

    # plot
    pca_1 = PCA(n_components=3)
    pca_1.fit(concate_fr_B_0)
    concate_fr_transform_B_0 = pca_1.transform(concate_fr_B_0)

    subspace1 = concate_fr_transform_A_0
    subspace2 = concate_fr_transform_B_0
    angle_radians = subspace_angles(subspace1, subspace2)
    angle_degrees = np.degrees(angle_radians)
    print("Angle between pfc-MD subspaces:", angle_degrees, "degrees")

    degree = (angle_degrees[0]+angle_degrees[1]+angle_degrees[2])/3
    #degree = (angle_degrees[0]+angle_degrees[1])/2
    degree = angle_degrees[0]

    return degree


def angle_between_conB_conA1(figure_path,data_path,model_dir_1,model_dir_2,hp,
                    start_time=0,
                    end_time=0,neuron=None,):
    p_coh = 0.99
    batch_size=200


    ########################## generate the data ##############################

    trial_input_A_0, run_result_A_0 = generate_test_trial_context(model_dir=model_dir_1, context='con_A2B',hp=hp, task_name='HL_task',
                                                      batch_size=batch_size,cue=1, p_coh=p_coh)

    trial_input_B_0, run_result_B_0 = generate_test_trial_context(model_dir=model_dir_2, context='con_B2A', hp=hp,
                                                          task_name='HL_task',
                                                          batch_size=batch_size, cue=-1, p_coh=p_coh)

    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A_0.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    if neuron=='pfc':
        pfc_fr_A_0 = run_result_A_0.firing_rate_binder.detach().cpu().numpy()

    elif neuron == 'md':
        pfc_fr_A_0 = run_result_A_0.firing_rate_md.detach().cpu().numpy()

    fr_list_A_0 = list(pfc_fr_A_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_0 = np.concatenate(fr_list_A_0, axis=0)

    if neuron=='pfc':
        pfc_fr_B_0 = run_result_B_0.firing_rate_binder.detach().cpu().numpy()
    elif neuron == 'md':
        pfc_fr_B_0 = run_result_B_0.firing_rate_md.detach().cpu().numpy()



    fr_list_B_0 = list(pfc_fr_B_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_0 = np.concatenate(fr_list_B_0, axis=0)

    # plot
    pca_0 = PCA(n_components=3)
    pca_0.fit(concate_fr_A_0)
    concate_fr_transform_A_0 = pca_0.transform(concate_fr_A_0)

    # plot
    pca_1 = PCA(n_components=3)
    pca_1.fit(concate_fr_B_0)
    concate_fr_transform_B_0 = pca_1.transform(concate_fr_B_0)

    subspace1 = concate_fr_transform_A_0
    subspace2 = concate_fr_transform_B_0

    angle_radians = subspace_angles(subspace1, subspace2)
    angle_degrees = np.degrees(angle_radians)
    print("Angle between pfc-MD subspaces:", angle_degrees, "degrees")

    degree = (angle_degrees[0] + angle_degrees[1] + angle_degrees[2]) / 3
    #degree = (angle_degrees[0] + angle_degrees[1]) / 2
    degree = angle_degrees[0]

    return degree





def angle_between_conA1_conA(figure_path,data_path,model_dir_1,model_dir_2,hp,
                    start_time=0,
                    end_time=0,neuron=None,):
    p_coh = 0.99
    batch_size=100


    ########################## generate the data ##############################

    trial_input_A_0, run_result_A_0 = generate_test_trial_context(model_dir=model_dir_1, context='con_B2A',hp=hp, task_name='HL_task',
                                                      batch_size=batch_size,cue=-1, p_coh=p_coh)

    trial_input_B_0, run_result_B_0 = generate_test_trial_context(model_dir=model_dir_2, context='con_A', hp=hp,
                                                          task_name='HL_task',
                                                          batch_size=batch_size, cue=-1, p_coh=p_coh)

    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A_0.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    if neuron=='pfc':
        pfc_fr_A_0 = run_result_A_0.firing_rate_binder.detach().cpu().numpy()

    elif neuron == 'md':
        pfc_fr_A_0 = run_result_A_0.firing_rate_md.detach().cpu().numpy()

    fr_list_A_0 = list(pfc_fr_A_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_0 = np.concatenate(fr_list_A_0, axis=0)

    if neuron=='pfc':
        pfc_fr_B_0 = run_result_B_0.firing_rate_binder.detach().cpu().numpy()
    elif neuron == 'md':
        pfc_fr_B_0 = run_result_B_0.firing_rate_md.detach().cpu().numpy()



    fr_list_B_0 = list(pfc_fr_B_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_0 = np.concatenate(fr_list_B_0, axis=0)

    # plot
    pca_0 = PCA(n_components=3)
    pca_0.fit(concate_fr_A_0)
    concate_fr_transform_A_0 = pca_0.transform(concate_fr_A_0)

    # plot
    pca_1 = PCA(n_components=3)
    pca_1.fit(concate_fr_B_0)
    concate_fr_transform_B_0 = pca_1.transform(concate_fr_B_0)

    subspace1 = concate_fr_transform_A_0
    subspace2 = concate_fr_transform_B_0

    angle_radians = subspace_angles(subspace1, subspace2)
    #angle_radians = subspace_angle1(subspace1, subspace2)




    angle_degrees = np.degrees(angle_radians)
    print("Angle between conA1 and conA subspaces:", angle_degrees, "degrees")

    #degree = (angle_degrees[0] + angle_degrees[1]) / 2
    degree = (angle_degrees[0] + angle_degrees[1] + angle_degrees[2]) / 3
    degree = angle_degrees[0]


    return degree







def angle_between_different_context(figure_path,data_path,model_dir_A,model_dir_A2B,model_dir_B2A,hp,
                    start_time=0,
                    end_time=0,neuron=None,cue=None,p_coh=None):
    p_coh = p_coh
    batch_size=100

    c_vis=None
    c_aud =None
    ########################## generate the data ##############################

    trial_input_A, run_result_A = generate_test_trial_context(model_dir=model_dir_A, context='con_A', hp=hp,task_name='HL_task',
                                                batch_size=batch_size, cue=cue,c_vis=c_vis,c_aud=c_aud,p_coh=p_coh)


    if cue is None:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=cue,c_vis=c_vis,c_aud=c_aud,p_coh=p_coh)
    else:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=-cue,c_vis=c_vis,c_aud=c_aud,p_coh=p_coh)


    trial_input_B2A, run_result_B2A = generate_test_trial_context(model_dir=model_dir_B2A, context='con_B2A',hp=hp, task_name='HL_task',
                                                batch_size=batch_size,cue=cue,c_vis=c_vis,c_aud=c_aud,p_coh=p_coh)



    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    if neuron=='pfc':
        pfc_fr_A = run_result_A.firing_rate_binder.detach().cpu().numpy()[:,:,:256]
        print('pfc_fr_A', pfc_fr_A.shape)
        pfc_fr_A2B = run_result_A2B.firing_rate_binder.detach().cpu().numpy()[:,:,:256]
        pfc_fr_B2A = run_result_B2A.firing_rate_binder.detach().cpu().numpy()[:,:,:256]

    elif neuron == 'md':
        # pfc_fr_A = run_result_A.firing_rate_md.detach().cpu().numpy()[:,:,:12]
        # pfc_fr_A2B = run_result_A2B.firing_rate_md.detach().cpu().numpy()[:,:,:12]
        # pfc_fr_B2A = run_result_B2A.firing_rate_md.detach().cpu().numpy()[:,:,:12]

        # pfc_fr_A = run_result_A.firing_rate_md.detach().cpu().numpy()[:, :, 12:]
        # pfc_fr_A2B = run_result_A2B.firing_rate_md.detach().cpu().numpy()[:, :, 12:]
        # pfc_fr_B2A = run_result_B2A.firing_rate_md.detach().cpu().numpy()[:, :, 12:]

        pfc_fr_A = run_result_A.firing_rate_md.detach().cpu().numpy()[:, :, :]
        pfc_fr_A2B = run_result_A2B.firing_rate_md.detach().cpu().numpy()[:, :, :]
        pfc_fr_B2A = run_result_B2A.firing_rate_md.detach().cpu().numpy()[:, :, :]


    fr_list_A = list(pfc_fr_A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A = np.concatenate(fr_list_A, axis=0)

    fr_list_A2B = list(pfc_fr_A2B[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A2B = np.concatenate(fr_list_A2B, axis=0)

    fr_list_B2A = list(pfc_fr_B2A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B2A = np.concatenate(fr_list_B2A, axis=0)



    # plot
    pca_A = PCA(n_components=3)
    pca_A.fit(concate_fr_A)
    concate_fr_transform_A = pca_A.transform(concate_fr_A)

    pca_A2B = PCA(n_components=3)
    pca_A2B.fit(concate_fr_A2B)
    concate_fr_transform_A2B = pca_A2B.transform(concate_fr_A2B)

    pca_B2A = PCA(n_components=3)
    pca_B2A.fit(concate_fr_B2A)
    concate_fr_transform_B2A = pca_A.transform(concate_fr_B2A)

    #print('******* concate_fr_transform_A',concate_fr_transform_A.shape)

    explained_variance_ratio_A = pca_A.explained_variance_ratio_
    print('explained_variance_ratio_A', explained_variance_ratio_A)
    explained_variance_ratio_A2B = pca_A2B.explained_variance_ratio_
    print('explained_variance_ratio_A2B', explained_variance_ratio_A2B)

    explained_variance_ratio_B2A = pca_B2A.explained_variance_ratio_
    print('explained_variance_ratio_B2A', explained_variance_ratio_B2A)

    subspace1 = concate_fr_transform_A
    subspace2 = concate_fr_transform_A2B
    subspace3 = concate_fr_transform_B2A

    angle_radians_0 = subspace_angles(subspace1, subspace2)
    angle_radians_1 = subspace_angles(subspace2, subspace3)
    angle_radians_2 = subspace_angles(subspace1, subspace3)
    #angle_radians = subspace_angle1(subspace1, subspace2)


    angle_degrees_0 = np.degrees(angle_radians_0)
    angle_degrees_1 = np.degrees(angle_radians_1)
    angle_degrees_2 = np.degrees(angle_radians_2)



    degree_0 =angle_degrees_0[0]# (angle_degrees_0[0]+angle_degrees_0[1]+angle_degrees_0[2])/3
    degree_1 = angle_degrees_1[0]#(angle_degrees_1[0]+angle_degrees_1[1]+angle_degrees_1[2])/3
    degree_2 = angle_degrees_2[0]#(angle_degrees_2[0]+angle_degrees_2[1]+angle_degrees_2[2])/3
    print('angle_degrees_AtoB', angle_degrees_0)
    print('angle_degrees_BtoA1', angle_degrees_1)
    print('angle_degrees_A1toA',angle_degrees_2)


    return degree_0,degree_1,degree_2




def angle_eigenvalue_different_context(figure_path,data_path,model_dir_A,model_dir_A2B,model_dir_B2A,hp,
                    start_time=0,
                    end_time=0,neuron=None,cue=None,p_coh=None):
    p_coh = p_coh
    batch_size=100



    ########################## generate the data ##############################

    trial_input_A, run_result_A = generate_test_trial_context(model_dir=model_dir_A, context='con_A', hp=hp,task_name='HL_task',
                                                batch_size=batch_size, cue=cue, c_vis=0.2,c_aud=-0.2,p_coh=p_coh)


    if cue is None:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=cue,c_vis=0.2,c_aud=-0.2, p_coh=p_coh)
    else:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=-cue, c_vis=0.2,c_aud=-0.2,p_coh=p_coh)


    trial_input_B2A, run_result_B2A = generate_test_trial_context(model_dir=model_dir_B2A, context='con_B2A',hp=hp, task_name='HL_task',
                                                batch_size=batch_size,cue=cue, c_vis=0.2,c_aud=-0.2,p_coh=p_coh)



    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    if neuron=='pfc':
        pfc_fr_A = run_result_A.firing_rate_binder.detach().cpu().numpy()

        pfc_fr_A2B = run_result_A2B.firing_rate_binder.detach().cpu().numpy()
        pfc_fr_B2A = run_result_B2A.firing_rate_binder.detach().cpu().numpy()

    elif neuron == 'md':
        pfc_fr_A = run_result_A.firing_rate_md.detach().cpu().numpy()
        pfc_fr_A2B = run_result_A2B.firing_rate_md.detach().cpu().numpy()
        pfc_fr_B2A = run_result_B2A.firing_rate_md.detach().cpu().numpy()




    fr_list_A = list(pfc_fr_A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A = np.concatenate(fr_list_A, axis=0)

    fr_list_A2B = list(pfc_fr_A2B[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A2B = np.concatenate(fr_list_A2B, axis=0)

    fr_list_B2A = list(pfc_fr_B2A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B2A = np.concatenate(fr_list_B2A, axis=0)



    # plot
    print('concate_fr_A',concate_fr_A.shape)
    pca_A = PCA(n_components=2)
    pca_A.fit(concate_fr_A)
    data_cov_A = pca_A.get_covariance().T
    eigenvalue_A, eigenvector_A = np.linalg.eig(data_cov_A)  # eigenvalue: 特征值，eigenvector: 特征向量
    sorted_id_A = sorted(range(len(eigenvalue_A)), key=lambda k: eigenvalue_A[k], reverse=True)  # 返回降序排列好的特征值对应的索引
    w_A = np.array([eigenvector_A[sorted_id_A[0]], eigenvector_A[sorted_id_A[1]], eigenvector_A[sorted_id_A[2]]])
    print('w_A',w_A.shape)

    pca_A2B = PCA(n_components=2)
    pca_A2B.fit(concate_fr_A2B)
    data_cov_A2B = pca_A2B.get_covariance().T
    eigenvalue_A2B, eigenvector_A2B = np.linalg.eig(data_cov_A2B)  # eigenvalue: 特征值，eigenvector: 特征向量
    sorted_id_A2B = sorted(range(len(eigenvalue_A2B)), key=lambda k: eigenvalue_A2B[k], reverse=True)  # 返回降序排列好的特征值对应的索引
    w_A2B = np.array([eigenvector_A2B[sorted_id_A2B[0]], eigenvector_A2B[sorted_id_A2B[1]], eigenvector_A2B[sorted_id_A2B[2]]])
    print('w_A2B', w_A2B.shape)

    pca_B2A = PCA(n_components=2)
    pca_B2A.fit(concate_fr_B2A)
    data_cov_B2A = pca_B2A.get_covariance().T
    eigenvalue_B2A, eigenvector_B2A = np.linalg.eig(data_cov_B2A)  # eigenvalue: 特征值，eigenvector: 特征向量
    sorted_id_B2A = sorted(range(len(eigenvalue_B2A)), key=lambda k: eigenvalue_B2A[k], reverse=True)  # 返回降序排列好的特征值对应的索引
    w_B2A = np.array([eigenvector_B2A[sorted_id_B2A[0]], eigenvector_B2A[sorted_id_B2A[1]], eigenvector_B2A[sorted_id_B2A[2]]])

    print('w_B2A', w_B2A.shape)




    subspace1 = w_A.T
    subspace2 = w_A2B.T
    subspace3 = w_B2A.T

    angle_radians_0 = subspace_angles(subspace1, subspace2)
    angle_radians_1 = subspace_angles(subspace2, subspace3)
    angle_radians_2 = subspace_angles(subspace1, subspace3)
    #angle_radians = subspace_angle1(subspace1, subspace2)


    angle_degrees_0 = np.degrees(angle_radians_0)
    angle_degrees_1 = np.degrees(angle_radians_1)
    angle_degrees_2 = np.degrees(angle_radians_2)



    degree_0 = angle_degrees_0[0]
    degree_1 = angle_degrees_1[0]
    degree_2 = angle_degrees_2[0]
    print('angle_degrees_A', angle_degrees_0)
    print('angle_degrees_B', angle_degrees_1)
    print('angle_degrees_A1',angle_degrees_2)


    return degree_0,degree_1,degree_2








def angle_plane_between_different_context(figure_path,data_path,model_dir_A,model_dir_A2B,model_dir_B2A,hp,
                    start_time=0,
                    end_time=0,neuron=None,cue=None):
    p_coh = 0.99
    batch_size=1



    ########################## generate the data ##############################

    trial_input_A, run_result_A = generate_test_trial_context(model_dir=model_dir_A, context='con_A', hp=hp,task_name='HL_task',
                                                batch_size=batch_size, cue=cue, p_coh=p_coh)


    if cue is None:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=cue, p_coh=p_coh)
    else:
        trial_input_A2B, run_result_A2B = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
                                                                      task_name='HL_task',
                                                                      batch_size=batch_size, cue=-cue, p_coh=p_coh)


    trial_input_B2A, run_result_B2A = generate_test_trial_context(model_dir=model_dir_B2A, context='con_B2A',hp=hp, task_name='HL_task',
                                                batch_size=batch_size,cue=cue, p_coh=p_coh)



    start_projection = start_time
    end_projection = end_time
    start_time_0, _ = trial_input_A.epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    if neuron=='pfc':
        pfc_fr_A = run_result_A.firing_rate_binder.detach().cpu().numpy()
        pfc_fr_A2B = run_result_A2B.firing_rate_binder.detach().cpu().numpy()
        pfc_fr_B2A = run_result_B2A.firing_rate_binder.detach().cpu().numpy()

    elif neuron == 'md':
        pfc_fr_A = run_result_A.firing_rate_md.detach().cpu().numpy()
        pfc_fr_A2B = run_result_A2B.firing_rate_md.detach().cpu().numpy()
        pfc_fr_B2A = run_result_B2A.firing_rate_md.detach().cpu().numpy()




    fr_list_A = list(pfc_fr_A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A = np.concatenate(fr_list_A, axis=0)

    fr_list_A2B = list(pfc_fr_A2B[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A2B = np.concatenate(fr_list_A2B, axis=0)

    fr_list_B2A = list(pfc_fr_B2A[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B2A = np.concatenate(fr_list_B2A, axis=0)



    # plot
    pca_A = PCA(n_components=3)
    pca_A.fit(concate_fr_A)
    concate_fr_transform_A = pca_A.transform(concate_fr_A)

    pca_A2B = PCA(n_components=3)
    pca_A2B.fit(concate_fr_A2B)
    concate_fr_transform_A2B = pca_A2B.transform(concate_fr_A2B)

    pca_B2A = PCA(n_components=3)
    pca_B2A.fit(concate_fr_B2A)
    concate_fr_transform_B2A = pca_A.transform(concate_fr_B2A)

    print('******* concate_fr_transform_A',concate_fr_transform_A.shape)






    subspace1 = concate_fr_transform_A
    subspace2 = concate_fr_transform_A2B
    subspace3 = concate_fr_transform_B2A

    angle_radians_0 = subspace_angles(subspace1, subspace2)
    angle_radians_1 = subspace_angles(subspace2, subspace3)
    angle_radians_2 = subspace_angles(subspace1, subspace3)
    #angle_radians = subspace_angle1(subspace1, subspace2)


    angle_degrees_0 = np.degrees(angle_radians_0)
    angle_degrees_1 = np.degrees(angle_radians_1)
    angle_degrees_2 = np.degrees(angle_radians_2)



    degree_0 = angle_degrees_0[0]
    degree_1 = angle_degrees_1[0]
    degree_2 = angle_degrees_2[0]
    print('angle_degrees_0', angle_degrees_0)
    print('angle_degrees_1', angle_degrees_1)
    print('angle_degrees_2',angle_degrees_2)


    return degree_0,degree_1,degree_2












