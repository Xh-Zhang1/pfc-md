from __future__ import division
import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import math
import run
import tools



fs=10



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


def generate_test_trial(model_dir,hp,task_name,batch_size=1,
                        sparsity_level=0.0,
                        p_coh=0.9,
                        cue=None,
                        c_vis=None,
                        c_aud=None):

    hp['sparsity_HL']=sparsity_level
    hp['sparsity_RDM']=sparsity_level



    rng  = hp['rng']
    cue_scale=hp['cue_scale']
    gamma_bar_vis = rng.uniform(1.1, 1.1, batch_size)#kwargs['gamma_bar']
    gamma_bar_aud = rng.uniform(-1.1, -1.1, batch_size)#kwargs['gamma_bar']



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




    runnerObj = run.Runner(model_dir=model_dir, rule_name=task_name, hp=hp,is_cuda=False, noise_on=False,mode='test_for_pca')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            p_coh=p_coh,
                                            c_cue=c_cue,
                                            c_vis=c_vis,
                                            c_aud=c_aud)


    return trial_input, run_result


# Regression coefficients
# CHOICE     = 0
# VIS         = 1
# AUD         = 2
# CONSTANT    = 3
# nreg           = 4

RULE      = 0
COH         =1
CHOICE         = 2

CONSTANT    = 3
nreg        = 4
t_step=6




def Plot_statespace_cue_epoch_3coh_HL(data_path,model_name,model_dir,idx,hp,Q_task_name,task_name,sparsity_HL=0,
                                       start_projection_Q=0,
                                       end_projection_Q=0,
                                       start_projection=0,
                                       end_projection=1,
                                       p_cohs_plot=0):

    batch_size=500
    cue_scale = hp['cue_scale']

    # Q = PCA_3D_regress_HL(data_path=data_path, model_dir=model_dir, hp=hp,
    #                                                  batch_size=1,
    #                                                  start_projection=start_projection_Q,
    #                                                  end_projection=end_projection_Q)
    #
    #
    # np.save(data_path+'Q_HL'+str(start_projection_Q)+'_'+str(end_projection_Q)+'.npy', Q)

    Q = np.load(data_path+'Q_HL'+str(start_projection_Q)+'_'+str(end_projection_Q)+'.npy')

    context = 1

    Q = Q.T
    print('Q',Q.shape)

    p_cohs=p_cohs_plot
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[0.9,1.0,0.9,1.0]
    r_0s=[];r_1s=[];r_2s=[];r_3s=[];r_4s=[];r_5s=[];r_6s=[];r_7s=[]
    c=0.2
    context_RDM=1
    context_HL=1


    path = data_path+'Plot_statespace_cue_epoch_3coh_HL'
    tools.mkdir_p(path)


    firing_rate_0 = np.load(path+'firing_rate_0.npy')
    firing_rate_1 = np.load(path+'firing_rate_1.npy')
    firing_rate_2 = np.load(path+'firing_rate_2.npy')
    firing_rate_3 = np.load(path+'firing_rate_3.npy')
    firing_rate_4 = np.load(path+'firing_rate_4.npy')
    firing_rate_5 = np.load(path+'firing_rate_5.npy')



    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0,r_1,r_2,r_3,r_4,r_5])
    print('r_concante',r_concante.shape)

    p_vc={}
    for i in range(6):
        p_vc[str(i)] = r_concante[i,:,::4]
    print(p_vc.keys())

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------
    p = list(p_vc.values())[0]
    Xrule   = np.zeros_like(p[RULE])
    Xcoh    = np.zeros_like(p[COH])
    Xchoice = np.zeros_like(p[CHOICE])

    for p in p_vc.values():#len(p_vc)=6
        Xrule   += p[RULE]
        Xcoh    += p[COH]
        Xchoice += p[CHOICE]

    mean_rule = Xrule/len(p_vc)
    mean_coh    = Xcoh/len(p_vc)
    mean_choice    = Xchoice/len(p_vc)

    for cond, p in p_vc.items():
        p[RULE]   -= mean_rule
        p[COH]    -= mean_coh
        p[CHOICE] -= mean_choice

    #-------------------------------------------------------------------------------------
    # plot
    #-------------------------------------------------------------------------------------

    colors=['tab:blue','skyblue','lightsteelblue','black','gray','silver']
    #colors=['lime','g','tab:blue','b','lime','g','tab:blue','b']
    fig = plt.figure(figsize=(4,3.0))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    plt.title(model_name+task_name)
    #fig, ax = plt.subplots(figsize=(5,3.5))
    ax.axhline(y=0,linewidth=4,color='#E5E5E5')##DBDBDB
    ax.axvline(x=0,linewidth=4,color='#E5E5E5')
    i=0
    #coh=[0.9,1.0,0.9,1.0]
    #coh=[1.0,1.0,1.0,1.0]
    coh=p_cohs_plot

    xax = RULE
    yax = CHOICE
    cue_off_time = int((hp['cue_duration']/hp['dt'])/2)-2
    alphas = [1,1,1,1,1,1]

    lab=['RULE','COH','CHOICE','CONSTANT']

    for cond, p in p_vc.items():
        print('p.shape',p.shape)
        if i<3:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1)
            plt.plot(p[xax], p[yax],'o', color=colors[i],label=str(p_cohs[i]))
        else:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1)
            plt.plot(p[xax], p[yax],'o', color=colors[i], label=str(p_cohs[i-3]))

        i+=1
        plt.legend(fontsize=7)
        plt.plot(p[xax,0],   p[yax,0],'o', markersize=8,color='orange',alpha=1)

        plt.xlabel(lab[xax])
        plt.ylabel(lab[yax])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    if task_name=='RDM_task':
        plt.title(task_name+str(hp['sparsity_RDM']),fontsize=7)
    elif task_name=='HL_task':
        fig.savefig(hp['figure_path']+model_name+task_name+str(hp['sparsity_HL'])+'_cue.eps', format='eps', dpi=1000)

    plt.show()
    #fig.savefig(hp['figure_path']+task_name+'_seed'+str(hp['seed'])+'_cue2.eps', format='eps', dpi=1000)


def Plot_statespace_diff_sparsity_HL(data_path,model_name,model_dir,idx,hp,Q_task_name,task_name,p_coh=0.92,
                              cue=1,
                              start_projection_Q=0,
                              end_projection_Q=0,
                              start_projection=0,
                              end_projection=1,
                              p_cohs_plot=0):

    batch_size=500
    cue_scale = hp['cue_scale']

    Q = np.load(data_path+'Q_HL'+str(start_projection_Q)+'_'+str(end_projection_Q)+'.npy')
    context = 1

    Q = Q.T
    print('Q',Q.shape)

    p_cohs=p_cohs_plot
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[0.9,1.0,0.9,1.0]
    r_0s=[];r_1s=[];r_2s=[];r_3s=[];r_4s=[];r_5s=[];r_6s=[];r_7s=[]
    c=0.2
    context_RDM=1
    context_HL=1
    sparsity_levels=[0.0,0.6,0.8]


    path = data_path+'Plot_statespace_diff_sparsity_HL'
    tools.mkdir_p(path)


    firing_rate_0 = np.load(path+'firing_rate_0.npy')
    firing_rate_1 = np.load(path+'firing_rate_1.npy')
    firing_rate_2 = np.load(path+'firing_rate_2.npy')
    firing_rate_3 = np.load(path+'firing_rate_3.npy')
    firing_rate_4 = np.load(path+'firing_rate_4.npy')
    firing_rate_5 = np.load(path+'firing_rate_5.npy')


    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0,r_1,r_2,r_3,r_4,r_5])
    print('r_concante',r_concante.shape)

    p_vc={}
    for i in range(6):
        p_vc[str(i)] = r_concante[i,:,::4]
    print(p_vc.keys())

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------
    p = list(p_vc.values())[0]
    Xrule   = np.zeros_like(p[RULE])
    Xcoh    = np.zeros_like(p[COH])
    Xchoice = np.zeros_like(p[CHOICE])

    for p in p_vc.values():#len(p_vc)=6
        Xrule   += p[RULE]
        Xcoh    += p[COH]
        Xchoice += p[CHOICE]

    mean_rule = Xrule/len(p_vc)
    mean_coh    = Xcoh/len(p_vc)
    mean_choice    = Xchoice/len(p_vc)

    for cond, p in p_vc.items():
        p[RULE]   -= mean_rule
        p[COH]    -= mean_coh
        p[CHOICE] -= mean_choice

    #-------------------------------------------------------------------------------------
    # plot
    #-------------------------------------------------------------------------------------

    colors=['tab:blue','skyblue','lightsteelblue','black','gray','silver',]
    #colors=['lime','g','tab:blue','b','lime','g','tab:blue','b']
    fig = plt.figure(figsize=(4,3.0))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #fig, ax = plt.subplots(figsize=(5,3.5))
    ax.axhline(y=0,linewidth=4,color='grey',alpha=0.2)
    ax.axvline(x=0,linewidth=4,color='grey',alpha=0.2)
    i=0
    #coh=[0.9,1.0,0.9,1.0]
    #coh=[1.0,1.0,1.0,1.0]
    coh=p_cohs_plot

    xax = RULE
    yax = CHOICE
    alphas = [1,1,1,1,1,1]

    lab=['RULE','COH','CHOICE','CONSTANT']

    for cond, p in p_vc.items():
        print('p.shape',p.shape)
        if i<3:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i],alpha=alphas[i])
        else:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i], alpha=alphas[i])

        i+=1
        #plt.legend()
        plt.plot(p[xax,0],   p[yax,0],'o', markersize=8,color='orange',alpha=1)

        plt.xlabel(lab[xax])
        plt.ylabel(lab[yax])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    if task_name=='RDM_task':
        plt.title(task_name+':coh_'+str(p_cohs),fontsize=8)
        fig.savefig(hp['figure_path']+task_name+'_coh'+str(p_cohs)+'_cue.svg', format='svg', dpi=1000)
        fig.savefig(hp['figure_path']+'_coh'+str(p_cohs[0])+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    elif task_name=='HL_task':
        plt.title(task_name+':coh_'+str(p_cohs),fontsize=8)
        fig.savefig(hp['figure_path']+task_name+'_coh'+str(p_cohs)+'_cue.svg', format='svg', dpi=1000)
        fig.savefig(hp['figure_path']+'_coh'+str(p_cohs)+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    #plt.title(task_name+': '+model_name+'_'+str(idx)+'\n'+'seed'+str(hp['seed'])+'; coh='+str(p_cohs),fontsize=8)
    fig.savefig(hp['figure_path']+task_name+'_seed'+str(hp['seed'])+'_cue', format='svg', dpi=1000)
    fig.savefig(hp['figure_path']+hp['mask_type']+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    plt.show()
    #fig.savefig(hp['figure_path']+task_name+'_seed'+str(hp['seed'])+'_cue2.svg', format='svg', dpi=1000)

def generate_velocity(r_concante_cue):
    p_vc = {}
    for i in range(6):
        p_vc[str(i)] = r_concante_cue[i, :, ::4]

    # -------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by motion
    # -------------------------------------------------------------------------------------
    p = list(p_vc.values())[0]
    Xrule = np.zeros_like(p[RULE])
    Xcoh = np.zeros_like(p[COH])
    Xchoice = np.zeros_like(p[CHOICE])

    for p in p_vc.values():  # len(p_vc)=6
        Xrule += p[RULE]
        Xcoh += p[COH]
        Xchoice += p[CHOICE]

    mean_rule = Xrule / len(p_vc)
    mean_coh = Xcoh / len(p_vc)
    mean_choice = Xchoice / len(p_vc)

    for cond, p in p_vc.items():
        p[RULE] -= mean_rule
        p[COH] -= mean_coh
        p[CHOICE] -= mean_choice

    xax = RULE
    yax = CHOICE

    v_cue = []
    for cond, p in p_vc.items():
        i = int(cond)
        x = p[xax]
        y = p[yax]
        print('x.shape[0]',x.shape[0])
        vx = [x[i + 1] - x[i] for i in range(x.shape[0] - 1)]
        vy = [y[i + 1] - y[i] for i in range(y.shape[0] - 1)]
        v = [np.sqrt(vx[i] ** 2 + vy[i] ** 2) for i in range(x.shape[0] - 1)]

        v_cue.append(v)

    print(len(v_cue))

    return v_cue
def calculate_velocity_cue_HL_separate(data_path, model_name, model_dir, idx, hp, Q_task_name, task_name,
                                       sparsity_HL=0.0,
                                       cue=1,
                                       start_projection_Q=0,
                                       end_projection_Q=0,
                                       start_projection=0,
                                       end_projection=1,
                                       p_cohs_plot=0):
    hp['sparsity_HL'] = sparsity_HL

    batch_size = 500
    cue_scale = hp['cue_scale']

    Q = np.load(data_path + 'Q_HL' + str(start_projection_Q) + '_' + str(end_projection_Q) + '.npy')
    context = 1

    Q = Q.T
    print('Q', Q.shape)

    p_cohs = p_cohs_plot
    # p_cohs=[1.0,1.0,1.0,1.0]
    # p_cohs=[1.0,1.0,1.0,1.0]
    # p_cohs=[0.9,1.0,0.9,1.0]
    r_0s = [];
    r_1s = [];
    r_2s = [];
    r_3s = [];
    r_4s = [];
    r_5s = [];
    r_6s = [];
    r_7s = []
    c = 0.2
    context_RDM = 1
    context_HL = 1

    path = data_path + 'calculate_velocity_cue_HL'
    tools.mkdir_p(path)

    firing_rate_0 = np.load(path + 'firing_rate_0.npy')
    firing_rate_1 = np.load(path + 'firing_rate_1.npy')
    firing_rate_2 = np.load(path + 'firing_rate_2.npy')
    firing_rate_3 = np.load(path + 'firing_rate_3.npy')
    firing_rate_4 = np.load(path + 'firing_rate_4.npy')
    firing_rate_5 = np.load(path + 'firing_rate_5.npy')

    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0, r_1, r_2, r_3, r_4, r_5])
    print('r_concante', r_concante.shape)
    r_concante_cue = r_concante[:, :, 0:40]
    r_concante_delay = r_concante[:, :, 41:]

    print('r_concante_cue', r_concante_cue.shape)
    v_cue = generate_velocity(r_concante_cue)
    print('v_cue', len(v_cue))
    v_delay = generate_velocity(r_concante_delay)

    # -------------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------------
    x = range(9, 9 + 9, 1)
    colors = ['lightsteelblue', 'skyblue', 'tab:blue', 'silver', 'gray', 'black']
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])

    for i in range(6):
        plt.plot(v_cue[i], color=colors[i])
        plt.plot(v_cue[i], 'o', markersize=4, color=colors[i])

        plt.plot(x, v_delay[i], color=colors[i])
        plt.plot(x, v_delay[i], 'o', markersize=4, color=colors[i])

        plt.title(str(hp['model_idx']) + '_' + str(idx) + '_' + task_name + ';' + str(hp['seed']), fontsize=8)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=12)
        # plt.ylim([0,3.5])
        plt.yticks([0, 1, 2, 3], fontsize=12)
        plt.ylabel('Velocity', fontsize=12)

    fig.savefig(
        hp['figure_path'] + 'velocity_' + str(hp['model_idx']) + '_' + str(idx) + '_' + task_name + '_seed' + str(
            hp['seed']) + '_cue.png')
    fig.savefig(hp['figure_path'] + 'velocity_' + str(hp['model_idx']) + '_' + str(idx) + '_' +
                task_name + '_seed' + str(hp['seed']) + '_cue.eps', format='eps', dpi=1000)

    plt.show()




def Plot_statespace_cue_epoch_3coh_RDM(data_path,figure_path,model_name,model_dir,idx,hp,Q_task_name,task_name,sparsity_HL=0,
                                   start_projection_Q=0,
                                   end_projection_Q=0,
                                   start_projection=0,
                                   end_projection=1,
                                   p_cohs_plot=0):
    fig_path = os.path.join(figure_path, 'Plot_statespace_cue_epoch_3coh_RDM' + '/')
    tools.mkdir_p(fig_path)

    batch_size=500
    cue_scale = hp['cue_scale']

    # Q = PCA_3D_regress_RDM(data_path,model_dir=model_dir, hp=hp, batch_size=1,
    #                                                       start_projection=start_projection_Q,
    #                                                       end_projection=end_projection_Q)
    #
    # np.save(data_path+'Q_RDM'+str(start_projection_Q)+'_'+str(end_projection_Q)+'.npy', Q)
    Q = np.load(data_path+'Q_RDM'+str(start_projection_Q)+'_'+str(end_projection_Q)+'.npy')
    context = 1

    Q = Q.T
    print('Q',Q.shape)

    p_cohs=p_cohs_plot
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[0.9,1.0,0.9,1.0]
    r_0s=[];r_1s=[];r_2s=[];r_3s=[];r_4s=[];r_5s=[];r_6s=[];r_7s=[]
    c=0.2
    context_RDM=1
    context_HL=1




    path = data_path+'Plot_statespace_cue_epoch_3coh_RDM'
    tools.mkdir_p(path)

    firing_rate_0 = np.load(path+'firing_rate_0.npy')
    firing_rate_1 = np.load(path+'firing_rate_1.npy')
    firing_rate_2 = np.load(path+'firing_rate_2.npy')
    firing_rate_3 = np.load(path+'firing_rate_3.npy')
    firing_rate_4 = np.load(path+'firing_rate_4.npy')
    firing_rate_5 = np.load(path+'firing_rate_5.npy')



    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0,r_1,r_2,r_3,r_4,r_5])
    print('r_concante',r_concante.shape)

    p_vc={}
    for i in range(6):
        p_vc[str(i)] = r_concante[i,:,::4]
    print(p_vc.keys())

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------
    p = list(p_vc.values())[0]
    Xrule   = np.zeros_like(p[RULE])
    Xcoh    = np.zeros_like(p[COH])
    Xchoice = np.zeros_like(p[CHOICE])

    for p in p_vc.values():#len(p_vc)=6
        Xrule   += p[RULE]
        Xcoh    += p[COH]
        Xchoice += p[CHOICE]

    mean_rule = Xrule/len(p_vc)
    mean_coh    = Xcoh/len(p_vc)
    mean_choice    = Xchoice/len(p_vc)

    for cond, p in p_vc.items():
        p[RULE]   -= mean_rule
        p[COH]    -= mean_coh
        p[CHOICE] -= mean_choice

    #-------------------------------------------------------------------------------------
    # plot
    #-------------------------------------------------------------------------------------

    colors=['tab:blue','skyblue','lightsteelblue','black','gray','silver']
    #colors=['lime','g','tab:blue','b','lime','g','tab:blue','b']
    fig = plt.figure(figsize=(4,3.0))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #fig, ax = plt.subplots(figsize=(5,3.5))
    ax.axhline(y=0,linewidth=4,color='#E5E5E5')
    ax.axvline(x=0,linewidth=4,color='#E5E5E5')
    i=0
    #coh=[0.9,1.0,0.9,1.0]
    #coh=[1.0,1.0,1.0,1.0]
    coh=p_cohs_plot

    xax = RULE
    yax = CHOICE
    cue_off_time = int((hp['cue_duration']/hp['dt'])/2)-2
    alphas = [1,1,1,1,1,1]

    lab=['RULE','COH','CHOICE','CONSTANT']

    for cond, p in p_vc.items():
        print('p.shape',p.shape)
        if i<3:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i],alpha=alphas[i])
        else:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i], alpha=alphas[i])

        i+=1
        #plt.legend()
        plt.plot(p[xax,0],   p[yax,0],'o', markersize=8,color='orange',alpha=1)

        plt.xlabel(lab[xax])
        plt.ylabel(lab[yax])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    if task_name=='RDM_task':
        plt.title(task_name+str(hp['sparsity_RDM'])+'_'+str(hp['seed']),fontsize=7)
    elif task_name=='HL_task':
        plt.title(task_name+str(hp['sparsity_level']),fontsize=7)
    #plt.title(task_name+': '+model_name+'_'+str(idx)+'\n'+'seed'+str(hp['seed'])+'; coh='+str(p_cohs),fontsize=8)
    #fig.savefig(hp['figure_path']+task_name+'_seed'+str(hp['seed'])+'_cue', format='svg', dpi=1000)
    #fig.savefig(hp['figure_path']+hp['mask_type']+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    plt.show()
    fig.savefig(fig_path + task_name + '_seed' + str(hp['seed']) + '.png')
    fig.savefig(fig_path+task_name+'_seed'+str(hp['seed'])+'_cue2.eps', format='eps', dpi=1000)


def Plot_statespace_diff_sparsity_RDM(data_path,figure_path, model_name,model_dir,idx,hp,Q_task_name,task_name,p_coh=0.92,
                                     cue=1,
                                     start_projection_Q=0,
                                     end_projection_Q=0,
                                     start_projection=0,
                                     end_projection=1,
                                     p_cohs_plot=0):
    fig_path = os.path.join(figure_path, 'Plot_statespace_diff_sparsity_RDM' + '/')
    tools.mkdir_p(fig_path)

    batch_size=500
    cue_scale = hp['cue_scale']


    Q = np.load(data_path + 'Q_RDM' + str(start_projection_Q) + '_' + str(end_projection_Q) + '.npy')

    Q = Q.T
    print('Q',Q.shape)

    p_cohs=p_cohs_plot
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[1.0,1.0,1.0,1.0]
    #p_cohs=[0.9,1.0,0.9,1.0]
    r_0s=[];r_1s=[];r_2s=[];r_3s=[];r_4s=[];r_5s=[];r_6s=[];r_7s=[]
    c=0.2
    context=1
    sparsity_levels=[0.0,0.6,0.8]


    path = data_path+'Plot_statespace_diff_sparsity_RDM'
    tools.mkdir_p(path)


    firing_rate_0 = np.load(path+'firing_rate_0.npy')
    firing_rate_1 = np.load(path+'firing_rate_1.npy')
    firing_rate_2 = np.load(path+'firing_rate_2.npy')
    firing_rate_3 = np.load(path+'firing_rate_3.npy')
    firing_rate_4 = np.load(path+'firing_rate_4.npy')
    firing_rate_5 = np.load(path+'firing_rate_5.npy')


    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0,r_1,r_2,r_3,r_4,r_5])
    print('r_concante',r_concante.shape)

    p_vc={}
    for i in range(6):
        p_vc[str(i)] = r_concante[i,:,::4]
    print(p_vc.keys())

    #-------------------------------------------------------------------------------------
    # Motion context: motion vs. choice, sorted by motion
    #-------------------------------------------------------------------------------------
    p = list(p_vc.values())[0]
    Xrule   = np.zeros_like(p[RULE])
    Xcoh    = np.zeros_like(p[COH])
    Xchoice = np.zeros_like(p[CHOICE])

    for p in p_vc.values():#len(p_vc)=6
        Xrule   += p[RULE]
        Xcoh    += p[COH]
        Xchoice += p[CHOICE]

    mean_rule = Xrule/len(p_vc)
    mean_coh    = Xcoh/len(p_vc)
    mean_choice    = Xchoice/len(p_vc)

    for cond, p in p_vc.items():
        p[RULE]   -= mean_rule
        p[COH]    -= mean_coh
        p[CHOICE] -= mean_choice

    #-------------------------------------------------------------------------------------
    # plot
    #-------------------------------------------------------------------------------------

    colors=['tab:blue','skyblue','lightsteelblue','black','gray','silver',]
    #colors=['lime','g','tab:blue','b','lime','g','tab:blue','b']
    fig = plt.figure(figsize=(4,3.0))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #fig, ax = plt.subplots(figsize=(5,3.5))
    ax.axhline(y=0,linewidth=4,color='grey',alpha=0.2)
    ax.axvline(x=0,linewidth=4,color='grey',alpha=0.2)
    i=0
    #coh=[0.9,1.0,0.9,1.0]
    #coh=[1.0,1.0,1.0,1.0]
    coh=p_cohs_plot

    xax = RULE
    yax = CHOICE
    alphas = [1,1,1,1,1,1]

    lab=['RULE','COH','CHOICE','CONSTANT']

    for cond, p in p_vc.items():
        print('p.shape',p.shape)
        if i<3:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i],alpha=alphas[i])
        else:
            plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
            plt.plot(p[xax], p[yax],'o', color=colors[i], alpha=alphas[i])

        i+=1
        #plt.legend()
        plt.plot(p[xax,0],   p[yax,0],'o', markersize=8,color='orange',alpha=1)

        plt.xlabel(lab[xax])
        plt.ylabel(lab[yax])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #plt.ylim([-2.5,2.5])
    plt.xticks([])
    plt.yticks([])
    if task_name=='RDM_task':
        plt.title(task_name+'_seed'+str(hp['seed']),fontsize=8)
    elif task_name=='HL_task':
        plt.title(task_name+':coh_'+str(p_cohs),fontsize=8)
        fig.savefig(hp['figure_path']+task_name+'_coh'+str(p_cohs)+'_cue.svg', format='svg', dpi=1000)
        fig.savefig(hp['figure_path']+'_coh'+str(p_cohs)+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    #plt.title(task_name+': '+model_name+'_'+str(idx)+'\n'+'seed'+str(hp['seed'])+'; coh='+str(p_cohs),fontsize=8)
    fig.savefig(fig_path+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.eps', format='eps', dpi=1000)
    fig.savefig(fig_path+'_'+str(hp['model_idx'])+'_'+str(idx)+'_'+task_name+'_seed'+str(hp['seed'])+'_cue.png')
    plt.show()
    #fig.savefig(hp['figure_path']+task_name+'_seed'+str(hp['seed'])+'_cue2.svg', format='svg', dpi=1000)



def calculate_velocity_cue_RDM_separate(data_path,model_name,model_dir,idx,hp,Q_task_name,task_name,sparsity_RDM=0.0,
                              cue=1,
                              start_projection_Q=0,
                              end_projection_Q=0,
                              start_projection=0,
                              end_projection=1,
                              p_cohs_plot=0):
    hp['sparsity_RDM']=sparsity_RDM

    batch_size=500
    cue_scale = hp['cue_scale']

    Q = np.load(data_path + 'Q_RDM' + str(start_projection_Q) + '_' + str(end_projection_Q) + '.npy')

    Q = Q.T

    p_cohs=p_cohs_plot

    r_0s=[];r_1s=[];r_2s=[];r_3s=[];r_4s=[];r_5s=[];r_6s=[];r_7s=[]
    c=0.2
    context_RDM=1
    context_HL=1
    path = data_path + 'calculate_velocity_cue_RDM'
    tools.mkdir_p(path)


    firing_rate_0 = np.load(path+'firing_rate_0.npy')
    firing_rate_1 = np.load(path+'firing_rate_1.npy')
    firing_rate_2 = np.load(path+'firing_rate_2.npy')
    firing_rate_3 = np.load(path+'firing_rate_3.npy')
    firing_rate_4 = np.load(path+'firing_rate_4.npy')
    firing_rate_5 = np.load(path+'firing_rate_5.npy')

    r_0 = Q.dot(firing_rate_0)
    r_1 = Q.dot(firing_rate_1)
    r_2 = Q.dot(firing_rate_2)

    r_3 = Q.dot(firing_rate_3)
    r_4 = Q.dot(firing_rate_4)
    r_5 = Q.dot(firing_rate_5)

    r_concante = np.array([r_0, r_1, r_2, r_3, r_4, r_5])
    print('r_concante', r_concante.shape)
    r_concante_cue = r_concante[:, :, 0:40]
    r_concante_delay = r_concante[:, :, 41:]

    print('r_concante_cue', r_concante_cue.shape)
    v_cue = generate_velocity(r_concante_cue)
    print('v_cue', len(v_cue))
    v_delay = generate_velocity(r_concante_delay)


    x = range(9, 9 + 9, 1)
    colors = ['lightsteelblue', 'skyblue', 'tab:blue', 'silver', 'gray', 'black']
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])


    for i in range(6):
        plt.plot(v_cue[i], color=colors[i])
        plt.plot(v_cue[i], 'o', markersize=4, color=colors[i])

        plt.plot(x,v_delay[i], color=colors[i])
        plt.plot(x,v_delay[i], 'o', markersize=4, color=colors[i])

        plt.title(str(hp['model_idx']) + '_' + str(idx) + '_' + task_name + ';' + str(hp['seed']), fontsize=8)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=12)
        # plt.ylim([0,3.5])
        plt.yticks([0, 1, 2, 3,4], fontsize=12)
        plt.ylabel('Velocity', fontsize=12)

    plt.title(task_name, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=12)
    # plt.yticks([0,1,2,3,4], fontsize=12)
    # plt.ylim([0,4])
    plt.ylabel('Velocity', fontsize=12)




    # fig2=plt.figure()
    # for cond, p in p_vc.items():
    #     i=int(cond)
    #     print('p.shape',p.shape)
    #     plt.plot(p[xax], p[yax],    '-', color=colors[i], lw=1,alpha=alphas[i])
    #     plt.plot(p[xax], p[yax],'o', color=colors[i],alpha=alphas[i])
    #     #plt.legend()
    #     plt.plot(p[xax,0],   p[yax,0],'o', markersize=8,color='orange',alpha=1)
    #     plt.xlabel(lab[xax])
    #     plt.ylabel(lab[yax])

    fig.savefig(
        hp['figure_path'] + 'velocity_' + str(hp['model_idx']) + '_' + str(idx) + '_' + task_name + '_seed' + str(
            hp['seed']) + '_cue.png')
    fig.savefig(hp['figure_path'] + 'velocity_' + str(hp['model_idx']) + '_' + str(idx) + '_' +
                task_name + '_seed' + str(hp['seed']) + '_cue.eps', format='eps', dpi=1000)

    plt.show()



