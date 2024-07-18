import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import default
import tools
import neuron_activity_supple_lib


hp=default.get_default_hp(rule_name='HL_task',random_seed=1)

hp['stim_std']=0.1
hp['dropout_model']=0.0
hp['model_idx']=59
idx=9
hp['stim_std_test']=0.1

#=========================  plot =============================
data_path = os.path.join(hp['root_path'], 'Datas', 'fig7_MD_PFC_lib/')
tools.mkdir_p(data_path)

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'fig7_MD_PFC_lib/')
tools.mkdir_p(figure_path)

model_prefix = 'type3_256_200_softplus_sr5.0_0.1_drop0.0_'

cohs_HL  = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9])
cohs_RDM = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])


local_folder_name = os.path.join('/'+'model_'+str(0.92),  model_prefix+str(hp['model_idx']), str(idx))
model_dir = hp['root_path']+local_folder_name+'/'
epoch = neuron_activity_supple_lib.get_epoch(model_dir, hp)



#def sparsity_SOM2VIP_SOM2PV_RDM
batch_size=100
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

sparsity_SOM_VIP=np.load(data_path+'sparsity_SOM_VIP_RDM.npy')
sparsity_SOM_PV =np.load(data_path+'sparsity_SOM_PV_RDM.npy')


sparsity_SOM_VIP_mean = np.mean(sparsity_SOM_VIP,axis=0)
sparsity_SOM_PV_mean = np.mean(sparsity_SOM_PV,axis=0)
sparsity_SOM_VIP_std = np.std(sparsity_SOM_VIP,axis=0)
sparsity_SOM_PV_std = np.std(sparsity_SOM_PV,axis=0)


percent_zeros = 100*percent_zeros

fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])
title_name='RDM:SOM_VIP vs SOM_PV'

plt.errorbar(percent_zeros[:],sparsity_SOM_VIP_mean[:], sparsity_SOM_VIP_std[:], color = '#DDA0DD', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_VIP_mean[:], color = '#DDA0DD',label = "SOM-VIP")

plt.errorbar(percent_zeros[:],sparsity_SOM_PV_mean[:],  sparsity_SOM_PV_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_PV_mean[:], '-',color = '#E9967A', label = "SOM-PV")

plt.title(title_name, fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()


#sparsity_SOM2VIP_SOM2PV_HL
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

sparsity_SOM_VIP=np.load(data_path+'sparsity_SOM_VIP_HL.npy')
sparsity_SOM_PV =np.load(data_path+'sparsity_SOM_PV_HL.npy')

sparsity_SOM_VIP_mean = np.mean(sparsity_SOM_VIP,axis=0)
sparsity_SOM_PV_mean = np.mean(sparsity_SOM_PV,axis=0)
sparsity_SOM_VIP_std = np.std(sparsity_SOM_VIP,axis=0)
sparsity_SOM_PV_std = np.std(sparsity_SOM_PV,axis=0)


percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],sparsity_SOM_VIP_mean[:], sparsity_SOM_VIP_std[:], color = '#DDA0DD', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_VIP_mean[:], color = '#DDA0DD',label = "SOM-VIP")

plt.errorbar(percent_zeros[:],sparsity_SOM_PV_mean[:],  sparsity_SOM_PV_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_PV_mean[:], '-',color = '#E9967A', label = "SOM-PV")

plt.title('HL:SOM_VIP vs SOM_PV', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()




#scale_SOM2VIP_VIP2SOM_RDM
hp['stim_std_test']=0.1
hp['cue_delay'] = 800
scale_values=np.array([1,1.05,1.1,1.15,1.2])

hp['stim_cell']='MD1'
hp['sparsity_RDM']=0.8

scale_VIP_SOM_stim0=np.load(data_path+'scale_VIP_SOM_RDM_stim0.npy')
scale_VIP_SOM_stim1=np.load(data_path+'scale_VIP_SOM_RDM_stim1.npy')

scale_VIP_SOM_mean_stim0 = np.mean(scale_VIP_SOM_stim0,axis=0)
scale_VIP_SOM_std_stim0 = np.std(scale_VIP_SOM_stim0,axis=0)

scale_VIP_SOM_mean_stim1 = np.mean(scale_VIP_SOM_stim1,axis=0)
scale_VIP_SOM_std_stim1 = np.std(scale_VIP_SOM_stim1,axis=0)


percent_zeros = scale_values
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],scale_VIP_SOM_mean_stim0[:], scale_VIP_SOM_std_stim0[:], color = 'blue', fmt = 'o')
plt.plot(percent_zeros[:],scale_VIP_SOM_mean_stim0[:], color = 'blue',label = "stim MD1(VIP): 0")

plt.errorbar(percent_zeros[:],scale_VIP_SOM_mean_stim1[:], scale_VIP_SOM_std_stim1[:], color = 'red', fmt = 'o')
plt.plot(percent_zeros[:],scale_VIP_SOM_mean_stim1[:], color = 'red',label = "stim MD1(VIP): 1")

plt.title('RDM:sparsity_'+str(hp['sparsity_RDM']), fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([1,1.05,1.10,1.15,1.20], fontsize=12)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0], fontsize=12)

plt.xlabel('scale connection VIP-SOM', fontsize=12)
plt.ylabel('Performance of choice', fontsize=12)
plt.legend(fontsize=10)
plt.show()


#scale_SOM2VIP_VIP2SOM_HL
hp['cue_delay'] = 800
scale_values=np.array([1,1.05,1.1,1.15,1.2])
stim_value = 0.1
hp['stim_cell']='MD1'
hp['sparsity_HL']=0.8


scale_VIP_SOM_stim0=np.load(data_path+'scale_VIP_SOM_HL_stim0.npy')
scale_VIP_SOM_stim1=np.load(data_path+'scale_VIP_SOM_HL_stim1.npy')

scale_VIP_SOM_mean_stim0 = np.mean(scale_VIP_SOM_stim0,axis=0)
scale_VIP_SOM_std_stim0 = np.std(scale_VIP_SOM_stim0,axis=0)

scale_VIP_SOM_mean_stim1 = np.mean(scale_VIP_SOM_stim1,axis=0)
scale_VIP_SOM_std_stim1 = np.std(scale_VIP_SOM_stim1,axis=0)


percent_zeros = scale_values
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],scale_VIP_SOM_mean_stim0[:], scale_VIP_SOM_std_stim0[:], color = 'blue', fmt = 'o')
plt.plot(percent_zeros[:],scale_VIP_SOM_mean_stim0[:], color = 'blue',label = "stim MD1(VIP): 0")

plt.errorbar(percent_zeros[:],scale_VIP_SOM_mean_stim1[:], scale_VIP_SOM_std_stim1[:], color = 'red', fmt = 'o')
plt.plot(percent_zeros[:],scale_VIP_SOM_mean_stim1[:], color = 'red',label = "stim MD1(VIP):"+str(stim_value))

plt.title('HL:sparsity_'+str(hp['sparsity_HL']), fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([1,1.05,1.10,1.15,1.20], fontsize=12)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0], fontsize=12)
plt.xlabel('scale connection VIP-SOM', fontsize=12)
plt.ylabel('Performance of choice', fontsize=12)
plt.legend(fontsize=10)
plt.show()


#plot_EXC_scale_bi_SOMVIP_RDM
data_path_1 = os.path.join(data_path, 'plot_EXC_scale_bi_SOMVIP_RDM/')
tools.mkdir_p(data_path_1)
cue_on=epoch['cue_on'];cue_off=epoch['cue_off'];stim_on=epoch['stim_on'];response_on=epoch['response_on']

hp['stim_std_test']=0.1
hp['cue_delay'] = 800
scale_values=np.array([1,1.3])

hp['stim_cell']='MD1'
rm_type = 'scale_both_VIP_and_SOM'
hp['sparsity_RDM']=0.8


for scale_value in scale_values:
    pfc_mean_RDM_vis = np.load(data_path_1+str(scale_value)+'pfc_mean_RDM_vis.npy')
    pfc_mean_RDM_aud = np.load(data_path_1+str(scale_value)+'pfc_mean_RDM_aud.npy')


    #### plot
    font=12
    start = cue_on
    end = response_on

    EXC_idx_max = np.array([157, 134, 164,  66])
    alphas = np.linspace(0, 1, 5)
    _color_list = list(map(cm.rainbow, alphas))

    fig1, axs1 = plt.subplots(1, 2,figsize=(3.0, 1.5))

    y_lim_pfc =1.3
    j=-1
    for i in EXC_idx_max:
        j+=1

        axs1[0].plot(pfc_mean_RDM_vis[start+1:end,i],color=_color_list[j],label=str(i))
        axs1[1].plot(pfc_mean_RDM_aud[start+1:end,i],color=_color_list[j],label=str(i))
        axs1[0].set_ylim([0,y_lim_pfc])
        axs1[1].set_ylim([0,y_lim_pfc])
        axs1[0].set_title('vis',fontsize=font)
        #axs1[0].legend(fontsize=5)
        axs1[0].axis('off')
        axs1[1].axis('off')
        axs1[0].spines.left.set_visible(False)
        axs1[0].spines.right.set_visible(False)
        axs1[0].spines.top.set_visible(False)
        axs1[1].spines.left.set_visible(False)
        axs1[1].spines.right.set_visible(False)
        axs1[1].spines.top.set_visible(False)


        axs1[0].set_xticks([], fontsize=12)
        axs1[0].set_yticks([], fontsize=12)
        axs1[1].set_xticks([], fontsize=12)
        axs1[1].set_yticks([], fontsize=12)

    plt.show()



#plot_sparsity_SOM2PC_exc_activity
data_path_1 = os.path.join(data_path, 'plot_sparsity_SOM2PC_exc_activity/')
tools.mkdir_p(data_path_1)
cue_on=epoch['cue_on'];cue_off=epoch['cue_off']
stim_on=epoch['stim_on'];stim_off=epoch['stim_off']
response_on=epoch['response_on'];response_off=epoch['response_off']

percent_zeros=np.array([0.0,0.5])
hp['stim_std_test']=0.1
hp['cue_delay'] = 800
start = cue_on
end = response_off


for percent_zero in percent_zeros:
    pfc_mean_vis = np.load(data_path_1+str(percent_zero)+'pfc_mean_vis.npy')
    pfc_mean_aud = np.load(data_path_1+str(percent_zero)+'pfc_mean_aud.npy')

    EXC_max = np.argsort(np.mean(pfc_mean_aud[cue_off:stim_on,0:200],axis=0))
    EXC_idx_max = EXC_max[-4:]
    alphas = np.linspace(0, 1, 5)
    _color_list = list(map(cm.rainbow, alphas))

    fig1, axs1 = plt.subplots(1, 2,figsize=(3.0, 1.5))
    j=-1

    for i in EXC_idx_max:
        j+=1
        axs1[0].plot(pfc_mean_vis[start+1:end,i],color=_color_list[j],label=str(i))
        axs1[1].plot(pfc_mean_aud[start+1:end,i],color=_color_list[j],label=str(i))

        axs1[0].spines.left.set_visible(False)
        axs1[0].spines.right.set_visible(False)
        axs1[0].spines.top.set_visible(False)
        axs1[1].spines.left.set_visible(False)
        axs1[1].spines.right.set_visible(False)
        axs1[1].spines.top.set_visible(False)


        axs1[0].set_xticks([], fontsize=12)
        axs1[0].set_yticks([], fontsize=12)
        axs1[1].set_xticks([], fontsize=12)
        axs1[1].set_yticks([], fontsize=12)

    plt.show()



#plot_sparsity_SOM2PC_scatter_delay
data_path_1 = os.path.join(data_path, 'plot_sparsity_SOM2PC_scatter_delay/')
tools.mkdir_p(data_path_1)
cue_on=epoch['cue_on'];cue_off=epoch['cue_off']
stim_on=epoch['stim_on'];stim_off=epoch['stim_off']
response_on=epoch['response_on'];response_off=epoch['response_off']

percent_zeros=np.array([0.0,0.5])
hp['stim_std_test']=0.1
hp['cue_delay'] = 800
cohs_RDM = np.array([0.8])



for percent_zero in percent_zeros:
    fr_rnn_1 = np.load(data_path_1+str(percent_zero)+'fr_rnn_1.npy')
    fr_rnn_2 = np.load(data_path_1+str(percent_zero)+'fr_rnn_2.npy')
    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    ss=22
    alphas = np.linspace(0, 1, 5)
    colors=list(map(cm.rainbow, alphas))
    plt.scatter(fr_rnn_1[0:205],fr_rnn_2[0:205],marker=">",s=ss,color='tab:red')
    plt.scatter(fr_rnn_1[142],fr_rnn_2[142],marker=">",s=ss,color=colors[0])
    plt.scatter(fr_rnn_1[66],fr_rnn_2[66],marker=">",s=ss,color=colors[1])
    plt.scatter(fr_rnn_1[124],fr_rnn_2[124],marker=">",s=ss,color=colors[2])
    plt.scatter(fr_rnn_1[174],fr_rnn_2[174],marker=">",s=ss,color=colors[3])


    max =2.5
    plt.plot([0,max],[0,max],color='grey')
    plt.xlim([0,max])
    plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,0.5,1,1.5,2,2.5], fontsize=12)
    plt.yticks([0,0.5,1,1.5,2,2.5], fontsize=12)
    plt.xlabel('Attend to vision',fontsize=12)
    plt.ylabel('Attend to audition',fontsize=12)
    plt.show()

#plot_sparsity_SOM2PC_scatter_target
data_path_1 = os.path.join(data_path, 'plot_sparsity_SOM2PC_scatter_target/')
tools.mkdir_p(data_path_1)
cue_on=epoch['cue_on'];cue_off=epoch['cue_off']
stim_on=epoch['stim_on'];stim_off=epoch['stim_off']
response_on=epoch['response_on'];response_off=epoch['response_off']
percent_zeros=np.array([0.0,0.5])
hp['stim_std_test']=0.1
hp['cue_delay'] = 800



for percent_zero in percent_zeros:
    fr_rnn_1 = np.load(data_path_1+str(percent_zero)+'fr_rnn_1.npy')
    fr_rnn_2 = np.load(data_path_1+str(percent_zero)+'fr_rnn_2.npy')
    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    ss=22
    alphas = np.linspace(0, 1, 5)
    colors=list(map(cm.rainbow, alphas))
    plt.scatter(fr_rnn_1[0:205],fr_rnn_2[0:205],marker=">",s=ss,color='tab:red')
    plt.scatter(fr_rnn_1[142],fr_rnn_2[142],marker=">",s=ss,color=colors[0])
    plt.scatter(fr_rnn_1[66],fr_rnn_2[66],marker=">",s=ss,color=colors[1])
    plt.scatter(fr_rnn_1[124],fr_rnn_2[124],marker=">",s=ss,color=colors[2])
    plt.scatter(fr_rnn_1[174],fr_rnn_2[174],marker=">",s=ss,color=colors[3])


    max =2.5
    plt.plot([0,max],[0,max],color='grey')
    plt.xlim([0,max])
    plt.ylim([0,max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,0.5,1,1.5,2,2.5], fontsize=12)
    plt.yticks([0,0.5,1,1.5,2,2.5], fontsize=12)
    plt.xlabel('Attend to vision',fontsize=12)
    plt.ylabel('Attend to audition',fontsize=12)
    plt.show()




#plot_sparsity_SOM2PC_HL
data_path_1 = os.path.join(data_path, 'plot_sparsity_SOM2PC_HL/')
tools.mkdir_p(data_path_1)
cue_on=epoch['cue_on'];cue_off=epoch['cue_off']
stim_on=epoch['stim_on'];stim_off=epoch['stim_off']
response_on=epoch['response_on'];response_off=epoch['response_off']

percent_zeros=np.array([0.0,0.5])
hp['stim_std_test']=0.1
hp['cue_delay'] = 800


for percent_zero in percent_zeros:
    concate_transform_split = np.load(data_path_1+str(percent_zero)+'concate_transform_split.npy')

    fig1 = plt.figure(figsize=(4.5,3.3))
    ax = fig1.add_axes([0.15,0.15,0.75,0.75])
    k0=0
    k1=1
    k2=2
    fs=10
    batch_size=10


    for i in range(0, len(concate_transform_split)):
        if i<batch_size*2:
            ax.plot(concate_transform_split[i][:, k0], concate_transform_split[i][:, k1], linewidth=1,color='tab:blue',zorder=1)
        else:
            ax.plot(concate_transform_split[i][:, k0], concate_transform_split[i][:, k1], linewidth=1,color='tab:grey',zorder=1)
        ax.scatter(concate_transform_split[i][0, k0], concate_transform_split[i][0, k1],  linewidth=0.2,marker='o', color='orange',zorder=2)

    for i in range(0, len(concate_transform_split)):
        if i<batch_size:
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1],  linewidth=0.2,marker='o', color='blue',zorder=3)
        if batch_size<=i<batch_size*2:
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1],  linewidth=0.8,marker='*', color='blue',zorder=3)
        if 2*batch_size<=i<batch_size*3:
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1],  linewidth=0.2,marker='o', color='red',zorder=2)
        if 3*batch_size<=i<batch_size*4:
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1],  linewidth=0.8,marker='*', color='red',zorder=2)


    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()










