import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

import default
import tools


hp=default.get_default_hp(rule_name='HL_task',random_seed=1)
hp['stim_epoch']='no'
hp['mask_test'] = 'type3'
hp['stim_value']=0
hp['scale_PC_MD']=1
hp['dropout']=0.0
hp['dropout_model']=0.0
hp['cue_delay'] = 800

batch_size=100
hp['stim_std']=0.1
hp['dropout_model']=0.0
hp['model_idx']=59
idx=9


#=========================  plot =============================
data_path = os.path.join(hp['root_path'], 'Datas', 'fig5_stim_between_MD_inh/')#fig5_stim_between_MD_inh
tools.mkdir_p(data_path)

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'fig5_stim_between_MD_inh/')
tools.mkdir_p(figure_path)

model_prefix = 'type3_256_200_softplus_sr5.0_0.1_drop0.0_'

cohs_HL  = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9])
cohs_RDM = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])


local_folder_name = os.path.join('/'+'model_'+str(0.92),  model_prefix+str(hp['model_idx']), str(idx))
model_dir = hp['root_path']+local_folder_name+'/'


#plot_MD_uncert_RDM
increase_md1 = [0,100*0.8180125951766968, 100*1.3162562847137451, 100*2.074329137802124, 100*2.6742560863494873]
increase_md2 = [0,100*0.8180125951766968, 100*1.3162562847137451, 100*2.074329137802124, 100*2.6742560863494873]


stim_MD1_VIP_RDM = np.load(data_path+'stim_MD1_VIP_uncert_RDM.npy')
stim_MD2_PV_RDM = np.load(data_path+'stim_MD2_PV_uncert_RDM.npy')

stim_MD1_VIP_RDM_mean = np.mean(stim_MD1_VIP_RDM,axis=0)
stim_MD1_VIP_RDM_std = np.std(stim_MD1_VIP_RDM,axis=0)

stim_MD2_PV_RDM_mean = np.mean(stim_MD2_PV_RDM,axis=0)
stim_MD2_PV_RDM_std = np.std(stim_MD2_PV_RDM,axis=0)


####### plot ######
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(increase_md1,stim_MD1_VIP_RDM_mean, stim_MD1_VIP_RDM_std, color = 'tab:purple', fmt = 'o')
plt.plot(increase_md1,stim_MD1_VIP_RDM_mean, color = 'tab:purple', label = "MD-VIP")

plt.errorbar(increase_md2,stim_MD2_PV_RDM_mean, stim_MD2_PV_RDM_std, color = 'tab:orange', fmt = 'o')
plt.plot(increase_md2,stim_MD2_PV_RDM_mean, color = 'tab:orange', label = "MD-PV")


plt.title('stim_MD_uncert_RDM', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('increase of MD (%)', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
fig1.savefig(figure_path+'stim_MD_uncert_RDM.png')
plt.show()



#plot_MD_uncert_HL
increase_md1 = [0,100*0.3134201765060425, 100*0.566372275352478, 100*0.7768073081970215, 100*0.9175423383712769]
increase_md2 = [0,100*0.33227574825286865, 100*0.543458104133606, 100*0.723967432975769, 100*0.9136813879013062]


stim_MD1_VIP_HL = np.load(data_path+'stim_MD1_VIP_uncert_HL.npy')
stim_MD2_PV_HL = np.load(data_path+'stim_MD2_PV_uncert_HL.npy')

stim_MD1_VIP_HL_mean = np.mean(stim_MD1_VIP_HL,axis=0)
stim_MD1_VIP_HL_std = np.std(stim_MD1_VIP_HL,axis=0)

stim_MD2_PV_HL_mean = np.mean(stim_MD2_PV_HL,axis=0)
stim_MD2_PV_HL_std = np.std(stim_MD2_PV_HL,axis=0)


####### plot ######
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(increase_md1,stim_MD1_VIP_HL_mean, stim_MD1_VIP_HL_std, color = 'tab:purple', fmt = 'o')
plt.plot(increase_md1,stim_MD1_VIP_HL_mean, color = 'tab:purple', label = "MD-VIP")

plt.errorbar(increase_md2,stim_MD2_PV_HL_mean, stim_MD2_PV_HL_std, color = 'tab:orange', fmt = 'o')
plt.plot(increase_md2,stim_MD2_PV_HL_mean, color = 'tab:orange', label = "MD-PV")


plt.title('stim_MD_uncert_HL', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0,20,40,60,80,100], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('increase of MD (%)', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
fig1.savefig(figure_path+'stim_MD_uncert_HL.png')
plt.show()





#connect_MD1_VIP_two_task
hp['stim_cell'] = 'MD1'
hp['stim_task'] = 'HL_task'

percent_zeros=np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

connect_MD1_VIP_RDM=np.load(data_path+'connect_MD1_VIP_RDM.npy')
connect_MD1_VIP_HL =np.load(data_path+'connect_MD1_VIP_HL.npy')

connect_MD1_VIP_RDM_mean = np.mean(connect_MD1_VIP_RDM,axis=0)
connect_MD1_VIP_HL_mean = np.mean(connect_MD1_VIP_HL,axis=0)

connect_MD1_VIP_RDM_std = np.std(connect_MD1_VIP_RDM,axis=0)
connect_MD1_VIP_HL_std = np.std(connect_MD1_VIP_HL,axis=0)


percent_zeros = 100*percent_zeros
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros,connect_MD1_VIP_RDM_mean, connect_MD1_VIP_RDM_std, color = 'tab:blue', fmt = 'o')
plt.plot(percent_zeros,connect_MD1_VIP_RDM_mean, color = 'tab:blue', label = "RDM")

plt.errorbar(percent_zeros,connect_MD1_VIP_HL_mean,  connect_MD1_VIP_HL_std, color = 'tab:grey', fmt = 'o')
plt.plot(percent_zeros,connect_MD1_VIP_HL_mean, color = 'tab:grey', label = "HL")

plt.title('MD1_VIP in RDM and HL', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0,10,20,30,40,50,60,70,80,90], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()


#connect_MD1_VIP_stim_MD1_RDM():
hp['stim_cell'] = 'MD1'
percent_zeros=np.array([0.2,0.25,0.3,0.35,0.4,0.45,0.5])
stim_values=np.array([0.0,1,2,3,4,5,6])
perf_stim1_MD2_RDM_seed = []

connect_MD1_VIP_RDM_stim1 =np.load(data_path+'connect_MD1_VIP_RDM_stim1.npy')
connect_MD1_VIP_RDM_stim1_mean = np.mean(connect_MD1_VIP_RDM_stim1,axis=0)
connect_MD1_VIP_RDM_stim1_std = np.std(connect_MD1_VIP_RDM_stim1,axis=0)


percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(3,2))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],connect_MD1_VIP_RDM_stim1_mean[:],  connect_MD1_VIP_RDM_stim1_std[:], color = 'skyblue', fmt = 'o')
plt.plot(percent_zeros[:],connect_MD1_VIP_RDM_stim1_mean[:], '-.',color = 'skyblue', label = "stim MD")

plt.title('RDM:change MD1_VIP and stim_MD1', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([20,30,40,50], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#connect_MD2_PV_two_task():
hp['stim_cell'] = 'MD1'
hp['stim_task'] = 'HL_task'

percent_zeros=np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

connect_MD2_PV_RDM=np.load(data_path+'connect_MD2_PV_RDM.npy')
connect_MD2_PV_HL =np.load(data_path+'connect_MD2_PV_HL.npy')

connect_MD2_PV_RDM_mean = np.mean(connect_MD2_PV_RDM,axis=0)
connect_MD2_PV_HL_mean = np.mean(connect_MD2_PV_HL,axis=0)
connect_MD2_PV_RDM_std = np.std(connect_MD2_PV_RDM,axis=0)
connect_MD2_PV_HL_std = np.std(connect_MD2_PV_HL,axis=0)


percent_zeros = 100*percent_zeros
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros,connect_MD2_PV_RDM_mean, connect_MD2_PV_RDM_std, color = 'tab:blue', fmt = 'o')
plt.plot(percent_zeros,connect_MD2_PV_RDM_mean, color = 'tab:blue', label = "RDM")

plt.errorbar(percent_zeros,connect_MD2_PV_HL_mean,  connect_MD2_PV_HL_std, color = 'tab:grey', fmt = 'o')
plt.plot(percent_zeros,connect_MD2_PV_HL_mean, color = 'tab:grey', label = "HL")

plt.title('MD2_PV in RDM and HL', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0,10,20,30,40,50,60,70,80,90], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#connect_MD2_PV_stim_MD2_RDM
hp['stim_cell'] = 'MD2'
percent_zeros=np.array([0.2,0.25,0.3,0.35,0.4,0.45,0.5])
stim_values=np.array([0.0,1,2,3,4,5,6])

connect_MD2_PV_RDM_stim0=np.load(data_path+'connect_MD2_PV_RDM_stim0.npy')
connect_MD2_PV_RDM_stim1 =np.load(data_path+'connect_MD2_PV_RDM_stim1.npy')

connect_MD2_PV_RDM_stim0_mean = np.mean(connect_MD2_PV_RDM_stim0,axis=0)
connect_MD2_PV_RDM_stim1_mean = np.mean(connect_MD2_PV_RDM_stim1,axis=0)
connect_MD2_PV_RDM_stim0_std = np.std(connect_MD2_PV_RDM_stim0,axis=0)
connect_MD2_PV_RDM_stim1_std = np.std(connect_MD2_PV_RDM_stim1,axis=0)


percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(3,2))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],connect_MD2_PV_RDM_stim0_mean[:], connect_MD2_PV_RDM_stim0_std[:], color = 'tab:blue', fmt = 'o')
plt.plot(percent_zeros[:],connect_MD2_PV_RDM_stim0_mean[:], color = 'tab:blue')

plt.errorbar(percent_zeros[:],connect_MD2_PV_RDM_stim1_mean[:],  connect_MD2_PV_RDM_stim1_std[:], color = 'skyblue', fmt = 'o')
plt.plot(percent_zeros[:],connect_MD2_PV_RDM_stim1_mean[:], '-.',color = 'skyblue', label = "stim MD")

plt.title('RDM:change MD2_PV and stim_PV', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([20,30,40,50], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)

plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#sparsity_SOM2PC_PV2PC_RDM
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])

sparsity_SOM_PC=np.load(data_path+'sparsity_SOM_PC_RDM.npy')
sparsity_PV_PC =np.load(data_path+'sparsity_PV_PC_RDM.npy')

sparsity_SOM_PC_mean = np.mean(sparsity_SOM_PC,axis=0)
sparsity_PV_PC_mean = np.mean(sparsity_PV_PC,axis=0)
sparsity_SOM_PC_std = np.std(sparsity_SOM_PC,axis=0)
sparsity_PV_PC_std = np.std(sparsity_PV_PC,axis=0)


percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],sparsity_SOM_PC_mean[:], sparsity_SOM_PC_std[:], color = '#66CDAA', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_PC_mean[:], color = '#66CDAA',label = "SOM-PC")

plt.errorbar(percent_zeros[:],sparsity_PV_PC_mean[:],  sparsity_PV_PC_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PV_PC_mean[:], '-',color = '#E9967A', label = "PV-PC")

plt.title('RDM:SOM-PC vs PV-PC', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#sparsity_SOM2PC_PV2PC_HL
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])
sparsity_SOM_PC=np.load(data_path+'sparsity_SOM_PC_HL.npy')
sparsity_PV_PC =np.load(data_path+'sparsity_PV_PC_HL.npy')

sparsity_SOM_PC_mean = np.mean(sparsity_SOM_PC,axis=0)
sparsity_PV_PC_mean = np.mean(sparsity_PV_PC,axis=0)
sparsity_SOM_PC_std = np.std(sparsity_SOM_PC,axis=0)
sparsity_PV_PC_std = np.std(sparsity_PV_PC,axis=0)

percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],sparsity_SOM_PC_mean[:], sparsity_SOM_PC_std[:], color = '#66CDAA', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_SOM_PC_mean[:], color = '#66CDAA',label = "SOM-PC")

plt.errorbar(percent_zeros[:],sparsity_PV_PC_mean[:],  sparsity_PV_PC_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PV_PC_mean[:], '-',color = '#E9967A', label = "PV-PC")

plt.title('HL:SOM-PC vs PV-PC', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()


#sparsity_PC_to_inh_RDM
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])
sparsity_PC_VIP=np.load(data_path+'sparsity_PC_VIP_RDM.npy')
sparsity_PC_SOM=np.load(data_path+'sparsity_PC_SOM_RDM.npy')
sparsity_PC_PV =np.load(data_path+'sparsity_PC_PV_RDM.npy')

sparsity_PC_VIP_mean = np.mean(sparsity_PC_VIP,axis=0)
sparsity_PC_SOM_mean = np.mean(sparsity_PC_SOM,axis=0)
sparsity_PC_PV_mean = np.mean(sparsity_PC_PV,axis=0)
sparsity_PC_VIP_std = np.std(sparsity_PC_VIP,axis=0)
sparsity_PC_SOM_std = np.std(sparsity_PC_SOM,axis=0)
sparsity_PC_PV_std = np.std(sparsity_PC_PV,axis=0)

percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],sparsity_PC_VIP_mean[:], sparsity_PC_VIP_std[:], color = '#DDA0DD', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_VIP_mean[:], color = '#DDA0DD',label = "PC-VIP")

plt.errorbar(percent_zeros[:],sparsity_PC_SOM_mean[:],  sparsity_PC_SOM_std[:], color = '#66CDAA', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_SOM_mean[:], '-',color = '#66CDAA', label = "PC-SOM")

plt.errorbar(percent_zeros[:],sparsity_PC_PV_mean[:],  sparsity_PC_PV_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_PV_mean[:], '-',color = '#E9967A', label = "PC-PV")

plt.title('RDM:PC_VIP vs PC_SOM vs PC_PV', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0,10,20,30,40,50,60], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()




#sparsity_PC_to_inh_HL
percent_zeros=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])
sparsity_PC_VIP=np.load(data_path+'sparsity_PC_VIP_HL.npy')
sparsity_PC_SOM=np.load(data_path+'sparsity_PC_SOM_HL.npy')
sparsity_PC_PV =np.load(data_path+'sparsity_PC_PV_HL.npy')

sparsity_PC_VIP_mean = np.mean(sparsity_PC_VIP,axis=0)
sparsity_PC_SOM_mean = np.mean(sparsity_PC_SOM,axis=0)
sparsity_PC_PV_mean = np.mean(sparsity_PC_PV,axis=0)
sparsity_PC_VIP_std = np.std(sparsity_PC_VIP,axis=0)
sparsity_PC_SOM_std = np.std(sparsity_PC_SOM,axis=0)
sparsity_PC_PV_std = np.std(sparsity_PC_PV,axis=0)


percent_zeros = 100*percent_zeros
## inset figure
fig1 = plt.figure(figsize=(4.5,3.3))
ax = fig1.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(percent_zeros[:],sparsity_PC_VIP_mean[:], sparsity_PC_VIP_std[:], color = '#DDA0DD', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_VIP_mean[:], color = '#DDA0DD',label = "PC-VIP")

plt.errorbar(percent_zeros[:],sparsity_PC_SOM_mean[:],  sparsity_PC_SOM_std[:], color = '#66CDAA', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_SOM_mean[:], '-',color = '#66CDAA', label = "PC-SOM")

plt.errorbar(percent_zeros[:],sparsity_PC_PV_mean[:],  sparsity_PC_PV_std[:], color = '#E9967A', fmt = 'o')
plt.plot(percent_zeros[:],sparsity_PC_PV_mean[:], '-',color = '#E9967A', label = "PC-PV")

plt.title('HL:PC_VIP vs PC_SOM vs PC_PV', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0,10,20,30,40,50,60], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize=12)
plt.xlabel('Percent of zero-connection', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()

