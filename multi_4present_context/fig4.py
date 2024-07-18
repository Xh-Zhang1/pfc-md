import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import default
import tools



hp=default.get_default_hp(rule_name='HL_task',random_seed=1)

data_path = os.path.join(hp['root_path'], 'Datas', 'construct_between_MD_PFC_fig6_0/')
tools.mkdir_p(data_path)

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'Fig6_stim_MD/')
tools.mkdir_p(figure_path)
model_prefix = 'type3_256_200_softplus_sr5.0_0.1_drop0.0_'
cohs_HL  = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9])
cohs_RDM = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

x_bottom=0.2
x_top=0.9

#increase_delay_RDM
cue_decays=[800,900,1000,1100,1200]
data_path = os.path.join(hp['root_path'], 'Datas', 'construct_between_MD_PFC_fig6_1/')

RDM_no_adds_coh1 = np.load(data_path+'RDM_no_adds_coh1.npy')
RDM_no_adds_coh2 = np.load(data_path+'RDM_no_adds_coh2.npy')
RDM_no_adds_coh3 = np.load(data_path+'RDM_no_adds_coh3.npy')

mean_no_adds_coh1 = np.mean(RDM_no_adds_coh1,axis=0)
error_no_adds_coh1 = np.std(RDM_no_adds_coh1, axis=0)

mean_no_adds_coh2 = np.mean(RDM_no_adds_coh2,axis=0)
error_no_adds_coh2 = np.std(RDM_no_adds_coh2, axis=0)

mean_no_adds_coh3 = np.mean(RDM_no_adds_coh3,axis=0)
error_no_adds_coh3 = np.std(RDM_no_adds_coh3, axis=0)




fig = plt.figure(figsize=(5,4))
ax = fig.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(cue_decays,mean_no_adds_coh1, error_no_adds_coh1, color = 'tab:blue', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh1, color = 'tab:blue', label = "0.6")

plt.errorbar(cue_decays,mean_no_adds_coh2, error_no_adds_coh2, color = 'tab:orange', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh2, color = 'tab:orange', label = "0.7")

plt.errorbar(cue_decays,mean_no_adds_coh3, error_no_adds_coh3, color = 'tab:green', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh3, color = 'tab:green', label = "0.8")


plt.title('RDM decay', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([800,900,1000,1100,1200], fontsize=12)
plt.yticks([0.2, 0.4,0.6,0.8,1.0], fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()

#increase_delay_HL
data_path = os.path.join(hp['root_path'], 'Datas', 'construct_between_MD_PFC_fig6_1/')
cue_decays=[800,900,1000,1100,1200]

HL_no_adds_coh1 = np.load(data_path+'HL_no_adds_coh1.npy')
HL_no_adds_coh2 = np.load(data_path+'HL_no_adds_coh2.npy')
HL_no_adds_coh3 = np.load(data_path+'HL_no_adds_coh3.npy')

mean_no_adds_coh1 = np.mean(HL_no_adds_coh1,axis=0)
error_no_adds_coh1 = np.std(HL_no_adds_coh1, axis=0)

mean_no_adds_coh2 = np.mean(HL_no_adds_coh2,axis=0)
error_no_adds_coh2 = np.std(HL_no_adds_coh2, axis=0)

mean_no_adds_coh3 = np.mean(HL_no_adds_coh3,axis=0)
error_no_adds_coh3 = np.std(HL_no_adds_coh3, axis=0)


fig = plt.figure(figsize=(5,4))
ax = fig.add_axes([0.15,0.15,0.75,0.75])

plt.errorbar(cue_decays,mean_no_adds_coh1, error_no_adds_coh1, color = 'tab:blue', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh1, color = 'tab:blue', label = "0.6")

plt.errorbar(cue_decays,mean_no_adds_coh2, error_no_adds_coh2, color = 'tab:orange', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh2, color = 'tab:orange', label = "0.7")

plt.errorbar(cue_decays,mean_no_adds_coh3, error_no_adds_coh3, color = 'tab:green', fmt = 'o')
plt.plot(cue_decays,mean_no_adds_coh3, color = 'tab:green', label = "0.8")


plt.title('HL decay', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([800,900,1000,1100,1200], fontsize=12)
plt.yticks([0.2, 0.4,0.6,0.8,1.0], fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(fontsize=10)
plt.show()




def fit_data_HL(perf_0):
    mean_perf_0 = np.mean(perf_0,axis=0)
    error_perf_0 = np.std(perf_0, axis=0)# / np.sqrt(perf_0.shape[0])


    x_0 = cohs_HL
    y_para_0 = mean_perf_0
    err_0 = error_perf_0

    weights_0 = [1.0/max(_,0.01) for _ in err_0]
    def logistic_growth_0(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_0 = np.linspace(0.5, 0.9, 100)
    paras_0, paras_cov_0 = curve_fit(logistic_growth_0, x_0, y_para_0,absolute_sigma=True,sigma = weights_0)
    para_curve_0 = logistic_growth_0(x_plot_0, *paras_0)

    return x_0,y_para_0,err_0,x_plot_0,para_curve_0


#HL_add_md1_stimulation
data_path = os.path.join(hp['root_path'], 'Datas', 'construct_between_MD_PFC_fig6_0/')
tools.mkdir_p(data_path)
hp['stim_epoch']='delay_epoch'
stim_cell='MD1'

scale_PC_md=1
stim_value_0=0
stim_value_1=0.4
stim_value_2=0.7

perf_0 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')

x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_HL(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_HL(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_HL(perf_2)


fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "increase 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:blue', label = "increase 32%")

plt.errorbar(x_2,y_para_2, err_2, color = 'skyblue', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'skyblue', alpha=1,label = "increase 63%")


plt.title('HL_task', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()

#HL_add_md2_stimulation
stim_cell='MD2'

stim_value_0=0
stim_value_1=0.6
stim_value_2=1

perf_0 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')

x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_HL(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_HL(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_HL(perf_2)

##===================== plot =====================#####
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "increase 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:blue', label = "increase 71%")

plt.errorbar(x_2,y_para_2, err_2, color = 'skyblue', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'skyblue', alpha=1,label = "increase 80%")


plt.title('HL_task', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#HL_add_md2_stimulation_decrease
stim_cell='MD2'
stim_value_0=0
stim_value_1=-0.85
stim_value_2=-1.0

perf_0 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_HL_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')


x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_HL(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_HL(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_HL(perf_2)

##===================== plot =====================#####
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "increase 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:grey', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:grey', label = "increase 45%")

plt.errorbar(x_2,y_para_2, err_2, color = 'silver', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'silver', alpha=1,label = "increase 52%")


plt.title('HL_task: decrease stim md2 (md_to_PV_path)', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()


def fit_data_RDM(perf_0):
    mean_perf_0 = np.mean(perf_0,axis=0)
    error_perf_0 = np.std(perf_0, axis=0)# / np.sqrt(perf_0.shape[0])


    x_0 = cohs_RDM
    y_para_0 = mean_perf_0
    err_0 = error_perf_0

    weights_0 = [1.0/max(_,0.2) for _ in err_0]
    def logistic_growth_0(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_0 = np.linspace(0, 0.9, 100)
    paras_0, paras_cov_0 = curve_fit(logistic_growth_0, x_0, y_para_0,absolute_sigma=True,sigma = weights_0)
    para_curve_0 = logistic_growth_0(x_plot_0, *paras_0)


    return x_0,y_para_0,err_0,x_plot_0,para_curve_0

#RDM_add_md1_stimulation
stim_cell='MD1'
stim_value_0=0
stim_value_1=0.4
stim_value_2=0.7

perf_0 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')


x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_RDM(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_RDM(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_RDM(perf_2)

##===================== plot =====================#####
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "increase 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:blue', label = "increase 32%")

plt.errorbar(x_2,y_para_2, err_2, color = 'skyblue', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'skyblue', alpha=1,label = "increase 64%")


plt.title('RDM_task', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8,1.0], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()

#RDM_add_md2_stimulation
stim_cell='MD2'

stim_value_0=0
stim_value_1=0.2
stim_value_2=1.0

perf_0 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')

x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_RDM(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_RDM(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_RDM(perf_2)

##===================== plot =====================#####
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "increase 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:blue', label = "increase 54%")

plt.errorbar(x_2,y_para_2, err_2, color = 'skyblue', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'skyblue', alpha=1,label = "increase 82%")


plt.title('RDM_task', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8,1.0], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()



#RDM_add_md2_stimulation_decrease
stim_cell='MD2'
stim_value_0=0
stim_value_1=-0.85
stim_value_2=-0.9

perf_0 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_0)+'.npy')
perf_1 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_1)+'.npy')
perf_2 = np.load(data_path+'perf_RDM_'+stim_cell+'_stim'+str(stim_value_2)+'.npy')

x_0,y_para_0,err_0,x_plot_0,para_curve_0 = fit_data_RDM(perf_0)
x_1,y_para_1,err_1,x_plot_1,para_curve_1 = fit_data_RDM(perf_1)
x_2,y_para_2,err_2,x_plot_2,para_curve_2 = fit_data_RDM(perf_2)

fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0.2,0.15,0.75,0.75])

plt.errorbar(x_0,y_para_0, err_0, color = 'black', fmt = 'o')
plt.plot(x_plot_0, para_curve_0, color = 'black', label = "decrease 0%")

plt.errorbar(x_1,y_para_1, err_1, color = 'tab:grey', fmt = 'o')
plt.plot(x_plot_1, para_curve_1, color = 'tab:grey', label = "decrease 45%")

plt.errorbar(x_2,y_para_2, err_2, color = 'silver', alpha=1,fmt = 'o')
plt.plot(x_plot_2, para_curve_2, color = 'silver', alpha=1,label = "decrease 48%")


plt.title('RDM_task', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([x_bottom,x_top])
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8,1.0], fontsize=12)
plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
plt.ylabel('Proportion correct', fontsize=12)
plt.legend(fontsize=10)
plt.show()





#=========================  plot =============================

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

