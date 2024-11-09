"""Collections of tasks."""

from __future__ import division
import numpy as np
import math
import sys
import tools

import pdb
def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))


# generating values of x, y, c_mask specific for tasks
# config contains hyper-parameters used for generating tasks
class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, xtdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            xtdim: int, number of total time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.xtdim = xtdim

        # time major
        self.x = np.zeros((xtdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)

        self.cost_mask = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        # strength of input noise
        self._sigma_x = config['sigma_x'] * math.sqrt(2./self.config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths):
        """Add an input or stimulus output to the indicated channel.

        Args:
            loc_type: str type of information to be added
            loc_idx: index of channel
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
            gaussian_center: float. location of gaussian bump, only used if loc_type=='line_gaussian_input' or 'line_gaussian_output'
        """
        self.ons = ons
        self.offs = offs


        if loc_type == 'input':
            for i in range(self.batch_size):
                #print('@@@@@@@@ self.x',self.x[:,0,:])
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'out':

            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'cost_mask':

            for i in range(self.batch_size):
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]

        else:
            raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        #print('add_x_noise')
        self.x[:,:,:] += 2*self.config['rng'].randn(*self.x[:,:,:].shape) * self._sigma_x

        #self.x[:,:,:-2] += 2*self.config['rng'].randn(*self.x[:,:,:-2].shape) * self._sigma_x

    def add_angle_noise_vis(self,on,off,scale,p_coh):
        """Add input noise."""
        #print('add_angle_noise')
        dim = self.config['n_ring_RDM']
        num_noise = int(dim * p_coh)

        noise = self.config['rng'].uniform(-scale, scale, self.x[on:off,:,num_noise:dim].shape)
        self.x[on:off,:,num_noise:dim] = noise


    def add_angle_noise_aud(self,on,off,scale,p_coh):
        """Add input noise."""

        #print('add_angle_noise')
        dim = self.config['n_ring']
        num_noise = int(dim*self.config['p_coh'])
        noise = self.config['rng'].uniform(-scale, scale, self.x[on:off,:,num_noise+dim :2*dim].shape)
        self.x[on:off,:,num_noise+dim:2*dim ] = noise


def _RDM_task(config, mode,**kwargs):
    pass

def _HL_task(config, mode,**kwargs):
    '''
    Two stimuli are presented simultaneously. The network should indicate which stimulus is stronger.
    '''
    #print('************ perform HL_task')
    dt = config['dt']


    rng = config['rng']
    batch_size = kwargs['batch_size']
    scale_HL = config['scale_HL']



    # cue stim
    cue_time = int(config['cue_duration'])
    cue_delay_time = int(config['cue_delay'])
    cue_on = (rng.uniform(40, 40, batch_size)/dt).astype(int)
    cue_duration = int(cue_time/dt)#60
    cue_off = cue_on + cue_duration
    cue_delay = (rng.uniform(cue_delay_time, cue_delay_time, batch_size)/dt).astype(int)

    #stim epoch
    stim_time = int(config['stim'])
    stim1_on = cue_off+cue_delay
    stim1_during =  (rng.uniform(stim_time, stim_time, batch_size)/dt).astype(int)
    stim1_off = stim1_on + stim1_during
    response_on = stim1_off

    # response end time
    response_duration = int(config['response_time']/dt)
    response_off = response_on + response_duration

    xtdim = response_off
    #print('xtdim',xtdim)

    trial = Trial(config, xtdim.max(), batch_size)

    #p_cohs = np.random.choice([0.7,0.75,0.8,0.85,0.9])
    if config['mod_type']== 'training':
        p_cohs = rng.choice([0.7,0.75,0.8,0.85,0.9,0.95,1])
    elif config['mod_type']== 'testing':
        p_cohs = config['p_coh']

    gamma_bar_vis = rng.uniform(0.9, 1.1,batch_size)#kwargs['gamma_bar']
    gamma_bar_aud = rng.uniform(-0.9, -1.1,batch_size)#kwargs['gamma_bar']
    c_vis = rng.choice([-0.2,0.2], (batch_size,))
    c_aud = rng.choice([-0.2,0.2], (batch_size,))
    #print('gamma_bar_vis,c_vis:',gamma_bar_vis,c_vis)


    mean_vis_0 = gamma_bar_vis+c_vis
    mean_vis_1 = gamma_bar_vis-c_vis
    mean_aud_0 = gamma_bar_aud+c_aud
    mean_aud_1 = gamma_bar_aud-c_aud


    if mode == 'random':  # Randomly generate parameters
        vis_modality_stim_0 = rng.normal(loc=mean_vis_0, scale=1.0*config['stim_std'], size=batch_size)
        vis_modality_stim_1 = rng.normal(loc=mean_vis_1, scale=1.0*config['stim_std'], size=batch_size)
        aud_modality_stim_0 = rng.normal(loc=mean_aud_0, scale=1.0*config['stim_std'], size=batch_size)
        aud_modality_stim_1 = rng.normal(loc=mean_aud_1, scale=1.0*config['stim_std'], size=batch_size)
        c_cue = rng.choice([-1.0*scale_HL, 1.0*scale_HL], (batch_size,))
        p_coh = rng.choice([p_cohs])#


    elif mode == 'random_validate':  # Randomly generate parameters
        vis_modality_stim_0 = rng.normal(loc=mean_vis_0, scale=config['stim_std'], size=batch_size)
        vis_modality_stim_1 = rng.normal(loc=mean_vis_1, scale=config['stim_std'], size=batch_size)
        aud_modality_stim_0 = rng.normal(loc=mean_aud_0, scale=config['stim_std'], size=batch_size)
        aud_modality_stim_1 = rng.normal(loc=mean_aud_1, scale=config['stim_std'], size=batch_size)

        c_cue = rng.choice([-1.0*scale_HL, 1.0*scale_HL], (batch_size,))
        p_coh = rng.choice([p_cohs])#


    elif mode == 'test1':  # Randomly generate parameters

        p_coh = kwargs['p_coh']
        c_cue = kwargs['c_cue']
        if p_coh==0.5:
            p_coh = rng.choice([0.7,0.75,0.8,0.85,0.9,0.95,1])
        print('p_coh', p_coh)


        if not hasattr(c_cue, '__iter__'):
            c_cue = np.array([c_cue] * batch_size)
        print('c_cue',c_cue)

        gamma_bar_vis = rng.uniform(0.9, 1.1,batch_size)#kwargs['gamma_bar']
        gamma_bar_aud = rng.uniform(-0.9, -1.1,batch_size)#kwargs['gamma_bar']
        c_vis = rng.choice([-0.2,0.2], (batch_size,))
        c_aud = rng.choice([-0.2,0.2], (batch_size,))

        mean_vis_0 = gamma_bar_vis+c_vis
        mean_vis_1 = gamma_bar_vis-c_vis

        mean_aud_0 = gamma_bar_aud+c_aud
        mean_aud_1 = gamma_bar_aud-c_aud


        vis_modality_stim_0 = rng.normal(loc=mean_vis_0, scale=config['stim_std_test'], size=batch_size)
        vis_modality_stim_1 = rng.normal(loc=mean_vis_1, scale=config['stim_std_test'], size=batch_size)
        aud_modality_stim_0 = rng.normal(loc=mean_aud_0, scale=config['stim_std_test'], size=batch_size)
        aud_modality_stim_1 = rng.normal(loc=mean_aud_1, scale=config['stim_std_test'], size=batch_size)



    elif mode == 'test':
        p_coh = kwargs['p_coh']
        c_cue = kwargs['c_cue']
        c_vis = kwargs['c_vis']
        c_aud = kwargs['c_aud']
        if not hasattr(c_cue, '__iter__'):
            c_cue = np.array([c_cue] * batch_size)
        if not hasattr(c_vis, '__iter__'):
            c_vis = np.array([c_vis] * batch_size)
        if not hasattr(c_aud, '__iter__'):
            c_aud = np.array([c_aud] * batch_size)

        mean_vis_0 = gamma_bar_vis+c_vis
        mean_vis_1 = gamma_bar_vis-c_vis

        mean_aud_0 = gamma_bar_aud+c_aud
        mean_aud_1 = gamma_bar_aud-c_aud
        vis_modality_stim_0 = rng.normal(loc=mean_vis_0, scale=config['stim_std_test'], size=batch_size)
        vis_modality_stim_1 = rng.normal(loc=mean_vis_1, scale=config['stim_std_test'], size=batch_size)
        aud_modality_stim_0 = rng.normal(loc=mean_aud_0, scale=config['stim_std_test'], size=batch_size)
        aud_modality_stim_1 = rng.normal(loc=mean_aud_1, scale=config['stim_std_test'], size=batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    batch_0 = rng.uniform(0.00, 0.00, batch_size)#gamma_bar - c


    # vis_modality_stim = rng.choice([0, 1], (batch_size,))
    # aud_modality_stim = rng.choice([0, 1], (batch_size,))


    cue_dim = config['n_ring_HL']
    dim_start = config['n_ring_RDM']
    # cue input
    ############################## cue input
    num_zero=int(cue_duration*config['sparsity_HL'])
    num_nonzero= cue_duration-num_zero

    HL_sequences=[]
    for i in range(batch_size):
        HL_sequence_zero = 0.0*np.ones(num_zero)
        HL_sequence_h = c_cue[i]*np.ones(int(p_coh*num_nonzero))
        #print('HL_sequence_h',cue_duration,HL_sequence_h)
        HL_sequence_l = -c_cue[i]*np.ones(num_nonzero-int(p_coh*num_nonzero))
        #print('HL_sequence_l',HL_sequence_l)
        HL_sequence = np.concatenate((HL_sequence_h,HL_sequence_l,HL_sequence_zero),axis=0)
        np.random.shuffle(HL_sequence)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    #print('HL_sequences',HL_sequences)


    if config['mod_type'] == 'cue_test':
        #HL_sequences = tools.generate_seq_training(c_cue, config, cue_duration, p_coh, batch_size)
        if config['front_show'] == 'plus_type1':
            HL_sequences = tools.generate_seq_cue_plus_type1(c_cue, config, cue_duration, p_coh, batch_size)
            #print('HL_sequences', HL_sequences.shape, HL_sequences)
        if config['front_show'] == 'type2':
            HL_sequences = tools.generate_seq_cue_plus_type2(c_cue, config, cue_duration, p_coh, batch_size)
            #print('HL_sequences', HL_sequences.shape, HL_sequences)
        if config['front_show'] == 'type3':
            HL_sequences = tools.generate_seq_cue_plus_type3(c_cue, config, cue_duration, p_coh, batch_size)
            #print('HL_sequences', HL_sequences.shape, HL_sequences)
        if config['front_show'] == 'minus_type1':
            HL_sequences = tools.generate_seq_cue_minus_type1(c_cue, config, cue_duration, p_coh, batch_size)
            #print('HL_sequences', HL_sequences.shape, HL_sequences)


    #print('HL_sequences', HL_sequences.shape, HL_sequences)

    ############################## cue input
    context_RDM=1
    context_HL=1
    for i in range(dim_start,dim_start+cue_dim):
        trial.add('input', i, ons=cue_on, offs=cue_off, strengths=HL_sequences)

    trial.add('input', -4, ons=stim1_on, offs=stim1_off, strengths=vis_modality_stim_0)
    trial.add('input', -3, ons=stim1_on, offs=stim1_off, strengths=vis_modality_stim_1)
    trial.add('input', -2, ons=stim1_on, offs=stim1_off, strengths=aud_modality_stim_0)
    trial.add('input', -1, ons=stim1_on, offs=stim1_off, strengths=aud_modality_stim_1)

    #trial.add('input', -1, ons=cue_on, offs=response_off, strengths=trial.expand(context_HL))


    output_target0 = 1. * (c_cue > batch_0)*(mean_vis_0 > mean_vis_1)
    output_target1 = 1. * (c_cue > batch_0)*(mean_vis_0 <= mean_vis_1)
    output_target2 = 1. * (c_cue <= batch_0)*(np.abs(mean_aud_0) > np.abs(mean_aud_1))
    output_target3 = 1. * (c_cue <= batch_0)*(np.abs(mean_aud_0) <= np.abs(mean_aud_1))

    if config['switch_context']=='con_A2B':

            output_target0 = 1. * (c_cue < batch_0)*(mean_vis_0 > mean_vis_1)
            output_target1 = 1. * (c_cue < batch_0)*(mean_vis_0 <= mean_vis_1)
            output_target2 = 1. * (c_cue >= batch_0)*(np.abs(mean_aud_0) > np.abs(mean_aud_1))
            output_target3 = 1. * (c_cue >= batch_0)*(np.abs(mean_aud_0) <= np.abs(mean_aud_1))

    elif config['switch_context']=='con_B2A':

            output_target0 = 1. * (c_cue > batch_0)*(mean_vis_0 > mean_vis_1)
            output_target1 = 1. * (c_cue > batch_0)*(mean_vis_0 <= mean_vis_1)
            output_target2 = 1. * (c_cue <= batch_0)*(np.abs(mean_aud_0) > np.abs(mean_aud_1))
            output_target3 = 1. * (c_cue <= batch_0)*(np.abs(mean_aud_0) <= np.abs(mean_aud_1))


    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target0)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 2, ons=response_on, offs=response_off, strengths=output_target2)
    trial.add('out', 3, ons=response_on, offs=response_off, strengths=output_target3)
    #trial.add('out', 4, ons=response_on, offs=response_off, strengths=trial.expand(0.1*context_HL))


    ############################## mask

    if config['mask_start']=='cueOn':
        trial.add('cost_mask', 0, ons=cue_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 1, ons=cue_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 2, ons=cue_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 3, ons=cue_on, offs=response_off, strengths=trial.expand(1.))
    elif config['mask_start']=='delayOn':
        trial.add('cost_mask', 0, ons=cue_off, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 1, ons=cue_off, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 2, ons=cue_off, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 3, ons=cue_off, offs=response_off, strengths=trial.expand(1.))
    elif config['mask_start']=='stimOn':
        trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 2, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 3, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    elif config['mask_start']=='respOn':
        trial.add('cost_mask', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 1, ons=response_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 2, ons=response_on, offs=response_off, strengths=trial.expand(1.))
        trial.add('cost_mask', 3, ons=response_on, offs=response_off, strengths=trial.expand(1.))


    target_choice_A =      1*(c_cue > batch_0)*(mean_vis_0 > mean_vis_1) \
                          +2*(c_cue > batch_0)*(mean_vis_0 <= mean_vis_1) \
                          +3*(c_cue <= batch_0)*(np.abs(mean_aud_0) > np.abs(mean_aud_1)) \
                          +4*(c_cue <= batch_0)*(np.abs(mean_aud_0) <= np.abs(mean_aud_1))

    target_choice_B =      1*(c_cue < batch_0)*(mean_vis_0 > mean_vis_1) \
                          +2*(c_cue < batch_0)*(mean_vis_0 <= mean_vis_1) \
                          +3*(c_cue >= batch_0)*(np.abs(mean_aud_0) > np.abs(mean_aud_1)) \
                          +4*(c_cue >= batch_0)*(np.abs(mean_aud_0) <= np.abs(mean_aud_1))

    target_choice = target_choice_A


    if config['switch_context']=='con_A2B':
        if not config['start_switch']:
            target_choice = target_choice_A
        else:
            target_choice = target_choice_B

    elif config['switch_context']=='con_B2A':
        if not config['start_switch']:
            target_choice = target_choice_B
        else:
            target_choice = target_choice_A




    trial.epochs = {'fix': (None, stim1_on),
                    'cue_stimulus':(cue_on, cue_off),
                    'stimulus': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}
    #print('trial.epochs',trial.epochs)

    trial.strength_cue = c_cue
    trial.context_RDM = context_RDM
    trial.context_HL = context_HL
    trial.strength_0    = batch_0
    trial.mean_vis_0 = mean_vis_0
    trial.mean_vis_1 = mean_vis_1
    trial.mean_aud_0 = mean_aud_0
    trial.mean_aud_1 = mean_aud_1
    trial.gamma_bar_vis = gamma_bar_vis
    trial.gamma_bar_aud = gamma_bar_aud
    trial.c_vis = c_vis
    trial.c_aud = c_aud


    trial.batch_0 = batch_0

    trial.p_coh = p_coh

    trial.output_target0 = output_target0
    trial.output_target1 = output_target1
    trial.output_target2 = output_target2
    trial.output_target3 = output_target3

    trial.target_choice = target_choice

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial

def RDM_task(config, mode, **kwargs):
    return _RDM_task(config, mode, **kwargs)

def HL_task(config, mode, **kwargs):
    return _HL_task(config, mode, **kwargs)
# map string to function
rule_mapping = {
    'RDM_task': RDM_task,
    'HL_task': HL_task,
               }



def generate_trials(context_name, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    # print(rule)
    config = hp
    kwargs['noise_on'] = noise_on

    seed =config['seed']
    rng = config['rng']
    trial = rule_mapping[context_name](config, mode, **kwargs)
    p_coh = trial.p_coh
    # print('p_coh',p_coh)
    cue_on = trial.epochs['cue_stimulus'][0][0]
    cue_off = trial.epochs['cue_stimulus'][1][0]
    batch_size =  kwargs['batch_size']
    n_ring_RDM = config['n_ring_RDM']
    scale_RDM = config['scale_RDM']
    #print('****** scale_RDM',scale_RDM)

    shuffle_matrix = []
    for i in range(cue_off-cue_on):
        num = np.arange(n_ring_RDM)
        np.random.shuffle(num)
        shuffle_matrix.append(num)
    #print(shuffle_matrix)

    x_temp = np.zeros(shape=trial.x.shape,dtype=np.float32)

    if noise_on:
        trial.add_x_noise()
    if config['zero_input']=='yes':
        trial.x = np.zeros_like(trial.x)
        #print(trial.x.shape)
    #print('@@@@@HL_task trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:,0,:],decimals=2))



    return trial


# test the generation of trials
if __name__ == "__main__":
    import seaborn as sns
    from matplotlib import pyplot as plt

    import default

    print(sys.argv)
    print(len(sys.argv))

    min_interval = 100
    max_interval = 500
    num_interval = 3



    for rule_name in np.array(['HL_task']):#,'HL_task'
        print('********* rule_name:',rule_name)
        hp = default.get_default_hp(rule_name)
        trial = generate_trials(rule_name, hp, 'random_validate', batch_size=3, noise_on=False)
        if rule_name == 'RDM_task':

            print('@@@@@ trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[2:42,0,:],decimals=2))
            #print('@@@@@ trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[:,0,:],decimals=2))

            # print('@@@@@ trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:,1,:],decimals=2))
            # print('@@@@@ trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[:,1,:],decimals=2))

            # print('@@@@@ trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:60,2,:],decimals=2))
            # print('@@@@@ trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[65:80,2,:],decimals=2))


        elif rule_name == 'HL_task':

            print('@@@@@HL_task trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:,0,:],decimals=2))
            print('@@@@@HL_task trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[:,0,:],decimals=2))

            print('@@@@@HL_task trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:,1,:],decimals=2))
            print('@@@@@HL_task trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[:,1,:],decimals=2))

            # print('@@@@@ trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[20:60,2,:],decimals=2))
            # print('@@@@@ trial.y',trial.y[:,0,:].shape,'\n', np.around(trial.y[65:80,2,:],decimals=2))
