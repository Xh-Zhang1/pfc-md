import numpy as np
import os
import task


def input_output_n():
    # RDM_task: 72


    n_rule = 0
    n_present = 4
    n_ring_RDM = 0#
    n_ring_HL = 2#

    return n_ring_RDM,n_ring_HL,n_ring_RDM+n_ring_HL+n_present+n_rule, 4


def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
    save_dir = os.path.join(root_path, 'models_saved')
    figure_dir = os.path.join(root_path, 'z_figure')
    print('root_path',root_path)
    n_ring_RDM,n_ring_HL,n_input, n_output = input_output_n()


    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(100)
    else:
        seed = random_seed


    hp = {
        'root_path':root_path,
        'figure_dir': figure_dir,


        'n_ring_RDM':n_ring_RDM,
        'n_ring_HL':n_ring_HL,
        'e_prop':0.8,
        'rule_name': rule_name,
        # batch size for training
        'batch_size_train': 64, #128,#64,
        # batch_size for testing
        'batch_size_test': 512,
        # Type of RNNs: RNN
        'rnn_type': 'RNN',
        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation functions: relu, softplus
        'activation': 'softplus',
        # Time constant (ms)
        'tau': 20,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3,#0.25,#0.27,#0.3,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # a default weak regularization prevents instability
        'l1_firing_rate': 0,
        # l2 regularization on activity
        'l2_firing_rate': 0,
        # l1 regularization on weight
        'l1_weight': 0.0,
        # l2 regularization on weight
        'l2_weight': 0.0,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn':256,
        'n_md':30,
        # learning rate
        'learning_rate': 0.0005,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
        'is_EI': 'EI',
        'use_reset':'yes',
        'rdm_context':'rdm1',
        'hlcontext':'hl1',

        'scale_RDM':1.0,
        'scale_HL':1.0,
        'std_scale':0.0,
        'cue_scale':1.0,

        'stim_scale':1.0,
        'p_coh':0.92,
        'mask_start':'respOn',
        'p_md1':0.6,

        'cue_duration':1200,
        'cue_delay':300,
        'stim':100,

        'response_time':100,
        'model_idx':1,
        'gamma_noise':1.0,
        'stim_std':0.1,
        'input_mask':'input_mask',
        'dropout':0.0,
        'scale_random':1.0,
        'init_hh':'init2',

        'remove_mask':'no',
        'stim_epoch':'no',
        'mask_type':'no',
        'add_mask':'no',
        'mode_mask':'train',
        'sparsity_HL':0.0,
        'sparsity_RDM':0.0,
        'get_SNR':'no',
        'zero_input':'no',
        'mask_test' :'no',
        'stim_cell':'no',
        'stim_value':0,
        'switch_context':'con_A',
        'start_switch':True,

        'mod_type':'training',
        'front_show':'no',






    }

    return hp
