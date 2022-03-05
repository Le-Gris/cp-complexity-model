"""
Configuration file for simulations with benchmark stimuli. 
Constraints on autoencoder (dimensionality reduction). 
Saving weightings from inner layer. 
Inner representation CP measurements are non-linear. 
Sigmoid encoder.
No activation function decoder.
Sigmoid classifier.
"""

import json
import argparse
from collections import OrderedDict
import os.path as osp
from pathlib import Path

def parse_args():

    parser = argparse.ArgumentParser(description='Experiment configurations')
    parser.add_argument('-save', help='<save/path/file.json>', required=True)
    args = parser.parse_args()

    return args.save

def sim_config():
    
    # Set layers parameters
    layer_params = {'encoder_in': 8, 'encoder_out': 5, 'decoder_in': 5, 'decoder_out': 8, 'classifier_in': 5,
                    'classifier_out': 2}

    # Set sim parameters
    sim_params = OrderedDict({'train_ratio': 0.7, 'AE_epochs': 100, 
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-3,'AE_wd': 10e-5, 
                              'AE_thresh': 0.001, 'AE_patience': 5, 'class_epochs': 100, 'class_batch_size': 8, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'class_monitor': 'loss', 'class_thresh': 0.001, 'training': 'early_stop', 
                              'inplace_noise': True, 'rep_type': 'act', 'save_model': True, 'metric':'cosine', 'verbose': False})
    
    return layer_params, sim_params

def main(**kwargs):
    
    # Get filename to write to
    save_fname = kwargs['save_fname']

    # Experiment 1 config
    exp19 = {}
    
    # Data set
    exp19['dataset'] = {}
    exp19['sim'] = {}
    exp19['mode'] = 'benchmark'
    exp19['model'] = 'lin-decoder-sig'
    exp19['exp_name'] = 'exp19'
    exp19['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'data', 'benchmark_stimuli', 'N8'))
    exp19['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'results'))
    
    ## Macrofeature parameters
    exp19['dataset']['size'] = 1000
    exp19['dataset']['i'] = None
    exp19['dataset']['k'] = None
    exp19['dataset']['l'] = None
    exp19['dataset']['m'] = None
    exp19['dataset']['s'] = None
    exp19['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    exp19['dataset']['custom'] = None 
    
    # Simulation parameters
    exp19['sim']['layer_params'], exp19['sim']['sim_params'] = sim_config() 
    
    # Repetition parameter
    exp19['repeat'] = 0

    with open(save_fname, 'w') as f:
        json.dump(exp19, f, indent=3)

if __name__ == "__main__":
    save_fname = parse_args()
    main(save_fname=save_fname)
