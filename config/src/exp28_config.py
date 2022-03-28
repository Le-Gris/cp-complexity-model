"""
Configuration file for simulations with benchmark stimuli. 
Constraints on autoencoder (dimensionality reduction). 
Saving weightings from inner layer. 
Inner representation CP measurements are non-linear. 
Activation is sigmoidal all the way through.
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
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-2,'AE_wd': 10e-5, 
                              'AE_thresh': 0.1, '_patience': 4, 'class_epochs': 100, 'class_batch_size': 8, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'class_monitor': 'loss', 'class_thresh': 0.1, 'training': 'early_stop', 'eval_mode':'batch', 
                              'inplace_noise': True, 'rep_type': 'act', 'save_model': True, 'metric':'cosine', 'verbose': False, 'rep_diff':True})
    
    return layer_params, sim_params

def main(**kwargs):
    
    # Get filename to write to
    save_fname = kwargs['save_fname']

    # Experiment 1 config
    exp28 = {}
    
    # Data set
    exp28['dataset'] = {}
    exp28['sim'] = {}
    exp28['mode'] = 'benchmark'
    exp28['model'] = 'nn-sig'
    exp28['exp_name'] = 'exp28'
    exp28['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'data', 'benchmark_stimuli', 'N8'))
    exp28['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'results'))
    
    ## Macrofeature parameters
    exp28['dataset']['size'] = 1000
    exp28['dataset']['i'] = None
    exp28['dataset']['k'] = None
    exp28['dataset']['l'] = None
    exp28['dataset']['m'] = None
    exp28['dataset']['s'] = None
    exp28['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    exp28['dataset']['custom'] = None 
    
    # Simulation parameters
    exp28['sim']['layer_params'], exp28['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp28, f, indent=3)

if __name__ == "__main__":
    save_fname = parse_args()
    main(save_fname=save_fname)
