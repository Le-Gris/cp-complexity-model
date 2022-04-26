"""
Configuration file for simulations with benchmark stimuli. 
Constraints on autoencoder (dimensionality reduction). 
Saving weightings from inner layer. 
Inner representation CP measurements are non-linear. 
Activation is sigmoidal all the way through.
thresh = 1%
eval_mode = batch
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
    layer_params = {'encoder_in': 8, 'encoder_out': 7, 'encoder_in2': 7, 'encoder_out2': 5, 'decoder_in': 5, 'decoder_out': 8,
                    'classifier_in': 5, 'classifier_out': 2}

    # Set sim parameters
    sim_params = OrderedDict({'train_ratio': 0.75, 'AE_epochs': 200, 
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-3, 'AE_wd': 10e-5, 
                              'AE_thresh': 0.01, '_patience': 75, 'class_epochs': 200, 'class_batch_size': 8, 'class_lr': 10e-3, 
                              'class_wd': 10e-4, 'class_monitor': 'loss', 'class_thresh': 0.01, 'training': 'early_stop', 'eval_mode':'batch', 
                              'inplace_noise': True, 'rep_type': 'act', 'save_model': True, 'metric':'cosine', 'verbose': False, 'rep_diff':True})
    
    return layer_params, sim_params

def main(**kwargs):
    
    # Get filename to write to
    save_fname = kwargs['save_fname']

    # Experiment 1 config
    exp33 = {}
    
    # Data set
    exp33['dataset'] = {}
    exp33['sim'] = {}
    exp33['mode'] = 'benchmark'
    exp33['model'] = 'nn-sig2'
    exp33['exp_name'] = 'exp33'
    exp33['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'data', 'benchmark_stimuli', 'N8'))
    exp33['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'results'))
    
    ## Macrofeature parameters
    exp33['dataset']['size'] = 1000
    exp33['dataset']['i'] = None
    exp33['dataset']['k'] = None
    exp33['dataset']['l'] = None
    exp33['dataset']['m'] = None
    exp33['dataset']['s'] = None
    exp33['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    exp33['dataset']['custom'] = None 
    
    # Simulation parameters
    exp33['sim']['layer_params'], exp33['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp33, f, indent=3)

if __name__ == "__main__":
    save_fname = parse_args()
    main(save_fname=save_fname)
