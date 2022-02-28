# Tester for convnet model performance with hard categories from experiment 1. 
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
    layer_params = {'encoder_in': 2, 'encoder_out': 20, 'decoder_in': 20, 'decoder_out': 2, 'classifier_in': 20,
                    'classifier_out': 2}

    # Set sim parameters
    sim_params = OrderedDict({'train_ratio': 0.7, 'AE_epochs': 50, 
                              'AE_batch_size': 10, 'noise_factor': 0.1, 'AE_lr': 10e-3,'AE_wd': 10e-5, 
                              'AE_thresh': 0.001, 'AE_patience': 5, 'class_epochs': 50, 'class_batch_size': 10, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'class_monitor': 'loss', 'class_thresh': 0.001, 'training': 'early_stop', 
                              'inplace_noise': True, 'save_model': True, 'metric':'cosine', 'verbose': False})
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    exp4 = {}
    
    # Data set
    exp4['dataset'] = {}
    exp4['sim'] = {}
    exp4['mode'] = 'cont'
    exp4['model'] = 'nn'
    exp4['exp_name'] = 'exp4'
    exp4['data_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'data', 'continuous', 'exp4'))
    exp4['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'data', 'continuous', 'exp4', 'categories'))
    exp4['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'results'))
    
    ## Macrofeature parameters
    exp4['dataset']['size'] = 10000
    exp4['dataset']['i'] = None
    exp4['dataset']['k'] = None
    exp4['dataset']['l'] = None
    exp4['dataset']['m'] = None
    exp4['dataset']['s'] = None
    exp4['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    exp4['dataset']['custom'] = None 
    
    # Simulation parameters
    exp4['sim']['layer_params'], exp4['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp4, f, indent=3)

if __name__ == "__main__":
    main()
