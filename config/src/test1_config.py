# Test config file with regular nn model. Verify that directory structure, datasets and simulations are as intended.
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
    
    # Set layer parameters 
    layer_params = {'encoder_in': 256, 'encoder_out': 128, 'decoder_in':  128, 'decoder_out': 256, 'classifier_in': 128,
                    'classifier_out': 2}

    # Set sim parameters
    sim_params = OrderedDict({'encoder_out_name': 'lin1_encoder', 'train_ratio': 0.8, 'AE_epochs': 15,
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,'AE_wd': 10e-5,
                              'AE_thresh': None, 'AE_patience': None, 'class_epochs': 15, 'class_batch_size': 8, 'class_lr': 10e-2,
                              'class_wd': 10e-3, 'training':'fixed', 'inplace_noise': True,'save_model': True, 'metric': 'euclid', 'verbose': False})

    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 test config
    test1 = {}
    
    # Data set
    test1['sim'] = {}
    test1['dataset'] = {}
    test1['mode'] = 'binary'
    test1['mdoel'] = 'nn'
    test1['exp_name'] = 'test1'
    test1['data_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'data'))
    test1['save_dir'] = osp.abspath(osp.join(Path(__file__).parent,'..', '..', 'test', 'test_results', 'results'))
    
    ## Macrofeature parameters
    test1['dataset']['i'] = 1
    test1['dataset']['k'] = [1,3,5] 
    test1['dataset']['l'] = 32
    test1['dataset']['m'] = 8
    test1['dataset']['s'] = 0
    test1['dataset']['s_list'] = None
    
    ## Category parameters
    test1['dataset']['d'] = [1,5]
    test1['dataset']['pd_i'] = [1,3,5]
    test1['dataset']['pd'] = [0.0,0.1]

    ## Simulation parameters
    test1['sim']['layer_params'], test1['sim']['sim_params'] = sim_config()

    with open(save_fname, 'w') as f:
        json.dump(test1, f, indent=3)

if __name__ == "__main__":
    main()
