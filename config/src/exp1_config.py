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
    layer_params = {'encoder_in': 256, 'encoder_out': 128, 'decoder_in':  128, 'decoder_out': 256, 'classifier_in': 128, 
                    'classifier_out': 2}
    
    # Set sim parameters
    sim_params = OrderedDict({'encoder_out_name': 'lin1_encoder', 'train_ratio': 0.8, 'AE_epochs': 15, 
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,'AE_wd': 10e-5, 
                              'AE_thresh': None, 'AE_patience': None, 'class_epochs': 15, 'class_batch_size': 8, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'class_monitor': None, 'class_thresh': None, 'training': 'fixed', 
                              'inplace_noise': True, 'save_model': True, 'verbose': False, 'metric':'euclid'})
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    exp1 = {}
    
    # Data set
    exp1['dataset'] = {}
    exp1['sim'] = {}
    exp1['mode'] = 'binary'
    exp1['model'] = 'nn'
    exp1['exp_name'] = 'exp1'
    exp1['data_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'binary', 'exp1'))
    exp1['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'binary', 'exp1', 'categories'))
    exp1['save_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'results'))

    ## Macrofeature parameters
    exp1['dataset']['i'] = 16
    exp1['dataset']['k'] = [i for i in range(1,13)]
    exp1['dataset']['l'] = 32
    exp1['dataset']['m'] = 8
    exp1['dataset']['s'] = 0
    exp1['dataset']['s_list'] = None
    
    ## Category parameters
    exp1['dataset']['d'] = [i for i in range(2,12)] + [j for j in range(12, 30, 2)]
    exp1['dataset']['pd_i'] = [k for k in range(1,13)]
    exp1['dataset']['pd'] = [0.0, 0.1, 0.2, 0.4]
    
    # Simulation parameters
    exp1['sim']['layer_params'], exp1['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp1, f, indent=3)

if __name__ == "__main__":
    main()
