# Experiment 3 with convnet.
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
    layer_params = {'input_dim': 3, 'encoder_in_channels': 1, 'encoder_out_channels': 1, 'encoder_kernel': 2, 'stride': 1, 'padding': 2, 'decoder_in':  4, 'decoder_out': 3, 'classifier_in': 4,
                    'classifier_out': 2}

    # Set sim parameters
    sim_params = OrderedDict({'encoder_out_name': 'lin1_encoder', 'train_ratio': 0.8, 'AE_epochs': 45,
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,'AE_wd': 10e-5,
                              'AE_thresh': 0.001, 'AE_patience': 0.001, 'class_epochs': 45, 'class_batch_size': 8, 'class_lr': 10e-2,
                              'class_wd': 10e-3, 'class_monitor': 'acc', 'class_thresh': 85, 'training': 'early_stop',
                              'inplace_noise': True, 'save_model': True, 'metric':'cosine', 'verbose': False})    
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    exp3 = {}
    
    # Data set
    exp3['dataset'] = {}
    exp3['sim'] = {}
    exp3['mode'] = 'cont'
    exp3['model'] = 'conv'
    exp3['exp_name'] = 'exp3'
    exp3['data_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'continuous'))
    exp3['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'continuous', 'categories'))
    exp3['save_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'results'))

    ## Category parameters
    exp3['dataset']['num_cat'] = 8
    exp3['dataset']['overlap_params'] = []
    exp3['dataset']['meanA'] = []
    exp3['dataset']['meanB'] = []
    exp3['dataset']['devA'] = []
    exp3['dataset']['devB'] = [] 
    exp3['dataset']['size'] = 1000
    exp3['dataset']['dim'] = 3
    exp3['dataset']['threshold'] = []
    exp3['dataset']['rangeA'] = []
    exp3['dataset']['rangeB'] = []

    # Simulation parameters
    exp3['sim']['layer_params'], exp3['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp3, f, indent=3)

if __name__ == "__main__":
    main()
