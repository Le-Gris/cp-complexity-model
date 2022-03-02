# Experiment 2 with convnet.
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
    layer_params = {'input_dim': 256, 'encoder_in_channels': 1, 'encoder_out_channels': 1, 'encoder_kernel': 8, 'stride': 2, 'decoder_in':  125, 'decoder_out': 256, 'classifier_in': 125,
                    'classifier_out': 2}
    
    # Set sim parameters
    sim_params = OrderedDict({'train_ratio': 0.7, 'AE_epochs': 50,
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-4,'AE_wd': 10e-5,
                              'AE_thresh': 0.001, 'AE_patience': 0.001, 'class_epochs': 50, 'class_batch_size': 8, 'class_lr': 10e-2,
                              'class_wd': 10e-3, 'class_monitor': 'loss', 'class_thresh': 85, 'training': 'early_stop',
                              'inplace_noise': True, 'save_model': True, 'metric':'cosine', 'verbose': False})    
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    exp2 = {}
    
    # Data set
    exp2['dataset'] = {}
    exp2['sim'] = {}
    exp2['mode'] = 'binary'
    exp2['model'] = 'conv'
    exp2['exp_name'] = 'exp2'
    exp2['data_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'binary', 'exp2'))
    exp2['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'data', 'binary', 'exp2', 'categories'))
    exp2['save_dir'] = osp.abspath(osp.join(Path(__file__).parents[2], 'results'))

    ## Macrofeature parameters
    exp2['dataset']['i'] = 16
    exp2['dataset']['k'] = [i for i in range(1,13)]
    exp2['dataset']['l'] = 32
    exp2['dataset']['m'] = 8
    exp2['dataset']['s'] = 0
    exp2['dataset']['s_list'] = None
    
    ## Category parameters
    exp2['dataset']['d'] = [i for i in range(2,12)] + [j for j in range(12, 30, 2)]
    exp2['dataset']['pd_i'] = [k for k in range(1,13)]
    exp2['dataset']['pd'] = [0.0]
    
    # Simulation parameters
    exp2['sim']['layer_params'], exp2['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(exp2, f, indent=3)

if __name__ == "__main__":
    main()
