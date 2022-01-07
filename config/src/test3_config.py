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
    layer_params = {'encoder_in_channels': 1, 'encoder_out_channels': 1, 'encoder_kernel': 8, 'stride': 2, 'decoder_in':  125, 'decoder_out': 256, 'classifier_in': 125, 
                    'classifier_out': 2}
    
    # Set sim parameters
    sim_params = OrderedDict({'encoder_out_name': 'lin1_encoder', 'train_ratio': 0.8, 'AE_epochs': 35, 
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,'AE_wd': 10e-5, 
                              'AE_thresh': 0.001, 'AE_patience': 0.001, 'class_epochs': 35, 'class_batch_size': 8, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'training': 'early_stop', 'inplace_noise': True, 'save_model': True, 'metric':'cosine', 'verbose': False})
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    test2 = {}
    
    # Data set
    test2['dataset'] = {}
    test2['sim'] = {}
    test2['mode'] = 'binary_custom'
    test2['model'] = 'conv'
    test2['exp_name'] = 'test3'
    test2['data_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'data'))
    test2['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'results'))
    
    ## Macrofeature parameters
    test2['dataset']['i'] = 1
    test2['dataset']['k'] = [8,11,12]
    test2['dataset']['l'] = 32
    test2['dataset']['m'] = 8
    test2['dataset']['s'] = 0
    test2['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    test2['dataset']['custom'] = [[8,24,6,0.0], [11,12,11,0.0], [11,14,6,0.0], [11,22,10,0.0], [11,24,10,0.0], [12,22,10,0.0], [12,26,8,0.0]]
    
    # Simulation parameters
    test2['sim']['layer_params'], test2['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(test2, f, indent=3)

if __name__ == "__main__":
    main()
