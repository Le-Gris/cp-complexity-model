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
    layer_params = {'input_dim': 256, 'encoder_in_channels': 1, 'encoder_out_channels': 1, 'encoder_kernel': 8, 'stride': 2, 'decoder_in':  125, 'decoder_out': 256, 'classifier_in': 125, 
                    'classifier_out': 2}
    
    # Set sim parameters
    sim_params = OrderedDict({'train_ratio': 0.8, 'AE_epochs': 50, 
                              'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,'AE_wd': 10e-5, 
                              'AE_thresh': 0.001, 'AE_patience': 5, 'class_epochs': 50, 'class_batch_size': 8, 'class_lr': 10e-2, 
                              'class_wd': 10e-3, 'class_monitor': 'loss', 'class_thresh': 0.001, 'training': 'early_stop', 
                              'inplace_noise': True, 'save_model': True, 'metric':'cosine', 'verbose': False})
    
    return layer_params, sim_params

def main():
    
    # Get filename to write to
    save_fname = parse_args()

    # Experiment 1 config
    test3 = {}
    
    # Data set
    test3['dataset'] = {}
    test3['sim'] = {}
    test3['mode'] = 'binary_custom'
    test3['model'] = 'conv'
    test3['exp_name'] = 'test3'
    test3['data_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'data', 'test3'))
    test3['dataset_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'data', 'test3', 'categories'))
    test3['save_dir'] = osp.abspath(osp.join(Path(__file__).parent, '..', '..', 'test', 'test_results', 'results'))
    
    ## Macrofeature parameters
    test3['dataset']['size'] = 2048
    test3['dataset']['i'] = 1
    test3['dataset']['k'] = [8,11,12]
    test3['dataset']['l'] = 32
    test3['dataset']['m'] = 8
    test3['dataset']['s'] = 0
    test3['dataset']['s_list'] = None

    ## Category parameters
    ### In order, each element contains: k, d, pdi, pd
    test3['dataset']['custom'] = [[8,24,6,0.0], [11,12,11,0.0], [11,14,6,0.0], [11,22,10,0.0], [11,24,10,0.0], [12,22,10,0.0], [12,26,8,0.0]]
    
    # Simulation parameters
    test3['sim']['layer_params'], test3['sim']['sim_params'] = sim_config() 
    
    with open(save_fname, 'w') as f:
        json.dump(test3, f, indent=3)

if __name__ == "__main__":
    main()
