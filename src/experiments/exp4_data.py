import argparse
import json
import neuralnet as NN
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial import distance
import run_simulations as rs
import torch

def parse_args():

    parser = argparse.ArgumentParser(description='Generate plots for experiments')
    parser.add_argument('-c', help='<config.json>', required=True)
    args = parser.parse_args()
    return args.c

def cosine_distance(act1, act2):
    return (1 - np.dot(act1, act2)/(np.linalg.norm(act1)*np.linalg.norm(act2)))

def get_dist(nn, x):
    nn.eval()
    act = nn.forward(x).detach().numpy()
    dist = [cosine_distance(act[i], act[i+1]) for i in range(len(act)-1)]
    return dist

def main(**kwargs):

    config_fname = parse_args()
    config, exp_name, data_dir, save_dir, mode, model, size = rs.get_configuration(config_fname)
    
    sims = glob.glob(os.path.join(save_dir, exp_name, '*'))
    
    n_stims = 500
    xmin = -8
    xmax = 8
    x_ = np.linspace(xmin,xmax,n_stims)
    dx = x_[1] - x_[0]
    y = np.zeros(n_stims)
    x = np.stack([y,x_]).T
    x = torch.as_tensor(x, dtype=torch.float32)
    x = torch.nn.functional.normalize(x)
    
    layer_params = config['layer_params']

    for j, p in enumerate(sims):
        
        # Verify that dir
        if not os.path.isdir(p):
            continue
       
        # Neural net layer config
        encoder_config, decoder_config, classifier_config = rs.get_model_arch(model, layer_params)
        
        # Stimuli
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.nn.functional.normalize(x)

        # Init neural net
        nn = NN.Net(encoder_config, decoder_config,classifier_config)
        
        nn.eval()

        # Init
        path = os.path.join(p, 'model_checkpoints', 'init_checkpoint.pth')
        init_params = torch.load(path)
        nn.load_state_dict(init_params['state_dict'])
        
        ## Neural distances
        dist = get_dist(nn, x) 
        
        ## Plot
        plt.figure(1)
        plt.plot(x_[:-1], dist)
        plt.xlabel('x')
        plt.ylabel('neural distance')
        plt.savefig(os.path.join(p,'plots', 'init_neuraldist'))
        plt.close(1)

        # Before
        path = os.path.join(p, 'model_checkpoints', 'AE_checkpoint.pth')
        init_params = torch.load(path)
        nn.load_state_dict(init_params['model_state_dict'])
        
        ## Neural distances
        dist = get_dist(nn, x)

        ## Plot
        plt.figure(2)
        plt.plot(x_[:-1], dist)
        plt.xlabel('x')
        plt.ylabel('neural distance')
        plt.savefig(os.path.join(p,'plots', 'before_neuraldist'))
        plt.close(2)

        # After                        
        path = os.path.join(p, 'model_checkpoints', 'class_checkpoint.pth')
        init_params = torch.load(path)
        nn.load_state_dict(init_params['model_state_dict'])
        
        ## Neural distances
        dist = get_dist(nn, x)

        ## Plot
        plt.figure(3)
        plt.plot(x_[:-1], dist)
        plt.xlabel('x')
        plt.ylabel('neural distance')
        plt.savefig(os.path.join(p,'plots', 'after_neuraldist'))
        plt.close(3)
         

    
if __name__ == '__main__':
    main()


