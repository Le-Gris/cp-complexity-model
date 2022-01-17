__author__ = 'Solim LeGris'

# Imports
import os
from . import neuralnet as NN
import torch
from torch import nn, utils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import time
import argparse
import json
plt.ioff()

# Get simulation configuration filename
def parse_args():

    parser = argparse.ArgumentParser(description='Run simulation experiments')
    parser.add_argument('-c', help='<config.json>', required=True)
    args = parser.parse_args()
    return args.c

def get_model_arch(model, layer_params):

    if model == 'nn':
        encoder_config = OrderedDict({'lin1_encoder': nn.Linear(layer_params['encoder_in'], layer_params['encoder_out']),
                                      'norm1_encoder': nn.BatchNorm1d(layer_params['encoder_out']), 'sig1_encoder': nn.ReLU()})

        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(layer_params['decoder_in'], layer_params['decoder_out']),
                                      'norm1_decoder': nn.BatchNorm1d(layer_params['decoder_out']), 'sig2_decoder': nn.ReLU()})

        classifier_config = OrderedDict({'lin1_classifier': nn.Linear(layer_params['classifier_in'], layer_params['classifier_out']),
                                         'sig1_classifier': nn.Sigmoid()})
    elif model == 'conv':
        encoder_config = OrderedDict({'unflatten': nn.Unflatten(1, (1, layer_params['input_dim'])),
                                      'lin1_encoder': nn.Conv1d(layer_params['encoder_in_channels'], layer_params['encoder_out_channels'], kernel_size=layer_params['encoder_kernel'], stride=layer_params['stride']),
                                        'flatten': nn.Flatten(), 'norm1_encoder': nn.BatchNorm1d(layer_params['decoder_in']), 'sig1_encoder': nn.ReLU()})

        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(layer_params['decoder_in'], layer_params['decoder_out']),
                                      'norm1_decoder': nn.BatchNorm1d(layer_params['decoder_out']), 'sig2_decoder': nn.ReLU()})

        classifier_config = OrderedDict({'linear1_classifier': nn.Linear(layer_params['classifier_in'], layer_params['classifier_out']), 'sig1_classifier': nn.Sigmoid()})
    else:
        raise Exception('Incorrect model type: use \'conv\' or \'nn\'')

    return encoder_config, decoder_config, classifier_config

# Get json configuration as dict
def get_configuration(fname):
    
    with open(fname, 'r') as f:
        config = json.load(f)
    return config['sim'], config['exp_name'], config['data_dir'], config['save_dir'], config['mode'], config['model'], config['dataset']['size']

# Function to get stimuli from hard drive
def get_stimuli(path):
    
    categories = np.load(path)
    catA = categories['a']
    catB = categories['b']
    stimuli = torch.as_tensor(np.concatenate((catA, catB)), dtype=torch.float32)
    stimuli = nn.functional.normalize(stimuli)

    return stimuli

def create_dirstruct(save_dir, exp_name):

    # Path
    exp_save_dir = os.path.join(save_dir, exp_name)

    # Verify directory existence
    dir_exists(exp_save_dir)

    return exp_save_dir


def dir_exists(*args):

    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def get_dataset_info(path, mode='binary'):
    
    
    split_path = os.path.split(path)
    
    if mode == 'binary' or mode == 'binary_custom':
        setnum = split_path[1]
        setnum = setnum.split('.')[0]
        setnum = setnum.split('_')[1]
    
        catcode = split_path[0]
        catcode = os.path.split(catcode)[1]
    
        catcode = catcode + '-' + setnum
    elif mode == 'cont':
        catcode = split_path[1].split('_')[1]
    else: 
        raise Exception('Incorrect mode. Use mode=binary or mode=binary_custom or mode=cont')
    return catcode

def get_labels(categoryA_size, categorynA_size):

    labels = [np.array([1, 0], dtype=float) if x < categoryA_size else np.array([0, 1], dtype=float) for x in
              range(categoryA_size + categorynA_size)]
    return labels

def load_datasets(stimuli, labels, AE_batch_size, AE_epochs, class_batch_size, inplace_noise=False, train_ratio=0.7, noise_factor=0.05):

    # Get training and test set sizes
    train_size = int(train_ratio * len(stimuli))
    test_size = len(stimuli) - train_size

    # Split
    train, test = utils.data.random_split(stimuli, [train_size, test_size])
    train_idx = train.indices
    test_idx = test.indices

    # Get dataloaders classifier
    test_loader_cl = utils.data.DataLoader(dataset=utils.data.TensorDataset(stimuli[test_idx], labels[test_idx]), batch_size=class_batch_size)
    train_loader_cl = utils.data.DataLoader(dataset=utils.data.TensorDataset(stimuli[train_idx], labels[train_idx]), batch_size=class_batch_size)

    # Get dataloaders AE
    train_loaders_AE = []
    if inplace_noise:
        for e in range(AE_epochs):
            corrupt_stimuli = add_noise(tensors=torch.clone(stimuli[train_idx]), noise_factor=noise_factor)
            dataset = utils.data.TensorDataset(corrupt_stimuli, stimuli[train_idx])
            train_loader = utils.data.DataLoader(dataset=dataset, batch_size=AE_batch_size )
            train_loaders_AE.append(train_loader)
    else:
        train_loaders_AE.append(utils.data.TensorDataset(stimuli[train_idx], stimuli[train_idx]))
    test_loader_AE = utils.data.DataLoader(dataset=utils.data.TensorDataset(stimuli[test_idx], stimuli[test_idx]), batch_size=AE_batch_size)

    return train_loaders_AE, test_loader_AE, train_loader_cl, test_loader_cl, train_idx, test_idx

def add_noise(tensors, noise_factor=0.05):
    noise = torch.randn(tensors.size())
    corrupt_tensors = tensors + (noise_factor*noise)

    return corrupt_tensors

# Function that runs a simulation
def sim_run(sim_num, cat_code, encoder_config, decoder_config, classifier_config, encoder_out_name,
            stimuli, labels, train_ratio, AE_epochs, AE_batch_size, noise_factor, AE_lr, AE_wd, AE_patience, AE_thresh, class_epochs,
            class_batch_size, class_lr, class_wd, class_monitor, class_thresh, training, inplace_noise, save_path, save_model=True, metric='euclid', verbose=False):

    path = os.path.join(save_path, 'sim_' + str(sim_num))

    # Setup directory structure
    if not os.path.exists(path):
        os.makedirs(path)
        os.mkdir(os.path.join(path, 'plots'))
        os.mkdir(os.path.join(path, 'cp'))
        if save_model:
            os.mkdir(os.path.join(path, 'model_checkpoints'))

    # Setup neural net
    neuralnet = NN.Net(encoder_config=encoder_config, decoder_config=decoder_config,
                       classifier_config=classifier_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neuralnet = neuralnet.to(device)
    
    # Get dataloaders
    train_loaders_AE, test_loader_AE, train_loader_cl, test_loader_cl, train_idx, test_idx = load_datasets(stimuli=stimuli, labels=labels, AE_batch_size=AE_batch_size, 
                                                                                                           AE_epochs=AE_epochs, class_batch_size=class_batch_size, 
                                                                                                           inplace_noise=inplace_noise, train_ratio=train_ratio, 
                                                                                                           noise_factor=noise_factor)

    # Compute initial inner representations
    #TODO: change compute CP inputs, add categoricality computations using Kolmogorov-Smirnov
    initial = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True, metric=metric)
    np.savez_compressed(os.path.join(path, 'cp', 'cp_initial'), between=initial[0], withinA=initial[1], withinB=initial[2], inner=initial[3])
    if save_model:
        torch.save({'state_dict': neuralnet.state_dict()}, os.path.join(path, 'model_checkpoints', 'init_checkpoint.pth'))

    # Freeze classifier
    neuralnet.freeze(neuralnet.classifier)

    # Train autoencoder
    optimizer = torch.optim.Adam(
        [{'params': neuralnet.encoder.parameters()}, {'params': neuralnet.decoder.parameters()}], lr=AE_lr,
        weight_decay=AE_wd)
    scheduler = ReduceLROnPlateau(optimizer, patience=4)
    criterion = nn.MSELoss()
    running_loss_AE, test_loss_AE  = neuralnet.train_autoencoder(num_epochs=AE_epochs, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                                                 train_loaders=train_loaders_AE, test_loader=test_loader_AE, training=training, 
                                                                 patience=AE_patience, thresh=AE_thresh, verbose=verbose)

    # Delete temporary model (this should be moved to training class once implemented)
    if training == 'early_stop':
        if os.path.exists('./.best_model.pth'):
            os.remove('./.best_model.pth')
    
    # Plot AE training data
    plt.figure(1)
    plt.plot(running_loss_AE)
    plt.title('Autoencoder Running Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(path, 'plots', 'ae_rloss.png'))
    plt.close(1)

    plt.figure(2)
    plt.plot(test_loss_AE)
    plt.title('Autoencoder Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(path, 'plots','ae_tloss.png'))
    plt.close(2)

    # Freeze autoencoder
    neuralnet.freeze(neuralnet.decoder)

    # Compute CP and save
    before = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True, metric=metric)
    np.savez_compressed(os.path.join(path, 'cp', 'cp_before'), between=before[0], withinA=before[1], withinB=before[2], inner=before[3])
    if save_model:
        torch.save({'model_state_dict': neuralnet.state_dict()}, os.path.join(path, 'model_checkpoints', 'AE_checkpoint.pth')) 
    
    # Thaw classifier
    neuralnet.unfreeze(neuralnet.classifier)

    # Train classifier
    optimizer = torch.optim.Adam([{'params': neuralnet.classifier.parameters()}, {'params': neuralnet.encoder.parameters()}],
                                lr=class_lr, weight_decay=class_wd)
    scheduler = ReduceLROnPlateau(optimizer, patience=0)
    criterion = nn.MSELoss()
    running_loss, train_accuracy, test_loss, test_accuracy = neuralnet.train_classifier(num_epochs=class_epochs, optimizer=optimizer, criterion=criterion, 
                                                                                        scheduler=scheduler, train_loader=train_loader_cl, test_loader=test_loader_cl,
                                                                                        training=training, monitor=class_monitor, threshold=class_thresh, verbose=verbose)
    
    # Delete temporary model
    if training == 'early_stop':
        if os.path.exists('./.best_model.pth'):
            os.remove('./.best_model.pth')

    # Plot classifier training data
    plt.figure(3)
    plt.plot(running_loss)
    plt.title('Running Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path, 'plots', 'class_rloss.png'))
    plt.close(3)

    plt.figure(4)
    plt.plot(train_accuracy)
    plt.title('Training Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(path, 'plots','class_tr-acc.png'))
    plt.close(4)

    plt.figure(5)
    plt.plot(test_loss)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path, 'plots', 'class_tloss.png'))
    plt.close(5)

    plt.figure(6)
    plt.plot(test_accuracy)
    plt.title('Test Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path, 'plots', 'class_t-acc.png'))
    plt.close(6)

    # Compute CP and save
    after = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True, metric=metric)
    np.savez_compressed(os.path.join(path, 'cp','cp_after'), between=after[0], withinA=after[1], withinB=after[2], inner=after[3])
    if save_model:
        torch.save({'model_state_dict': neuralnet.state_dict()}, os.path.join(path, 'model_checkpoints', 'class_checkpoint.pth')) 

    # Stack autoencoder and classifier training and testing data
    ae_data = np.stack([running_loss_AE, test_loss_AE])
    class_data = np.stack([running_loss, train_accuracy, test_loss, test_accuracy])

    code = []
    code.append(cat_code)

    # Save data
    np.savez_compressed(os.path.join(path, 'sim_' + str(sim_num)), ae=ae_data, classifier=class_data, code=code)

def main(**kwargs):
    
    # Load configuration and other parameters
    if 'config_fname' in kwargs:
        config_fname = kwargs['config_fname']
    else:
        config_fname = parse_args()
    config, exp_name, data_dir, save_dir, mode, model, size = get_configuration(config_fname)
    
    # Config
    sim_params = config['sim_params']
    layer_params = config['layer_params']
    

    # Get save path and setup directory structure
    save_path = create_dirstruct(save_dir, exp_name) 
    sim_params['save_path'] = save_path
    
    # Dataset paths
    path_cat = os.path.join(data_dir, 'categories')
    category_paths = glob.glob(os.path.join(path_cat, '*'))
    datasets = []
    for path in category_paths:
        d = glob.glob(os.path.join(path, '*'))
        datasets += d
    
    # Setup timer
    timer = []
    total = len(datasets)
    print('Total number of simulations left to run: {}'.format(total))

    for j, p in enumerate(datasets):
        
        # Timer
        s = time.time()
        
        # Get labels
        labels = torch.as_tensor(get_labels(size, size), dtype=torch.float32)
        
        # Get stimuli
        stimuli = get_stimuli(p)

        # Neural net layer config
        encoder_config, decoder_config, classifier_config = get_model_arch(model, layer_params)
                         
        sim_params['encoder_config'] = encoder_config
        sim_params['decoder_config'] = decoder_config
        sim_params['classifier_config'] = classifier_config

        # Simulation specific info
        catcode = get_dataset_info(p, mode)
        sim_params['sim_num'] = j
        sim_params['cat_code'] = catcode 
        sim_params['stimuli'] = stimuli
        sim_params['labels'] = labels
            
        # Run simulation
        sim_run(**sim_params)

        # End of simulation
        total -= 1
        e = time.time()
        timer.append(e-s)

        if j < 5:
            mean = np.mean(timer)
            print('Average run time: {}'.format(mean))
        else:
            if j % 25 == 0:
                mean = np.mean(timer)
                print('Average run time: {}'.format(mean))

        time_left = mean*total/3600
        print('Number of simulations to run: {} \t Estimated time left: {}h'.format(total, time_left))

if __name__ == "__main__":
    main()
