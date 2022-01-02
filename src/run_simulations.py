# Imports
import os
import src.neuralnet as NN
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import time
import argparse
plt.ioff()

# Get simulation configuration file
def parse_args():

    parser = argparse.ArgumentParser(description='Run simulation experiments')
    parser.add_argument('-c', help='<config.json>', required=True)
    args = parser.parse_args()
    return args.config

def get_configuration(fname):

    with open(fname, 'r') as f:
        config = json.load(f)
    return config['sim']

# Function to get stimuli from hard drive
def get_stimuli(cat_path):
    
    categories = np.load(cat_path)
    catA = categories['a']
    catB = categories['b']
    stimuli = torch.as_tensor(np.concatenate((catA, catB)), dtype=torch.float32)
    stimuli = nn.functional.normalize(stimuli)

    return stimuli

def labels(categoryA_size, categorynA_size):

    labels = [np.array([1, 0], dtype=float) if x < categoryA_size else np.array([0, 1], dtype=float) for x in
              range(categoryA_size + categorynA_size)]
    return labels

# Function that runs a simulation
def sim_run(sim_num, cat_code, encoder_config, decoder_config, classifier_config, encoder_out_name,
            stimuli, labels, train_ratio, AE_epochs, AE_batch_size, noise_factor, AE_lr, AE_wd, class_epochs,
            class_batch_size, class_lr, class_wd, inplace_noise, save_path, verbose=False):

    path = os.path.join(save_path, 'sim_' + str(sim_num))

    try:
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'plots'))
        os.mkdir(os.path.join(path, 'cp'))
    except:
        pass

    # Setup neural net
    neuralnet = NN.Net(encoder_config=encoder_config, decoder_config=decoder_config,
                       classifier_config=classifier_config)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    neuralnet = neuralnet.to(device)

    # Compute initial inner representations
    initial = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True)
    np.savez_compressed(os.path.join(path, 'cp', 'cp_initial'), between=initial[0], withinA=initial[1], withinB=initial[2], weights=initial[3], inner=initial[4])

    # Freeze classifier
    neuralnet.freeze(neuralnet.classifier)

    # Train autoencoder
    optimizer = torch.optim.Adam(
        [{'params': neuralnet.encoder.parameters()}, {'params': neuralnet.decoder.parameters()}], lr=AE_lr,
        weight_decay=AE_wd)
    scheduler = ReduceLROnPlateau(optimizer, patience=0)
    criterion = nn.MSELoss()
    running_loss_AE, test_loss_AE = neuralnet.train_autoencoder(AE_epochs, stimuli, AE_batch_size, noise_factor,
                                                                optimizer, criterion, scheduler, inplace_noise,
                                                                verbose=verbose)

    # Plot AE training data
    plt.figure(1)
    plt.plot(running_loss_AE)
    plt.title('Autoencoder Running Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.paht.join(path, 'plots', 'ae_rloss.png'))
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
    before = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True)
    np.savez_compressed(os.path.join(path, 'cp', 'cp_before'), between=before[0], withinA=before[1], withinB=before[2], weights=before[3], inner=before[4])

    # Thaw classifier
    neuralnet.unfreeze(neuralnet.classifier)

    # Train classifier
    optimizer = torch.optim.Adam([{'params': neuralnet.classifier.parameters()}, {'params': neuralnet.encoder.parameters()}],
                                lr=class_lr, weight_decay=class_wd)
    scheduler = ReduceLROnPlateau(optimizer, patience=0)
    criterion = nn.MSELoss()
    running_loss, train_accuracy, test_loss, test_accuracy = neuralnet.train_classifier(class_epochs, train_ratio,
                                                                                        stimuli, labels,
                                                                                        class_batch_size, optimizer,
                                                                                        criterion, scheduler, verbose)

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
    after = neuralnet.compute_cp(stimuli=stimuli, layer_name=encoder_out_name, inner=True)
    np.savez_compressed(os.path.join(path, 'cp','cp_after'), between=after[0], withinA=after[1], withinB=after[2], weights=after[3], inner=after[4])

    # Stack autoencoder and classifier training and testing data
    ae_data = np.stack([running_loss_AE, test_loss_AE])
    class_data = np.stack([running_loss, train_accuracy, test_loss, test_accuracy])

    code = []
    code.append(cat_code)

    # Save data
    np.savez_compressed(os.path.join(path, 'sim' + str(sim_num)), ae=ae_data, classifier=class_data, code=code)

def main():
    
    config_fname = parse_args()
    
    config = get_configuration(config_fname)
    









    category_paths = glob.glob(os.path.join(path_cat, '*'))

    timer = []

    total = len(category_paths)
    print('Total number of simulations left to run: {}'.format(total))

    num_list = np.arange(simruns_done+1, simruns_done+1+len(category_paths))
    
    for j, c_dir in enumerate(category_paths):
        
        ### Simulation specific information to add:
        #### simnum, cat-code, stimuli, labels, savepath











        s = time.time()

        # Set labels
        labels = torch.as_tensor(labels(2048, 2048), dtype=torch.float32)

        # Neural net layer config
        encoder_config = OrderedDict({'lin1_encoder': nn.Linear(256, 128), 'norm1_encoder': nn.BatchNorm1d(128),
                                      'sig1_encoder': nn.ReLU()})
        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(128, 256), 'norm1_decoder': nn.BatchNorm1d(256),
                                      'sig2_decoder': nn.ReLU()})
        classifier_config = OrderedDict({'lin1_classifier': nn.Linear(128, 2), 'sig1_classifier': nn.Sigmoid()})

        # Get paths in category directory
        paths = glob.glob(os.path.join(c_dir, '*'))


        # Only choose sets where s = 0
        if len(paths) < 32:
            p = np.random.choice(paths, 1)[0]
        else:
            p = np.random.choice(paths[:32], 1)[0]

        # Get stimuli
        load_from = os.path.join(p)
        stimuli = get_stimuli(load_from)

        setnum = os.path.split(p)[1]
        setnum = setnum.split('.')[0]
        setnum = setnum.split('_')[1]

        # Set sim parameters
        sim_params = OrderedDict(
            {'sim_num': num_list[j], 'cat_code': c_dir[-11:] + '-' + setnum, 'encoder_config': encoder_config, 'decoder_config': decoder_config,
             'classifier_config': classifier_config, 'encoder_out_name': 'lin1_encoder', 'stimuli': stimuli,
             'labels': labels, 'train_ratio': 0.8, 'AE_epochs': 15, 'AE_batch_size': 8, 'noise_factor': 0.1, 'AE_lr': 10e-5,
             'AE_wd': 10e-5, 'class_epochs': 15, 'class_batch_size': 8, 'class_lr': 10e-2, 'class_wd': 10e-3,
             'inplace_noise': True, 'save_path': save_path, 'verbose': False})

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
            if j % 100 == 0:
                mean = np.mean(timer)
                print('Average run time: {}'.format(mean))

        time_left = mean*total/3600
        print('Number of simulations to run: {} \t Estimated time left: {}h'.format(total, time_left))

if __name__ == "__main__":
    main()
