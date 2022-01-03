__author__ = 'Solim LeGris'

# Imports
import torch
import torch.nn as nn
from torch import utils
import numpy as np

class Net(nn.Module):

    def __init__(self, encoder_config, decoder_config, classifier_config):
        
        """
        :param encoder_config: an ordered dictionary containing layer configuration for the encoder
        :param decoder_config: an ordered dictionary containing layer configuration for the decoder
        :param classifier_config: an ordered dictionary containing layer configuration for the classifier
        """
        super(Net, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(encoder_config)

        # Decoder
        self.decoder = nn.Sequential(decoder_config)

        # Classifier
        self.classifier = nn.Sequential(classifier_config)

    def forward(self, x, decode=False, classifier=False):
        """

        :param x: data
        :param decode: Flag for decoder forward pass
        :param classifier: Flag for classifier forward pass
        :return: output of forward pass
        """
        x = self.encoder(x)
        if decode:
            x = self.decoder(x)
        if classifier:
            x = self.classifier(x)

        return x

    def train_autoencoder(self, num_epochs, stimuli, batch_size, noise_factor, optimizer, criterion, scheduler,
                          inplace_noise=False, verbose=False):
        """
        This function trains the autoencoder portion of the neural net model

        :param num_epochs: number of epochs to train for
        :param stimuli:
        :param batch_size:
        :param noise_factor:
        :param verbose: Print training and testing information to screen
        :param conv:
        :param scheduler:
        :param self: an instantiation of a neural net
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :return: three arrays containing the training loss per batch, the training loss per epoch
        and the evaluation loss
        """

        running_loss = []
        test_loss = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load test dataset
        testloader = load_dataset_AE(stimuli=torch.clone(stimuli), batch_size=batch_size, noise_factor=0.0)
        train_loaders = []
        if inplace_noise:
            for t in range(num_epochs):
                train_loaders.append(load_dataset_AE(stimuli=torch.clone(stimuli), batch_size=batch_size,
                                                     noise_factor=noise_factor))
        else:
            train_loaders.append(load_dataset_AE(stimuli=torch.clone(stimuli), batch_size=batch_size,
                                                     noise_factor=noise_factor))
        
        for epoch in range(num_epochs):

            cur_running_loss = 0.0

            # Load training dataset with new noise or same
            if inplace_noise:
                trainloader = train_loaders[epoch]
            else:
                trainloader = train_loaders[0]

            for i, (stimuli, target) in enumerate(trainloader):

                stimuli = stimuli.to(device)
                target = target.to(device)

                # Train mode
                self.train()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                x = self.forward(stimuli, decode=True)
                loss = criterion(x, target)

                # Backward pass
                loss.backward()
                optimizer.step()


                # Add loss of current pass
                cur_running_loss += loss.detach().data.cpu().numpy()

            # Save running loss
            cur_running_loss /= len(trainloader)
            running_loss.append(cur_running_loss)

            # Eval loss
            eval_loss = self.evaluate_AE(testloader, criterion)
            test_loss.append(eval_loss)

            if verbose:
                print('Autoencoder, epoch {} --> Running Loss: {} \t Eval loss: {}'.format(epoch, cur_running_loss, eval_loss))

            # Scheduler
            scheduler.step(eval_loss)

        return running_loss, test_loss

    def train_classifier(self, num_epochs, train_ratio, stimuli, labels, batch_size, optimizer, criterion, scheduler,
                         verbose=False):
        """
        Train the classifier portion of the neural net.

        :param num_epochs: number of epochs to train the classifier for
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :param train_ratio: portion of the dataset to be used for training versus testing
        :param stimuli: dataset
        :param labels: labels for dataset
        :param batch_size:
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :param scheduler: monitor loss progression per epoch with patience parameter
        :param verbose: print training and testing information to screen
        :return: four arrays containing training and testing information
        """

        # Data to save
        running_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []

        # Determine device to use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load datasets
        trainloader, testloader = load_dataset_class(train_ratio=train_ratio, stimuli=stimuli, labels=labels,
                                                           batch_size=batch_size)
        for epoch in range(num_epochs):

            # Save running loss
            cur_loss = 0.0

            for i, (stimuli, labels) in enumerate(trainloader):

                # Pass training data to device
                stimuli = stimuli.to(device)
                labels = labels.to(device)

                # Train mode
                self.train()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                x = self.forward(stimuli, classifier=True)
                loss = criterion(x, labels)

                # Backward pass
                loss.backward(retain_graph=True)
                optimizer.step()

                # Save and add loss of current pass
                cur_loss += loss.item()

            # Running loss
            cur_loss /= len(trainloader)
            running_loss.append(cur_loss)

            # Train accuracy
            _, train_acc = self.evaluate_classifier(trainloader, criterion)
            train_accuracy.append(train_acc)

            # Eval loss and accuracy
            eval_loss, eval_acc = self.evaluate_classifier(testloader, criterion)
            test_loss.append(eval_loss)
            test_accuracy.append(eval_acc)

            if verbose:
                print('Classifier, epoch {} --> Test loss: {} \t Test accuracy: {}'.format(epoch, eval_loss, eval_acc))

            # Scheduler
            scheduler.step(eval_loss)

        return running_loss, train_accuracy, test_loss, test_accuracy

    @torch.no_grad()
    def evaluate_AE(self, dataloader, criterion):
        """
        Function that evaluates the performance of the autoencoder

        :param dataloader:
        :param criterion:
        :return:
        """
        # Device to use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set model to eval mode
        self.eval()

        # Save loss
        total_loss = 0.0

        # for i, (stimuli, target) in enumerate(dataloader):
        for i in range(len(dataloader)):
            stimuli, target = dataloader.__iter__().__next__()

            if torch.cuda.is_available():
                stimuli = stimuli.to(device)
                target = target.to(device)

            pred = self.forward(stimuli, decode=True)
            test_loss = criterion(pred, target)
            total_loss += test_loss.item()
        total_loss /= len(dataloader)

        return total_loss

    @torch.no_grad()
    def evaluate_classifier(self, dataloader, criterion):
        """
        Function that evaluates the performance of the classifier

        :param dataloader: dataset on which to evaluate the classifier
        :param criterion:
        :return:
        """
        # Device to use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set model to eval mode
        self.eval()

        # Save data
        correct = 0
        total = 0
        total_loss = 0.0

        for i, (x, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)

            y_pred = self.forward(x, classifier=True)
            test_loss = criterion(y_pred, y)
            total_loss += test_loss
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            _, target = torch.max(y, 1)
            correct += torch.where(predicted == target)[0].shape[0]

        total_loss /= len(dataloader)
        accuracy = 100 * correct / total

        return total_loss.cpu(), accuracy

    def freeze(self, network):
        """
        Freeze layers in network to prevent training parameters

        :param network: a list of parameters to freeze
        """
        for p in network.parameters():
            p.requires_grad = False

    def unfreeze(self, network):
        """
        Unfreeze layers in network to prevent training parameters

        :param network: a list of parameters to unfreeze
        :return: unfrozen layers as a list of parameters
        """

        for i, p in enumerate(network.parameters()):
            p.requires_grad = True

    @torch.no_grad()
    def compute_cp(self, stimuli, layer_name, save=True, inner=False):

        # Set model to eval mode
        self.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Send stimuli to device
        stimuli = stimuli.to(device)

        # Get weight matrix
        state_dict = self.encoder.state_dict()
        W = torch.clone(state_dict[layer_name+'.weight']).to(device)
        W = W/torch.norm(W, dim=None)

        # Compute inner representations for A and B
        in_rep = torch.matmul(W, stimuli.T).T

        # Get index that separates categories
        half_index = int(in_rep.shape[0]/2)

        # Compute distances between and within
        between = torch.cdist(in_rep[:half_index], in_rep[half_index:], p=2)
        withinA = torch.cdist(in_rep[:half_index], in_rep[:half_index], p=2)
        withinB = torch.cdist(in_rep[half_index:], in_rep[half_index:], p=2)

        # Create return array
        return_arr = []

        # Compute mean by removing the diagonal values for within since we don't care about it
        withinA = withinA.sum()/(withinA.numel() - withinA.size()[0])
        withinB = withinB.sum()/(withinB.numel() - withinB.size()[0])
        return_arr.append(between.mean().cpu().item())
        return_arr.append(withinA.cpu().item())
        return_arr.append(withinB.cpu().item())

        if save:
            return_arr.append(W.cpu().numpy())

        if inner:
            return_arr.append(in_rep.cpu().numpy())

        return return_arr

def load_dataset_AE(stimuli, batch_size, noise_factor=0.05):

    # Training set for autoencoder
    if noise_factor > 0.0:
        corrupt_stimuli = add_noise(tensors=torch.clone(stimuli), noise_factor=noise_factor)
        dataset = utils.data.TensorDataset(corrupt_stimuli, stimuli)
    else:
        dataset = utils.data.TensorDataset(stimuli, stimuli)
    train_loader = utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    return train_loader

def load_dataset_class(train_ratio, stimuli, labels, batch_size):

    # Training and test sets for classifier
    dataset = utils.data.TensorDataset(stimuli, labels)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = utils.data.random_split(dataset, [train_size, test_size])
    train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def add_noise(tensors, noise_factor=0.05):
    noise = torch.randn(tensors.size())
    corrupt_tensors = tensors + (noise_factor*noise)

    return corrupt_tensors
