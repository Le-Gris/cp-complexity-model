__author__ = 'Solim LeGris'

# Imports
import torch
from torch import nn, utils
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import math

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

    def train_autoencoder(self, num_epochs, optimizer, criterion, scheduler, train_loaders, test_loader, numA, numB, rep_diff=False, verbose=False, 
                            eval_mode='epoch', eval_freq=5, training='fixed', patience=3, thresh=0.01, dataset=None):
        """
        This function trains the autoencoder portion of the neural net model

        :param self: an instantiation of a neural net
        :param num_epochs: number of epochs to train for or max number of epochs when in \'early_stop\' mode 
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :param train_loaders: list of dataloaders (either 1 or num_epochs)
        :param test_loader: dataloader for loss
        :param scheduler: learning rate scheduler
        :param verbose: Print training and testing information to stdout
        :param training: \'fixed\' or \'early_stop\'
        :param patience: number of epochs to wait before stopping training
        :param thresh: threshold value for which it is considered that change in loss is significant 
        :return: three arrays containing the training loss per batch, the training loss per epoch
        and the evaluation loss
        """
        
        # Initial setup
        running_loss = []
        test_loss = []
        batch_counter = 0
        rep_diffs = []
        trained = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log = ''
        
        # Training mode setup
        if training == 'early_stop':
            # Keep track of best validation loss
            best_loss = [math.inf, -1]
            patience_count = 0
        
        # get initial sdm score
        if rep_diff:
            w, b, sdm = self.prog_diff(dataset.clone(), numA, numB)
            rep_diffs.append([w,b,sdm])

        # Training 
        for epoch in range(num_epochs):
            log += f'Autoencoder epoch {epoch+1}\n'
            cur_running_loss = 0.0

            # Load training dataset with in-place noise or static noise
            if len(train_loaders) > 1:
                trainloader = train_loaders[epoch]
            else:
                trainloader = train_loaders[0]

            for i, (stimuli, target) in enumerate(trainloader):
                
                log += f'\tBatch {i+1}\n' 
                
                # Send inputs and labels to device
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

                if eval_mode == 'batch' and ((batch_counter+1)%eval_freq) == 0:
                    log+= f'\tBatch eval mode:\n'
                    
                    # Save running loss
                    cur_running_loss /= eval_freq
                    running_loss.append(cur_running_loss)

                    # Evaluate on test set and get eval loss
                    eval_loss = self.evaluate_AE(test_loader, criterion)
                    test_loss.append(eval_loss)

                    # Get representation difficulty score
                    if rep_diff:
                        w, b, sdm = self.prog_diff(dataset.clone(), numA, numB)
                        rep_diffs.append([w,b, sdm])

                    if training == 'early_stop':
                        log += f'\t - Current best loss: {best_loss[0]:.2f}, {best_loss[1]}\n'
                        log += f'\t - Current test loss: {eval_loss:.2f}\n'
                        if eval_loss < best_loss[0]:
                            # Verify change in loss against relative threshold
                            relative_criterion = thresh*best_loss[0]
                            if best_loss[0] - eval_loss < relative_criterion:
                                patience_count += 1
                            else:
                                patience_count = 0
                            best_loss = [eval_loss, len(test_loss)-1]
                            torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')
                        else:
                            patience_count += 1
                        log += f'\t - Patience count: {patience_count}\n'
                        if patience_count >= patience:
                            # End training
                            trained = True
                            break
                batch_counter += 1
                
            if trained:
                break
                                    
            if eval_mode == 'epoch':
                # Save running loss
                cur_running_loss /= len(trainloader)
                running_loss.append(cur_running_loss)

                # Eval loss
                eval_loss = self.evaluate_AE(test_loader, criterion, test_dropout)
                test_loss.append(eval_loss)
                
                # Get representation difficulty score
                if rep_diff:
                    sdm = self.prog_diff(dataset.clone(), numA, numB)
                    rep_diffs.append(sdm)
                
                if training == 'early_stop':
                    if eval_loss < best_loss[0]:
                        # Verify change in loss against relative threshold
                        relative_criterion = thresh*best_loss[0]
                        if best_loss[0] - eval_loss < relative_criterion:
                            patience_count +=1
                        else:
                            patience_count = 0
                        best_loss = [eval_loss, epoch]
                        torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')
                    else:
                        patience_count += 1

                    if patience_count >= patience:
                        # End training
                        break

                if verbose:
                    print('Autoencoder, epoch {} --> Running Loss: {} \t Eval loss: {}'.format(epoch, cur_running_loss, eval_loss))
            # Scheduler
            scheduler.step(eval_loss)
        
        if eval_loss != best_loss[0]:
            # Load best model
            best_model = torch.load('./.best_model.pth')
            self.load_state_dict(best_model['model_state_dict'])

        # Return data up until best model
        full_test_loss = [l for l in test_loss]
        running_loss = running_loss[:best_loss[1]+1] 
        test_loss = test_loss[:best_loss[1]+1]

        return running_loss, test_loss, full_test_loss, rep_diffs, log

    def train_classifier(self, num_epochs, optimizer, criterion, scheduler, train_loader, test_loader, training, monitor, threshold, numA, numB, rep_diff=False, 
                        eval_mode='epoch', eval_freq=5, patience=4, verbose=False, dataset=None):
        """
        Train the classifier portion of the neural net.

        :param num_epochs: number of epochs to train the classifier for, max_value when in 'early_stop mode'
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :param labels: labels for dataset
        :param batch_size: size of batch for training
        :param optimizer: the optimizer to use to train the neural net
        :param criterion: the criterion to use for the loss
        :param scheduler: monitor loss progression per epoch with patience parameter
        :param training: training type \'fixed\' or \'early_stop\'
        :param monitor: stop training depeding on \'loss\' or \'acc\' 
        :param threshold: value of threshold at which to stop training
        :param verbose: print training and testing information to stdout
        :return: four arrays containing training and testing information
        """

        # Data to save
        running_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        rep_diffs = []
        trained = False
        batch_counter = 0
        log = ''
        
        # Determine device to use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Monitor best model in case threshold is never reached
        if training == 'early_stop':
            if monitor == 'loss':
                best = [math.inf, -1]
                patience_count = 0
            elif monitor == 'acc':
                best = [0, -1]
        
        # get initial sdm score
        if rep_diff:
            w, b, sdm = self.prog_diff(dataset.clone(), numA, numB)
            rep_diffs.append([w,b,sdm])

        for epoch in range(num_epochs):
            
            log += f'Classifier epoch {epoch+1}\n'
            
            # Save running loss
            cur_loss = 0.0

            for i, (stimuli, labels) in enumerate(train_loader):
                
                log += f'\tBatch {i+1}\n'
                
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
                cur_loss += loss.detach().data.cpu().numpy()
            
                if eval_mode == 'batch' and ((batch_counter+1)%eval_freq) == 0:
                    
                    log+= f'\tBatch eval mode:\n'

                    # Running loss
                    cur_loss /= eval_freq
                    running_loss.append(cur_loss)

                    # Train accuracy
                    _, train_acc = self.evaluate_classifier(train_loader, criterion)
                    train_accuracy.append(train_acc)

                    # Eval loss and accuracy
                    eval_loss, eval_acc = self.evaluate_classifier(test_loader, criterion)
                    test_loss.append(eval_loss)
                    test_accuracy.append(eval_acc)
                    

                    # Get representation difficulty score
                    if rep_diff:
                        w, b, sdm = self.prog_diff(dataset.clone(), numA, numB)
                        rep_diffs.append([w, b, sdm])
                    
                    # Monitor
                    if training == 'early_stop':
                        log += f'\t - Current best {monitor}: {best[0]:.2f}, {best[1]}\n'
                        log += f'\t - Current test loss: {eval_loss:.2f}\n'
                        if monitor == 'loss':
                            if eval_loss <= best[0]:
                                # Look at relative difference, not absolute
                                relative_criterion = threshold*best[0]
                                if best[0] - eval_loss < relative_criterion:
                                    patience_count += 1
                                else:
                                    patience_count = 0
                                best = [eval_loss, len(test_loss) -1]
                                torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')
                            else:
                                # Loss is larger than previous best
                                patience_count += 1
                            log += f'\t - Patience count: {patience_count}\n'            
                            if patience_count >= patience:
                                trained = True
                                break
                                
                        elif monitor == 'acc':
                            if eval_acc >= threshold:
                                trained = True
                                break
                            elif eval_acc > best[0]:
                                best = [eval_acc, len(test_loss) -1]
                                torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')
                
                batch_counter += 1
                
            if trained:
                break

            if eval_mode == 'epoch':
                # Running loss
                cur_loss /= len(train_loader)
                running_loss.append(cur_loss)

                # Train accuracy
                _, train_acc = self.evaluate_classifier(train_loader, criterion)
                train_accuracy.append(train_acc)

                # Eval loss and accuracy
                eval_loss, eval_acc = self.evaluate_classifier(test_loader, criterion)
                test_loss.append(eval_loss)
                test_accuracy.append(eval_acc)
                
                # Get representation difficulty
                if rep_diff:
                    sdm = self.prog_diff(dataset.clone(), numA, numB)
                    rep_diffs.append(sdm)

                if verbose:
                    print('Classifier, epoch {} --> Test loss: {} \t Test accuracy: {}'.format(epoch, eval_loss, eval_acc))
        
                # Monitor 
                if training == 'early_stop':
                    if monitor == 'loss':
                        if eval_loss <= best[0]:
                            # Look at relative difference, not absolute
                            relative_criterion = threshold*best[0]
                            if best[0] - eval_loss < relative_criterion:
                                patience_count += 1
                            else:
                                patience_count = 0
                            best = [eval_loss, epoch]
                            torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')
                        else:
                            # Loss is larger than previous best
                            patience_count += 1
                        if patience_count >= patience:
                            break
                    elif monitor == 'acc':
                        if eval_acc >= threshold:
                            break
                        elif eval_acc > best[0]:
                            best = [eval_acc, epoch]
                            torch.save({'model_state_dict': self.state_dict()}, './.best_model.pth')

            # Scheduler
            scheduler.step(eval_loss)
        
        if eval_loss != best[0] or eval_acc != best[0]:
            # Load best model
            best_model = torch.load('./.best_model.pth')
            self.load_state_dict(best_model['model_state_dict'])
            
        # Return data up until best model
        test_loss_full = [x for x in test_loss]
        running_loss = running_loss[:best[1]+1]
        train_accuracy = train_accuracy[:best[1]+1]
        test_loss = test_loss[:best[1]+1]
        test_accuracy = test_accuracy[:best[1]+1]

        return running_loss, train_accuracy, test_loss, test_accuracy, test_loss_full, rep_diffs, log

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
    def compute_cp(self, stimuli, inner=False, metric='euclid', rep_type='act'):

        # Set model to eval mode
        self.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Send stimuli to device
        stimuli = stimuli.to(device)

        if rep_type == 'act':
            in_rep = self.forward(stimuli)
        elif rep_type == 'lin':
            # Index of last layer
            top_layer_idx = len(self.encoder)-1
             
            # Iterate through all layers except last one
            in_rep = stimuli.clone().detach()
            for k, layer in enumerate(self.encoder.children()):
                if k == top_layer_idx:
                    break

                in_rep = layer(in_rep)


        # Get index that separates categories
        half_index = int(in_rep.shape[0]/2)

        # Compute pairwise distances between and within
        if metric == 'euclid':
            in_rep = F.normalize(in_rep)            
            between = torch.cdist(in_rep[:half_index], in_rep[half_index:], p=2)
            withinA = torch.cdist(in_rep[:half_index], in_rep[:half_index], p=2)
            withinB = torch.cdist(in_rep[half_index:], in_rep[half_index:], p=2)

            # Send to numpy
            between = between.numpy()
            withinA = withinA.numpy()
            withinB = withinB.numpy()
        elif metric == 'cosine':
            # Need pairwise distance calculation
            if torch.cuda.is_available():
                in_rep = in_rep.cpu()
            between = distance.cdist(in_rep[:half_index], in_rep[half_index:], metric=metric)
            withinA = distance.cdist(in_rep[:half_index], in_rep[:half_index], metric=metric)
            withinB = distance.cdist(in_rep[half_index:], in_rep[half_index:], metric=metric)
            # Remove self-similarity values    
            np.fill_diagonal(withinA, 0)
            np.fill_diagonal(withinB, 0)
        else:
            raise Exception('Invalid distance metric: use \'euclid\' or \'cosine\'')
        
        # Return array init
        return_arr = []
        
        # Compute mean by removing the diagonal values for within since we don't care about it
        withinA = withinA.sum()/(withinA.size - withinA.shape[0])
        withinB = withinB.sum()/(withinB.size - withinB.shape[0])
        between = np.mean(between) 
        
        return_arr.append(between)
        return_arr.append(withinA)
        return_arr.append(withinB)

        if inner:
            return_arr.append(in_rep.cpu().numpy())
        else:
            return_arr.append(None)

        return return_arr

    @torch.no_grad()
    def prog_diff(self, stimuli, numA, numB):
        
        # Eval mode and device setup
        self.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Send stimuli to device
        stimuli = stimuli.to(device)
        
        # Get inner representation
        in_rep = self.forward(stimuli)
        
        # Need pairwise cosine similarity calculation
        if torch.cuda.is_available():
                in_rep = in_rep.cpu()
        
        between = distance.cdist(in_rep[:numA], in_rep[numA:], metric='cosine')
        withinA = distance.cdist(in_rep[:numA], in_rep[:numA], metric='cosine')
        withinB = distance.cdist(in_rep[numA:], in_rep[numA:], metric='cosine')

        # Remove self-similarity/distance values
        np.fill_diagonal(withinA, 0)
        np.fill_diagonal(withinB, 0)
        
        # Compute SDM
        assert withinA.shape[0] == withinA.shape[1]
        assert withinB.shape[0] == withinB.shape[1]
        avg_w = (np.sum(withinA)/(withinA.size-withinA.shape[0]))*numA/(numA+numB) + (np.sum(withinB)/(withinB.size-withinB.shape[0]))*numB/(numA+numB)
        avg_b = np.mean(between)
        sdm = avg_w/avg_b

        return avg_w, avg_b, sdm

    def sample(self, stimuli, sample_dropout, n_samples):
        #TODO: complete sampling procedure
        """
        Sample representation space using dropout
        """
        #TODO: consider using sample_dropout list
        
        # Eval mode
        self.eval()

        # Device setup
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Send stimuli to device
        stimuli = stimuli.to(device)

        # Set Dropout "layers" to train and p to sample_dropout
        for m in self.encoder.children():
            pass
        sample = []
        return sample

