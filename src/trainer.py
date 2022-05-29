import torch
from torch import nn, optim
import math

class trainer:
    """
    This class implements a training regime for neural networks. It also allows for the evaluation
    of the model.
    """
    def __init__(self, net):
        self.net = net
    
    def train(self, train_set, test_set, num_epochs, device, lr, wd, train_type='auto'):
        
        # Initial setup
        running_loss = []
        test_loss = []
        batch_counter = 0
        rep_diffs = []
        trained = False
        log = ''
        
        # Criterion, optimizer (maybe make those inputs) and trained parameters 
        if train_type == 'auto':
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
            
            self.net.encoder.requires_grad_ = True
            self.net.decoder.requires_grad_ = True
            self.net.classifier.requires_grad_ = False
            
            train_phase = 'Autoencoder'
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
            
            self.net.encoder.requires_grad_ = True
            self.net.decoder.requires_grad_ = False
            self.net.classifier.requires_grad_ = True
            
            train_phase = 'Classifier'

        
        # Early stop setup
        # TODO
        best_loss = [math.inf, -1]
        patience_count = 0

        # Training 
        print(f'{train_phase} training...')
        for epoch in range(num_epochs):
            
            log += f'{train_phase} epoch {epoch+1}\n'
            cur_running_loss = 0.0
            
            for i, (x, labels) in enumerate(train_set):
                
                log += f'\tBatch {i+1}\n' 
                
                # Send data to device
                x = x.to(device)
                if train_type != 'auto':
                    labels = labels.to(device)

                # Train mode
                self.net.train()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass and error computation
                if train_type == 'auto':
                    pred = self.net.forward(x, decode=True)
                    loss = criterion(pred, x)
                else:
                    pred = self.net.forward(x, classify=True)
                    loss = criterion(pred, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()

                # Add loss of current pass
                cur_running_loss += loss.item()
                
            # Save running loss
            cur_running_loss /= len(train_set)
            running_loss.append(cur_running_loss)

            # Eval loss
            eval_loss, acc = self.evaluate(test_set, criterion, train_type, device)
            test_loss.append(eval_loss)
            
            # Verbose
            print(f'Epoch {epoch}\n\tRunning Loss:{cur_running_loss}\n\tEval loss: {eval_loss}')
            if train_type != 'auto':
                print(f'\n\tAccuracy: {acc}')
                
    @torch.no_grad()
    def evaluate(self, test_set, criterion, train_type, device):
        
        # Set model to eval mode
        self.net.eval()

        # Save loss
        total_loss = 0.0
        
        # Classifier data
        if train_type != 'auto':
            correct = 0
            total = 0

        for x, labels in iter(test_set):
            
            # Send to device
            x = x.to(device)
            if train_type != 'auto':
                labels = labels.to(device)
            
            # Forward pass and error computation
            if train_type == 'auto':
                pred = self.net.forward(x, decode=True)
                test_loss = criterion(pred, x)
            else:
                pred = self.net.forward(x, classify=True)
                test_loss = criterion(pred, labels)
                _, predicted = torch.max(pred, 1)
                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                
            total_loss += test_loss.item()
        
        eval_loss = total_loss/len(test_set)
        
        if train_type != 'auto':
            acc = correct/total 
        else:
            acc = None

        return eval_loss, acc