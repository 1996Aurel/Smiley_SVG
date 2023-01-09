from .utils import EarlyStopping     #The import work in VS code but not in jupyter ! 
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(VAE, train_dataloader, val_dataloader, criterion, optimizer, epochs, beta = 1.0):
    train_loss = []
    val_loss = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience = 10, verbose = True)
    
    for epoch in range(epochs):
        train_current_loss = 0
        val_current_loss = 0
        
        #### TRAINING ######
        VAE.train() # prep model for training
        for x in train_dataloader:
            x = x.to(device) 
            optimizer.zero_grad()
            x_hat = VAE(x.float())
            loss = criterion(x_hat, x.float()) + beta * VAE.encoder.kl
            #loss = ((x - x_hat)**2).sum()  + VAE.encoder.kl
            loss.backward()
            optimizer.step()
            train_current_loss += loss.item()
            
        ##### VALIDATION #####
        VAE.eval()
        for x in val_dataloader:
            x = x.to(device)
            x_hat = VAE(x.float())
            val_current_loss += (criterion(x_hat, x.float())+ beta * VAE.encoder.kl).item()
            #val_current_loss = (((x - x_hat)**2).sum()  + VAE.encoder.kl).item()
        
        train_current_loss /= len(train_dataloader)
        train_loss.append(train_current_loss)
        val_current_loss /= len(val_dataloader)
        val_loss.append(val_current_loss)
        print('Epoch {} of {}, Train Loss: {:.3f}, Val Loss:{:.3f}'.format(
            epoch + 1, epochs, train_current_loss, val_current_loss))
        
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_current_loss, VAE)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    VAE.load_state_dict(torch.load('checkpoint.pt'))

    return VAE, train_loss, val_loss 
