### The model used for VAE smiley ####

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np

class Variational_Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Variational_Encoder, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 40)
        self.linear3 = nn.Linear(40,80)
        self.linear4 = nn.Linear(80,40)
        self.linear5 = nn.Linear(40,20)
        self.linear6 = nn.Linear(20,5)
        self.linear7 = nn.Linear(5, latent_dims)
        self.linear8 = nn.Linear(5, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x)) 
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        mu =  self.linear7(x)
        sigma = torch.exp(self.linear8(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Variational_Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Variational_Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 5)
        self.linear2 = nn.Linear(5, 20)
        self.linear3 = nn.Linear(20, 40)
        self.linear4 = nn.Linear(40,80)
        self.linear5 = nn.Linear(80,40)
        self.linear6 = nn.Linear(40,20)
        self.linear7 = nn.Linear(20,10)     

    def forward(self, z):
        z = F.relu(self.linear1(z)) # relu(x) = max(0,x)
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        z = F.relu(self.linear6(z))
        z = self.linear7(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Variational_Encoder(latent_dims)
        self.decoder = Variational_Decoder(latent_dims)  

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
