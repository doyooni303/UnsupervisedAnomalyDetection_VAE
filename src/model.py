import os, sys
from typing import Union, List

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.ReLU = nn.ReLU()

        self.training = True

    def forward(self, x):
        h_ = self.ReLU(self.FC_input(x))
        h_ = self.ReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU(0.2)

    def forward(self, x):
        h = self.ReLU(self.FC_hidden(x))
        h = self.ReLU(self.FC_hidden2(h))

        x_hat = self.Tanh(self.FC_output(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, latent_dim, dec_hidden_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, enc_hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, dec_hidden_dim, input_dim)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
