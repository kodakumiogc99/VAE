import torch
import torch.nn as nn
import numpy as np


class LSTM_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, bidirection):
        super(LSTM_ENCODER, self).__init__()
        if bidirection == True:
            self.bidirection = 2
        else:
            self.bidirection = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_i = nn.Embedding(29, hidden_size)
        self.embedding_c = nn.Embedding(4, 8)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirection)
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logv = nn.Linear(hidden_size, latent_size)

    def iniHidden(self, sz):
        return torch.zeros(self.num_layers*self.bidirection, sz, (self.hidden_size - 8), device='cuda')

    def forward(self, input_, condition):
        sz = input_.size(0)

        embedded_i = self.embedding_i(input_).to('cuda')
        embedded_i = embedded_i.transpose(0, 1)

        hidden = self.iniHidden(sz)
        embedded_c = self.embedding_c(condition).to('cuda')
        embedded_c = embedded_c.transpose(0, 1)
        embedded_c = embedded_c.repeat(self.num_layers*self.bidirection, 1, 1)

        hidden = torch.cat((hidden, embedded_c), dim=2)

        output, hidden = self.encoder_lstm(embedded_i, (hidden, hidden))

        hidden = hidden[0]

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps

        kld = -0.5 * torch.mean(1 + logv - mean.pow(2) - logv.exp()).to('cuda')

        return z, kld


class LSTM_DECODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, bidirection):
        super(LSTM_DECODER, self).__init__()
        if bidirection == True:
            self.bidirection = 2
        else:
            self.bidirection = 1
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirection)
        self.latent2hidden = nn.Linear(latent_size+8, hidden_size)
        self.outputs2char = nn.Linear(hidden_size*self.bidirection, 29)
        self.embedding_i = nn.Embedding(29, hidden_size)
        self.embedding_c = nn.Embedding(4, 8)
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, char, lattent, condition, concat=False):
        if concat:
            embedded_c = self.embedding_c(condition).transpose(0, 1).repeat(self.num_layers * self.bidirection, 1, 1)
            hidden = torch.cat((lattent, embedded_c), dim=2)
            hidden = self.latent2hidden(hidden)
            hidden = (hidden, hidden)
        else:
            hidden = lattent

        embedded_i = self.embedding_i(char)

        output, hidden = self.decoder_lstm(embedded_i, hidden)

        cross = self.outputs2char(output)
        output = torch.argmax(cross, dim=-1)
        return output, hidden, cross
