import torch
import torch.nn as nn
from torch import optim
from model import LSTM_ENCODER, LSTM_DECODER
from DataLoader import train_dataset, test_dataset
import argparse
import numpy as np
from train import train, GaussianTest


def main(epoch, lr, hidden_size, num_workers, batch_size, num_layers, latent_size, bidirection):
    train_data = train_dataset('train.txt', num_workers, batch_size)
    encoder_loader, decoder_loader = test_dataset('test.txt',num_workers, batch_size)
    """
    LSTM_CVAE(char_size, hidden_size, num_layers, batch_size, latent_size)
    """
    encoder = LSTM_ENCODER(17, hidden_size, num_layers, latent_size, bidirection)
    decoder = LSTM_DECODER(17, hidden_size, num_layers, latent_size, bidirection)
    encoder.load_state_dict(torch.load('net/encoder_31.pth'))
    decoder.load_state_dict(torch.load('net/decoder_31.pth'))
    encoder_optim = optim.SGD(encoder.parameters(), lr=lr, momentum=0.9)
    decoder_optim = optim.SGD(decoder.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train(train_data, encoder_loader, decoder_loader, encoder, decoder, encoder_optim, decoder_optim, criterion,
          epoch, num_workers, bidirection, batch_size, latent_size, num_layers)


if __name__ == '__main__':
    # np.random.seed(0)
    # torch.random.manual_seed(0)
    # torch.cuda.random.manual_seed_all(0)
    parser = argparse. ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--bidirection', type=bool, default=True)
    args = parser.parse_args()
    main(args.epoch, args.lr, args.hidden_size, args.num_workers, args.batch_size,
         args.num_layers, args.latent_size, args.bidirection)
