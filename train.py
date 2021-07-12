import torch
import torch.nn as nn
from DataLoader import train_dataset
import numpy as np
from torch import optim
import math
from utils import index2word, GaussianBatchNoise, GaussianScore, Bleu
import random


def GaussianTest(decoder, path, num_layers, bidirection, batch_size, latent_size, num_workers,
                 num_of_word_to_test):
    decoder = decoder.to('cuda')

    if bidirection == True:
        bi = 2
    else:
        bi = 1

    testnoise = GaussianBatchNoise(batch_size, num_layers, bi, latent_size,
                                   num_workers, num_of_word_to_test)
    decoder.eval()
    testlist = []
    for index, (latent, tense) in enumerate(testnoise):

        latent = latent.to('cuda')
        tense = tense.to('cuda')
        tense = tense.unsqueeze(1)

        sz = latent.size(0)

        SOS = []
        v_list = []

        for i in range(sz):
            SOS.append(1)
        SOS = torch.IntTensor(np.array(SOS)).to('cuda')
        SOS = SOS.reshape(1, sz)

        latent = latent.transpose(0, 1)

        output, hidden, _ = decoder(SOS, latent, tense, True)

        v_list = output.transpose(0, 1).cpu().numpy()

        for i in range(1, 17):
            output, hidden, _ = decoder(output, hidden, tense, False)
            v_list = np.concatenate((v_list, output.transpose(0, 1).cpu().numpy()), axis=1)

        temp = tense.cpu().numpy()
        words = []

        for i, word in zip(temp, v_list):
            words.append(index2word(word))
            if i == 3:
                testlist.extend([words])
                words = []
    print(f'GAUSSIAN\n {testlist}')

    return GaussianScore(testlist, path)


def BLEUScore(encoder_loader, decoder_loader, encoder, decoder):
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')
    encoder.eval()
    decoder.eval()
    predict = []
    answer = []
    for index, ((encoders, encoders_tense), (decoders, decoders_tense)) in enumerate(zip(encoder_loader, decoder_loader)):
        encoders = encoders.to('cuda')
        decoders = decoders.to('cuda')
        encoders_tense = encoders_tense.to('cuda')
        decoders_tense = decoders_tense.to('cuda')

        latent, kld = encoder(encoders, encoders_tense)
        SOS = []
        sz = latent.size(1)
        for i in range(sz):
            SOS.append(1)
        SOS = torch.LongTensor(np.array(SOS)).to('cuda')
        SOS = SOS.reshape(1, sz)

        output, hidden, cross = decoder(SOS, latent, decoders_tense, True)

        v_array = output.transpose(0, 1).cpu().numpy()

        for i in range(1, 17):
            output, hidden, cross = decoder(output, hidden, decoders_tense, False)
            v_array = np.concatenate((v_array, output.transpose(0, 1).cpu().numpy()), axis=1)

        for i in range(sz):
            predict.append(index2word(v_array[i]))
            answer.append(index2word(decoders[i].cpu().numpy()))
    print(f'BLEU:')
    print(f'Predict: {predict}')
    print(f'Answer: {answer}')

    return Bleu(predict, answer)


def train(data, encoder_loader, decoder_loader, encoder, decoder, encoder_optim,
          decoder_optim, loss_func, epoch, num_workers,
          bidirection, batch_size, latent_size, num_layers):
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')
    teacher_forcing_ration = 0.0
    ratio = 0
    for eps in range(5225, epoch):

        if eps < 200:
            ratio += 0.01
            ratio = round(ratio, 2)
        else:
            ratio = 2
        if eps < 100:
            teacher_forcing_ration = 0
        elif eps < 1000:
            teacher_forcing_ration = 1.0
        else:
            teacher_forcing_ration = 0.5
        encoder.train()
        decoder.train()
        epoch_entropy = 0
        epoch_kld = 0
        ep = 0
        predict = []
        answer = []
        for index, (vocabu, tense) in enumerate(data):
            vocabu = vocabu.to('cuda')
            tense = tense.to('cuda')

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            z, kld = encoder(vocabu, tense)

            use_teacher_forcing = True if random.random() < teacher_forcing_ration else False

            SOS = []
            sz = z.size(1)
            for i in range(sz):
                SOS.append(1)
            SOS = torch.IntTensor(np.array(SOS)).to('cuda')
            SOS = SOS.reshape(1, sz)

            output, hidden, cross = decoder(SOS, z, tense, True)

            character_entropy = 0

            v_array = []
            v_array = output.transpose(0, 1).cpu().numpy()
            character_entropy += loss_func(cross.squeeze(), vocabu[:, 0])

            for i in range(1, 17):
                if use_teacher_forcing:
                    output, hidden, cross = decoder(vocabu[:, i].unsqueeze(0), hidden, tense, False)
                else:
                    output, hidden, cross = decoder(output, hidden, tense, False)

                v_array = np.concatenate((v_array, output.transpose(0, 1).cpu().numpy()), axis=1)
                character_entropy += loss_func(cross.squeeze(), vocabu[:, i])

            epoch_entropy += character_entropy
            epoch_kld += kld
            total_entropy = character_entropy + ratio * kld

            total_entropy.backward()
            encoder_optim.step()
            decoder_optim.step()

            for i in range(sz):
                predict.append(index2word(v_array[i]))
                answer.append(index2word(vocabu[i].cpu().numpy()))
            ep = index + 1

        BScore = BLEUScore(encoder_loader, decoder_loader, encoder, decoder)
        GScore = GaussianTest(decoder, 'train.txt', num_layers,
                              bidirection, batch_size, latent_size,
                              num_workers, 100)
        epoch_entropy = epoch_entropy / ep
        epoch_kld = epoch_kld / ep

        print(f'eps:{eps} entropy:{epoch_entropy} kld:{epoch_kld} ratio:{ratio} tf:{teacher_forcing_ration} BLEU:{BScore} Gassian:{GScore}\n')

        with open('record/record.txt', 'a') as fp:
            fp.write(f'eps:{eps} entropy:{epoch_entropy} kld:{epoch_kld} ratio:{ratio} tf:{teacher_forcing_ration} BLEU:{BScore} Gassian:{GScore}\n')

        torch.save(encoder.state_dict(), f'net/encoder_{eps % 100}.pth')
        torch.save(decoder.state_dict(), f'net/decoder_{eps % 100}.pth')
