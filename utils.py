from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def index2word(str1):
    trans1 = ''
    for c in str1:
        if c == 0:
            trans1 += ''
        elif c == 1:
            trans1 += ''
        elif c == 2:
            break
        else:
            trans1 += chr(c + 94)
    return trans1


def ComputeBleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    return sentence_bleu([reference], output, weights=weights,
                         smoothing_function=cc.method1)


def Bleu(predict, answer):
    score = 0
    for p, a in zip(predict, answer):
        score += ComputeBleu(p, a)
    return score/len(predict)


def GaussianNoise(num_layers, bidirection, latent_size):

    # np.random.seed(0)
    # torch.random.manual_seed(0)
    # torch.cuda.random.manual_seed_all(0)

    normal_latents = torch.empty([1, num_layers * bidirection, latent_size])

    normal_latents = normal_latents.normal_(mean=0, std=1)

    return normal_latents


def GaussianScore(words, path):
    search_list = []
    score = 0
    yourpath = path
    with open(yourpath, 'r') as fp:
        for line in fp:
            search = line.split(' ')
            search[3] = search[3].strip('\n')
            search_list.extend([search])
        for t in words:
            for i in search_list:
                if t == i:
                    score += 1
    return score/len(words)


def GaussianBatchNoise(batch_size, num_layers, bidirection, latent_size, num_workers,
                       num_of_word_to_test):
    latent = None
    tense = []

    for indx in range(num_of_word_to_test):
        noise = GaussianNoise(num_layers, bidirection, latent_size)

        for i in range(4):
            if latent == None:
                latent = noise
            else:
                latent = torch.cat((latent, noise), 0)
            tense.append(i)

    tense = torch.IntTensor(np.array(tense))

    dataset = TensorDataset(latent, tense)

    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    return dataloader
