from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch


"""
SOS   A   B   C   D   E   F   G   H   I    J    K   L   M
 1    3   4   5   6   7   8   9   10   11   12   13  14  15

 N    O   P   Q   R   S   T   U   V   W    X    Y   Z  EOS
16   17  18  19  20  21  22  23  24  25   26   27  28  2

PAD
0

"""


def train_dataset(file: str, num_workers: int, batch_size: int):
    with open(file, 'r') as f:
        vectors = []
        labels = []
        for line in f:
            for i in range(4):
                vector = []
                label = []
                char = line.split()[i]
                vector.append(int(1))
                for j in char:
                    vector.append(int(ord(j.lower()) - 94 ))
                vector.append(int(2))
                while(len(vector) < 17):
                    vector.append(int(0))
                label.append(i)
                vectors.append(np.array(vector))
                labels.append(np.array(label))
        vectors = torch.LongTensor(vectors)
        labels = torch.LongTensor(labels)
        dataset = TensorDataset(vectors, labels)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    return dataloader

def test_dataset(file: str, num_workers: int, batch_size: int):
    with open(file, 'r') as f:
        encoders = []
        encoders_tense = []
        decoders = []
        decoders_tense = []
        for i, (line) in enumerate(f):
            encoder = []
            decoder = []
            tense = []
            char = line.split()[0]

            if i % 2 == 0:
                encoder.append(int(1))
                for j in char:
                    encoder.append(int(ord(j.lower()) - 94 ))
                encoder.append(int(2))
                while(len(encoder) < 17):
                    encoder.append(int(0))
                tense.append(int(line.split()[1]))
                encoders_tense.append(np.array(tense))
                encoders.append(np.array(encoder))

            else:
                decoder. append(int(1))
                for j in char:
                    decoder.append(int(ord(j.lower()) - 94 ))
                decoder.append(int(2))
                while(len(decoder) < 17):
                    decoder.append(int(0))
                tense.append(int(line.split()[1]))
                decoders_tense.append(np.array(tense))
                decoders.append(np.array(decoder))

        encoders = torch.LongTensor(encoders)
        decoders = torch.LongTensor(decoders)
        encoders_tense = torch.LongTensor(encoders_tense)
        decoders_tense = torch.LongTensor(decoders_tense)
        encoder_set = TensorDataset(encoders, encoders_tense)
        decoder_set = TensorDataset(decoders, decoders_tense)
        encoder_loader = DataLoader(encoder_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        decoder_loader = DataLoader(decoder_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    return encoder_loader, decoder_loader

if __name__ == '__main__':
    print('enter\n')
    test = test_dataset('test.txt', 12, 1)
    for a, b in test:
        print(f'{a},{b}\n')

