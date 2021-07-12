import matplotlib.pyplot as plt
import numpy as np

"""
1: cross entropy
2: KLD
3: KLD ration
4: teacher forcing ration
5: BLEU Score
6: Gaussian Score
"""


def plotresult():
    ce = []
    kld = []
    ration = []
    tf = []
    bs = []
    gs = []
    with open('./record/record.txt', 'r') as f:
        for line in f:
            ce.append(float(line.split()[1].split(':')[1]))
            kld.append(float(line.split()[2].split(':')[1]))
            ration.append(float(line.split()[3].split(':')[1]))
            tf.append(float(line.split()[4].split(':')[1]))
            bs.append(float(line.split()[5].split(':')[1]))
            gs.append(float(line.split()[6].split(':')[1]))
    lines = []
    fig, ax1 = plt.subplots()

    plt.title('CVAE with KLD monotonic', fontsize=24)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    lines += ax1.plot(ce, label='Cross Entropy', color='black')
    lines += ax1.plot(kld, label='KL Divergence', color='red')
    ax1 = ax1.twinx()
    lines += ax1.plot(ration, label='KLD Ratio')
    lines += ax1.plot(tf, label='Teacher Forcing Ratio')
    lines += ax1.plot(bs,'.' ,label='BLEU Score')
    lines += ax1.plot(gs,'.' ,label='Gaussian Score', color='yellow')
    ax1.set_ylabel('Score/Weight')
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper right')
    plt.show()


if __name__ == '__main__':
    plotresult()
