#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize(feat, labels, title, filename):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title(title)
    plt.savefig(f"./images/{filename}.jpg")
    plt.draw()
    plt.pause(0.001)


def visualize_tsne(feat, phrases, title, filename):
    indices_to_annotate = [528, 1830, 1369, 1009, 35, 754, 462, 756, 1451, 639]
    # indices_to_annotate = np.random.choice(feat.shape[0], 10, replace=False)
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(feat)
    print(f"t-SNE done! Time elapsed: {time.time() - time_start} seconds")
    plt.ion()
    plt.clf()
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    plt.plot(x, y, '.', c='#00ffff')
    for i in indices_to_annotate:
        print(f"Phrase {i}: {phrases[i]}")
        plt.annotate(str(i), (x[i], y[i]))
    plt.title(title)
    plt.savefig(f"./images/{filename}.jpg")
    plt.draw()
    plt.pause(0.001)


def plot_dists(dists, title, filename):
    plt.ion()
    plt.clf()
    plt.plot(dists, '.')
    plt.title(title)
    plt.savefig(f"./images/{filename}.jpg")
    plt.draw()
    plt.pause(0.001)
