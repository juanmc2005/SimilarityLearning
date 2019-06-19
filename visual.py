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


def visualize_tsne(feat, title, filename):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(feat)
    print(f"t-SNE done! Time elapsed: {time.time() - time_start} seconds")
    plt.ion()
    plt.clf()
    plt.plot(tsne_results[:, 0], tsne_results[:, 1], '.', c='#009900')
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
