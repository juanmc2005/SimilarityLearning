#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
          '#ff00ff', '#990000', '#999900', '#009900', '#009999']


def visualize(feat, labels, title, dir_path, filename):
    plt.ion()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=colors[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def visualize_tsne_neighbors(feat, phrases, distance, title, dir_path, filename):
    feat_unique, phrases_unique = [], []
    phrases_seen = set()
    for i in range(len(feat)):
        if phrases[i] not in phrases_seen:
            feat_unique.append(feat[i])
            phrases_unique.append(phrases[i])
            phrases_seen.add(phrases[i])
    feat = np.vstack(feat_unique)
    phrases = phrases_unique

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, metric=distance.to_sklearn_metric())
    tsne_results = tsne.fit_transform(feat)
    print(f"t-SNE done! Time elapsed: {time.time() - time_start} seconds")

    plt.ion()
    plt.clf()
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    if phrases is not None:
        with open(join(dir_path, f"{filename}-reference.txt"), 'w') as reffile:
            centers = [1061, 999, 782, 2518, 94]
            # centers = np.random.choice(feat.shape[0], 6, replace=False)
            nn = NearestNeighbors(n_neighbors=5, metric=distance.to_sklearn_metric())
            nn.fit(tsne_results)
            distances, indices = nn.kneighbors(tsne_results[centers, :])
            for i in range(len(centers)):
                c = centers[i]
                reffile.write(f"Phrase {c}: {phrases[c]}\nNeighbors:\n")
                for j in range(len(indices[i])):
                    nid = indices[i, j]
                    if nid != c:
                        reffile.write(f"\tPhrase {nid} at distance {distances[i, j]}: {phrases[nid]}\n")
                inds = [indices[i, j] for j in range(len(indices[i]))]
                plt.plot(x[inds], y[inds], '.', c=colors[i])
            plt.legend([str(center) for center in centers], loc='upper right')
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def plot_pred_hists(dists, y_true, title, dir_path, filename):
    bins = np.arange(0, 1, step=0.005)
    plt.ion()
    plt.clf()
    plt.hist([dist for dist, y in zip(dists, y_true) if y == 1], bins, alpha=0.5, label='Same', color='green')
    plt.hist([dist for dist, y in zip(dists, y_true) if y == 0], bins, alpha=0.5, label='Different', color='red')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def visualize_train_loss(exp_path: str, title: str, filename: str):
    with open(join(exp_path, 'loss.log'), 'r') as loss_file:
        losses = [float(line.strip()) for line in loss_file.readlines()]
        plt.ion()
        plt.clf()
        plt.plot(losses, c='red')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.savefig(join(exp_path, f"{filename}.jpg"))
        plt.draw()
        plt.pause(0.001)


def visualize_train_metric(exp_path: str, title: str, filename: str):
    with open(join(exp_path, 'metric.log'), 'r') as metric_file:
        metrics = [float(line.strip()) for line in metric_file.readlines()]
        plt.ion()
        plt.clf()
        plt.plot(metrics, c='green')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.savefig(join(exp_path, f"{filename}.jpg"))
        plt.draw()
        plt.pause(0.001)
