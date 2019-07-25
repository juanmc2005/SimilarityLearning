#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

cmap = plt.get_cmap('tab10')
COLORS = [cmap(i) for i in range(10)]

# colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
#           '#ff00ff', '#990000', '#999900', '#009900', '#009999']


def visualize(feat, labels, title, legend, dir_path, filename):
    plt.ion()
    plt.clf()
    for i in range(len(np.unique(labels))):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=COLORS[i])
    plt.legend(legend, loc='best')
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
                plt.plot(x[inds], y[inds], '.', c=COLORS[i])
            plt.legend([str(center) for center in centers], loc='upper right')
    plt.axhline(y=0, color='grey', ls=':')
    plt.axvline(x=0, color='grey', ls=':')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def visualize_tsne_speaker(feat, y, unique_labels, distance, title, dir_path, filename):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, metric=distance.to_sklearn_metric())
    tsne_results = tsne.fit_transform(feat)
    print(f"t-SNE done! Time elapsed: {time.time() - time_start} seconds")
    plt.ion()
    plt.clf()
    plt.figure(figsize=(14, 10))
    legends = []
    for label, color in zip(unique_labels, COLORS):
        # Calculate distances between original embeddings
        dists = pdist(feat[y == label, :], metric=distance.to_sklearn_metric())
        legends.append(f"{label}: μ={np.mean(dists):.2f} σ={np.std(dists):.2f}")
        # Plot t-SNE embeddings
        curfeat_tsne = tsne_results[y == label, :]
        plt.plot(curfeat_tsne[:, 0], curfeat_tsne[:, 1], '.', c=color)
    plt.legend(legends, loc='best', fontsize='medium')
    plt.axhline(y=0, color='grey', ls=':')
    plt.axvline(x=0, color='grey', ls=':')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def plot_pred_hists(dists, y_true, title, dir_path, filename):
    bins = np.arange(0, 1, step=0.005)
    plt.ion()
    plt.clf()
    plt.hist([dist for dist, y in zip(dists, y_true) if y == 1], bins,
             alpha=0.5, density=True, label='Same', color='green', histtype='stepfilled')
    plt.hist([dist for dist, y in zip(dists, y_true) if y == 0], bins,
             alpha=0.5, density=True, label='Different', color='red', histtype='stepfilled')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def plot_det_curve(fpr, fnr, title, dir_path, filename):
    fpr, fnr = 100 * fpr, 100 * fnr
    min_tick, max_tick = 0.1, 40
    plt.ion()
    plt.clf()
    fig, ax = plt.subplots()
    plt.plot(fpr, fnr)
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [min_tick, 0.2, 0.5, 1, 2, 5, 10, 20, max_tick]
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([min_tick, max_tick, min_tick, max_tick])
    plt.xlabel('False Positive Rate (in %)')
    plt.ylabel('False Negative Rate (in %)')
    plt.title(title)
    plt.savefig(join(dir_path, f"{filename}.jpg"))
    plt.draw()
    plt.pause(0.001)


def visualize_logs(exp_path: str, log_file_name: str, metric_name: str, bottom: float,
                   top: float, color: str, title: str, plot_file_name: str):
    with open(join(exp_path, log_file_name), 'r') as log_file:
        data = [float(line.strip()) for line in log_file.readlines()]
        plt.ion()
        plt.clf()
        plt.plot(range(1, len(data) + 1), data, c=color)
        if bottom is not None and top is not None:
            plt.ylim(bottom, top)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.savefig(join(exp_path, f"{plot_file_name}.jpg"))
        plt.draw()
        plt.pause(0.001)


def visualize_logs_against(files: list, chain: list, metric_name: str, colors: list,
                           legends: list, title: str, plot_file_path: str):
    data = []
    for file_path in files:
        with open(file_path, 'r') as file:
            data.append([float(line.strip()) for line in file.readlines()])
    start_points = [1]
    for i in range(0, len(files) - 1):
        start_points.append(len(data[i]) if chain[i] else 1)
    plt.ion()
    plt.clf()
    for start, line, c in zip(start_points, data, colors):
        plt.plot(range(start, start + len(line)), line, c=c)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend(legends, loc='best')
    plt.title(title)
    plt.savefig(plot_file_path)
    plt.draw()
    plt.pause(0.001)
