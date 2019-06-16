import argparse
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from sts import utils as sts


def load_partition(path, partition):
    with open(join(path, partition, 'a.toks')) as afile, \
            open(join(path, partition, 'b.toks')) as bfile, \
            open(join(path, partition, 'sim.txt')) as simfile:
        a = [line.strip() for line in afile.readlines()]
        b = [line.strip() for line in bfile.readlines()]
        sim = [float(line.strip()) for line in simfile.readlines()]
        return a, b, sim


def clusterize(sents_a, sents_b, scores, threshold):
    segment_a = sts.SemEvalSegment(sents_a)
    segment_b = sts.SemEvalSegment(sents_b)
    clusters = segment_a.clusters(segment_b, scores, threshold)
    return clusters, len(clusters)


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='Path to SemEval dataset')
args = parser.parse_args()

atrain, btrain, simtrain = load_partition(args.path, 'train')
adev, bdev, simdev = load_partition(args.path, 'dev')
atest, btest, simtest = load_partition(args.path, 'test')
sents_a = atrain + adev + atest
sents_b = btrain + bdev + btest
scores = simtrain + simdev + simtest
sents_a, sents_b, scores = sts.unique_pairs(sents_a, sents_b, scores)

total_sents = len(set(sents_a + sents_b))

kept_sents, nclusters, means, maxs, ts = [], [], [], [], []
for i in range(5, 55, 5):
    threshold = i / 10
    clusters, nclass = clusterize(sents_a, sents_b, scores, threshold)
    sizes = [len(cluster) for cluster in clusters]
    kept_sents.append(sum(sizes))
    nclusters.append(nclass)
    means.append(np.mean(sizes))
    maxs.append(max(sizes))
    ts.append(threshold)

plt.figure(figsize=(15,5))
plt.title('SemEval Cluster Analysis')
plt.plot(ts, kept_sents, color='LightBlue')
plt.plot(ts, nclusters, color='DarkGreen')
plt.axhline(y=total_sents, color='red', linestyle='--')
plt.legend(['Sentences', 'Clusters', 'Total'], loc='upper right')
plt.xlabel('Thresholds')
plt.savefig('images/t-cluster-coverage.png')

plt.figure(figsize=(15,5))
plt.title('SemEval Cluster Analysis')
plt.plot(ts, means, color='orange')
plt.plot(ts, maxs, color='red')
plt.legend(['Mean Cluster Size', 'Max Cluster Size'], loc='upper right')
plt.xlabel('Thresholds')
plt.ylabel('N Sentences')
plt.savefig('images/t-cluster-size.png')