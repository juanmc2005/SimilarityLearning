#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
from random import randint

import matplotlib.pyplot as plt
from tqdm import tqdm

from sts.stats.utils import SentIndex, Segment, MergeSegment


def unique_pairs(xs, ys, scores):
    seen = set()
    xunique, yunique, sunique = [], [], []
    for x, y, score in zip(xs, ys, scores):
        if (x, y) not in seen:
            seen.add((x, y))
            seen.add((y, x))
            xunique.append(x)
            yunique.append(y)
            sunique.append(score)
    return xunique, yunique, sunique


def plot_scores(scores, filename=None):
    sorted_scores = sorted(scores)
    plt.figure(figsize=(15,5))
    plt.title('Golden Rating Distribution')
    plt.plot(sorted_scores, color='DarkGreen')
    plt.axhline(y=1, color='LightBlue', linestyle='--')
    plt.axhline(y=2, color='yellow', linestyle='--')
    plt.axhline(y=3, color='orange', linestyle='--')
    plt.axhline(y=4, color='red', linestyle='--')
    plt.legend(['Rating'], loc='upper left')
    plt.xlabel('Sentence Pairs')
    plt.ylabel('Golden Rating')
    if filename is not None:
        plt.savefig(filename)


def anchor_related_sents(anchor, pairs):
    anchor_pairs = [(x, y) for x, y in pairs if x == anchor or y == anchor]
    related = []
    for pos1, pos2 in anchor_pairs:
        if pos1 != anchor:
            related.append(pos1)
        else:
            related.append(pos2)
    return related


def triplets(sents, pos_pairs, neg_pairs):
    anchors, positives, negatives = [], [], []
    for anchor in tqdm(sents):
        for positive in anchor_related_sents(anchor, pos_pairs):
            for negative in anchor_related_sents(anchor, neg_pairs):
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
    return anchors, positives, negatives


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--partition', type=str, help='train / dev / test')
parser.add_argument('-di', '--dumpindex', type=bool, default=False, help='Dump the global index G for every sentence (A and B)')
parser.add_argument('-sp', '--saveplots', type=bool, default=False, help='Whether to save duplicate stats plots')
args = parser.parse_args()

base_path = f"../../sts-all/{args.partition}/"
index_path = f"{base_path}general-index.txt"

with open(f"{base_path}a.toks", 'r') as file_a,\
         open(f"{base_path}b.toks", 'r') as file_b,\
         open(f"{base_path}sim.txt", 'r') as score_file:

    # Read sentences and create global index
    sents_a = [line.strip() for line in file_a.readlines()]
    sents_b = [line.strip() for line in file_b.readlines()]
    scores = [float(line.strip()) for line in score_file.readlines()]
    sents_a, sents_b, scores = unique_pairs(sents_a, sents_b, scores)
    sents_all = sents_a + sents_b
    global_index = SentIndex('G', sents_all)
    
    # Extract duplicates which appear in A and B
    segment_both = MergeSegment('a+b', sents_all, sents_a, sents_b)
    
    # Dump index if requested
    if args.dumpindex:
        global_index.dump(index_path)
    
    # Calculate stats and dump duplicate information
    segment_a = Segment('a', sents_a)
    segment_b = Segment('b', sents_b)
    
    # Plot score distribution
    print('Plotting score distribution...')
    plot_scores(scores, f"../images/{args.partition}-score-dist.jpg" if args.saveplots else None)
    
    # Generate positive and negative pairs
    pos, neg = segment_a.pos_neg_pairs(segment_b, scores, threshold=(2, 3))
    print(f"Total Positive Pairs: {len(pos)}")
    print(f"Total Negative Pairs: {len(neg)}")
    print(f"Mean Positive Pairs by Sentence: {len(pos) / len(sents_a)}")
    print(f"Mean Negative Pairs by Sentence: {len(neg) / len(sents_a)}")
    print(f"\nPositive Example: {random.choice(pos)}")
    print(f"\nNegative Example: {random.choice(neg)}\n")
    
    # Generate Triplets
    anchors, positives, negatives = triplets(sents_all, pos, neg)
    print(f"Total Triplets: {len(anchors)}")
    for choice in [randint(0, len(anchors)) for _ in range(5)]:
        print(f"\nTriplet Example: ({anchors[choice]}, {positives[choice]}, {negatives[choice]})")
    print()
    
    for segment, other_segment in [(segment_a, segment_b), (segment_b, segment_a)]:
        print(f"Analyzing segment {segment}...")
        dup_dmp_path = f"{base_path}duplicate-dump-{args.partition}-{segment}.txt"
        non_dup_dmp_path = f"{base_path}non-duplicate-dump-{args.partition}-{segment}.txt"
        plot_path = f"../images/sts-dup-stats-{args.partition}-{segment}.jpg" if args.saveplots else None
        segment.plot_dup_stats(args.partition, plot_path)
        segment.compare_and_dump_dups(other_segment, global_index,
                                      segment_both.dups, scores, dup_dmp_path)
        segment.compare_and_dump_non_dups(other_segment, global_index,
                                          segment_both.dups, scores, non_dup_dmp_path)

    # Calculate joint stats and dump joint duplicate information
    print(f"Analyzing segment {segment}...")
    dup_dmp_path = f"{base_path}duplicate-dump-{args.partition}-{segment_both}.txt"
    non_dup_dmp_path = f"{base_path}non-duplicate-dump-{args.partition}-{segment_both}.txt"
    plot_path = f"../images/sts-dup-stats-{args.partition}-{segment_both}.jpg" if args.saveplots else None
    segment_both.plot_dup_stats(args.partition, plot_path)
    segment_both.dump_dups(global_index, dup_dmp_path)
    segment_both.dump_non_dups(global_index, non_dup_dmp_path)








