#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import argparse


class SentIndex:
    
    def __init__(self, unique_sents):
        self.index = {i: sent for i, sent in enumerate(unique_sents)}
        self.reversed = {sent: i for i, sent in enumerate(unique_sents)}
        
    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self.index[arg]
        elif isinstance(arg, str):
            return self.reversed[arg]
        else:
            raise ValueError('Can only take an int or a string as index')


def build_sentence_index(xs):
    seen = set()
    unique = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return SentIndex(unique)


def get_bar_data(indexed_items, counter):
    icounts, counts = [], []
    for i, s in indexed_items:
        icounts.append(i)
        counts.append(counter[s])
    return icounts, counts


def partition_dups(xs):
    seen = set()
    unique, dups = [], []
    for x in xs:
        if x not in seen:
            seen.add(x)
            unique.append(x)
        elif x not in dups:
            dups.append(x)
    return [s for s in unique if s not in dups], dups


def dup_stats(sents, dup_index, segment, partition):
    counter = Counter(sents)
    icounts, counts = get_bar_data(dup_index.index.items(), counter)
    plt.figure(figsize=(15,5))
    plt.title(f"Partition: {partition[0].upper()}{partition[1:].lower()} - Segment: {segment.upper()}")
    plt.xticks(np.arange(len(icounts), step=8))
    plt.bar(icounts, counts, color='SkyBlue')
    plt.legend(['Occurrences'], loc='upper right')
    plt.savefig(f"./images/sts-dup-stats-{partition}-{segment.upper()}.jpg")
    

def dump_sent_relations(sents, other_sents, scores, global_index, index, other_segment, filename):
    with open(filename, 'w') as out:
        for i, s in index.index.items():
            igeneral = global_index[s]
            original_indices = [j for j, x in enumerate(sents) if x == s]
            out.write(f"L{i}-G{igeneral}:\t'{s}'\n")
            out.write(f"Relates in {other_segment.upper()} to:\n")
            for j in original_indices:
                other = other_sents[j]
                out.write(f"\tG{global_index[other]:05d}\t\t{scores[j]:.3f}\t\t'{other}'\n")
            out.write('-' * 100 + '\n')


def dump_index(index, filename):
    with open(filename, 'w') as out:
        for i, sent in index.index.items():
            out.write(f"G{i:05d}: {sent}\n")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--partition', type=str, help='train / dev / test')
    parser.add_argument('-s', '--segment', type=str, help='a / b')
    parser.add_argument('--dumpindex', type=bool, default=False, help='Dump the global index G for every sentence (A and B)')
    args = parser.parse_args()
    base_path = f"../sts2017/{args.partition}/"
    other_segment = 'b' if args.segment == 'a' else 'a'
    with open(f"{base_path}{args.segment}.toks", 'r') as file,\
         open(f"{base_path}{other_segment}.toks", 'r') as other_file,\
         open(f"{base_path}sim.txt", 'r') as score_file:
        sents = [line.strip() for line in file.readlines()]
        other_sents = [line.strip() for line in other_file.readlines()]
        scores = [float(line.strip()) for line in score_file.readlines()]
        global_index = build_sentence_index(sents + other_sents if args.segment == 'a' else other_sents + sents)
        nondups, dups = partition_dups(sents)
        non_dup_index = build_sentence_index(nondups)
        dup_index = build_sentence_index(dups)
        dup_stats(sents, dup_index, args.segment, args.partition)
        dump_sent_relations(sents, other_sents, scores, global_index, dup_index, other_segment,
                            f"{base_path}duplicate-dump-{args.partition}-{args.segment.upper()}.txt")
        dump_sent_relations(sents, other_sents, scores, global_index, non_dup_index, other_segment,
                            f"{base_path}non-duplicate-dump-{args.partition}-{args.segment.upper()}.txt")
        if args.dumpindex:
            dump_index(global_index, f"{base_path}general-index.txt")
        











