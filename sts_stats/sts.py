#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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


def dups_in_both(dups, first, second):
    result = []
    for x in dups:
        a = x in first
        b = x in second
        if a and b:
            result.append(x)
    return result


def plot_stats(sents, index, partition, segment, filename, step=8):
    counter = Counter(sents)
    icounts, counts = get_bar_data(index.index.items(), counter)
    plt.figure(figsize=(15,5))
    plt.title(f"Partition: {partition[0].upper()}{partition[1:].lower()} - Segment: {segment}")
    plt.xticks(np.arange(len(icounts), step=step))
    plt.bar(icounts, counts, color='SkyBlue')
    plt.legend(['Occurrences'], loc='upper right')
    if filename is not None:
        plt.savefig(filename)


class SentIndex:
    
    def __init__(self, code, sents):
        seen = set()
        unique = []
        for x in sents:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        self.index = {i: sent for i, sent in enumerate(unique)}
        self.reversed = {sent: i for i, sent in enumerate(unique)}
        self.code = code
        
    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self.index[arg]
        elif isinstance(arg, str):
            return self.reversed[arg]
        else:
            raise ValueError('Can only take an int or a string as index')
    
    def dump(self, filename, verbose=True):
        if verbose:
            print(f"Dumping index to {filename}...")
        with open(filename, 'w') as out:
            for i, sent in self.index.items():
                out.write(f"{self.code}{i:05d}: {sent}\n")


class Segment:
    
    def __init__(self, code, sents):
        self.code = code
        self.sents = sents
        self.nondups, self.dups = partition_dups(sents)
        self.non_dup_index = SentIndex(f"ND{code.upper()}", self.nondups)
        self.dup_index = SentIndex(f"D{code.upper()}", self.dups)
    
    def _compare_and_dump_index(self, index, other_segment, global_index, both_dups, scores, filename):
        with open(filename, 'w') as out:
            for i, s in index.index.items():
                igeneral = global_index[s]
                original_indices = [j for j, x in enumerate(self.sents) if x == s]
                out.write(f"{index.code}{i}-{global_index.code}{igeneral:05d}:\t'{s}'\n")
                out.write(f"Relates in {other_segment} to:\n")
                for j in original_indices:
                    other = other_segment.sents[j]
                    global_id = f"{global_index.code}{global_index[other]:05d}"
                    dup_category = 'No '
                    if other in both_dups:
                        dup_category = f"{self}+{other_segment}"
                    elif other in self.dup_index.reversed:
                        dup_category = f"{self}  "
                    elif other in other_segment.dup_index.reversed:
                        dup_category = f"{other_segment}  "
                    out.write(f"\t{global_id}\t\tDuplicate: {dup_category}\t\t{scores[j]:.3f}\t\t'{other}'\n")
                out.write('-' * 100 + '\n')
    
    def __str__(self):
        return self.code.upper()
    
    def plot_dup_stats(self, partition, filename):
        plot_stats(self.sents, self.dup_index, partition, self, filename)
    
    def compare_and_dump_dups(self, other_segment, global_index,
                              both_dups, scores, filename, verbose=True):
        if verbose:
            print(f"Dumping duplicate info to {filename}...")
        self._compare_and_dump_index(self.dup_index, other_segment,
                                     global_index, both_dups, scores, filename)
    
    def compare_and_dump_non_dups(self, other_segment, global_index,
                                  both_dups, scores, filename, verbose=True):
        if verbose:
            print(f"Dumping non-duplicate info to {filename}...")
        self._compare_and_dump_index(self.non_dup_index, other_segment,
                                     global_index, both_dups, scores, filename)


class MergeSegment:
    
    def __init__(self, code, sents, sents_a, sents_b):
        self.code = code
        self.sents = sents
        self.nondups, dups = partition_dups(self.sents)
        self.dups = dups_in_both(dups, sents_a, sents_b)
        self.non_dup_index = SentIndex(f"ND{code.upper()}", self.nondups)
        self.dup_index = SentIndex(f"D{code.upper()}", self.dups)
    
    def __str__(self):
        return self.code.upper()
    
    def _dump(self, index, global_index, filename):
        with open(filename, 'w') as out:
            for i, s in index.index.items():
                igeneral = global_index[s]
                out.write(f"{index.code}{i}-{global_index.code}{igeneral:05d}:\t'{s}'\n")
    
    def plot_dup_stats(self, partition, filename):
        plot_stats(self.sents, self.dup_index, partition, self, filename, step=16)
    
    def dump_dups(self, global_index, filename, verbose=True):
        if verbose:
            print(f"Dumping duplicate info to {filename}...")
        self._dump(self.dup_index, global_index, filename)
    
    def dump_non_dups(self, global_index, filename, verbose=True):
        if verbose:
            print(f"Dumping non-duplicate info to {filename}...")
        self._dump(self.non_dup_index, global_index, filename)

