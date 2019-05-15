#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from sts import SentIndex, Segment, MergeSegment


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
    print(f"A size : {len(sents_a)}")
    print(f"B size : {len(sents_b)}")
    print(f"Scores size : {len(scores)}")
    segment_a = Segment('a', sents_a)
    segment_b = Segment('b', sents_b)
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








