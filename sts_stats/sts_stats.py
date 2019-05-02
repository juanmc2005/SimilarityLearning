#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats


def get_dups(xs):
    seen = set()
    dups = []
    for x in xs:
        if x not in seen:
            seen.add(x)
        else:
            dups.append(x)
    return dups


def count_unique(xs):
    seen = set()
    unique = 0
    for x in xs:
        if x not in seen:
            seen.add(x)
            unique += 1
    return unique


def print_duplicate_source(dups, first, second):
    #str_list = ""
    a_dups, b_dups, both_dups = 0, 0, 0
    for x in dups:
        #str_list += x
        a = x in first
        b = x in second
        if a and b:
            both_dups += 1
        elif a:
            a_dups += 1
        else:
            b_dups += 1
    #print(f"Duplicated sentences:\n{str_list}")
    print(f"Duplicates from A:\t\t{a_dups}\nDuplicates in B:\t\t{b_dups}\nDuplicates in both:\t\t{both_dups}")


def print_occurrence_stats(dups, first, second, verbose=True):
    max_dup = 0
    max_sent = ""
    occurrences = []
    for x in dups:
        if verbose:
            print('-' * 120)
        
        a = x in first
        b = x in second
        
        if a and b:
            cnt = first.count(x) + second.count(x)
            if verbose:
                i = first.index(x)
                j = second.index(x)
                print(f"The phrase '{x.strip()}' appears {first.count(x)} times in A, and {second.count(x)} times in B")
                print(f"In A, it is related to the phrase '{second[i].strip()}'{', which is also a duplicate' if second[i] in dups else ''}")
                print(f"In B, it is related to the phrase '{first[j].strip()}'{', which is also a duplicate' if first[j] in dups else ''}")
        elif a:
            cnt = first.count(x)
            if verbose:
                i = first.index(x)
                print(f"The phrase '{x.strip()}' appears {first.count(x)} times in A, and it doesn't appear in B")
                print(f"It is related to the phrase '{second[i].strip()}'{', which is also a duplicate' if second[i] in dups else ''}")
        else:
            cnt = second.count(x)
            if verbose:
                j = second.index(x)
                print(f"The phrase '{x.strip()}' appears {second.count(x)} times in B, and it doesn't appear in A")
                print(f"It is related to the phrase '{first[j].strip()}'{', which is also a duplicate' if first[j] in dups else ''}")
        
        occurrences.append(cnt)
        if cnt > max_dup:
            max_dup = cnt
            max_sent = x

    print("\n\n" + "-" * 30 + " Duplicate Stats " + "-" * 30)
    print(f"Max Occurrences:\t\t{max_dup} times '{max_sent.strip()}'")
    print(f"Mean Occurrences:\t\t{np.mean(occurrences)}")
    print(f"Median Occurrences:\t\t{np.median(occurrences)}")
    print(f"Mode Occurrences:\t\t{stats.mode(occurrences)[0][0]}")
    print(f"Total Duplicate Occurrences:\t{sum(occurrences)}")
    print("-" * 77)


with open('../sts2017/train/a.toks', 'r') as a_file, open('../sts2017/train/b.toks', 'r') as b_file:
    first = a_file.readlines()
    second = b_file.readlines()
    sentences = first + second
    dups = get_dups(sentences)
    
    print_occurrence_stats(dups, first, second, verbose=True)
    
    print(f"Total Sentences:\t\t{len(sentences)}")
    print(f"Unique Duplicates:\t\t{len(dups)}")
    print(f"Unique Sentences:\t\t{count_unique(sentences)}")
    
    print_duplicate_source(dups, first, second)
    print()
