import time
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.cluster import SpectralClustering
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from numpy import linalg as LA
import scipy
from scipy.sparse import csgraph
import math
from scipy.sparse.linalg import eigsh
import random
from tqdm import tqdm
import json


def composite_function(f, g):
    return lambda x: f(g(x))


with open("remapped_sequential_data.txt", "r") as f:
    data = f.read()

data = data.split("\n")[:-1]
data = [d.split(" ")[1:-2] for d in data]

words = [[int(one_d) - 1 for one_d in d] for d in data]
original_vocab = set([a for one_word in words for a in one_word])

print("compute item pair frequency")

pair_freqs = defaultdict(int)
for word in words:
    pairs = combinations(word, 2)
    for pair in pairs:
        pair_freqs[tuple(pair)] += 1
sorted_pair_freqs = sorted(pair_freqs.items(), key=lambda x: x[1], reverse=True)

print("compute item frequency")

occurrences = Counter([a for one_word in words for a in one_word])
occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
matrix_size = len(occurrences)

print("compute adjacency matrix")
adjacency_matrix = np.zeros((matrix_size, matrix_size))
for pair, freq in sorted_pair_freqs:

    adjacency_matrix[pair[0], pair[1]] = freq
    adjacency_matrix[pair[1], pair[0]] = freq


maximum_cluster_size = 500
number_of_clusters = 20

begin_time = time.time()
clustering = SpectralClustering(
    n_clusters=number_of_clusters,
    assign_labels="cluster_qr",
    random_state=0,
    affinity="precomputed",
).fit(adjacency_matrix)
end_time = time.time()
used_time = end_time - begin_time
print("used time to compute it is {} seconds".format(used_time))

labels = clustering.labels_.tolist()
item_CF_index_map = {i: [str(label)] for i, label in enumerate(labels)}
group_labels = []
for i in range(number_of_clusters):
    group_labels.append(labels.count(i))


def one_further_indexing(
    which_group,
    group_labels,
    sorted_pair_freqs,
    number_of_clusters,
    item_CF_index_map,
    reverse_fcts,
    mode,
):

    one_subcluster_items = [
        item for item, l in enumerate(group_labels) if l == which_group
    ]

    subcluster_pairs = [
        sorted_pair_freq
        for sorted_pair_freq in sorted_pair_freqs
        if sorted_pair_freq[0][0] in one_subcluster_items
        and sorted_pair_freq[0][1] in one_subcluster_items
    ]

    item_map = {
        old_item_index: i for i, old_item_index in enumerate(one_subcluster_items)
    }
    reverse_item_map = {
        i: old_item_index for i, old_item_index in enumerate(one_subcluster_items)
    }

    remapped_subcluster_pairs = [
        (
            (item_map[subcluster_pair[0][0]], item_map[subcluster_pair[0][1]]),
            subcluster_pair[1],
        )
        for subcluster_pair in subcluster_pairs
    ]


    sub_matrix_size = len(item_map)
    sub_adjacency_matrix = np.zeros((sub_matrix_size, sub_matrix_size))
    for pair, freq in remapped_subcluster_pairs:
        sub_adjacency_matrix[pair[0], pair[1]] = freq
        sub_adjacency_matrix[pair[1], pair[0]] = freq

    numberofclusters = number_of_clusters


    sub_clustering = SpectralClustering(
        n_clusters=numberofclusters,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(sub_adjacency_matrix)
    sub_labels = sub_clustering.labels_.tolist()


    reversal = lambda x: x
    for reverse_fct in reverse_fcts:
        reversal = composite_function(
            reverse_fct, reversal
        )

    for i, label in enumerate(sub_labels):
        item_CF_index_map[reversal(reverse_item_map[i])].append(str(label))


    new_reverse_fcts = [lambda y: reverse_item_map[y]] + reverse_fcts

    return sub_labels, remapped_subcluster_pairs, item_CF_index_map, new_reverse_fcts


level_one = labels
level_two = []
level_three = []
level_four = []
level_five = []
level_six = []
level_seven = []
level_eight = []
N = number_of_clusters
M = maximum_cluster_size
reverse_fcts = [lambda x: x]
for a in range(N):
    if level_one.count(a) > M:
        (
            a_labels,
            remapped_a_cluster_pairs,
            item_CF_index_map,
            level_two_reverse_fcts,
        ) = one_further_indexing(
            a, labels, sorted_pair_freqs, N, item_CF_index_map, reverse_fcts, 2
        )
        level_two.append((a, a_labels))

        for b in range(N):
            if a_labels.count(b) > M:
                (
                    b_labels,
                    remapped_b_cluster_pairs,
                    item_CF_index_map,
                    level_three_reverse_fcts,
                ) = one_further_indexing(
                    b,
                    a_labels,
                    remapped_a_cluster_pairs,
                    N,
                    item_CF_index_map,
                    level_two_reverse_fcts,
                    3,
                )
                level_three.append((a, b, b_labels))

                for c in range(N):
                    if b_labels.count(c) > M:
                        (
                            c_labels,
                            remapped_c_cluster_pairs,
                            item_CF_index_map,
                            level_four_reverse_fcts,
                        ) = one_further_indexing(
                            c,
                            b_labels,
                            remapped_b_cluster_pairs,
                            N,
                            item_CF_index_map,
                            level_three_reverse_fcts,
                            4,
                        )
                        level_four.append((a, b, c, c_labels))

                        for d in range(N):
                            if c_labels.count(d) > M:
                                (
                                    d_labels,
                                    remapped_d_cluster_pairs,
                                    item_CF_index_map,
                                    level_five_reverse_fcts,
                                ) = one_further_indexing(
                                    d,
                                    c_labels,
                                    remapped_c_cluster_pairs,
                                    N,
                                    item_CF_index_map,
                                    level_four_reverse_fcts,
                                    5,
                                )
                                level_five.append((a, b, c, d, d_labels))

                                for e in range(N):
                                    if d_labels.count(e) > M:
                                        (
                                            e_labels,
                                            remapped_e_cluster_pairs,
                                            item_CF_index_map,
                                            level_six_reverse_fcts,
                                        ) = one_further_indexing(
                                            e,
                                            d_labels,
                                            remapped_d_cluster_pairs,
                                            N,
                                            item_CF_index_map,
                                            level_five_reverse_fcts,
                                            6,
                                        )
                                        level_six.append((a, b, c, d, e, e_labels))

                                        for f in range(N):
                                            if e_labels.count(f) > M:
                                                (
                                                    f_labels,
                                                    remapped_f_cluster_pairs,
                                                    item_CF_index_map,
                                                    level_seven_reverse_fcts,
                                                ) = one_further_indexing(
                                                    f,
                                                    e_labels,
                                                    remapped_e_cluster_pairs,
                                                    N,
                                                    item_CF_index_map,
                                                    level_six_reverse_fcts,
                                                    7,
                                                )
                                                level_seven.append(
                                                    (a, b, c, d, e, f, f_labels)
                                                )

                                                for g in range(N):
                                                    if f_labels.count(g) > M:
                                                        (
                                                            g_labels,
                                                            remapped_g_cluster_pairs,
                                                            item_CF_index_map,
                                                            level_eight_reverse_fcts,
                                                        ) = one_further_indexing(
                                                            g,
                                                            f_labels,
                                                            remapped_f_cluster_pairs,
                                                            N,
                                                            item_CF_index_map,
                                                            level_seven_reverse_fcts,
                                                            8,
                                                        )
                                                        level_eight.append(
                                                            (
                                                                a,
                                                                b,
                                                                c,
                                                                d,
                                                                e,
                                                                f,
                                                                g,
                                                                g_labels,
                                                            )
                                                        )


with open(
    "c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size), "w"
) as f:
    json.dump(item_CF_index_map, f)


with open(
    "c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size), "r"
) as f:
    data = json.load(f)


count = {}
for k, v in data.items():
    if tuple(v) not in count:
        count[tuple(v)] = 1
    else:
        count[tuple(v)] += 1
for k, v in count.items():
    if v > maximum_cluster_size:
        print(k)
        print(v)


new_data = {}
for k, v in data.items():
    ids = []
    for i in range(len(v)):
        id = "-".join(v[: i + 1])
        ids.append(id)
    new_data[k] = ids


ordered_positions = []
for i in tqdm(range(max([len(v) for v in new_data.values()]))):
    one_layer_positions = []
    for k, v in new_data.items():
        if len(v) > i:
            if v[i] not in one_layer_positions:
                one_layer_positions.append(v[i])
    one_layer_positions = sorted(one_layer_positions)
    ordered_positions += one_layer_positions


start = 0
renumber_map = {}
for p in ordered_positions:
    renumber_map[p] = start % maximum_cluster_size
    start += 1

renumbered_data = {k: [renumber_map[item] for item in v] for k, v in new_data.items()}


count = {}
for k, v in renumbered_data.items():
    if tuple(v) not in count:
        count[tuple(v)] = 1
    else:
        count[tuple(v)] += 1
for k, v in count.items():
    if v > maximum_cluster_size:
        print(k)
        print(v)

regroup = [(k, v) for k, v in renumbered_data.items()]
regroup = sorted(regroup, key=lambda x: x[1])


final_data = {}
start = 0
for k, v in regroup:
    if count[tuple(v)] > 1:
        final_data[k] = v + [start % maximum_cluster_size]
        start += 1
    else:
        final_data[k] = v


final_count = {}
for k, v in final_data.items():
    if tuple(v) not in final_count:
        final_count[tuple(v)] = 1
    else:
        final_count[tuple(v)] += 1
for k, v in final_count.items():
    if v > 1:
        print(k)
        print(v)


with open("computed_optimal_{}_CF_index.json".format(maximum_cluster_size), "w") as f:
    json.dump(final_data, f)


with open(
    "c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size), "r"
) as f:
    data = json.load(f)


count = {}
for k, v in data.items():
    if tuple(v) not in count:
        count[tuple(v)] = 1
    else:
        count[tuple(v)] += 1
for k, v in count.items():
    if v > maximum_cluster_size:
        print(k)
        print(v)


reformed_item_CF_index_map = {}
for item, labels in data.items():
    reformed_item_CF_index_map[item] = [
        "-".join(labels[: i + 1]) for i in range(len(labels))
    ]


enumeration_by_group = {}
full_index = {}
vocab = []
for item, clusters in reformed_item_CF_index_map.items():
    if tuple(clusters) not in enumeration_by_group:
        full_index[item] = "".join(
            ["<A" + c + ">" for c in clusters] + ["<A0" + str(0) + ">"]
        )
        enumeration_by_group[tuple(clusters)] = 1
        vocab += ["<A" + c + ">" for c in clusters] + ["<A0" + str(0) + ">"]
    else:
        full_index[item] = "".join(
            ["<A" + c + ">" for c in clusters]
            + ["<A0" + str(enumeration_by_group[tuple(clusters)]) + ">"]
        )
        vocab += ["<A" + c + ">" for c in clusters] + [
            "<A0" + str(enumeration_by_group[tuple(clusters)]) + ">"
        ]
        enumeration_by_group[tuple(clusters)] += 1


with open(
    "computed_{}_{}_CF_index.json".format(
        number_of_clusters, maximum_cluster_size
    ),
    "w",
) as f:
    json.dump([full_index, vocab], f)


with open(
    "c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size), "r"
) as f:
    data = json.load(f)


count = {}
for k, v in data.items():
    if tuple(v) not in count:
        count[tuple(v)] = 1
    else:
        count[tuple(v)] += 1
for k, v in count.items():
    if v > maximum_cluster_size:
        print(k)
        print(v)


final_data = {}
met_category = {}
for k, v in data.items():
    if count[tuple(v)] > 1:
        if tuple(v) in met_category:
            met_category[tuple(v)] += 1
        else:
            met_category[tuple(v)] = 0
        i = met_category[tuple(v)]
        final_data[k] = v + ["0" + k]
    else:
        final_data[k] = v


computed_data = {}
vocabulary = []
for k, v in final_data.items():
    final_string = ""
    for i in range(len(v)):
        if i != len(v) - 1:
            final_string += "<A{}>".format("-".join(v[: i + 1]))
            vocabulary.append("<A{}>".format("-".join(v[: i + 1])))
        else:
            final_string += "<A{}>".format(v[i])
            vocabulary.append("<A{}>".format(v[i]))
    computed_data[k] = final_string


with open(
    "computed_no_repetition_{}_{}_CF_index.json".format(
        number_of_clusters, maximum_cluster_size
    ),
    "w",
) as f:
    json.dump([computed_data, vocabulary], f)
