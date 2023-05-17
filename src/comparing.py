import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)
NUM_NEIGHBORS = config['NUM_NEIGHBORS']


def closest(query_emb, lookup_emb, num_neighbors):
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    distances = cosine_similarity(query_emb, lookup_emb)
    abs_distances = np.abs(distances)
    min_dist_indices = np.argsort(-abs_distances)
    min_dist_indices = min_dist_indices[:, :num_neighbors]
    minimal_dist = []
    for i in range(min_dist_indices.shape[0]):
        min_dist = abs_distances[i][min_dist_indices[i]]
        minimal_dist.append(min_dist)
    minimal_dist = np.vstack(minimal_dist)
    return min_dist_indices, minimal_dist


def find_neighbors(query_emb, lookup_emb, lookup_table):
    min_dist_indices, min_dist = closest(query_emb, lookup_emb, NUM_NEIGHBORS)
    list_terms = []
    for i in range(min_dist_indices.shape[0]):
        terms = lookup_table.iloc[min_dist_indices[i]]
        terms['distances'] = min_dist[i]
        list_terms.append(terms)
    terms = pd.concat(list_terms)

    return terms


def coverage(query_embeddings, lookup_embeddings):
    distance_matrix = cosine_similarity(query_embeddings, lookup_embeddings)
    return distance_matrix
