from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import numpy as np


def create_clustering(X, cls, **kwargs):
    algorithm = cls(**kwargs)
    prediction = algorithm.fit_predict(X)
    prediction = pd.DataFrame(prediction, columns=['cluster'])
    prediction['cluster'] = prediction['cluster'].astype("string")
    return prediction


def get_silhouette_score(X, labels):
    score = silhouette_score(X, labels, metric='euclidean')
    return score


def test_clusters(X, cls, test_grid, **kwargs):
    scores = []
    for name, value in test_grid:
        prediction = create_clustering(X, cls, **kwargs, name=value)
        labels = prediction.values
        silhouette_score = get_silhouette_score(X, labels)
        scores.append(silhouette_score)
    return scores


def grid_search_clustering(embeddings, algorithm, params):
    results = []

    for param_set in ParameterGrid(params):
        if algorithm == 'kmeans':
            clustering = KMeans(n_clusters=param_set['n_clusters'], random_state=0)
        elif algorithm == 'dbscan':
            clustering = DBSCAN(eps=param_set['eps'], min_samples=param_set['min_samples'])
        else:
            raise ValueError("Invalid algorithm name. Must be either 'kmeans' or 'dbscan'.")

        clustering.fit(embeddings)
        labels = clustering.labels_

        if -1 in labels:
            # Exclude noise points (-1) from silhouette score calculation for DBSCAN
            score = silhouette_score(embeddings, labels, metric='euclidean', sample_size=None, random_state=0)
        else:
            score = silhouette_score(embeddings, labels)

        result = {
            'params': param_set,
            'silhouette_score': score,
            'labels': labels
        }

        results.append(result)

    return results


if __name__ == '__main__':
    df = pd.read_csv('data/frames_encoding.csv')
    X = df.iloc[:, 3:].values
    kmeans_params = {'n_clusters': [80, 90, 100]}
    dbscan_params = {'eps': [2.0, 2.5, 3.0, 4.0, 5.0], 'min_samples': [2]}

    # Perform grid search for K-means
    kmeans_results = grid_search_clustering(X, 'kmeans', kmeans_params)

    # Perform grid search for DBSCAN
    dbscan_results = grid_search_clustering(X, 'dbscan', dbscan_params)


    # Print the results for K-means
    print("K-means results:")
    for result in kmeans_results:
        print("Parameters:", result['params'])
        print("Silhouette Score:", result['silhouette_score'])
        print("Labels:", result['labels'])
        print()

    # Print the results for DBSCAN
    print("DBSCAN results:")
    for result in dbscan_results:
        print("Parameters:", result['params'])
        print("Silhouette Score:", result['silhouette_score'])
        print("Labels:", result['labels'])
        print()
