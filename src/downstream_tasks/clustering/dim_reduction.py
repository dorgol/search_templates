from sklearn.manifold import TSNE
import pandas as pd


def tsne_data(embeddings, n_components=2):

    tsne = TSNE(n_components=n_components, random_state=0)
    projections = tsne.fit_transform(embeddings)
    projections = pd.DataFrame(projections, columns=['x', 'y'])
    return projections
