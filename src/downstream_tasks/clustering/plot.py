import numpy as np
import pandas as pd
import plotly.express as px
from src.downstream_tasks.clustering import dim_reduction as dr
from src.downstream_tasks.clustering import clustering as cl


def plot_tsne_data(embeddings, meta_data=None):

    df = dr.tsne_data(embeddings)
    df = pd.concat([df, meta_data], axis=1)

    fig = px.scatter(
        df, x='x', y='y',
        hover_name='cluster',
        hover_data=['cluster', 'name'] if meta_data is not None else None,
        title="Terms",
        color='cluster' if meta_data is not None else None,
        color_discrete_sequence=px.colors.qualitative.G10,
        template='plotly_white'

    )
    fig.update_layout(
        title_font_family="Poppins",
        title_font_color="black",
    )
    fig.update_layout(
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.show()

    # fig.write_html(OUTPUTS_PATH + plot_name + '.html')


def main():
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    df = pd.read_csv('data/frames_encoding.csv')
    X = df.iloc[:, 3:].values
    clusters = cl.create_clustering(X, KMeans, n_clusters=90)
    meta = pd.concat([pd.DataFrame(df.name, columns=['name']), clusters], axis=1).reset_index(drop=True)
    plot_tsne_data(X, meta)
    # plot_tsne_data(embeddings, terms, "search_and_fit", add_size=False)


if __name__ == '__main__':
    main()
