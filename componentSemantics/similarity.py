import glob
import os
import re

import igraph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE

from utils import check_dir

matplotlib.matplotlib_fname()
import numpy

sns.set_theme(style="ticks")


def load_embeddings(path):
    embeddings = {}
    with open(path, "rt", encoding="utf8") as inf:
        for line in inf:
            splitLines = line.split()
            word = splitLines[0]
            embedding = numpy.array([float(value) for value in splitLines[1:]])
            embeddings[word] = embedding

    return embeddings


def plot_heatmap(maxtrix, project, method, embedding, out):
    mask = numpy.zeros_like(maxtrix)
    mask[numpy.triu_indices_from(mask)] = True

    ax = plt.axes()

    seaborn.heatmap(maxtrix, mask=mask)  # , vmin=0.70, vmax=0.99)
    # ax.set_title(f"{project} - {method} - {embedding}")
    out = os.path.join(out, f"heatmap_{project}_{method}_{embedding}.pdf")
    plt.savefig(out)
    plt.clf()


def plot_seaborns(df, project, method, embedding, out):
    ax = plt.axes()
    import matplotlib
    colors = {}
    classes = set(df['y'].tolist())

    for i, color in zip(classes, igraph.drawing.colors.ClusterColoringPalette(len(classes))):
        colors[str(i)] = matplotlib.colors.to_hex(color)

    fig = seaborn.scatterplot(data=df, x="PC1", y="PC2", hue="y", palette=colors)
    if len(colors) > 12:
        fig.legend_.remove()
        # plt.legend([],[], frameon=False)

    # ax.set_title(f"{project} - {method} - {embedding}")
    out = os.path.join(out, f"TSNE_{project}_{method}_{embedding}.pdf")
    plt.savefig(out)
    plt.clf()


def visualize(embeddings, classes, project, method, embedding, out):
    points = TSNE(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(points, columns=["PC1", "PC2"])
    df["y"] = classes
    plot_seaborns(df, project, method, embedding, out)


def community_features(path, embeddings, level):
    level = {"package": "name", "document": "filePath", "TFIDF": "filePath", "CodeGPT": "filePath",
             "BERT-Tokens": "filePath"}[level]

    x = os.path.join(path, "**", "*.graphml")
    files = glob.glob(x, recursive=True)

    communities = []
    features = []
    skipped = []

    for file in files:
        i = str(re.search("comm_(\d+).graphml", file, re.IGNORECASE).group(1))
        doc_emb = []

        subcommunity = igraph.Graph.Read_GraphML(file)

        for node in subcommunity.vs:
            if "." == node[level]:
                continue

            node_features = numpy.array(embeddings[node[level]]).astype(numpy.float)
            doc_emb.append(node_features)
            features.append(node_features)
            communities.append(i)

        if not doc_emb:
            skipped.append(i)

    data = pd.DataFrame(zip(features, communities), columns=["features", "classes"])
    return data, skipped


def aggregate(data, method="sum"):
    if method == "mean":
        data = data.groupby('classes')["features"].apply(np.mean).reset_index(name='features')
    elif method == "sum":
        data = data.groupby('classes')["features"].apply(np.sum).reset_index(name='features')
    else:
        raise ValueError("Method of aggregation not defined.")

    data["classes"] = pd.to_numeric(data["classes"])
    return data.sort_values("classes").set_index("classes", drop=False)


def load_dependencies(path, skipped):
    dependencies = numpy.loadtxt(path, dtype=int, delimiter=",")
    norm_dependencies = dependencies / dependencies.sum().sum()
    skipped = np.array(skipped).astype(int)
    dependencies = norm_dependencies
    dependencies = numpy.delete(dependencies, skipped, axis=0)
    dependencies = numpy.delete(dependencies, skipped, axis=1)
    return dependencies


def communities_similarities(features):
    avgs = []
    for i, community in features.groupby("classes"):
        comm_feat = sklearn.metrics.pairwise.cosine_similarity(community["features"].tolist())
        iterate_indices = numpy.tril_indices(comm_feat.shape[0])
        n = 0
        tot = 0
        for r, c in zip(*iterate_indices):
            tot += comm_feat[r, c]
            n += 1

        mean = tot / n
        avgs.append(mean)
    return numpy.mean(avgs), numpy.std(avgs)


def main(project, method, embedding):
    embedding_path = f"../data/embeddings/{embedding}/{project}.vec"
    embeddings = load_embeddings(embedding_path)
    graph = f"../data/graphs/{method}/raw/{project}/"

    plot_out = f"../data/plots/analysis/"
    check_dir(plot_out)

    features, skipped = community_features(graph, embeddings, embedding)

    visualize(features['features'].tolist(), features["classes"].tolist(), project, method, embedding, plot_out)

    aggregated_features = aggregate(features)
    similarities = sklearn.metrics.pairwise.cosine_similarity(
        numpy.array(aggregated_features["features"].tolist()))
    plot_heatmap(similarities, project, method, embedding, plot_out)

    path = f"../data/graphs/projects/{project}/comm_dependencies_{method}.csv"

    dependencies = load_dependencies(path, skipped)
    assert dependencies.shape == similarities.shape, print(dependencies.shape, similarities.shape)

    dep_sim, sims = get_depsim(dependencies, similarities)

    communities_sim = communities_similarities(features)
    print("Community Similarity", f"{communities_sim[0]:.4f}\pm{communities_sim[1]:.4f}")
    print("Project", f"{numpy.mean(sims):.4f}\pm{numpy.std(sims):.4f}")
    glob_sims = sklearn.metrics.pairwise.cosine_similarity(features["features"].tolist())

    iterate_indices = numpy.tril_indices(glob_sims.shape[0])

    tot = []
    for r, c in zip(*iterate_indices):
        tot.append(glob_sims[r, c])

    print("GLOBAL Project", f"{numpy.mean(glob_sims):.4f}\pm{numpy.std(glob_sims):.4f}")

    cosine_distance = sklearn.metrics.pairwise.cosine_distances(features["features"].tolist())

    silhouette = sklearn.metrics.silhouette_score(cosine_distance, features['classes'].tolist(),
                                                  metric="precomputed")

    print(f"Similarity Silhouette {silhouette:.4f}")

    df = pd.DataFrame(dep_sim, columns=["similarity", "dependency"])
    corr = df.corr()
    print(f"Correlation {corr['similarity'][1]:.4f}")


def get_depsim(dependencies, similarities):
    dep_sim = []
    sims = []
    iterate_indices = numpy.tril_indices(dependencies.shape[0])
    col_skip = np.invert(dependencies.any(axis=0))
    row_skip = np.invert(dependencies.any(axis=1))
    for i, j, in zip(*iterate_indices):
        if i == j and col_skip[i] and row_skip[j]:
            continue

        simil = similarities[i, j]
        sumx = dependencies[i, j] + dependencies[j, i]
        dep_sim.append((sumx, simil))
        sims.append(similarities[i, j])

    return dep_sim, sims


if __name__ == '__main__':
    methods = ["leiden", "infomap"]
    embeddings = ["TFIDF"]
    for embedding in embeddings:
        for method in methods:
            for project in ["antlr4", "avro", "openj9"]:
                print("Processing", project, method, embedding)
                main(project, method, embedding)
                print("=" * 60)
