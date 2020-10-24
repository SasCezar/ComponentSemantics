import os

import numpy


def check_dir(path):
    project_path = os.path.join(path)
    if not os.path.exists(project_path):
        os.makedirs(project_path)


clean_edges = ["isChildOf", "isImplementationOf", "nestedTo", "belongsTo", "implementedBy", "definedBy",
               "containerIsAfferentOf", "unitIsAfferentOf"]


def clean_graph(graph):
    graph.es['weight'] = graph.es['Weight']
    delete = [x.index for x in graph.vs if "$" in x['name']]
    graph.delete_vertices(delete)
    for edge_label in clean_edges:
        graph.es.select(labelE=edge_label).delete()

    graph.vs.select(_degree=0).delete()

    return graph


def load_stopwords(path):
    stopwords = set()
    with open(path, "rt", encoding="utf8") as inf:
        for line in inf:
            stopwords.add(line.strip())

    return stopwords


def load_embeddings(path):
    embeddings = {}
    with open(path, "rt", encoding="utf8") as inf:
        for line in inf:
            splitLines = line.split()
            word = splitLines[0]
            embedding = numpy.array([float(value) for value in splitLines[1:]])
            embeddings[word] = embedding

    return embeddings
