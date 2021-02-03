import os

import numpy


def check_dir(path):
    project_path = os.path.join(path)
    if not os.path.exists(project_path):
        os.makedirs(project_path)


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
