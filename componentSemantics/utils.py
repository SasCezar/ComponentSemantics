import os

import numpy
import pandas

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


def load_embeddings(path, skip=0) -> dict:
    embeddings = dict()
    with open(path, "rt", encoding="utf8") as inf:
        if skip:
            next(inf)
        for line in inf:
            splitLines = line.split()
            word = splitLines[0]
            embedding = numpy.array([float(value) for value in splitLines[1:]])
            embeddings[word] = embedding

    return embeddings


def load_projects(path, filename):
    df = pandas.read_csv(os.path.join(path, filename))
    data = pandas.DataFrame()
    data['names'] = df["project.link"].apply(lambda x: x.strip("/").split("/")[-1])
    data['labels'] = df['category.name'].fillna("NA")
    data = data.dropna()
    data = data[~data['names'].duplicated(keep="first")]

    categorical = pandas.Categorical(data["labels"], ordered=False)
    data["labels_id"] = categorical.codes
    mapping = dict(zip(data["labels"], data["labels_id"]))

    return data, mapping

def load_projects_new(path, filename, level='1st level'):
    df = pandas.read_csv(os.path.join(path, filename))
    data = pandas.DataFrame()
    data['names'] = df["project.link"].apply(lambda x: x.strip("/").split("/")[-1])
    data['labels'] = df[level].fillna("NA")
    data = data.dropna()
    data = data[~data['names'].duplicated(keep="first")]

    categorical = pandas.Categorical(data["labels"], ordered=False)
    data["labels_id"] = categorical.codes
    mapping = dict(zip(data["labels"], data["labels_id"]))

    return data, mapping