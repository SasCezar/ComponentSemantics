import glob
import os
import re

import igraph
import sourcy
import spacy
from more_itertools import flatten
from nltk.corpus import stopwords
from tqdm import tqdm


def fix(param):
    return param.replace("/home/sasce/PycharmProjects/ComponentSemantics/data/projects/",
                         "/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data/repositories/")


def get_files(graph):
    paths = []
    for node in graph.vs:
        paths.append(fix(node["filePath"]))

    return paths


def split_camel(name):
    splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return splitted


def check_dir(path):
    project_path = os.path.join(path)
    if not os.path.exists(project_path):
        os.makedirs(project_path)


def split_camel(name):
    splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return splitted


def name_to_sentence(name):
    tokens = name.split(".")[2:]
    clean = []

    for token in tokens:
        clean.extend(split_camel(token))
        #clean.append(token)

    return [x.lower() for x in clean]
    #return clean

stop = set(stopwords.words('english'))

def check_vocab():
    in_path = "../data/arcanOutput/"
    out_path = "../data/embeddings/"
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project)) if "elasticsearch" not in project]

    scp = sourcy.load("java")

    # nlp = spacy.load("../../data/models/codebert-base")

    files_vocab = set()
    package_vocab = set()
    docs = set()
    for project in projects:
        print(project)
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]

        graph = igraph.Graph.Read_GraphML(filepath)
        files = [x for x in get_files(graph) if os.path.isfile(x)]

        packages = [name_to_sentence(x["name"]) for x in graph.vs]
        packages = flatten(packages)
        package_vocab.update(packages)
        for file in tqdm(files):
            filename = fix(file)
            with open(filename, "rt", encoding="utf8") as inf:
                text = inf.read()

            docs.add(text)
            doc = scp(text)

            ids = [split_camel(x.token) for x in doc.identifiers]
            #ids = [x.token for x in doc.identifiers]
            ids = [x for x in set(flatten(ids))]
            ids = [x for x in ids if x not in stop]

            files_vocab.update(ids)

    #nlp = spacy.load("en_trf_bertbaseuncased_lg")
    nlp = spacy.load("../../data/models/codebert-base")
    spacy_vocab = set(nlp.vocab.strings)
    print(list(spacy_vocab)[:10])
    spacy_tokens = set()

    for doc in tqdm(docs):
        tokens = nlp(doc)
        for token in tokens:
            spacy_tokens.update(token.text)
    package_int = spacy_vocab.intersection(package_vocab)
    files_int = spacy_vocab.intersection(files_vocab)
    spacy_int = spacy_vocab.intersection(spacy_tokens)
    print(package_vocab)
    #print(files_vocab)
    print("Vocab package", len(package_vocab), "Coverage", len(package_int), "Percentage", len(package_int)/len(package_vocab)*100)
    print("Files package", len(files_vocab), "Coverage", len(files_int), "Percentage", len(files_int)/len(files_vocab)*100)
    print("Files package", len(spacy_tokens), "Coverage", len(spacy_int), "Percentage", len(spacy_int)/len(spacy_tokens)*100)

check_vocab()