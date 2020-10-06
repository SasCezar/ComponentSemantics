import glob
import os
import re

import igraph
import sourcy
import spacy

from more_itertools import flatten
from tqdm import tqdm


from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

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


def extract_features():
    in_path = "../../data/arcanOutput/"
    out_path = "../../data/embeddings/"
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project)) if "elasticsearch" not in project]

    scp = sourcy.load("java")
    use_gpu = spacy.require_gpu()

    # nlp = spacy.load("../../data/models/codebert-base")
    nlp = spacy.load("en_trf_bertbaseuncased_lg")

    for project in projects:
        print(project)
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))[0]

        graph = igraph.Graph.Read_GraphML(filepath)
        files = [x for x in get_files(graph) if os.path.isfile(x)]
        check_dir(out_path)
        with open(os.path.join(out_path, f"{project}.vec"), "wt", encoding="utf8") as outf:
            for file in tqdm(files):
                filename = fix(file)
                with open(filename, "rt", encoding="utf8") as inf:
                    text = inf.read()

                doc = scp(text)

                ids = [split_camel(x.token) for x in doc.identifiers]
                ids = [x.lower() for x in set(flatten(ids))]
                ids = [x for x in ids if x not in stop]
                doc_embeddings = nlp(" ".join(ids)).vector

                rep = " ".join([str(x) for x in doc_embeddings.tolist()])
                line = file + " " + rep + "\n"
                outf.write(line)


if __name__ == '__main__':
    extract_features()
