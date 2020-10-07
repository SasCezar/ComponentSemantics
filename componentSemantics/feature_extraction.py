import os
import re
from abc import abstractmethod, ABC

import sourcy
import spacy
from more_itertools import flatten
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np

class FeatureExtraction(ABC):
    def __init__(self, model="en_trf_bertbaseuncased_lg"):
        self.nlp = spacy.load(model)
        self.level = ""

    @abstractmethod
    def get_embeddings(self, graph):
        pass

    @staticmethod
    def split_camel(name):
        splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        return splitted

    @staticmethod
    def save_features(features, path, file):
        out = os.path.join(path, file)
        with open(out, "wt", encoding="utf8") as outf:
            for name, cleanded, embedding in features:
                rep = " ".join([str(x) for x in embedding.tolist()])
                line = name + " " + rep + "\n"
                outf.write(line)


class PackageFeatureExtraction(FeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg"):
        super().__init__(model)
        self.nlp = spacy.load(model)
        self.level = "package"

    def get_embeddings(self, graph):
        for node in tqdm(graph.vs, leave=False):
            name = node['name']
            name, clean = self.name_to_sentence(name)

            if not clean:
                clean = node['name']

            embedding = self.nlp(clean).vector
            yield name, clean, embedding

    def name_to_sentence(self, name):
        tokens = name.split(".")[2:]
        clean = []

        for token in tokens:
            clean.extend(self.split_camel(token))

        return name, " ".join(clean).lower()


class DocumentFeatureExtraction(FeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg"):
        super().__init__(model)
        self.nlp = spacy.load(model)
        self.level = "document"
        self.scp = sourcy.load("java")
        self.stop = set(stopwords.words('english'))

    def get_embeddings(self, graph):
        for node in tqdm(graph.vs, leave=False):
            path = node['filePath']
            if not os.path.isfile(self.fix(path)):
                continue
            text = self.read_file(self.fix(path))

            doc = self.scp(text)

            ids = [self.split_camel(x.token) for x in doc.identifiers]
            ids = [x.lower() for x in set(flatten(ids))]
            ids = [x for x in ids if x not in self.stop]
            embedding = self.nlp(" ".join(ids)).vector

            yield path, path, embedding

    def name_to_sentence(self, name):
        tokens = name.split(".")[2:]
        clean = []

        for token in tokens:
            clean.extend(self.split_camel(token))

        return name, " ".join(clean).lower()

    @staticmethod
    def read_file(filename):
        with open(filename, "rt", encoding="utf8") as inf:
            text = inf.read()

        return text

    @staticmethod
    def fix(path):
        return path.replace("SemanticGraphEmbedding", "ComponentSemantics")


class TfidfFeatureExtraction(DocumentFeatureExtraction):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.level = "TFIDF"

    def get_embeddings(self, graph):
        documents = []
        files = []
        for node in tqdm(graph.vs, leave=False):
            path = node['filePath']
            if not os.path.isfile(self.fix(path)):
                continue
            text = self.read_file(self.fix(path))

            doc = self.scp(text)

            ids = [self.split_camel(x.token) for x in doc.identifiers]
            ids = [x.lower() for x in set(flatten(ids))]
            ids = [x for x in ids if x not in self.stop]

            doc = self.nlp(" ".join(ids))

            documents.append(" ".join([token.lemma_ for token in doc]))
            files.append(path)

        X = self.vectorizer.fit_transform(documents).todense()
        for file, vector in zip(files, X):
            yield file, file, np.array(vector.tolist()[0])
