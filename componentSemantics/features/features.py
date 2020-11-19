import os
import re
from abc import abstractmethod, ABC
from collections import Counter

import numpy as np
import sourcy
import spacy
from more_itertools import flatten
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from csio.graph_load import ArcanGraphLoader
from utils import check_dir

import fasttext as ft


class FeatureExtraction(ABC):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method=None, stopwords=None):
        try:
            self.nlp = spacy.load(model, disable=["ner", "textcat", "parser"])
        except:
            pass
        self.method = method
        if not stopwords:
            stopwords = set()
        self.stopwords = stopwords

    @abstractmethod
    def get_embeddings(self, graph):
        raise NotImplemented()

    @staticmethod
    def split_camel(name):
        splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        return splitted

    @staticmethod
    def save_features(features, path, file):
        out = os.path.join(path, file)
        with open(out, "wt", encoding="utf8") as outf:
            for name, cleanded, embedding in features:
                if not isinstance(embedding, list):
                    embedding = embedding.tolist()
                rep = " ".join([str(x) for x in embedding])
                line = name + " " + rep + "\n"
                outf.write(line)

    def extract(self, project_name, graph_path, out_path):
        graph = ArcanGraphLoader().load(graph_path)
        features_out = os.path.join(out_path, "embeddings", self.method)
        features = self.get_embeddings(graph)
        check_dir(features_out)
        features_name = f"{project_name}.vec"
        self.save_features(features, features_out, features_name)


class PackageFeatureExtraction(FeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="package", stopwords=None):
        super().__init__(model, method, stopwords)
        self.nlp = spacy.load(model)

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
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="document", preprocess=True, stopwords=None):
        super().__init__(model, method, stopwords)
        self.scp = sourcy.load("java")
        self.preprocess = preprocess

    def get_embeddings(self, graph):
        for node in tqdm(graph.vs, leave=False):
            path = node['filePathReal']

            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            text = " ".join(identifiers)
            embedding = self.nlp(text).vector

            yield path, path, embedding

    @staticmethod
    def read_file(filename):
        with open(filename, "rt", encoding="utf8") as inf:
            text = inf.read()

        return text

    def get_identifiers(self, path):
        text = self.read_file(path)

        doc = self.scp(text)

        ids = [self.split_camel(x.token) for x in doc.identifiers]
        ids = [x.lower() for x in set(flatten(ids)) if x.lower() not in self.stopwords]

        return ids


class TfidfFeatureExtraction(DocumentFeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="TFIDF", stopwords=None):
        super().__init__(model, method, stopwords)
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def get_embeddings(self, graph):
        documents = []
        files = []
        for node in tqdm(graph.vs, leave=False):
            path = node['filePathReal']
            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            doc = self.nlp(" ".join(identifiers))

            documents.append(" ".join([token.lemma_ for token in doc]))
            files.append(path)

        X = self.vectorizer.fit_transform(documents).todense()
        for file, vector in zip(files, X):
            yield file, file, np.array(vector.tolist()[0])


class DocumentAndCommentsFeatureExtraction(DocumentFeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="comments", stopwords=None):
        super().__init__(model, method, stopwords)

    def get_identifiers(self, path):
        text = self.read_file(path)
        doc = self.scp(text)

        comments = [token.token for token in doc.comments]

        clean_comments = []
        for comment in comments:
            if "Copyright" in comment or "License" in comment:
                continue
            comment_doc = self.nlp(comment)
            clean_comment = [token.text for token in comment_doc if token.text not in self.stopwords
                             and str(token.text).isalpha()]

            clean_comments.extend(clean_comment)

        ids = [self.split_camel(x.token) for x in doc.identifiers]
        ids = [x.lower() for x in set(flatten(ids)) if x.lower() not in self.stopwords]
        clean_comments.extend(ids)

        return clean_comments


class WordFrequencies(DocumentFeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="WordCount", stopwords=None):
        super().__init__(model, method, stopwords)

    def get_embeddings(self, graph):
        for node in tqdm(graph.vs, leave=False):
            path = node['filePathReal']
            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            doc = self.nlp(" ".join(identifiers))

            wc = Counter([token.text for token in doc if token.text.lower() not in self.stopwords])

            embedding = wc.most_common()
            yield path, path, embedding


class FastTextExtraction(DocumentFeatureExtraction):
    def __init__(self, model="wiki.en.bin", method="fastText", preprocess=True, stopwords=None):
        super().__init__(model, method, stopwords)
        self.nlp = ft.load_model(model)
        self.scp = sourcy.load("java")
        self.preprocess = preprocess

    def get_embeddings(self, graph):
        for node in tqdm(graph.vs, leave=False):
            path = node['filePathReal']

            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            text = " ".join(identifiers)
            embedding = self.nlp.get_sentence_vector(text)

            yield path, path, embedding


class VocabCountFeatureExtraction(DocumentFeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="VocabCount", stopwords=None):
        super().__init__(model, method, stopwords)

    def get_embeddings(self, graph):
        vocab = set()
        for node in tqdm(graph.vs, leave=False):
            path = node['filePathReal']
            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            doc = self.nlp(" ".join(identifiers))

            vocab.update([token.lemma_ for token in doc])

        return vocab

    def extract(self, project_name, graph_path, out_path):
        graph = ArcanGraphLoader().load(graph_path)
        features_out = os.path.join(out_path, "embeddings", self.method)
        vocab = self.get_embeddings(graph)
        check_dir(features_out)
        features_name = f"{project_name}.vec"
        self.save_features(vocab, features_out, features_name)

    @staticmethod
    def save_features(features, path, file):
        out = os.path.join(path, file)
        with open(out, "wt", encoding="utf8") as outf:
            outf.write(str(len(features)))
