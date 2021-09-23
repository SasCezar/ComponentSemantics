import random
import warnings
from pprint import pprint

import numpy as np
import shap
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from multiset import Multiset
import seaborn as sn
import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from datasetEval.dataloader import lact, diSipio

memory = Memory('cache', verbose=0)


def analyze(model, explainer_class, X, y, feature_names, top_info_words=50, sample=0):
    rand_i = random.sample(range(0, len(X)), 100)
    X = X.iloc[rand_i]
    y = [y[i] for i in rand_i]
    explainer = explainer_class(model.predict_proba, X)
    predicted_y = model.predict(X)

    y_map = {}
    for i in set(y):
        if i not in y_map:
            y_map[i] = len(y_map)

    if sample:
        # Subsample the number of analyzed examples in the dataset
        select = random.sample(range(0, len(X)), sample)
        features = X.iloc[select]
    else:
        features = X

    pprint(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features)
        # shap_interaction_values = explainer.shap_interaction_values(features)
        expected_value = explainer.expected_value
        print('expected', expected_value)
    from collections import defaultdict

    informative_words = defaultdict(lambda: Multiset())
    for sample_id in range(len(shap_values[0])):
        y_id = y_map[y[sample_id]]
        y_tilde = predicted_y[sample_id]
        if y_id == y_map[y_tilde]:

            ind = np.argpartition(shap_values[y_id][sample_id], -top_info_words)[-top_info_words:]
            sample_features = [feature_names[i] for i in ind# if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
                               if shap_values[y_id][sample_id][i] > 0]  # + expected_value[y_id] > np.mean(expected_value)]
            # sample_features_1 = [feature_names[i] for i in ind if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
            # assert sample_features_1 == sample_features

            informative_words[y[sample_id]].update(sample_features)

    return informative_words


@memory.cache
def TFIDF(dataset, **kwargs):
    corpus, y = dataset
    clean_corpus = []
    for sample in corpus:
        clean_corpus.append(' '.join([x if len(x) > 3 else '' for x in sample.split(' ') ]))
    corpus = clean_corpus
    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(corpus).todense()
    features = vectorizer.get_feature_names()
    X = pd.DataFrame(X, columns=features)

    return X, y, features


def train(model, X, y, test_size=0.3, random_state=1337):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    trained_model = model().fit(X_train, y_train)
    performance = {'train_score': trained_model.score(X_train, y_train), 'test_score': trained_model.score(X_test, y_test)}
    trained_model = model().fit(X, y)
    return trained_model, performance


def get_intersections(info_words):
    labels = list(info_words.keys())
    n = len(labels)
    matrix = np.eye(n)
    indexes = np.tril_indices(n, k=-1)
    for i, j in zip(*indexes):
        a = info_words[labels[i]]
        b = info_words[labels[j]]
        intersection = a.intersection(b)
        union = a.union(b)
        IoU = len(intersection) / len(union) if len(union) else 0
        matrix[i, j] = IoU
        matrix[j, i] = IoU

    return matrix, labels


def analyze_all():
    datasets = [(diSipio, {'path':'/home/sasce/Downloads/Classifications Dataset/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'})]
    #datasets = [(lact, {'path': '/home/sasce/Downloads/Classifications Dataset/LACT/msr09-data/43-software-systems',
    #                    'data': 'LACT'})]
    models = [(MultinomialNB, shap.KernelExplainer)]

    #models = [(DecisionTreeClassifier, shap.TreeExplainer)]
    feature_extractor = (TFIDF, {'max_features': 500, 'max_df': 0.9, 'min_df': 0.1, 'token_pattern': r"(?u)\b\w\w{3,}\b"})
    for reader, args in datasets:
        dataset = reader.read(**args)
        X, y, feature_names = feature_extractor[0](dataset, **feature_extractor[1])
        for model, explainer in models:
            trained_model, performance = train(model, X, y)
            info_words = analyze(trained_model, explainer, X, y, feature_names)
            pprint(info_words)
            print(performance)
            intersection_matrix, labels = get_intersections(info_words)

            hm = sn.heatmap(data=intersection_matrix, xticklabels=labels, yticklabels=labels, annot=True)
            plt.show()


if __name__ == '__main__':
    analyze_all()
