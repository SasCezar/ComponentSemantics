import csv
import random
import warnings
from collections import defaultdict
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

from datasetEval.dataloader import lact, diSipio

memory = Memory('cache', verbose=0)


def analyze(model, explainer_class, X, y, feature_names, top_info_words=50, sample=0):
    explainer = explainer_class(model)
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

    # pprint(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features)
        # shap_interaction_values = explainer.shap_interaction_values(features)
        expected_value = explainer.expected_value
        # print('expected', expected_value)

    informative_words = defaultdict(lambda: Multiset())
    for sample_id in range(len(shap_values[0])):
        y_id = y_map[y[sample_id]]
        y_tilde = predicted_y[sample_id]
        if y_id == y_map[y_tilde]:
            ind = np.argpartition(shap_values[y_id][sample_id], -top_info_words)[-top_info_words:]
            sample_features = [feature_names[i] for i in ind
                               # if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
                               if
                               shap_values[y_id][sample_id][i] > 0]  # + expected_value[y_id] > np.mean(expected_value)]
            # sample_features_1 = [feature_names[i] for i in ind if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
            # assert sample_features_1 == sample_features

            informative_words[y[sample_id]].update(sample_features)

    return informative_words


@memory.cache
def TFIDF(dataset, **kwargs):
    corpus, y = dataset
    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(corpus).todense()
    features = vectorizer.get_feature_names()
    X = pd.DataFrame(X, columns=features)

    return X, y, features


def train(model, X, y, test_size=0.3, random_state=1337):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    trained_model = model(max_depth=5).fit(X_train, y_train)
    performance = {'train_score': trained_model.score(X_train, y_train),
                   'test_score': trained_model.score(X_test, y_test)}
    trained_model = model(max_depth=5).fit(X, y)
    performance['all_score'] = trained_model.score(X, y)
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


def plot_wordcloud(name, informative_words, topn=10):
    res = []
    for cl in informative_words:
        sorted_count = sorted(list(informative_words[cl].items()), key=lambda x: -x[1])
        words = sorted_count[:topn]
        if len(words) == 0:
            words.append(['', 0])
        for word in words:
            res.append({'word': word[0], 'label': cl, 'count': word[1]})
    data = pd.DataFrame(res)

    hm = sn.catplot(x="word", y="count",
                    hue="label", col="label",
                    data=data, kind="bar", sharex=False, dodge=False)

    hm.savefig(name, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()


def save_heatmap(name, intersection_matrix, labels):
    df = pd.DataFrame.from_records(intersection_matrix, columns=labels)
    df.to_csv(name, index=False)
    return


def save_info_words(path, informative_words):
    with open(path, 'wt') as outf:
        writer = csv.writer(outf)
        writer.writerow(['class', 'word', 'count'])
        for l in informative_words:
            words_count = sorted(list(informative_words[l].items()), key=lambda x: -x[1])
            for word, count in words_count:
                writer.writerow([l, word, count])


def save_intersections(path, dataset, model, intersection_matrix, labels):
    iterate_indices = np.tril_indices(intersection_matrix.shape[0], k=-1)
    ints = []
    for i, j in zip(*iterate_indices):
        ints.append([dataset, model, labels[i], labels[j], intersection_matrix[i, j]])

    with open(path, 'wt') as outf:
        writer = csv.writer(outf)
        for line in ints:
            writer.writerow(line)


def analyze_all():
    # datasets = [(diSipio, {'path':'/home/sasce/Downloads/Classifications Dataset/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'}, 'sipio')]
    datasets = [#(lact, {'path': '/home/sasce/Downloads/Classifications Dataset/LACT/msr09-data/41-software-systems',
                #        'data': 'MUDA'}, 'MUDA'),
                #(lact, {'path': '/home/sasce/Downloads/Classifications Dataset/LACT/msr09-data/43-software-systems',
                #        'data': 'LACT'}, 'LACT'),
                (diSipio, {
                    'path': '/home/sasce/Downloads/Classifications Dataset/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'},
                 'sipio')]
    models = [(xgboost.XGBClassifier, shap.TreeExplainer, 'xgb')]
    # models = [(DecisionTreeClassifier, shap.TreeExplainer, 'decisiontree')]
    n_features = 1000
    max_df = 0.9
    min_df = 0.1
    n_sample = 5
    feature_extractor = (
        TFIDF, {'max_features': n_features, 'max_df': max_df, 'min_df': min_df, 'token_pattern': r"(?u)\b\w\w{3,}\b"},
        'tfidf')
    for reader, args, name in datasets:
        dataset = reader.read(**args)
        feat_name = feature_extractor[2]
        X, y, feature_names = feature_extractor[0](dataset, **feature_extractor[1])
        for model, explainer, mod_name in models:
            informative_words = defaultdict(lambda: Multiset())
            scores_path = f'{name}_perf_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.csv'
            with open(scores_path, 'wt') as outf:
                writer = csv.writer(outf)
                writer.writerow(['run', 'train', 'test'])
                for i in range(n_sample):
                    trained_model, performance = train(model, X, y)
                    info_words = analyze(trained_model, explainer, X, y, feature_names)
                    for c in info_words:
                        informative_words[c].update(info_words[c])

                    print(performance)
                    writer.writerow([i, performance['train_score'],
                                     performance['test_score'], performance['all_score']])

            info_words_name = f'{name}_words_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.csv'
            save_info_words(info_words_name, informative_words)
            intersection_matrix, labels = get_intersections(informative_words)
            intersection_name = f'{name}_intersections_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.csv'
            save_intersections(intersection_name, name, mod_name, intersection_matrix, labels)

            pprint(informative_words)

            heatmap_name = f'{name}_hm_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.pdf'
            save_heatmap(heatmap_name.replace('.pdf', '.csv'), intersection_matrix, labels)
            # plot_heatmap(heatmap_name, intersection_matrix, labels)

            # wordcloud_name = f"{name}_bp_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.pdf"
            # plot_wordcloud(wordcloud_name, informative_words, topn=5)


def plot_heatmap(heatmap_name, intersection_matrix, labels):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    hm = sn.heatmap(data=intersection_matrix, xticklabels=list(range(len(labels))),
                    yticklabels=list(range(len(labels))), cbar=False)
    fig.savefig(heatmap_name, format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    analyze_all()
