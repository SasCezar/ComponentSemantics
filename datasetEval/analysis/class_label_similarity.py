import csv

import fasttext as ft
import numpy as np
import pandas
from gensim.models import KeyedVectors
from sklearn import metrics


def clean(term):
    a = str(term).strip('$').replace("-", ' ').lower().split(" ")
    return a


def get_embeddings_label_so(term):
    words = clean(term)
    t = []
    skipped = 0
    for x in words:
        if x in lm:
            t.append(lm.get_vector(x.lower()))
        else:
            t.append(np.random.uniform(low=-1, high=1, size=(200,)))
            skipped += 1

    embedding = np.mean(t, axis=0)
    print('Skipped Labels', skipped)
    return [embedding], []


def get_embeddings_so(terms):
    embeddings = []
    skipped = 0
    mask = []
    for term in terms:
        if term in lm:
            embeddings.append(lm.get_vector(term))
            mask.append(True)
        else:
            skipped += 1
            mask.append(False)

    print('Skipped Words', skipped)
    return embeddings, mask


def get_embeddings_fasttext(terms):
    if isinstance(terms, str):
        terms = [terms]
    embeddings = []
    for term in terms:
        if isinstance(term, float):
            print(term, dataset)
        embedding = lm.get_sentence_vector(str(term))
        embeddings.append(embedding)

    return embeddings, [True] * len(embeddings)


#model = 'fasttext'
model = 'SO'
if model == 'fasttext':
    lm = ft.load_model('/home/sasce/PycharmProjects/ComponentSemantics/data/models/fastText/wiki.en.bin')
    get_embeddings = get_embeddings_fasttext
    get_embeddings_label = get_embeddings_fasttext
else:
    lm = KeyedVectors.load_word2vec_format('/home/sasce/PycharmProjects/ComponentSemantics/scripts/SO_vectors_200.bin',
                                           binary=True)
    get_embeddings = get_embeddings_so
    get_embeddings_label = get_embeddings_label_so


def load_words(path):
    df = pandas.read_csv(path)
    df_group = df.groupby('class')
    labels = dict(list(df_group))

    label_words = {}

    for label in labels:
        group = pandas.DataFrame(df_group.get_group(label))
        group.drop('class', axis=1, inplace=True)
        label_words[label] = list(group.to_records(index=False))

    return label_words


def get_label_score(label, words):
    label_emb, _ = get_embeddings_label(label)
    weights = [x[1] for x in words]
    words = [x[0] for x in words]

    words_emb, mask = get_embeddings(words)
    weights = [x for i, x in enumerate(weights) if mask[i]]
    similarities = metrics.pairwise.cosine_similarity(label_emb, words_emb)

    average = np.average(similarities[0], weights=weights)

    return average


def get_labels_scores(label_words):
    label_scores = {}
    for label in label_words:
        words = label_words[label]
        score = get_label_score(label, words)
        label_scores[label] = score

    return label_scores


def save_label_scores(path, words, scores):
    with open(path, 'wt') as outf:
        writer = csv.writer(outf)
        writer.writerow(['dataset', 'classifier', 'embeddings', 'label', 'score', 'words'])
        for label in scores:
            writer.writerow([dataset, classifier, model, label, scores[label], words[label]])


dataset = 'LACT'
classifier = 'xgb'
n_sample = 5
feat_name = 'tfidf'
n_features = 1000
max_df = 0.9
min_df = 0.1
datasets = ['MUDA', 'LACT', 'sipio', 'HiGitClassAI-0', 'HiGitClassAI-1', 'HiGitClassBIO-0',
            'HiGitClassBIO-1', 'OURS', 'AwesomeJava']
classifiers = ['xgb', 'MNB']
if __name__ == '__main__':
    for x in datasets:
        for k in classifiers:
            for s in [1, 5]:
                if k == 'MNB' and s == 5:
                    continue
                dataset = x
                classifier = k
                n_sample = s
                file = f'{dataset}_words_{classifier}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.csv'
                path = f'/home/sasce/PycharmProjects/ComponentSemantics/datasetEval/analysis/{file}'
                label_words = load_words(path)
                label_scores = get_labels_scores(label_words)
                save_label_scores(path.replace('.csv', f'_label_scores_{model}.csv'), label_words, label_scores)
