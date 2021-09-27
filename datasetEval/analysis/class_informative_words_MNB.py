import csv
import warnings
from collections import defaultdict
from pprint import pprint

import numpy as np
import shap
from joblib import Memory, Parallel, delayed
from multiset import Multiset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

from datasetEval.analysis.class_informative_words import plot_heatmap, save_info_words, save_intersections, \
    save_heatmap, TFIDF, get_intersections
from datasetEval.dataloader import diSipio, lact

memory = Memory('cache', verbose=0)


def analyze(model, explainer_class, X, y, feature_names, top_info_words=50, sample=10):
    # X, y = resample(X, y, stratify=y, n_samples=sample)
    explainer = explainer_class(model.predict_proba, X)

    y_map = {}
    for i in set(y):
        if i not in y_map:
            y_map[i] = len(y_map)

    # pprint(features)
    informative_words = defaultdict(lambda: Multiset())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # X, y = resample(X, y, stratify=y, n_samples=sample)
        #for i in range(0, len(X), sample):
            #X_sample = X[i:i + sample] if i + sample < len(X) else X[i:]
            #y_sample = y[i:i + sample] if i + sample < len(X) else y[i:]

        res = Parallel(n_jobs=3)(delayed(extract_info_words)(explainer, model, X, y, feature_names, y_map,
                                                   top_info_words, i, sample) for i in range(0, len(X), sample))

        for r in res:
            for i in r:
                informative_words[i].update(r[i])

    return informative_words


def extract_info_words(explainer, model, X, y, feature_names, y_map, top_info_words, i, sample):
    X_sample = X[i:i + sample] if i + sample < len(X) else X[i:]
    y_sample = y[i:i + sample] if i + sample < len(X) else y[i:]
    informative_words = defaultdict(lambda: Multiset())
    shap_values = explainer.shap_values(X_sample)
    predicted_y = model.predict(X_sample)
    # shap_interaction_values = explainer.shap_interaction_values(features)
    expected_value = explainer.expected_value
    # print('expected', expected_value)
    for sample_id in range(len(shap_values[0])):
        y_id = y_map[y_sample[sample_id]]
        y_tilde = predicted_y[sample_id]
        if y_id == y_map[y_tilde]:
            ind = np.argpartition(shap_values[y_id][sample_id], -top_info_words)[-top_info_words:]
            sample_features = [feature_names[i] for i in ind
                               # if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
                               if
                               shap_values[y_id][sample_id][
                                   i] > 0]  # + expected_value[y_id] > np.mean(expected_value)]
            # sample_features_1 = [feature_names[i] for i in ind if shap_values[y_id][sample_id][i] + expected_value[y_id] > np.mean(expected_value)]
            # assert sample_features_1 == sample_features

            informative_words[y_sample[sample_id]].update(sample_features)
    return informative_words


def train(model, X, y, test_size=0.3, random_state=1337):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    trained_model = model().fit(X_train, y_train)
    performance = {'train_score': trained_model.score(X_train, y_train),
                   'test_score': trained_model.score(X_test, y_test)}
    trained_model = model().fit(X, y)
    performance['all_score'] = trained_model.score(X, y)
    return trained_model, performance


def analyze_all():
    # datasets = [(diSipio, {'path':'/home/sasce/Downloads/Classifications Dataset/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'}, 'sipio')]
    datasets = [
        # (lact, {'path': '../datasets/LACT/msr09-data/41-software-systems',
        #        'data': 'MUDA'}, 'MUDA'),
         (lact, {'path': '../datasets/LACT/msr09-data/43-software-systems',
                'data': 'LACT'}, 'LACT'),
        #(diSipio, {
        #    'path': '../datasets/Di Sipio/evaluation/evaluation structure/ten_folder_100/root'},
        # 'sipio')
    ]
    # models = [(xgboost.XGBClassifier, shap.TreeExplainer, 'xgb')]
    models = [(MultinomialNB, shap.KernelExplainer, 'MNB')]
    # models = [(DecisionTreeClassifier, shap.TreeExplainer, 'decisiontree')]
    n_features = 1000
    max_df = 0.9
    min_df = 0.1
    n_sample = 1
    feature_extractor = (
        TFIDF, {'max_features': n_features, "stop_words": {'english'},
                'max_df': max_df, 'min_df': min_df, 'token_pattern': r"(?u)\b\w\w{3,}\b"}, 'tfidf')

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
                for i in range(1):
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
            plot_heatmap(heatmap_name, intersection_matrix, labels)

            # wordcloud_name = f"{name}_bp_{mod_name}_{n_sample}_{feat_name}_{n_features}_{max_df}_{min_df}.pdf"
            # plot_wordcloud(wordcloud_name, informative_words, topn=5)


if __name__ == '__main__':
    analyze_all()
