import csv

import pandas


def get_avarage_intersections(file):
    df = pandas.read_csv(file, header=0, names=['dataset', 'model', 'label1', 'label2', 'intersection'])
    df_cp = df.copy(deep=True)
    df.cp = df_cp.rename(columns={'label2': 'label1', 'label1': 'label2'})

    joined = df.append(df_cp)
    avgs = joined.groupby(['label1']).mean()

    res = avgs.to_dict()['intersection']

    return res


def load_label_sim(file):
    similarities = pandas.read_csv(file)
    similarities = similarities[['label', 'score']]
    similarities = similarities.set_index('label')

    res = similarities.to_dict()['score']

    return res


def save_int_sim(path, class_avg_intersections, class_label_similarity):
    with open(path, 'wt') as outf:
        writer = csv.writer(outf)
        labels = set(class_label_similarity).union(set(class_avg_intersections.keys()))
        writer.writerow(['dataset', 'classifier', 'embeddings', 'label', 'intersection', 'sim'])
        for label in labels:
            intersection = class_avg_intersections[label] if label in class_avg_intersections else 0
            sim = class_label_similarity[label] if label in class_label_similarity else 0

            writer.writerow([dataset, classifier, model, label, intersection, sim])


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
model = 'SO'

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
                try:
                    class_avg_intersections = get_avarage_intersections(path.replace('words', 'intersections'))
                    class_label_similarity = load_label_sim(path.replace('.csv', f'_label_scores_{model}.csv'))

                    save_int_sim(path.replace('.csv', f'_labels_sim_uniq_{model}.csv'), class_avg_intersections,
                                 class_label_similarity)
                except Exception as e:
                    print(e)
                    print('Skipped', dataset)
