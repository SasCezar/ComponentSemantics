import csv

import pandas


def load_projects(path, level='1st'):
    df = pandas.read_csv(path)
    data = pandas.DataFrame()
    data['name'] = df["project.link"].apply(lambda x: x.strip("/").split("/")[-1])
    data['label'] = df[f'{level} level'].fillna("Miscellaneous")
    data = data.dropna()
    data = data[~data['name'].duplicated(keep="first")]
    data = data.set_index('name')

    projects = data.to_dict()['label']
    return projects


def load_files(projects, embeddings_path):
    documents = []
    labels = []

    for project in projects:
        try:
            doc = ''
            with open(embeddings_path + '/' + project + '.vec') as outf:
                reader = csv.reader(outf, delimiter=' ')
                for word, count in reader:
                    if len(word) > 3:
                        words = [word] * int(count)
                        doc = doc + ' '.join(words)
            if len(doc):
                documents.append(doc)
                labels.append(projects[project])
        except:
            continue

    return documents, labels


def read(path, embeddings_path, level):
    projects = load_projects(path, level)

    documents, labels = load_files(projects, embeddings_path)

    return documents, labels


if __name__ == '__main__':
    path = '/home/sasce/PycharmProjects/ComponentSemantics/componentSemantics/java-projects.final.csv'
    embeddings_path = '/home/sasce/PycharmProjects/ComponentSemantics/data/embeddings/terms-count'

    docs, lbls = read(path, embeddings_path, '3rd')
    print(len(lbls), len(set(lbls)))
    print(len(docs))