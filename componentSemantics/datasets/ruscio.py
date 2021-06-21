import glob
import os


def read(path):
    categories = glob.glob(os.path.join(path, '*', ''))

    documents = []
    labels = []
    for category in categories:
        samples = glob.glob(os.path.join(path, category, '*.txt'))
        label = category.split('/')[-2]
        for sample in samples:
            with open(sample, 'rt', encoding='utf8') as inf:
                documents.append(' '.join(inf.readlines()))
                labels.append(label)

    return documents, labels


def extract_csv(text, lab, out_path):
    import pandas

    df = pandas.DataFrame({'text': text, 'label': lab})

    X = df['text']

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(X)
    X_mat = vectorizer.transform(X)
    X_mat = X_mat.todense()

    out = pandas.DataFrame(X_mat, columns=vectorizer.get_feature_names())
    out.rename(columns={"data": "xdatax"})
    out['y'] = df['label']

    out.to_csv(out_path, index=False)


if __name__ == '__main__':
    a, b = read('/home/sasce/Downloads/ten_folder_100/all')
    extract_csv(a,b, 'ruscio.csv')
