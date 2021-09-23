import glob
import os
import re

import ftfy


def read(path):
    categories = glob.glob(os.path.join(path, '*', ''))

    documents = []
    labels = []
    for category in categories:
        samples = glob.glob(os.path.join(path, category, '*.txt'))
        label = category.split('/')[-2]
        for sample in samples:
            with open(sample, 'rt') as inf:
                text = ftfy.fixes.decode_escapes(' '.join(inf.readlines()))
                #text = re.sub('`\\\\x(\w)?\d+', ' ', text)
                #text = inf.readlines()
                documents.append(text)
                labels.append(label)

    return documents, labels