import json


def read(path, level=0):
    documents = []
    labels = []
    with open(path, 'rt') as outf:
        for line in outf:
            obj = json.loads(line)
            documents.append(obj['text'])
            labels.append(obj['labels'][level])

        return documents, labels