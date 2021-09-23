def read(path, data='LACT'):
    documents = []
    labels = []
    names_path = path + ".names"
    content_path = path + ".corpus"
    with open(names_path, 'rt') as names_f, \
            open(content_path) as content_f:
        next(content_f)
        for name_line, content_line in zip(names_f, content_f):
            n_id, label_name = name_line.split("   ")
            if data == 'LACT':
                prog, label, name = label_name.split("_", maxsplit=2)
            else:
                label, name = label_name.split("_", maxsplit=1)

            documents.append(content_line)
            labels.append(label)

    return documents, labels