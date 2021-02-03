from os import listdir
from os.path import isfile, join
import numpy as np

if __name__ == '__main__':
    path = "../data_out/embeddings/VocabCount"
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    size = []
    with open("all.txt", "wt", encoding="utf8") as outf:
        outf.write("size" + "\n")
        for file in files:
            with open(file, "rt", encoding="utf8") as inf:
                try:
                    num = inf.readlines()[0]
                except:
                    continue
                outf.write(num + "\n")
                size.append(int(num))

    mean = np.mean(size)
    std = np.std(size)
    print(mean, std, len(size))
