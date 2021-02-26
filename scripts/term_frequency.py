import csv
import glob
import sys
import traceback

from tqdm import tqdm

from features.features import *
from utils import load_embeddings


def extract_features(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    feature = AllTermCount()

    skipped = 0
    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))
        if not filepath:
            continue
        filepath = filepath[0]
        try:
            feature.extract(project, filepath, out_path)
        except:
            traceback.print_exc(file=sys.stdout)
            print(project)
            skipped += 1
            pass

    print(skipped)

    with open('terms_count.txt', 'wt', encoding='utf8') as outf:
        writer = csv.writer(outf, delimiter=' ')
        for term, count in feature.counter.most_common():
            row = [term, count]
            writer.writerow(row)

def count(in_path):
    projects = [os.path.join(in_path, project) for project in os.listdir(in_path)
                if os.path.isfile(os.path.join(in_path, project))]

    counter = Counter()
    for file in projects:
        terms = load_embeddings(file).keys()
        counter.update(terms)

    with open('word_in_projects.txt', 'wt', encoding='utf8') as outf:
        writer = csv.writer(outf, delimiter=' ')
        for term, count in counter.most_common():
            writer.writerow([term, count])

if __name__ == '__main__':
    #extract_features("../data/arcanOutput/", "../data/")
    count('../data/embeddings/terms-count/')
