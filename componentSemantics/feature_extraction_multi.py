import glob
import sys
import traceback

from more_itertools import chunked

from features.extract import extract
from features.features import *
from utils import load_stopwords

from joblib import Parallel, delayed


def extract_features(in_path, repo_path, out_path, feature, workers):
    projects = sorted([project for project in os.listdir(in_path)
                       if os.path.isdir(os.path.join(in_path, project))])

    chunks = list(chunked(projects, len(projects) // workers))

    Parallel(n_jobs=workers)(delayed(extract)(repo_path, feature, in_path, out_path, False, chunks[i])
                             for i in range(workers))


if __name__ == '__main__':
    arcan_path = sys.argv[1]
    repo_path = sys.argv[2]
    out_path = sys.argv[3]
    feature = sys.argv[4]
    splits = int(sys.argv[5])
    extract_features("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics2/data/arcanOutput",
                     "/media/cezarsas/Data/PyCharmProjects/ComponentSemantics2/data/repositories/",
                     "../data_test/",
                     feature,
                     splits)
