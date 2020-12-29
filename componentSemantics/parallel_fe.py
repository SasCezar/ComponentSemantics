import functools
import multiprocessing
import sys
import time

from more_itertools import chunked

from features.extract import extract
from features.features import *


def extract_features(in_path, repo_path, out_path, feature):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    num_workers = 6

    chunks = chunked(projects, len(projects) / 2)

    extract_partial = functools.partial(extract, repo_path, feature, in_path, out_path, False)
    pool = multiprocessing.Pool(num_workers)

    start = time.time()
    for res in pool.imap_unordered(extract_partial, chunks):
        print("Skipped {} (Time elapsed: {}s)".format(res, int(time.time() - start)))

    pool.terminate()


if __name__ == '__main__':
    feature = sys.argv[1]
    extract_features("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics2/data/arcanOutput",
                     "/media/cezarsas/Data/PyCharmProjects/ComponentSemantics2/data/repositories/",
                     "../data_test/",
                     feature)
