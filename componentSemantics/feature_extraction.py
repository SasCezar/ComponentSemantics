import glob
import os

from tqdm import tqdm

from features.features import TfidfFeatureExtraction, PackageFeatureExtraction, DocumentFeatureExtraction
from utils import load_stopwords


def extract_features(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    path = "resources/java/stopwords.txt"
    stopwords = load_stopwords(path)

    features = [
        PackageFeatureExtraction(stopwords=stopwords),
        TfidfFeatureExtraction(stopwords=stopwords),
        DocumentFeatureExtraction(stopwords=stopwords)
    ]

    skipped = 0
    for feature in features:
        for project in tqdm(projects):
            filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))
            if not filepath:
                continue
            filepath = filepath[0]
            try:
                feature.extract(project, filepath, out_path)
            except:
                print(project)
                skipped += 1
                pass

        print(skipped)


if __name__ == '__main__':
    extract_features("../data/arcanOutput/", "../data/")
