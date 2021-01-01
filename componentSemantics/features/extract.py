import glob
import subprocess
import sys
import traceback

from features.features import *
from utils import load_stopwords

features = {
    "package": "PackageFeatureExtraction(stopwords=stopwords)",
    "tfidf": "TfidfFeatureExtraction(stopwords=stopwords)",
    "BERT": "DocumentFeatureExtraction(stopwords=stopwords)",
    "fasttext": "FastTextExtraction(model='../data/models/fastText/wiki.en.bin', stopwords=stopwords)",
    "code2vec": "Code2VecExtraction(model='../data/models/code2vec/token_vecs.txt', stopwords=stopwords)"
}


def git_checkout(repo_path, project, sha):
    p = subprocess.Popen(["rm", "--force", "./.git/index.lock"], cwd=os.path.join(repo_path, project))
    p.wait()
    p = subprocess.Popen(["git", "checkout", "--force", sha], cwd=os.path.join(repo_path, project))
    p.wait()
    return


def extract(repo_path, feature_name, in_path, out_path, clean_graph, projects):
    stopwords = load_stopwords("resources/java/stopwords.txt")
    feature = eval(features[feature_name])

    skipped = 0
    print("Doing stuff", projects)
    for project in projects:
        filepaths = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))
        if not filepaths:
            continue
        for filepath in filepaths:
            try:
                base = os.path.basename(filepath)
                filename = os.path.splitext(base)[0]
                sha = filename.split("-")[-1]
                num = filename.split("-")[-2]
                git_checkout(repo_path, project, sha)
                feature.extract(project, filepath, out_path, sha=sha, num=num, clean_graph=clean_graph)
            except:
                traceback.print_exc(file=sys.stdout)
                print(project)
                skipped += 1
                pass

    return skipped
