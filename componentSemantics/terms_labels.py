import os
import sys
import traceback

from analysis.annotation import IdentifiersTokens
from utils import load_embeddings

if __name__ == '__main__':
    in_path = "../data/"
    out_path = "../data/"

    embedding = 'id-code2vec'
    labels = load_embeddings('../results/label_embedding.vec')
    analysis = IdentifiersTokens(in_path, out_path, labels, visualize=False)

    arc_in = os.listdir(os.path.join(in_path, "arcanOutput"))
    projects = [project for project in arc_in
                if os.path.isdir(os.path.join(in_path, "arcanOutput", project))]

    skipped = 0
    counted = 0

    sizes = []
    nums = []
    seen = set()

    for project in projects:
        print("Processing", project)
        try:
            res = analysis.analyze(project, embedding)

            counted += 1
        except:
            traceback.print_exc(file=sys.stdout)
            skipped += 1
            pass

    print("Counted", counted)
    print("Skipped", skipped)
