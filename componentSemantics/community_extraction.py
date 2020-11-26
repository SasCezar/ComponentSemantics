import glob
import os
import shutil
import sys
import traceback

from tqdm import tqdm

from community.community import Leiden, Infomap
from community.extraction import CommunityExtraction


def extract_communities(in_path, out_path):
    projects = [project for project in os.listdir(in_path)
                if os.path.isdir(os.path.join(in_path, project))]

    algorithms = {
        "leiden": Leiden(),
        "infomap": Infomap()
    }

    extractor = CommunityExtraction(algorithms)
    skipped = 0
    for project in tqdm(projects):
        filepath = glob.glob(os.path.join(in_path, project, "dep-graph-*.graphml"))
        if not filepath:
            continue
        filepath = filepath[0]
        try:
            extractor.extract(project, filepath, out_path)
        except:
            traceback.print_exc(file=sys.stdout)
            print(project)
            skipped += 1
            pass
    print(skipped)


if __name__ == '__main__':
    shutil.rmtree("../data/graphs", ignore_errors=True)
    extract_communities("../data/arcanOutput/", "../data/")
