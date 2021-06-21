import igraph

from csio.graph_load import ProjectLoader
import glob


class FilesProjectLoader(ProjectLoader):
    def __init__(self, filter=None, extension='.java'):
        if filter is None:
            filter = []
        self.filter = filter
        self.extension = extension

    def load(self, path):
        files = glob.glob(path + '/**/*.txt', recursive=True)
        relatives = [file.replace(path, '') for file in files]
        graph = igraph.Graph.DictList
        return graph