from pyserini import *


class AnseriniRetriever:

    def __init__(self, indexPath: str):
        self.indexPath = indexPath

    def getTopKPassages(self, K: int, passageType: str):
        pass
