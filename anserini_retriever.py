from transformers import *


class AnseriniRetriever:

    def __init__(self, indexPath: str, bertPath: str):
        self.indexPath = indexPath
        self.tokenizer = AutoTokenizer.from_pretrained(bertPath, do_lower_case=False)
        self.model = AutoModel.from_pretrained(bertPath)

    def getTopKPassages(self):
        pass
