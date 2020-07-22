from transformers import *


class BERTReader:

    def __init__(self, bertPath: str):
        self.bertPath = bertPath
        self.tokenizer = AutoTokenizer.from_pretrained(bertPath, do_lower_case=False)
        self.model = AutoModel.from_pretrained(bertPath)

    def execute(self):
        pass
