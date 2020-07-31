from transformers import *
import torch
import numpy


def calcCosineSim(embeddings1, embeddings2):
    embeddings1 = embeddings1 / torch.sqrt((embeddings1 ** 2).sum(1, keepdims=True))
    embeddings2 = embeddings2 / torch.sqrt((embeddings2 ** 2).sum(1, keepdims=True))
    sim = (embeddings1.unsqueeze(1) * embeddings2.unsqueeze(0)).sum(-1)
    return torch.max(sim).item()


class BERTReader:

    def __init__(self, bertPath: str, mix: float, query: str, document: str, docid: str, score: float):
        self.bertPath = bertPath
        self.tokenizer = AutoTokenizer.from_pretrained(bertPath, do_lower_case=False)
        self.model = AutoModel.from_pretrained(bertPath)
        self.mix = mix
        self.query = query
        self.doc = document
        self.docid = docid
        self.score = score

    def getEmbeddingsFromBERT(self, text):
        textIDS = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        # textWords = self.tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

        nChunks = int(numpy.ceil(float(textIDS.size(1)) / 512))
        states = []

        for ci in range(nChunks):
            # textIDs_ = textIDs[0, 1 + ci * 512:1 + (ci + 1) * 512]
            textIDS_ = textIDS[0, ci * 512:(ci + 1) * 512]
            torch.cat([textIDS[0, 0].unsqueeze(0), textIDS_])
            if textIDS[0, -1] != textIDS[0, -1]:
                torch.cat([textIDS, textIDS[0, -1].unsqueeze(0)])

            with torch.no_grad():
                state = self.model(textIDS_.unsqueeze(0))[0]
                state = state[:, 1:-1, :]
            states.append(state)

        state = torch.cat(states, axis=1)
        return state[0]

    def execute(self):
        queryEmbedding = self.getEmbeddingsFromBERT(self.query)
        documentEmbedding = self.getEmbeddingsFromBERT(self.doc)
        cosineSim = calcCosineSim(queryEmbedding, documentEmbedding)
        interpScore = self.mix * cosineSim + (1 - self.mix) * self.score
        res = {
            "docid": self.docid,
            "interpScore": interpScore,
        }
        return res
