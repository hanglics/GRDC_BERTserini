from anserini_retriever import *
from bert_reader import *


def getScoresFromRetriever(CONF, topK, enableRM3, query):
    retriever = AnseriniRetriever(CONF, int(topK), enableRM3)
    hits = retriever.getRawDocumentHits(query)
    collection = []
    for each in hits:
        temp = {
            "docid": each.docid,
            "score": each.score,
            "content": retriever.searcher.doc(each.docid).contents()
        }
        collection.append(temp)
    return collection


def getScoresFromBERTReader(CONF, query, collection, bertVersion):
    finalRanking = []
    bertVersion = int(bertVersion)
    if bertVersion is 1:
        bertPath = CONF["BERT_BASE"]
    elif bertVersion is 2:
        bertPath = CONF["BERT_SQUAD_1.1"]
    elif bertVersion is 3:
        bertPath = CONF["BERT_SQUAD_2.0"]
    else:
        bertPath = CONF["BERT_SQUAD_1.1"]
    mix = CONF["LINEAR_MIX"]
    print("Loading pretrained BERT Model...")
    for each in collection:
        print("Processing " + str(collection.index(each) + 1) + "/" + str(len(collection)))
        document = each["content"]
        docid = each["docid"]
        score = each["score"]
        reader = BERTReader(bertPath, mix, query, document, docid, score)
        res = reader.execute()
        each["interpScore"] = res["interpScore"]
        finalRanking.append(each)
    return finalRanking
