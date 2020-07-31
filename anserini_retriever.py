from pyserini.search import SimpleSearcher


class AnseriniRetriever:

    def __init__(self, CONF, K: int, enableRM3: bool):
        self.CONF = CONF
        self.indexPath = CONF["INDEX"]
        self.searcher = SimpleSearcher(self.indexPath)
        self.K = K
        self.enableRM3 = enableRM3

    def getRawDocumentHits(self, query: str):
        k1 = self.CONF["BM25"]["K1"]
        b = self.CONF["BM25"]["B"]
        if self.enableRM3:
            fb_terms = self.CONF["RM3"]["FB_TERMS"]
            fb_docs = self.CONF["RM3"]["FB_DOCS"]
            original_query_weight = self.CONF["RM3"]["ORIGINAL_QUERY_WEIGHT"]
            self.searcher.set_bm25(k1, b)
            self.searcher.set_rm3(fb_terms, fb_docs, original_query_weight)
            hits = self.searcher.search(query, self.K)
            return hits
        else:
            self.searcher.set_bm25(k1, b)
            hits = self.searcher.search(query, self.K)
            return hits
