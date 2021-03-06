from middleware import *
import json

print("--------------------------------")
print("Loading Config...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
print("Loaded")
print("--------------------------------")


def BERTSerini():
    topK = input("Top K Passages (Int Only): ")
    enableRM3 = input("Enable RM3? (Y/N) ")
    if enableRM3 is "Y" or "y":
        enableRM3 = True
    else:
        enableRM3 = False
    print("Use which version of BERT? ")
    print("1. BERT BASE")
    print("2. BERT BASE Fine Tuned on SQuAD 1.1")
    print("3. BERT BASE Fine Tuned on SQuAD 2.0")
    bertVersion = input("Select: ")
    query = input("Input query: ")
    collection = getScoresFromRetriever(CONF, topK, enableRM3, query)
    res = getScoresFromBERTReader(CONF, query, collection, bertVersion)
    res = sorted(res, key=lambda x: x["interpScore"], reverse=True)
    for each in res:
        print(f'{res.index(each) + 1:2} {each["docid"]:15}  {each["interpScore"]:.5f}')


def BERTPRFTOPK():
    topK = input("Top K Passages (Int Only): ")
    enableRM3 = input("Enable RM3? (Y/N) ")
    if enableRM3 is "Y" or "y":
        enableRM3 = True
    else:
        enableRM3 = False
    query = input("Input query: ")
    collection = getScoresFromRetriever(CONF, topK, enableRM3, query)
    newQuery = query
    for each in collection:
        newQuery += each["title"] + " "
    topKBERT = input("Top K Passages For BERT Reader (Int Only): ")
    print("Use which version of BERT? ")
    print("1. BERT BASE")
    print("2. BERT BASE Fine Tuned on SQuAD 1.1")
    print("3. BERT BASE Fine Tuned on SQuAD 2.0")
    bertVersion = input("Select: ")
    newCollection = getScoresFromRetriever(CONF, topKBERT, enableRM3, newQuery)
    res = getScoresFromBERTReader(CONF, newQuery, newCollection, bertVersion)
    res = sorted(res, key=lambda x: x["interpScore"], reverse=True)
    for each in res:
        print(f'{res.index(each) + 1:2} {each["docid"]:15}  {each["interpScore"]:.5f}')


def main():
    print("Perform BERTserini or BERT Top K PRF?")
    print("1. BERTserini")
    print("2. BERT Top K PRF")
    selection = input("Select: ")
    if int(selection) is 1:
        BERTSerini()
    elif int(selection) is 2:
        BERTPRFTOPK()
    else:
        print("Wrong input.")


if __name__ == '__main__':
    main()
