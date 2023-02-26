# -*- coding: utf-8 -*-
import numpy as np
from model import graph
from config import GlobalConfig

import logging
logging.basicConfig(filename="server.log", level=logging.ERROR, filemode='a')

from opensearchpy import OpenSearch

client = OpenSearch(
    GlobalConfig.OPENSEARCH_SERVICE_URL,
    http_auth = GlobalConfig.OPENSEARCH_AUTH,
    use_ssl = False,
    verify_certs = False,
)

def calcEmbedding(model, jsonData):
    result = {}
    result = []
    for g in enumerate(jsonData['list']):
        func_result = {}
        embedding = model.predict(g)
        func_result['embedding'] = embedding[0].tolist()
        result.append(func_result)

    return result

def searchSimilarEmbedding(query):
    resp = client.search(
        index = "test",
        body = {
            "size": 5,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query,
                        "k": 5,
                    }
                }
            }
        },
        request_timeout = 0
    )

    hits = resp['hits']
    recalls = hits['hits']
    candidates = []
    for recall in recalls:
        candidates.append({
            'score': recall['_score'],
            'item': recall['_source']['item'],
        })

    result = {
        'max_score': hits['max_score'],
        'candidates': candidates
    }
    return result

def fetchResultByEmbedding(data):
    searchResult = []
    for func in data['func_list']:
        funcname = func['func_name']
        embedding = func['embedding']
        result = searchSimilarEmbedding(embedding)
        searchResult.append({
            'funcname': funcname,
            'result': result
        })
    return searchResult