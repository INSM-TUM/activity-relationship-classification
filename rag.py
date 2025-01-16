import json

import chromadb
import replicate
from textsplitter import textsplitter
from flashrank import Ranker, RerankRequest

import logging
from util import *

logging.getLogger("httpx").setLevel(logging.WARNING)


class Rag:
    def __init__(self, collection_name: str, process_desc: str):
        self.text_splitter = textsplitter.TextSplitter(max_token_size=50, remove_stopwords=False)
        chroma_client = chromadb.Client()
        if any(filter(lambda collection : collection.name == collection_name, chroma_client.list_collections())):
            self.collection = chroma_client.get_collection(collection_name)
        else:
            log(f'Creating new collection "{collection_name}"')
            self.collection = chroma_client.create_collection(collection_name)
            self._load_embeddings(process_desc)
        self.ranker = Ranker()

    def _load_embeddings(self, process_desc: str):
        chunks = self.text_splitter.split_text(process_desc)

        output = replicate.run(
            "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
            input={
                "text_batch": json.dumps(chunks)
            }
        )

        embeddings = list()
        for i in range(len(chunks)):
            embeddings.append(output[i]["embedding"])
        self.collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in range(len(chunks))])

    def return_related(self, query_activities: list) -> list:
        output = replicate.run(
            "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
            input={
                "text": ' and '.join(query_activities)
            }
        )

        passages = list()
        query_results = self.collection.query(
                query_embeddings=output[0]["embedding"],
                n_results=10
            )
        for i in range(len(query_results['documents'][0])):
            passages.append({"id": query_results['ids'][0][i], "text": query_results['documents'][0][i]})

        rerank_request = RerankRequest(query=' and '.join(query_activities), passages=passages)
        reranked_passages = self.ranker.rerank(rerank_request)
        context = list()
        for passage in reranked_passages[:5]:
            context.append(passage['text'])
        return context
