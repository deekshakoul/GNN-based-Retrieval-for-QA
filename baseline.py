import numpy as np
import os
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

class BM25Retriever:
    """
    Sparse retrieval using BM25 algorithm
    """
    def __init__(self, passages):
        self.tokenized_corpus = None
        self.bm25 = None
        self.passages = passages
        self.index_file = "data/bm25_index.pkl"

        if os.path.exists(self.index_file):
            with open(self.index_file, "rb") as f:
                self.bm25 = pickle.load(f)

    def index(self, save=False):
        if not self.bm25 or save:            
            tokenized_corpus = []
            for passage in self.passages:
                tokens = passage.split()
                tokenized_corpus.append(tokens)
            
            self.tokenized_corpus = tokenized_corpus
            self.bm25 = BM25Okapi(tokenized_corpus)

            with open(self.index_file, "wb") as f:
                pickle.dump(self.bm25, f)
    
    def retrieve(self, query, k=10):
        """
        Retrieve top-k passages for a given query
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        top_passages = [self.passages[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        return top_passages, top_scores, top_indices
    

class DenseRetriever:
    """
    Dense retrieval using pre-trained sentence transformers
    """
    def __init__(self, passages, model_name = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.passages = passages
        self.passage_embeddings = None
        self.index_file = "data/DPR_index.pt"
        if os.path.exists(self.index_file):
            self.passage_embeddings = torch.load(self.index_file)

    def index(self, save=False):
        if not self.passage_embeddings or save:
            self.passage_embeddings = self.model.encode(self.passages, convert_to_tensor=True)
            torch.save(self.passage_embeddings, self.index_file)
    
    def retrieve(self, query, k=10):
        """
        Retrieve top-k passages for a given query
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.passage_embeddings.cpu().numpy()
        )[0]
        top_indices = np.argsort(scores)[::-1][:k]
        top_passages = [self.passages[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        return top_passages, top_scores, top_indices
