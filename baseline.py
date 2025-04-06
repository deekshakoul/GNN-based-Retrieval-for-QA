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
        if os.path.exists("data/bm25_index.pkl"):
            with open("data/bm25_index.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = None


    def index(self, save=False):
        if save and os.path.exists("data/bm25_index.pkl"):
            return self.bm25
        tokenized_corpus = []
        for passage in self.passages:
            tokens = passage.split()
            tokenized_corpus.append(tokens)
        
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

        if save:
            with open("data/bm25_index.pkl", "wb") as f:
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
    def __init__(self, passages, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.passages = passages
        if os.path.exists("data/DPR_index.pt"):
            self.passage_embeddings = torch.load("data/DPR_index.pt")
        else:
            self.passage_embeddings = None

    
    def index(self, save=False):
        if save and os.path.exists("data/DPR_index.pt"):
            return self.passage_embeddings
        self.passage_embeddings = self.model.encode(self.passages, convert_to_tensor=True)

        if save:
            torch.save(self.passage_embeddings, "data/DPR_index.pt")
    
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
    
# print("Initializing retrievers...")

# output_dir = "index"
# bm25_index_path = os.path.join(output_dir, "bm25_index.pkl")
# dense_index_path = os.path.join(output_dir, "dense_index.pkl")

# all_passages = []
# for item in train_dataset:
#     all_passages.extend([p["text"] for p in item["passages"]])

# bm25_retriever = BM25Retriever()
# if os.path.exists(bm25_index_path):
#     print("Loading existing BM25 index...")
#     bm25_retriever.load_index(bm25_index_path)
# else:
#     print("Building BM25 index...")
#     bm25_retriever.index(all_passages, save_path=bm25_index_path)

# dense_retriever = DenseRetriever()
# if os.path.exists(dense_index_path):
#     print("Loading existing Dense index...")
#     dense_retriever.load_index(dense_index_path)
# else:
#     print("Building Dense index...")
#     dense_retriever.index(all_passages, save_path=dense_index_path)
