import numpy as np
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BM25Retriever:
    """
    Sparse retrieval using BM25 algorithm
    """
    def __init__(self):
        self.tokenized_corpus = None
        self.bm25 = None
        self.passages = None
    
    def index(self, passages):
        """
        Index the corpus of passages
        """
        self.passages = passages
        tokenized_corpus = []
        for passage in passages:
            tokens = passage.split()
            tokenized_corpus.append(tokens)
        
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)
    
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
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.passage_embeddings = None
        self.passages = None
    
    def index(self, passages):
        """
        Index the corpus of passages by computing embeddings
        """
        self.passages = passages
        self.passage_embeddings = self.model.encode(passages, convert_to_tensor=True)
    
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