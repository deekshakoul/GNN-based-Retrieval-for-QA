from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
DEVICE="cuda"
DATA_FOLDER_PATH="data/data_ids/"
dev = json.load(open(DATA_FOLDER_PATH+"dev.json"))    
test = json.load(open(DATA_FOLDER_PATH+"test.json"))
train = json.load(open(DATA_FOLDER_PATH+"train.json"))
print(f"train: {len(train)} val:{len(dev)} test:{len(test)}")
'''
a. create GoP
b. create a GNN class
c. retrieve
d. input to LLM
e. evaluation
'''

class SentenceTransformerModel(torch.nn.Module):
    def __init__(self):
        super(SentenceTransformerModel, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)#, device_map = "auto")
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            model_output = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
class GoP:
    def __init__(self):
        dev_passages = self.create_passages(dev)
        self.dev_nodes = len(dev_passages)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sentence_transformer = SentenceTransformerModel()
        self.threshold = 0.5
        embeddings, edges  = self.embedding_based_graph(dev_passages)
        self.create_pyg_graph(embeddings, edges)

    
    def create_passages(self, data):
        passages = []
        map_passage_entityid = {}
        for i in tqdm(range(len(data))):
            for title, list_sents in data[i]["context"]:
                passage = title + "\t" + " ".join(list_sents)
                passages.append(passage)     
            if i == 50:
                break       
        return passages

    def same_question_based_graph(self, passages):
        pass
    
    def common_keywords_based_graph(self, passages):
        pass

    def fetch_embeddings(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        dataset = TensorDataset(encoded_input['input_ids'], encoded_input['attention_mask'])
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        all_embeddings = []
        for batch in loader:
            input_ids, attention_mask = batch
            output = self.sentence_transformer(input_ids.to(DEVICE), attention_mask.to(DEVICE))
            all_embeddings.append(output)
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings

    def embedding_based_graph(self, passages):
        embeddings = self.fetch_embeddings(passages)
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = normalized @ normalized.T 
        edge_mask = (similarity_matrix > self.threshold) & (~torch.eye(len(passages), dtype=bool, device=similarity_matrix.device))
        src, dst = torch.where(edge_mask)
        edges = list(zip(src.tolist(), dst.tolist()))
        return embeddings, edges 

    def create_pyg_graph(self, embeddings, edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(embeddings.cpu().numpy(), dtype=torch.float)
        return Data(x=x, edge_index=edge_index)        


# gop = GoP()