from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from itertools import permutations, combinations


from itertools import permutations, combinations
from selective_sampler import SelectiveSampler

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
DEVICE="cuda"
DATA_FOLDER_PATH="data/data_ids/"
dev = json.load(open(DATA_FOLDER_PATH+"dev.json")) 

title_passage_dict = {}
for i in range(len(dev)):
    for idx, list_ in enumerate(dev[i]["context"]):
        title, list_sents = list_
        passage = " ".join(list_sents)
        title_passage_dict[title] = passage

with open("dev_title_passage_dict", "w") as fp: json.dump(title_passage_dict, fp)

train = json.load(open(DATA_FOLDER_PATH+"train.json"))
print(f"train: {len(train)} val:{len(dev)}")

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
    '''
        generate -  (q1, G1, positive_ids1)
                    (q2, G2, positive_ids2)
                    positive_ids: supporting_facts
                    G: from context
    '''
    def __init__(self):
        train_data = self.create_data(train)
        dev_data = self.create_data(dev)
        print("completed train and dev data creation")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sentence_transformer = SentenceTransformerModel()
        self.threshold = 0.55
        self.match_words_threshold = 20
        print(f"\n\n ==================== \n\n")

        self.create_and_save_graphs(dev_data, "dev")
        self.create_and_save_graphs(train_data, "train")
        self.selective_sampler = SelectiveSampler()

    
    def create_data(self, data):
        '''
            _id:
                query
                passsages
                positive_ids
        '''
        data_dict = {}
        for i in tqdm(range(len(data))):
            key = data[i]["_id"]
            passages = []
            pos_ids = []
            supporting_titles = []
            titles = []
            supporting_facts = data[i]["supporting_facts"]
            for fact in supporting_facts:
                supporting_titles.append(fact[0])
            for idx, list_ in enumerate(data[i]["context"]):
                title, list_sents = list_
                passage = title + "\t" + " ".join(list_sents)
                passages.append(passage)
                titles.append(title)     
                if title in supporting_titles:
                    pos_ids.append(idx) 
            data_dict[key] = {
                            "query": data[i]["question"],
                            "passages": passages, #list of passages
                            "positive_ids": pos_ids,
                            "passage_titles": titles
            }
            # import pdb; pdb.set_trace()     
            # if i == 10:
            #     break
        return data_dict

    def same_question_based_graph(self, passages):
        pass
    
    def common_keywords_based_edges(self, passages):
        tokenized = [set(p.lower().split()) for p in passages]
        matching_pairs = []
        for i, j in combinations(range(len(passages)), 2):
            common_words = tokenized[i] & tokenized[j]
            if len(common_words) > self.match_words_threshold:
                matching_pairs.append((i, j))
        return matching_pairs

    def fetch_embeddings(self, passages_per_query):
        encoded_input = self.tokenizer(passages_per_query, padding=True, truncation=True, return_tensors='pt') # go upto max pasage length 
        output = self.sentence_transformer(encoded_input['input_ids'].to(DEVICE), encoded_input['attention_mask'].to(DEVICE))
        return output
    
    def embedding_based_graph(self, passages, positive_ids):
        # common_words_edges = self.common_keywords_based_edges(passages)
        embeddings = self.fetch_embeddings(passages) # number_passages_in_context x 768
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = normalized @ normalized.T 
        edge_mask = (similarity_matrix > self.threshold) & (~torch.eye(len(passages), dtype=bool, device=similarity_matrix.device))
        src, dst = torch.where(edge_mask)
        edges = list(zip(src.tolist(), dst.tolist()))
        pos_edges = list(permutations(positive_ids, 2))  # a,b b,a
        edges.extend(pos_edges)
        # edges.extend(common_words_edges)
        edges = list(set(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(embeddings.cpu().numpy(), dtype=torch.float)
        pos_ids_tensor =  torch.tensor(positive_ids, dtype=torch.long)
        selective_sampled_edges = self.selective_sampler(x, edge_index)
        return Data(x=x, edge_index=edge_index, positive_ids=pos_ids_tensor, selective_sampled_edges = selective_sampled_edges)  

    def create_and_save_graphs(self, data, name):
        graph_dict = {}
        for key in tqdm(data):
            graph_dict[key] = {}
            graph  = self.embedding_based_graph(data[key]["passages"], data[key]["positive_ids"])
            graph_dict[key]["graph"]= graph 
            graph_dict[key]["positive_ids"] = data[key]["positive_ids"]
            graph_dict[key]["query"] = data[key]["query"]
            if name == "dev":
                graph_dict[key]["passage_titles"] = data[key]["passage_titles"]
        torch.save(graph_dict, f"data/{name}_graph_dict_Heuristics2.pt")
        print(f"saved - {name}_graph_dict data")
gop = GoP()
