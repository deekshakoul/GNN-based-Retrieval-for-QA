from torch_geometric.utils import negative_sampling, subgraph, to_networkx
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool

from transformers import AutoTokenizer

from utils import GraphDatasetContrastive, infonce, GCN, SentenceTransformerModel

SEED = 2025

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
model_path ="/mnt/nas/mohitsinghtomar/project_experiments/ir_assignment/assignment_3/saved_model/gnn_model_num_layer_2_aggregate_sum.pt"
data_path = "/mnt/nas/deekshak/ir3/GNN-based-Retrieval-for-QA/data/dev_graph_dict.pt"

Batch_size = 1
Aggregate = "sum"
Num_layers = 2
Num_epochs = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data = torch.load(data_path,  weights_only=False)

input_query_list = []
encoded_input_list = []
graph_data_list = []
positive_doc_mask_list = []

map_index_title = {}
index = 0
tmp = []
pos_ids = []
for key,val in tqdm(data.items()):
    input_query_list.append(val['query'])
    graph_data_list.append(val['graph'])
    num_docs = val['graph'].x.shape[0]
    temp_positive_ids = val['graph'].positive_ids
    temp_pos_doc_mask = torch.zeros(num_docs)
    temp_pos_doc_mask[temp_positive_ids] = 1
    
    pos_ids.append(temp_positive_ids)
    
    positive_doc_mask_list.append(temp_pos_doc_mask)
    for pass_title in val["passage_titles"]:
        map_index_title[index] = pass_title
        index += 1
        if pass_title == "Nitin Bose": tmp.append(index-1)
        # if pass_title == "Nitin Bose":
        #     import pdb; pdb.set_trace()
    assert len(val["passage_titles"]) == 10

    # if key in ['479265fa0bdc11eba7f7acde48001122', '7d9d5d720bdb11eba7f7acde48001122', '89af0df60bd911eba7f7acde48001122', '7b0e01820bdc11eba7f7acde48001122']:
    #     print(index)
    #     import pdb; pdb.set_trace()
    
positive_doc_mask = torch.stack(positive_doc_mask_list)
encoded_input = tokenizer(input_query_list, return_tensors="pt", truncation=True, padding=True)

 
gnn_model = GCN(input_dim= 768, hidden_dim=768, output_dim=768, num_layers=Num_layers, dropout=0.1, aggr=Aggregate)
gnn_model = gnn_model.to(device)
gnn_model.load_state_dict(torch.load(model_path))
gnn_model.eval()

sentencetransformer_model = SentenceTransformerModel()
sentencetransformer_model = sentencetransformer_model.to(device)
sentencetransformer_model.eval()



dev_loader = DataLoader(GraphDatasetContrastive(encoded_input['input_ids'], encoded_input['attention_mask'], 
                                                  graph_data_list, positive_doc_mask), batch_size = Batch_size, shuffle = False)

contrastive_loss = infonce()
total_loss = 0.0

assert_passages_len_per_query =  10

query_embeddings = []
passage_embeddings = []
index = 0
for batch_ in tqdm(dev_loader):
    query_ids = batch_[0].to(device)
    query_mask = batch_[1].to(device)
    graph_data = batch_[2].to(device)
    feature_matrix = graph_data.x
    edge_index = graph_data.edge_index
    positive_doc_mask = batch_[3].to(device)
    batch = graph_data.batch

    query_emb = sentencetransformer_model(query_ids, query_mask)
    with torch.no_grad():
        positive_passage_emb, pass_embeddings = gnn_model(feature_matrix = feature_matrix, edge_index = edge_index, positive_doc_mask = positive_doc_mask, batch = batch)
    query_embeddings.append(query_emb)
    passage_embeddings.append(pass_embeddings)
    # if  tmp[0]-10 < index < tmp[0]:
    #     import pdb; pdb.set_trace()
    # if  tmp[1]-10 < index < tmp[1]:
    #     import pdb; pdb.set_trace()
    index += 10

    # loss = contrastive_loss(query_emb, positive_passage_emb)
    # total_loss += loss.item()
    # graph_embedding bs x 768
    # query_emb 1 x 768
    # passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
    # query_emb = F.normalize(query_emb, p=2, dim=1)
    # Compute cosine similarity: [10 x 768] @ [768 x 1] => [10 x 1]
    # cosine_sim = torch.matmul(passage_embeddings, query_emb.T).squeeze()
    # sorted_indices = torch.argsort(cosine_sim, descending=True)
all_query_embeddings = torch.cat(query_embeddings, dim=0)
all_passage_embeddings = torch.cat(passage_embeddings, dim=0)

all_query_embeddings = F.normalize(all_query_embeddings, p=2, dim=1)
all_passage_embeddings = F.normalize(all_passage_embeddings, p=2, dim=1) 
cosine_scores = torch.matmul(all_query_embeddings, all_passage_embeddings.T)


## evaluation ##
top_k_temp = 20  # or any number of top passages you want
top_k=10
topk_scores, topk_indices = torch.topk(cosine_scores, k=top_k_temp, dim=1)  # [5000, k]

final_topk_indices = []
for i in tqdm(range(topk_indices.size(0))):  # for each query
    seen_titles = set()
    top10 = []
    for idx in topk_indices[i]:
        idx = idx.item()
        title = map_index_title.get(idx, None)
        if title == None:
            assert Exception
        if title and title not in seen_titles:
            seen_titles.add(title)
            top10.append(idx)
        if len(top10) == top_k:
            break
    final_topk_indices.append(top10)

f1_all = []
mrr_all = []
for i in tqdm(range(len(final_topk_indices))): 
    st = i*10
    en = st + 10
    pos_indices = [pos+10*i for pos in pos_ids[i]]

    topk_indices_q = final_topk_indices[i]

    hit = []
    mrr = 0
    for rank, ind in enumerate(topk_indices_q):
        if st < ind < en and ind in pos_indices:
            hit.append(ind)
            if len(hit) == 1:
                mrr = 1/(rank+1)
    mrr_all.append(mrr)

    tp = len(hit)
    fp = len(topk_indices_q) - tp
    fn = len(pos_indices) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    f1_all.append(f1)

    

print(f"MRR: {np.mean(mrr_all):.4f}")
print(f"F1: {np.mean(f1_all):.4f}")
