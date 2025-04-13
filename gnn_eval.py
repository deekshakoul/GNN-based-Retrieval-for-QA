from dataclasses import dataclass
from torch_geometric.utils import negative_sampling, subgraph, to_networkx
import torch
import json
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool

from transformers import AutoTokenizer, HfArgumentParser

from utils import GraphDatasetContrastive, infonce, GCN, SentenceTransformerModel

SEED = 2025

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

@dataclass
class EvalArguments:
    gcn_model_path: str
    data_path: str 
    aggregate: str 
    num_layers: int
    gnn_retrived_file_name: str

encoder_name = 'sentence-transformers/all-mpnet-base-v2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1

with open('dev_title_passage_dict.json') as f:
    title_passage_dev = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(encoder_name)

def create_dataloader(args):
    data = torch.load(args.data_path,  weights_only=False)
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
        assert len(val["passage_titles"]) == 10

    positive_doc_mask = torch.stack(positive_doc_mask_list)
    encoded_input = tokenizer(input_query_list, return_tensors="pt", truncation=True, padding=True)

    dev_loader = DataLoader(GraphDatasetContrastive(encoded_input['input_ids'], encoded_input['attention_mask'], 
                                                  graph_data_list, positive_doc_mask), batch_size = BATCH_SIZE, shuffle = False)
    
    return dev_loader, pos_ids, map_index_title, input_query_list

 
def load_model(args):
    gnn_model = GCN(input_dim= 768, hidden_dim=768, output_dim=768, num_layers=args.num_layers, dropout=0.1, aggr=args.aggregate)
    gnn_model = gnn_model.to(device)
    gnn_model.load_state_dict(torch.load(args.gcn_model_path))
    gnn_model.eval()
    sentencetransformer_model = SentenceTransformerModel()
    sentencetransformer_model = sentencetransformer_model.to(device)
    sentencetransformer_model.eval()
    return gnn_model, sentencetransformer_model


def evaluate(args):
    loader, pos_ids, map_index_title, input_query_list = create_dataloader(args)
    gnn_model, sentencetransformer_model = load_model(args)
    query_embeddings = []
    passage_embeddings = []
    index = 0
    for batch_ in tqdm(loader):
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
        index += 10
    all_query_embeddings = torch.cat(query_embeddings, dim=0)
    all_passage_embeddings = torch.cat(passage_embeddings, dim=0)

    all_query_embeddings = F.normalize(all_query_embeddings, p=2, dim=1)
    all_passage_embeddings = F.normalize(all_passage_embeddings, p=2, dim=1) 
    cosine_scores = torch.matmul(all_query_embeddings, all_passage_embeddings.T)


    ## evaluation ##
    top_k_temp = 20
    top_k = 10
    icl_exs_num = 10
    icl_exs = evaluate_metrics(top_k, top_k_temp, icl_exs_num, cosine_scores, pos_ids, map_index_title)


    top_k_temp = 20
    top_k = 5
    icl_exs_num = 10
    icl_exs_del = evaluate_metrics(top_k, top_k_temp, icl_exs_num, cosine_scores, pos_ids, map_index_title)

    return icl_exs, input_query_list

def evaluate_metrics(top_k, top_k_temp, icl_exs_num, cosine_scores, pos_ids, map_index_title):
    final_topk_indices = []
    topk_scores, topk_indices = torch.topk(cosine_scores, k=top_k_temp, dim=1)  # [5000, k]
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
        if len(top10) < top_k:
            assert Exception
        final_topk_indices.append(top10)
    f1_all = []
    mrr_all = []
    gold_passages = []
    for i in tqdm(range(len(final_topk_indices))): 
        st = i*10
        en = st + 10
        pos_indices = [pos+10*i for pos in pos_ids[i]]
        topk_indices_q = final_topk_indices[i]
        # retrieved passages to be passed in prompt to T5 
        gold_ = [title_passage_dev[map_index_title[i]] for i in topk_indices_q[:icl_exs_num]]
        gold_passages.append(gold_)

        hit = []
        mrr = 0
        for rank, ind in enumerate(topk_indices_q):
            if st <= ind < en and ind in pos_indices:
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

    print(f"top: {top_k}, MRR: {np.mean(mrr_all):.4f}")
    print(f"top: {top_k}, F1: {np.mean(f1_all):.4f}")

    return gold_passages

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args()
    print(args)
    icl_examples, queries = evaluate(args)

    dd = {}
    for i in range(len(queries)):
        query = queries[i]
        dd[query] = icl_examples[i]


with open(f"{args.gnn_retrived_file_name}", "w") as f:
    json.dump(dd, f, indent=4)
    

