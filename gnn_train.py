from torch_geometric.utils import negative_sampling, subgraph, to_networkx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from sentence_transformers import SentenceTransformer
import pdb
from dataclasses import dataclass

SEED = 2025

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

@dataclass
class RunArguments:
    Aggregate: str
    Num_layers: int
    Num_epochs: int
    Batch_size: int
    selective_sampling: bool
    input_data_path: str
    output_data_path: str

parser = HfArgumentParser((RunArguments))
args = parser.parse_args()

print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : ', device)

# For aggregation check https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data = torch.load(args.input_data_path,  weights_only=False)

input_query_list = []
encoded_input_list = []
graph_data_list = []
positive_doc_mask_list = []

for key,val in tqdm(data.items()):
     input_query_list.append(val['query'])
     curr_node_emb = val['graph'].x
     if(args.selective_sampling):
        curr_edge = val['graph'].selective_sampled_edges
     else:
        curr_edge = val['graph'].edge_index

     curr_pos_ids = val['graph'].positive_ids
     temp_graph_data = Data(x=curr_node_emb, edge_index = curr_edge, positive_ids=curr_pos_ids)
     
     graph_data_list.append(temp_graph_data)

     num_docs = val['graph'].x.shape[0]
     temp_pos_doc_mask = torch.zeros(num_docs)
     temp_pos_doc_mask[curr_pos_ids] = 1
     
     positive_doc_mask_list.append(temp_pos_doc_mask)
    
positive_doc_mask = torch.stack(positive_doc_mask_list)
encoded_input = tokenizer(input_query_list, return_tensors="pt", truncation=True, padding=True)

class SentenceTransformerModel(torch.nn.Module):
    def __init__(self):
        super(SentenceTransformerModel, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device)#, device_map = "auto")
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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, aggr = None, return_embeds=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList([GCNConv(in_channels = input_dim, out_channels = hidden_dim, aggr = aggr) if i==0
                                          else GCNConv(in_channels = hidden_dim, out_channels = hidden_dim, aggr = aggr) if i<num_layers-1
                                          else GCNConv(in_channels = hidden_dim, out_channels = output_dim, aggr = aggr) for i in range(num_layers)])

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = hidden_dim) for i in range(num_layers-1)])
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, feature_matrix, edge_index, positive_doc_mask, batch):
        """
        feature matrix shape : [num_nodes, embedding_dim]
        edge_index shape : [2, num_edges]
        positive_doc_mask : [batch, num_docs] binary mask where its value is 1 where there is a positive doc
        """
        out = None
        for idx, conv_layer in enumerate(self.convs):

            num_layers = len(self.convs)

            if(idx == 0):
                out = conv_layer(feature_matrix, edge_index)
                out = self.bns[idx](out)
                out = torch.nn.functional.relu(out)
                out = torch.nn.functional.dropout(out, p = self.dropout, training = True)

            elif(idx>0 and idx<=num_layers-2):
                out = conv_layer(out, edge_index)
                out = self.bns[idx](out)
                out = torch.nn.functional.relu(out)
                out = torch.nn.functional.dropout(out, p = self.dropout, training = True)

            elif(idx == num_layers-1):
                out = conv_layer(out, edge_index)

        positive_doc_mask = positive_doc_mask.reshape(batch.shape[0], -1)
        pos_doc_emb = out * positive_doc_mask

        positive_passages_emb = global_mean_pool(pos_doc_emb, batch)
        all_passages_emb = out
        return positive_passages_emb, all_passages_emb
    
gnn_model = GCN(input_dim= 768, hidden_dim=768, output_dim=768, num_layers=args.Num_layers, dropout=0.1, aggr=args.Aggregate)
# taking input, hidden and output dim to be same for calculating scoring function in loss

sentencetransformer_model = SentenceTransformerModel()

gnn_model = gnn_model.to(device)
sentencetransformer_model = sentencetransformer_model.to(device)

for param in sentencetransformer_model.parameters():
    param.requires_grad = False

sentencetransformer_trainable_params = sum(p.numel() for p in sentencetransformer_model.parameters() if p.requires_grad)

assert sentencetransformer_trainable_params == 0

# infonce loss taken from https://github.com/paridhimaheshwari2708/GraphSSL/blob/main/loss.py 
class infonce(nn.Module):
    def __init__(self):
        super(infonce, self).__init__()

        self.tau = 0.5
        self.norm = True
    
    def forward(self, embed_anchor, embed_positive):
        batch_size = embed_anchor.shape[0]
        sim_matrix = torch.einsum("ik,jk->ij", embed_anchor, embed_positive)

        if(self.norm):
            embed_anchor_abs = embed_anchor.norm(dim=1)
            embed_positive_abs = embed_positive.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum("i,j->ij", embed_anchor_abs, embed_positive_abs)
        
        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        new_loss = -torch.log(loss).mean()
        return new_loss

class GraphDatasetContrastive(Dataset):
    def __init__(self, query_ids, query_attention_mask, graph_data_list, positive_doc_mask):
        self.query_ids = query_ids
        self.query_attention_mask = query_attention_mask
        self.graph_data = graph_data_list
        self.positive_doc_mask = positive_doc_mask

    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        query_idx = self.query_ids[idx]
        query_attention_mask_idx = self.query_attention_mask[idx]
        graph_data_idx = self.graph_data[idx]
        pos_doc_idx = self.positive_doc_mask[idx]
        
        return query_idx, query_attention_mask_idx,  graph_data_idx, pos_doc_idx

# batching documentation - https://pytorch-geometric.readthedocs.io/en/2.4.0/notes/batching.html 
train_loader = DataLoader(GraphDatasetContrastive(encoded_input['input_ids'], encoded_input['attention_mask'], 
                                                  graph_data_list, positive_doc_mask), batch_size = args.Batch_size, shuffle = False)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr = 1e-3)
contrastive_loss = infonce()

def train(loader):
    gnn_model.train()
    sentencetransformer_model.eval()

    epoch_loss = 0.0
    
    for data in tqdm(loader):
        optimizer.zero_grad()
        
        query_ids = data[0].to(device)
        query_mask = data[1].to(device)
        graph_data = data[2].to(device)
        
        feature_matrix = graph_data.x
        edge_index = graph_data.edge_index
        positive_doc_mask = data[3].to(device)
        batch = graph_data.batch

        query_emb = sentencetransformer_model(query_ids, query_mask)
        positive_passage_emb, _ = gnn_model(feature_matrix = feature_matrix, edge_index = edge_index, positive_doc_mask = positive_doc_mask, batch = batch)
        
        loss = contrastive_loss(query_emb, positive_passage_emb)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

best_epoch_loss = 1e6

for epoch in range(args.Num_epochs):

    

    print("\n=============Epoch : ", epoch)
    epoch_loss = train( train_loader)

    print('Epoch loss : ', epoch_loss)

    if(epoch_loss < best_epoch_loss):
        best_epoch_loss = epoch_loss
        
        if(args.selective_sampling):
            torch.save(gnn_model.state_dict(), f'{args.output_data_path}/gnn_model_num_layer_{args.Num_layers}_aggregate_{args.Aggregate}_selective_sampling' + '.pt')
        else:
            torch.save(gnn_model.state_dict(), f'{args.output_data_path}/gnn_model_num_layer_{args.Num_layers}_aggregate_{args.Aggregate}' + '.pt')

        print("model saved\n")
