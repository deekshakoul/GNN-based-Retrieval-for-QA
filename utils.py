import torch
from transformers import AutoModel
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import torch.nn.functional as F

SEED=2025

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphDatasetContrastive(torch.utils.data.Dataset):
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



class infonce(torch.nn.Module):
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
        # pdb.set_trace()
        return new_loss


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
        positive_doc_mask : [2, num_docs] binary mask where its value is 1 where there is a positive doc
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

        final_emb = global_mean_pool(pos_doc_emb, batch)

        return final_emb, out
