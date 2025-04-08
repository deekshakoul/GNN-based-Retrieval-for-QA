from torch_geometric.utils import degree
import numpy as np
import torch
from tqdm import tqdm
import pdb

class SelectiveSampler():
    def __call__(self, node_embedding, edge_index):
        
        num_nodes = node_embedding.shape[0]
        num_edges = edge_index.shape[1]

        assert num_edges == len(edge_index[0])

        num_samples = int(0.6 * num_edges)

        node_degree = degree(index = edge_index[0], num_nodes=num_nodes)
        # normalized_node_degree = node_degree / torch.sum(node_degree)
        row_edge_index = [node_degree[edge_index[0][j]].item() for j in range(num_edges)] 

        assert len(row_edge_index) == num_edges
        # pdb.set_trace()
        
        row_edge_index = torch.tensor(row_edge_index)
        row_edge_index_prob = row_edge_index / torch.sum(row_edge_index)
        row_edge_index_prob_list = np.array(row_edge_index_prob) 

        assert len(row_edge_index_prob_list) == num_edges
        # pdb.set_trace()

        sample_indices = np.random.choice(num_edges, size = num_samples, replace=False, p = row_edge_index_prob_list)
        sampled_row = edge_index[0][sample_indices]
        sampled_col = edge_index[1][sample_indices]

        sampled_edge_index = torch.stack([sampled_row, sampled_col], dim = 0)
        # pdb.set_trace()

        return sampled_edge_index



