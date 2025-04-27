import logging
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Sigmoid, Tanh, Softmax, LayerNorm
import torch.nn.functional as F
from torch import nn
from mygenai.models.layers import EquivariantMPNNLayer
class GraphDecoder(Module):
    def __init__(self, latent_dim, emb_dim=32, num_nodes=29):
        """
        Notes
        -----
        - Stripped the model down to a minimal version
        """
        super().__init__()

        self.num_nodes = num_nodes

        # Projection from latent space to node embeddings
        # L -> (n/b)d
        self.latent_to_nodes = nn.Linear(latent_dim, num_nodes * emb_dim)
        self.emb_dim = emb_dim

        # Edge classifier takes concatenated embeddings of two nodes
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 5)  # 4 bond types + 1 for no bond
        )

    def forward(self, z):
        """
        z: (batch_size, num_nodes, latent_dim)
        Returns:
            edge_attr_logits: raw logits for edge prediction (batch_size, num_nodes, num_nodes)
        """
        batch_size = z.size(0)

        # Project from latent to node embeddings (b,L) -> (b, (n/b)d)
        h = self.latent_to_nodes(z)
        # Reshape
        h = h.view(batch_size, self.num_nodes, self.emb_dim)

        # Compute all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)

        edge_input = torch.cat([h_i, h_j], dim=-1)  # (batch_size, num_nodes, num_nodes, 2 * hidden_dim)
        edge_attr_logits = self.edge_mlp(edge_input)  # (batch_size, num_nodes, num_nodes)

        # Enforce symmetry: take average of i->j and j->i predictions
        edge_attr_logits = 0.5 * (edge_attr_logits + edge_attr_logits.transpose(1, 2))

        return edge_attr_logits  # sigmoid + mask + BCEWithLogitsLoss outside

    def predict_edges(self, z, threshold=0.5):
        edge_attr_logits = self(z)
        # convert to probabilities using softmax along last dimension (bond types)
        edge_probs = F.softmax(edge_attr_logits, dim=-1)
        return edge_probs
