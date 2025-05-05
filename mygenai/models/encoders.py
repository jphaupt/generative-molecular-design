import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.nn import global_mean_pool, GCNConv, GINEConv

from mygenai.models.layers import EquivariantMPNNLayer

# n: Total number of nodes across all graphs in batch
# d_n: Input node feature dimension (= 5 + "no atom")
# d: Embedding dimension (= 64)
#    Use same initial embedding and hidden layer dimensions
# b: Batch size (number of graphs)
# L: Latent dimension (= 32)

class GraphEncoder(Module):
    def __init__(self, emb_dim=32, node_feat_dim=6, latent_dim=32, num_layers=4):
        """Encoder module for graph property prediction
        """
        super().__init__()

        # input shape (n, d_n) -> (n, d)
        self.lin_in = Linear(node_feat_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(emb_dim, emb_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # projections to latent space
        self.mu = Linear(emb_dim, latent_dim)
        self.log_var = Linear(emb_dim, latent_dim)

        # Property prediction (only one: homo-lumo gap)
        self.property_predictor = Sequential(
            Linear(latent_dim, emb_dim),
            ReLU(),
            # BatchNorm1d(emb_dim),
            Linear(emb_dim, 1)
        )

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG (complete) graphs

        Returns:
            mu: Latent mean (batch_size, latent_dim)
            log_var: Latent log variance (batch_size, latent_dim)
            property_pred: Predicted property (batch_size, 1)
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        # pos = data.pos

        for conv in self.convs:
            # Message passing layer
            # h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)
            h = conv(h, data.edge_index)

        # Pool to graph level
        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        # Get latent parameters and property prediction
        mu = self.mu(h_graph)
        log_var = self.log_var(h_graph)
        property_pred = self.property_predictor(mu)

        return mu, log_var, property_pred
