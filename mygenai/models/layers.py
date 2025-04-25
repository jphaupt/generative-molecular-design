import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add', max_distance=2.0, min_distance=0.8):
        """
        Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\\oplus` (sum/mean/max)
            max_distance: (float) - fixed maximum distance for normalization
            min_distance: (float) - fixed minimum distance for normalization
        """
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.max_distance = max_distance
        self.min_distance = min_distance

        self.mlp_scalar = Sequential(
            Linear(2 * emb_dim + edge_dim + 1, emb_dim),  # +1 for normalized distance
            BatchNorm1d(emb_dim),
            ReLU()
        )
        self.mlp_vector = Sequential(
            Linear(2 * emb_dim + edge_dim + 1, 1),  # Input: [h_i, h_j, edge_attr, normalized dist]
            BatchNorm1d(1),
            ReLU()
        )

        # update MLPs
        self.mlp_h = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, 1), BatchNorm1d(1), ReLU()
        )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        return self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        r_ij = pos_j - pos_i  # Equivariant relative positions
        dist = torch.norm(r_ij, dim=-1, keepdim=True)  # Compute raw distances
        dist = (dist - self.min_distance) / (self.max_distance - self.min_distance)  # Normalize to [0, 1]

        # Scalar message (invariant features)
        scalar_inputs = torch.cat([h_i, h_j, edge_attr, dist], dim=-1)
        scalar_msg = self.mlp_scalar(scalar_inputs)

        # Vector message (equivariant coordinates)
        # vector message should only depend on rotation/translation invariant quantities
        vector_inputs = torch.cat([h_i, h_j, edge_attr, dist], dim=-1)
        scale = self.mlp_vector(vector_inputs) # (e, 1)
        vector_msg = scale * r_ij # (e, 3)

        return scalar_msg, vector_msg

    def aggregate(self, inputs, index):
        scalar_msgs, vector_msgs = inputs
        scalar_aggr = scatter(scalar_msgs, index, dim=self.node_dim, reduce=self.aggr)
        vector_aggr = scatter(vector_msgs, index, dim=self.node_dim, reduce=self.aggr)
        return scalar_aggr, vector_aggr

    def update(self, aggr_out, h, pos):
        scalar_aggr, vector_aggr = aggr_out

        # Update node features (h)
        h_update = self.mlp_h(torch.cat([h, scalar_aggr], dim=-1))

        # Update node positions (pos)
        scale = self.mlp_pos(scalar_aggr) # (n, 1)
        pos_update = pos + scale * vector_aggr  # (n, 3)

        return h_update, pos_update

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
