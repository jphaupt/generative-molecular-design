import logging
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Sigmoid

class ConditionalDecoder(Module):
    def __init__(self, latent_dim=32, emb_dim=64, out_node_dim=11, out_edge_dim=4):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initial projection from latent+property space
        # expect one property
        self.lin_latent = Linear(latent_dim + 1, emb_dim)

        # Node feature generation
        self.node_decoder = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, out_node_dim)
        )

        # Position generation
        self.pos_decoder = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, 3)
        )

        # Number of nodes predictor
        self.num_nodes_predictor = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, 1)
        )

        # Edge prediction
        self.edge_existence = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, 1),
            Sigmoid()
        )

        self.edge_features = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, out_edge_dim)
        )

    def forward(self, z, target_property, batch_size):
        self.logger.debug(f"Input shapes - z: {z.shape}, target_property: {target_property.shape}")

        # Make sure target_property has correct shape for concatenation
        if target_property.dim() == 3:
            target_property = target_property.squeeze(1)
        if target_property.dim() == 1:
            target_property = target_property.unsqueeze(1)

        z_cond = torch.cat([z, target_property], dim=1)
        h = self.lin_latent(z_cond)

        # Predict number of nodes per graph
        num_nodes = self.num_nodes_predictor(h).sigmoid() * 30 + 5  # 5-35 nodes
        num_nodes = num_nodes.long()

        node_features_list = []
        positions_list = []

        for i in range(batch_size):
            n = num_nodes[i].item()  # Convert tensor to integer
            h_expanded = h[i:i+1].expand(n, -1)

            # Generate node features and positions
            node_feat = self.node_decoder(h_expanded)
            pos = self.pos_decoder(h_expanded)

            node_features_list.append(node_feat)
            positions_list.append(pos)

        node_features = torch.cat(node_features_list, dim=0)
        positions = torch.cat(positions_list, dim=0)

        self.logger.debug(f"Output shapes - node_features: {node_features.shape}, positions: {positions.shape}")

        return node_features, positions, num_nodes
