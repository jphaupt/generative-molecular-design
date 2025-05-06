import logging
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import to_dense_adj

from mygenai.models.encoders import GraphEncoder
from mygenai.models.decoders import GraphDecoder

class GraphVAE(Module):
    def __init__(self, node_feat_dim=6, emb_dim=32, latent_dim=32):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim=node_feat_dim, emb_dim=emb_dim,
                                    latent_dim=latent_dim)
        self.decoder = GraphDecoder(latent_dim, emb_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, node_feats):
        mu, logvar, property_pred = self.encoder(node_feats)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, node_feats.num_real_atoms)
        return logits, mu, logvar, property_pred

    def loss_function(self, edge_attr_logits, batch):
        # Create ground-truth adjacency matrices using PyG utility
        adj_target = to_dense_adj(
            batch.edge_index,
            batch=batch.batch,
            max_num_nodes=edge_attr_logits.size(1),
            edge_attr=batch.edge_attr
        )

        # Create masks for bonds we want to exclude from loss
        batch_size, n_nodes = edge_attr_logits.shape[0], edge_attr_logits.shape[1]
        device = edge_attr_logits.device  # Get the device of input tensors

        # diagonal mask (self-bonds)
        diag_mask = torch.eye(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1).bool()

        # Padding node mask
        padding_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool, device=device)

        if hasattr(batch, 'num_real_atoms'):
            for b in range(batch_size):
                num_real = batch.num_real_atoms[b]
                # Create padding_nodes tensor on the right device
                padding_nodes = torch.arange(n_nodes, device=device) >= num_real
                # Use boolean operations on device-matched tensors
                padding_mask[b] = padding_nodes.unsqueeze(0) | padding_nodes.unsqueeze(1)

        # Combined mask of bonds to ignore in loss
        ignore_mask = diag_mask | padding_mask

        # Create inverse mask for bonds we want to consider
        consider_mask = ~ignore_mask

        # Expand consider_mask to match dimensions for indexing
        consider_mask_expanded = consider_mask.unsqueeze(-1).expand(-1, -1, -1, adj_target.size(-1))

        # Calculate loss only on non-masked elements
        pred_flat = torch.masked_select(edge_attr_logits, consider_mask_expanded)
        target_flat = torch.masked_select(adj_target, consider_mask_expanded)

        loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)

        return loss

# TODO interpret molecular graph function
