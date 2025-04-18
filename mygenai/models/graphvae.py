import logging
import torch
import torch.nn.functional as F
from torch.nn import Module

from mygenai.models.encoders import Encoder
from mygenai.models.decoders import ConditionalDecoder

class PropertyConditionedVAE(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, latent_dim=32):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.encoder = Encoder(emb_dim, in_dim, edge_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, emb_dim, in_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, data, target_property=None):
        logger = logging.getLogger('PropertyConditionedVAE')

        # Encode
        mu, log_var, property_pred = self.encoder(data)
        logger.debug(f"Encoder outputs - mu: {mu.shape}, log_var: {log_var.shape}, property_pred: {property_pred.shape}")

        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        logger.debug(f"Sampled z shape: {z.shape}")

        # Use predicted property if target not provided
        if target_property is None:
            target_property = property_pred
        else:
            # if all properties provided, extract just the HOMO-LUMO gap
            if target_property.size(1) != 1:
                target_property = target_property[:, 4:5]
        logger.debug(f"Target property shape before squeeze: {target_property.shape}")

        # # Ensure target_property is 2D
        # if len(target_property.shape) == 3:
        #     target_property = target_property.squeeze(1)
        # logger.debug(f"Target property shape after squeeze: {target_property.shape}")

        # Decode
        node_features, positions, num_nodes = self.decoder(
            z, target_property, data.batch.max().item() + 1
        )
        logger.debug(f"Decoder outputs - features: {node_features.shape}, positions: {positions.shape}")

        return node_features, positions, mu, log_var, property_pred, num_nodes

    def loss_function(self, node_features, positions, num_nodes, data, mu, log_var,
                    property_pred, property_weight=1.0):
        logger = logging.getLogger(self.__class__.__name__)

        # Log shapes for debugging
        logger.debug(f"Property prediction shape: {property_pred.shape}")
        logger.debug(f"Target property shape: {data.y.shape}")

        # Get batch size
        batch_size = data.batch.max().item() + 1

        # Reconstruction loss (with proper masking for variable size graphs)
        recon_loss = 0
        start_idx = 0
        total_nodes = 0

        for i, n in enumerate(num_nodes):
            n_orig = (data.batch == i).sum()
            n_gen = n.item()
            nodes_to_compare = min(n_gen, n_orig)
            total_nodes += nodes_to_compare

            if nodes_to_compare > 0:
                # Node feature reconstruction - use sum reduction
                recon_loss += F.mse_loss(
                    node_features[start_idx:start_idx + nodes_to_compare],
                    data.x[data.batch == i][:nodes_to_compare],
                    reduction='sum'  # Sum within each graph
                )

                # Position reconstruction - use sum reduction
                recon_loss += F.mse_loss(
                    positions[start_idx:start_idx + nodes_to_compare],
                    data.pos[data.batch == i][:nodes_to_compare],
                    reduction='sum'  # Sum within each graph
                )

            start_idx += n_gen

        # Normalize reconstruction loss by total nodes compared
        if total_nodes > 0:
            recon_loss = recon_loss / total_nodes

        # KL divergence (already normalized by batch size)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

        # # Property prediction loss - ensure shapes match
        # if property_pred.shape != data.y.shape:
        #     logger.warning(f"Property shape mismatch: pred={property_pred.shape}, target={data.y.shape}")
        #     # Fix the shape of either property_pred or data.y to match
        #     if property_pred.size(1) == 1 and data.y.size(1) > 1:
        #         # Need to modify your model to output the correct shape (19 columns)
        #         # As a temporary fix, repeat the single value to match the target width
        #         property_pred = property_pred.expand(-1, data.y.size(1))

        target_property = data.y[:, 4:5]  # HOMO-LUMO gap, keep dimension as [batch_size, 1]
        prop_loss = F.mse_loss(property_pred, target_property, reduction='mean')

        # Combine losses with scaling factors
        # Use smaller coefficients to prevent overflow
        total_loss = recon_loss + 0.01 * kl_loss + 0.1 * property_weight * prop_loss

        # Add guard against NaN or Inf
        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss detected! recon={recon_loss}, kl={kl_loss}, prop={prop_loss}")
            # Return a backup loss that won't break training
            return torch.tensor(1000.0, device=total_loss.device, requires_grad=True)

        # Log component values for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Losses - recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, prop: {prop_loss.item():.4f}, total: {total_loss.item():.4f}")

        return total_loss

    def generate_molecule(self, target_property, num_samples=1):
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
            target = torch.ones(num_samples, 1).to(z.device) * target_property # single value for homo-lumo gap

            # Generate
            node_features, positions, num_nodes = self.decoder(z, target, num_samples)

            return node_features, positions, num_nodes
