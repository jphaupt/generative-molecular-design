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
        """
        Forward pass through the VAE.

        Args:
            data: PyG Data batch
            target_property: Optional target property (used during generation)

        Returns:
            node_features, positions, mu, log_var, property_pred, num_nodes
        """
        logger = logging.getLogger('PropertyConditionedVAE')

        # Safely log input information
        batch_size = data.batch.max().item() + 1
        logger.debug(f"Input data - batch_size: {batch_size}, nodes: {data.x.shape[0]}")
        if target_property is not None:
            logger.debug(f"Forward called with target_property shape: {target_property.shape}")
        else:
            logger.debug("Forward called without target_property (None)")

        # Encode
        mu, log_var, property_pred = self.encoder(data)
        logger.debug(f"Encoder outputs - mu: {mu.shape}, log_var: {log_var.shape}, property_pred: {property_pred.shape}")

        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        logger.debug(f"Sampled z shape: {z.shape}")

        # For conditioning the decoder:
        if self.training and target_property is None:
            # Use true property from data (teacher forcing)
            decoder_property = data.y[:, 4:5]  # HOMO-LUMO gap
            logger.debug(f"Using teacher forcing with property shape: {decoder_property.shape}")
        elif target_property is not None:
            # Use provided target property (for generation or specific conditioning)
            decoder_property = target_property
            if decoder_property.size(1) != 1:
                decoder_property = decoder_property[:, 4:5]
            logger.debug(f"Using provided target property, shape after processing: {decoder_property.shape}")
        else:
            # During validation without teacher forcing, use encoder prediction
            decoder_property = property_pred
            logger.debug(f"Using encoder prediction for property, shape: {decoder_property.shape}")

        # IMPORTANT: Decode with decoder_property, NOT target_property
        node_features, positions, num_nodes = self.decoder(
            z, decoder_property, batch_size
        )

        logger.debug(f"Decoder outputs - features: {node_features.shape}, positions: {positions.shape}")

        return node_features, positions, mu, log_var, property_pred, num_nodes

    def loss_function(self, node_features, positions, num_nodes, data, mu, log_var,
                    property_pred, property_weight=1.0, recon_weight=1.0, kl_weight=0.1):
        logger = logging.getLogger(self.__class__.__name__)

        # Check for value instability early
        if torch.isnan(node_features).any() or torch.isnan(positions).any():
            logger.error("NaN values detected in model outputs!")
            return torch.tensor(1000.0, device=node_features.device, requires_grad=True)

        # Check for extremely large values early
        max_node_value = node_features.abs().max().item()
        max_pos_value = positions.abs().max().item()
        if max_node_value > 100 or max_pos_value > 100:
            logger.error(f"Large values detected! Features: {max_node_value}, Positions: {max_pos_value}")
            # Add debugging info to pinpoint the issue
            logger.error(f"Node features range: {node_features.min().item()} to {node_features.max().item()}")
            logger.error(f"Positions range: {positions.min().item()} to {positions.max().item()}")
            return torch.tensor(1000.0, device=node_features.device, requires_grad=True)

        # Log shapes for debugging
        logger.debug(f"Property prediction shape: {property_pred.shape}")
        logger.debug(f"Target property shape: {data.y.shape}")

        # Get batch size
        batch_size = data.batch.max().item() + 1

        # Reconstruction loss (with proper masking for variable size graphs)
        recon_loss = 0
        start_idx = 0
        total_nodes = 0

        # Node feature and position losses with consistent handling
        feature_loss = 0.0
        position_loss = 0.0

        for i, n in enumerate(num_nodes):
            n_orig = (data.batch == i).sum()
            n_gen = n.item()
            nodes_to_compare = min(n_gen, n_orig)
            total_nodes += nodes_to_compare

            if nodes_to_compare > 0:
                # Split losses for better scaling
                feature_loss += F.mse_loss(
                    node_features[start_idx:start_idx + nodes_to_compare],
                    data.x[data.batch == i][:nodes_to_compare],
                    reduction='mean'  # Use mean within each graph
                )

                position_loss += F.mse_loss(
                    positions[start_idx:start_idx + nodes_to_compare],
                    data.pos[data.batch == i][:nodes_to_compare],
                    reduction='mean'  # Use mean within each graph
                )

            start_idx += n_gen

        # Normalize by number of graphs (not total nodes)
        feature_loss = feature_loss / batch_size
        position_loss = position_loss / batch_size

        # Weight the position loss differently than feature loss
        # since positions have different scale than node features
        recon_loss = feature_loss + position_loss

        # KL divergence (already normalized by batch size)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

        # Property prediction loss - this is for the encoder's prediction ability
        target_property = data.y[:, 4:5]  # HOMO-LUMO gap
        prop_loss = F.mse_loss(property_pred, target_property, reduction='mean')

        # normalize each term more consistently
        kl_loss = kl_loss / batch_size
        prop_loss = prop_loss  # Already normalized by mean

        # Combine losses with scaling factors
        # Use smaller coefficients to prevent overflow
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss + property_weight * prop_loss

        # Add guard against NaN or Inf
        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss detected! recon={recon_loss}, kl={kl_loss}, prop={prop_loss}")
            # Return a backup loss that won't break training
            return torch.tensor(1000.0, device=total_loss.device, requires_grad=True)

        # log component values
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
