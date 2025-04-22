import logging
import torch
import torch.nn.functional as F
from torch.nn import Module

from mygenai.models.encoders import Encoder
from mygenai.models.decoders import ConditionalDecoder

class PropertyConditionedVAE(Module):
    """
    Variational Autoencoder (VAE) for property-conditioned molecular generation.

    Attributes
    ----------
    encoder : Encoder
        The encoder module for encoding input graphs into latent space.
    decoder : ConditionalDecoder
        The decoder module for generating molecular graphs from latent space.
    latent_dim : int
        Dimensionality of the latent space.
    """

    def __init__(self, num_layers=4, emb_dim=64, in_dim=5, edge_dim=4, latent_dim=32, max_distance=2.0, min_distance=0.8):
        """
        Initialize the PropertyConditionedVAE.

        Parameters
        ----------
        num_layers : int, optional
            Number of layers in the encoder and decoder, by default 4.
        emb_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        in_dim : int, optional
            Dimensionality of the input node features, by default 11.
        edge_dim : int, optional
            Dimensionality of the input edge features, by default 4.
        latent_dim : int, optional
            Dimensionality of the latent space, by default 32.
        max_distance : float, optional
            Fixed maximum distance for normalization, by default 2.0.
        min_distance : float, optional
            Fixed minimum distance for normalization, by default 0.8.
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.encoder = Encoder(emb_dim, in_dim, edge_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, emb_dim, in_dim, max_distance=max_distance, min_distance=min_distance)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        Perform the reparameterization trick to sample from the latent space.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent space distribution.
        log_var : torch.Tensor
            Log variance of the latent space distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        # The small epsilon prevents underflow for very negative log_var
        std = torch.exp(0.5 * log_var) + 1e-10
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, target_property=None):
        """
        Forward pass through the VAE.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data batch.
        target_property : torch.Tensor, optional
            Target property for conditioning, by default None.

        Returns
        -------
        tuple
            Tuple containing:
            - node_features (torch.Tensor): Predicted node features.
            - distances (torch.Tensor): Predicted distances for edges.
            - directions (torch.Tensor): Predicted direction vectors for edges.
            - edge_features (torch.Tensor): Predicted edge properties.
            - num_nodes (torch.Tensor): Predicted number of nodes.
            - edge_existence (torch.Tensor): Predicted edge existence probabilities.
            - mu (torch.Tensor): Latent mean.
            - log_var (torch.Tensor): Latent log variance.
            - property_pred (torch.Tensor): Predicted property.
        """
        logger = logging.getLogger('PropertyConditionedVAE')

        # Safely log input information
        batch_size = data.batch.max().item() + 1
        logger.debug(f"Input data - batch_size: {batch_size}, nodes: {data.x.shape[0]}")
        if target_property is not None:
            logger.debug(f"Forward called with target_property shape: {target_property.shape}")
        else:
            logger.debug("Forward called without target_property (None)")

        # Add small noise to node positions during training
        if self.training:
            data.pos = data.pos + torch.randn_like(data.pos) * 0.01

        # Encode
        mu, log_var, property_pred = self.encoder(data)
        logger.debug(f"Encoder outputs - mu: {mu.shape}, log_var: {log_var.shape}, property_pred: {property_pred.shape}")

        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        logger.debug(f"Sampled z shape: {z.shape}")

        # Determine decoder property
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

        # Decode
        node_features, distances, directions, edge_features, num_nodes, edge_existence = self.decoder(
            z, decoder_property, data
        )

        logger.debug(
            f"Decoder outputs - "
            f"node_features: {node_features.shape}, "
            f"distances: {distances.shape}, "
            f"directions: {directions.shape}, "
            f"edge_features: {edge_features.shape}, "
            f"num_nodes: {num_nodes}, "
            f"edge_existence: {edge_existence.shape}"
        )

        return node_features, distances, directions, edge_features, num_nodes, edge_existence, mu, log_var, property_pred

    def loss_function(self, node_features, distances, directions, edge_features, num_nodes, edge_existence, data, mu, log_var,
                      property_pred, property_weight=1.0, recon_weight=1.0, kl_weight=0.1):
        """
        Compute the VAE loss with detailed logging for debugging.

        Parameters
        ----------
        node_features : torch.Tensor
            Predicted node features.
        distances : torch.Tensor
            Predicted distances for edges.
        directions : torch.Tensor
            Predicted direction vectors for edges.
        edge_features : torch.Tensor
            Predicted edge properties.
        num_nodes : torch.Tensor
            Predicted number of nodes.
        edge_existence : torch.Tensor
            Predicted edge existence probabilities.
        data : torch_geometric.data.Data
            Ground truth graph data.
        mu : torch.Tensor
            Latent mean.
        log_var : torch.Tensor
            Latent log variance.
        property_pred : torch.Tensor
            Predicted property.
        property_weight : float, optional
            Weight for property prediction loss, by default 1.0.
        recon_weight : float, optional
            Weight for reconstruction loss, by default 1.0.
        kl_weight : float, optional
            Weight for KL divergence loss, by default 0.1.

        Returns
        -------
        torch.Tensor
            Total loss.
        """
        logger = logging.getLogger(self.__class__.__name__)

        # Node feature reconstruction loss
        node_loss = F.cross_entropy(node_features, data.x.argmax(dim=-1))
        logger.debug(f"Node feature loss: {node_loss.item():.6f}")

        # Edge existence loss
        edge_existence_loss = F.binary_cross_entropy(edge_existence.squeeze(-1), data.edge_existence.float())
        logger.debug(f"Edge existence loss: {edge_existence_loss.item():.6f}")

        # Filter edges that exist in the ground truth
        existing_edges = data.edge_existence.bool()

        # Distance reconstruction loss
        ground_truth_distances = torch.norm(data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]], dim=1, keepdim=True)
        normalized_gt_distances = (ground_truth_distances - self.decoder.min_distance) / (self.decoder.max_distance - self.decoder.min_distance)
        distance_loss = F.mse_loss(
            distances[existing_edges],
            normalized_gt_distances[existing_edges]
        )

        # Direction reconstruction loss
        relative_directions = data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
        ground_truth_directions = relative_directions / (ground_truth_distances + 1e-10)
        direction_loss = 1 - F.cosine_similarity(directions, ground_truth_directions[existing_edges], dim=1).mean()

        # Bond type loss
        edge_classes = data.edge_attr[existing_edges].argmax(dim=-1)
        edge_loss = F.cross_entropy(edge_features, edge_classes)
        logger.debug(f"Edge feature loss: {edge_loss.item():.6f}")

        # Number of nodes loss
        num_nodes_loss = F.mse_loss(
            num_nodes,
            torch.bincount(data.batch, minlength=data.batch.max().item() + 1).float()
        )
        logger.debug(f"Number of nodes loss: {num_nodes_loss.item():.6f}")

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        logger.debug(f"KL divergence loss: {kl_loss.item():.6f}")

        # Property prediction loss
        target_property = data.y[:, 4:5]
        property_loss = F.mse_loss(property_pred, target_property)
        logger.debug(f"Property prediction loss: {property_loss.item():.6f}")

        component_weights = {
            'node': 50.0,
            'edge_existence': 20.0,
            'distance': 100.0,  # Make this very important
            'direction': 50.0,
            'edge': 50.0,
            'num_nodes': 10.0
        }
        recon_loss = (component_weights['node'] * node_loss +
                      component_weights['edge_existence'] * edge_existence_loss +
                      component_weights['distance'] * distance_loss +
                      component_weights['direction'] * direction_loss +
                      component_weights['edge'] * edge_loss +
                      component_weights['num_nodes'] * num_nodes_loss)
        logger.debug(f"Reconstruction loss: {recon_loss.item():.6f}")

        # Log individual loss components
        logger.info(f"Loss components - Node: {node_loss:.6f}, Edge: {edge_loss:.6f}, "
                    f"Edge Existence: {edge_existence_loss:.6f}, Distance: {distance_loss:.6f}, "
                    f"Direction: {direction_loss:.6f}, Num Nodes: {num_nodes_loss:.6f}, "
                    f"KL: {kl_loss:.6f}, Property: {property_loss:.6f}")

        # Total loss
        total_loss = (recon_weight * recon_loss +
                      kl_weight * kl_loss +
                      property_weight * property_loss)
        logger.debug(f"Total loss: {total_loss.item():.6f}")

        return total_loss

    def generate_molecule(self, target_property, num_samples=1):
        """TODO"""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
            target = torch.ones(num_samples, 1).to(z.device) * target_property # single value for homo-lumo gap

            # Generate
            node_features, positions, num_nodes = self.decoder(z, target, num_samples)

            return node_features, positions, num_nodes
