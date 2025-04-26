import logging
import torch
import torch.nn.functional as F
from torch.nn import Module

from mygenai.models.encoders import GraphEncoder
from mygenai.models.decoders import GraphDecoder

class GraphVAE(Module):
    def __init__(self, node_feat_dim=5, emb_dim=32, latent_dim=32):
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
        logits = self.decoder(z)
        return logits, mu, logvar, property_pred

    def loss_function(self, logits, node_feats):
        # Reconstruction loss (assuming node_feats are real-valued)
        recon_loss = F.mse_loss(logits, node_feats, reduction='sum')
        return recon_loss
