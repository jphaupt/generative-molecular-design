# ensure all tensor dimensions match throughout the models

import pytest
import torch
from mygenai.models.encoders import GraphEncoder
from mygenai.models.decoders import GraphDecoder
from mygenai.models.graphvae import GraphVAE

def test_encoder_dimensions(v0_model, v0_single_batch):
    mu, logvar, property_predictor = v0_model.encoder(v0_single_batch)
    batch_size = v0_single_batch.num_graphs
    emb_dim = 32
    assert mu.shape == (batch_size, emb_dim), f"Expected mu shape {(batch_size, emb_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, emb_dim), f"Expected logvar shape {(batch_size, emb_dim)}, got {logvar.shape}"
    assert property_predictor.shape == (batch_size, 1), f"Expected property_predictor shape {(batch_size, 1)}, got {property_predictor.shape}"

def test_decoder_dimensions(v0_model, v0_single_batch, v0_random_latent_vector):
    batch_size = v0_single_batch.num_graphs
    edge_logits = v0_model.decoder(v0_random_latent_vector)
    assert edge_logits.shape == (batch_size, v0_model.decoder.num_nodes, v0_model.decoder.num_nodes), \
        f"Expected edge_logits shape {(batch_size, v0_model.decoder.num_nodes, v0_model.decoder.num_nodes)}, got {edge_logits.shape}"

def test_graphvae_dimensions(v0_model, v0_single_batch):
    logits, mu, logvar, property_pred = v0_model(v0_single_batch)
    batch_size = v0_single_batch.num_graphs
    emb_dim = 32
    assert logits.shape == (batch_size, v0_model.decoder.num_nodes, v0_model.decoder.num_nodes), \
        f"Expected logits shape {(batch_size, v0_model.decoder.num_nodes, v0_model.decoder.num_nodes)}, got {logits.shape}"
    assert mu.shape == (batch_size, emb_dim), f"Expected mu shape {(batch_size, emb_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, emb_dim), f"Expected logvar shape {(batch_size, emb_dim)}, got {logvar.shape}"
    assert property_pred.shape == (batch_size, 1), f"Expected property_pred shape {(batch_size, 1)}, got {property_pred.shape}"

def test_graphvae_reparam_dimensions(v0_model, v0_single_batch, device):
    # create dummy mu and logvar
    batch_size = v0_single_batch.num_graphs
    emb_dim = 32
    mu = torch.randn(batch_size, emb_dim).to(device)
    logvar = torch.randn(batch_size, emb_dim).to(device)

    z = v0_model.reparameterize(mu, logvar)
    assert z.shape == (batch_size, emb_dim), f"Expected z shape {(batch_size, emb_dim)}, got {z.shape}"

    # also check that z is different from mu (i.e. that reparameterization worked)
    assert not torch.allclose(z, mu), "Reparameterization did not work as expected: z is equal to mu"
