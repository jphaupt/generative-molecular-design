# ensure all tensor dimensions match throughout the models

import pytest
import torch
from mygenai.models.encoders import GraphEncoder
from mygenai.models.decoders import GraphDecoder
from mygenai.models.graphvae import GraphVAE

def test_encoder_dimensions(v0_model, v0_single_batch):
    mu, logvar, property_predictor = v0_model.encoder(v0_single_batch)
    batch_size = v0_single_batch.num_graphs
    assert False # Placeholder for actual dimension checks
