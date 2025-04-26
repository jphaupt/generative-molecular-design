import pytest
import torch

from mygenai.models.layers import EquivariantMPNNLayer

def test_equivariant_mpnn_layer_init():
    """Test that EquivariantMPNNLayer initializes correctly"""
    layer = EquivariantMPNNLayer(emb_dim=64, edge_dim=4)
    assert layer.emb_dim == 64
    assert layer.edge_dim == 4
    assert layer.aggr == 'add'

def test_rot_trans_equivariance(single_batch, random_orthogonal_matrix, device):
    """Test rotation and translation equivariance of EquivariantMPNNLayer"""
    batch = single_batch.to(device)
    layer = EquivariantMPNNLayer(emb_dim=11, edge_dim=4).to(device)

    # First forward pass
    h_update1, pos_update1 = layer(batch.x, batch.pos, batch.edge_index, batch.edge_attr)

    # Apply rotation and translation
    Q = random_orthogonal_matrix.to(device)
    t = torch.rand(3, device=device)

    # clone batch to avoid modifying the original data
    batch_rotated = batch.clone()
    batch_rotated.pos = batch.pos @ Q + t

    # Second forward pass
    h_update2, pos_update2 = layer(batch_rotated.x, batch_rotated.pos, batch_rotated.edge_index, batch_rotated.edge_attr)

    # Node features should be invariant to rotation and translation
    assert torch.allclose(h_update1, h_update2, atol=1e-4)

    # Position updates should be equivariant to rotation and translation
    # This means: rotate_and_translate(pos_update1) = pos_update2
    pos_update1_transformed = pos_update1 @ Q + t
    assert torch.allclose(pos_update1_transformed, pos_update2, atol=1e-4)
