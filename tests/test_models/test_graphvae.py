import pytest
import torch

from mygenai.models.graphvae import PropertyConditionedVAE

def test_rot_trans_invariance(single_batch, random_orthogonal_matrix, device):
    """Test rotation and translation invariance of PropertyConditionedVAE"""
    batch = single_batch.to(device)
    model = PropertyConditionedVAE(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, latent_dim=32).to(device)

    # Set model to eval mode to make outputs deterministic
    model.eval()

    # First forward pass
    with torch.no_grad():
        outputs1 = model(batch)
        node_features1, distances1, directions1, edge_features1, num_nodes1, mu1, log_var1, property_pred1 = outputs1

    # Apply rotation and translation
    Q = random_orthogonal_matrix.to(device)
    t = torch.rand(3, device=device)
    batch_rotated = batch.clone()
    batch_rotated.pos = batch.pos @ Q + t

    # Second forward pass
    with torch.no_grad():
        outputs2 = model(batch_rotated)
        node_features2, distances2, directions2, edge_features2, num_nodes2, mu2, log_var2, property_pred2 = outputs2

    # Check that latent representations and property predictions are invariant
    assert torch.allclose(mu1, mu2, atol=1e-4), "Latent means are not invariant to rotation/translation."
    assert torch.allclose(log_var1, log_var2, atol=1e-4), "Latent log variances are not invariant to rotation/translation."
    assert torch.allclose(property_pred1, property_pred2, atol=1e-4), "Property predictions are not invariant to rotation/translation."
