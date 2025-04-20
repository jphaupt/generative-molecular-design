import pytest
import torch
from mygenai.models.decoders import ConditionalDecoder
import torch_geometric

@pytest.fixture
def decoder():
    """Fixture to initialize the ConditionalDecoder."""
    return ConditionalDecoder(latent_dim=32, emb_dim=64, out_node_dim=11, out_edge_dim=4)


def test_output_dimensions(decoder, single_batch):
    """
    Test that the decoder outputs tensors with the correct dimensions.
    """
    z = torch.randn(single_batch.batch.max().item() + 1, 32)  # Latent space embedding
    target_property = torch.randn(z.size(0), 1)  # Target property

    # Forward pass
    node_features, distances, directions, edge_features, num_nodes = decoder(z, target_property, single_batch)

    assert node_features.shape == (single_batch.num_nodes, 11), "Node features have incorrect shape."
    assert distances.shape == (single_batch.edge_index.size(1), 1), "Distances have incorrect shape."
    assert directions.shape == (single_batch.edge_index.size(1), 3), "Directions have incorrect shape."
    assert edge_features.shape == (single_batch.edge_index.size(1), 4), "Edge features have incorrect shape."
    assert num_nodes.shape == (z.size(0),), "Number of nodes has incorrect shape."


def test_output_ranges(decoder, single_batch):
    """
    Test that the decoder outputs are within expected ranges.
    """
    z = torch.randn(single_batch.batch.max().item() + 1, 32)  # Latent space embedding
    target_property = torch.randn(z.size(0), 1)  # Target property

    # Forward pass
    node_features, distances, directions, edge_features, num_nodes = decoder(z, target_property, single_batch)

    assert torch.all(distances >= 0), "Distances contain negative values."

    norms = torch.norm(directions, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Directions are not unit vectors."

    assert torch.all(edge_features >= 0), "Edge features contain negative values."
    assert torch.all(edge_features <= 1), "Edge features exceed 1."

    assert torch.all(num_nodes >= 0), "Number of nodes contains negative values."


def test_batch_processing(decoder, dataloader):
    """
    Test that the decoder works correctly with batched inputs.
    """
    batch = next(iter(dataloader))  # Get a batch from the dataloader
    z = torch.randn(batch.batch.max().item() + 1, 32)  # Latent space embedding
    target_property = torch.randn(z.size(0), 1)  # Target property

    # Forward pass
    node_features, distances, directions, edge_features, num_nodes = decoder(z, target_property, batch)

    assert node_features.shape == (batch.num_nodes, 11), "Node features have incorrect shape for batched input."
    assert distances.shape == (batch.edge_index.size(1), 1), "Distances have incorrect shape for batched input."
    assert directions.shape == (batch.edge_index.size(1), 3), "Directions have incorrect shape for batched input."
    assert edge_features.shape == (batch.edge_index.size(1), 4), "Edge features have incorrect shape for batched input."
    assert num_nodes.shape == (z.size(0),), "Number of nodes has incorrect shape for batched input."


def test_edge_cases(decoder, device):
    """
    Test edge cases such as very small or large graphs.
    """
    # Move the decoder to the appropriate device
    decoder = decoder.to(device)
    decoder.eval()  # Use evaluation mode to avoid batch size issues

    # Create latent space embedding and target property on the same device
    z = torch.randn(1, 32, device=device)  # Latent space embedding
    target_property = torch.randn(1, 1, device=device)  # Target property

    # Very small graph (1 node)
    small_data = torch_geometric.data.Data(
        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
        batch=torch.zeros(1, dtype=torch.long, device=device),
        num_nodes=1,
    )
    node_features, distances, directions, edge_features, num_nodes = decoder(z, target_property, small_data)
    assert node_features.shape == (1, 11), "Node features shape incorrect for small graph."
    assert distances.shape == (0, 1), "Distances shape incorrect for small graph."
    assert directions.shape == (0, 3), "Directions shape incorrect for small graph."
    assert edge_features.shape == (0, 4), "Edge features shape incorrect for small graph."
    assert num_nodes.shape == (1,), "Number of nodes shape incorrect for small graph."

    # Large graph (100 nodes)
    num_nodes_ground_truth = 100  # Ground truth number of nodes
    large_data = torch_geometric.data.Data(
        edge_index=torch.combinations(torch.arange(num_nodes_ground_truth, device=device), r=2).t(),
        batch=torch.zeros(num_nodes_ground_truth, dtype=torch.long, device=device),
        num_nodes=num_nodes_ground_truth,
    )
    node_features, distances, directions, edge_features, num_nodes = decoder(z, target_property, large_data)
    assert node_features.shape == (num_nodes_ground_truth, 11), "Node features shape incorrect for large graph."
    assert distances.shape == (large_data.edge_index.size(1), 1), "Distances shape incorrect for large graph."
    assert directions.shape == (large_data.edge_index.size(1), 3), "Directions shape incorrect for large graph."
    assert edge_features.shape == (large_data.edge_index.size(1), 4), "Edge features shape incorrect for large graph."
    assert num_nodes.shape == (1,), "Number of nodes shape incorrect for large graph."

    # Check that the predicted number of nodes is reasonable
    assert torch.all(num_nodes >= 0), "Predicted number of nodes contains negative values."
