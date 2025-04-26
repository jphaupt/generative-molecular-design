# shared fixtures
import pytest
import torch
import numpy as np
from scipy.stats import ortho_group
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from mygenai.utils.transforms import CompleteGraph

@pytest.fixture
def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q

@pytest.fixture
def device():
    """Get available device (CPU or CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
