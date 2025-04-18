import sys
import torch
import numpy as np
import unittest
from torch_geometric.data import Data, Batch
import logging

# Test framework for PropertyConditionedVAE in 03_train_mp_graphvae.ipynb


# Import the model from the parent directory
sys.path.append('..')
from models.property_vae import PropertyConditionedVAE  # Adjust import path if needed

class TestPropertyConditionedVAE(unittest.TestCase):
    def setUp(self):
        # Set up a small model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PropertyConditionedVAE(
            node_features=11,
            hidden_dim=16,  # Smaller dimension for faster tests
            latent_dim=8,   # Smaller dimension for faster tests
        ).to(self.device)

        # Create a small test batch with 3 molecules
        self.test_batch = self._create_test_batch()

    def _create_test_batch(self):
        # Create a small batch of 3 test molecules
        graphs = []

        # Molecule 1: Small molecule with 5 nodes
        x1 = torch.randn(5, 11)  # Node features
        pos1 = torch.randn(5, 3)  # 3D positions
        edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                    [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        y1 = torch.tensor([1.5])  # Property value
        graphs.append(Data(x=x1, pos=pos1, edge_index=edge_index1, y=y1))

        # Molecule 2: Medium molecule with 7 nodes
        x2 = torch.randn(7, 11)
        pos2 = torch.randn(7, 3)
        edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]], dtype=torch.long)
        y2 = torch.tensor([2.7])
        graphs.append(Data(x=x2, pos=pos2, edge_index=edge_index2, y=y2))

        # Molecule 3: Small molecule with 4 nodes
        x3 = torch.randn(4, 11)
        pos3 = torch.randn(4, 3)
        edge_index3 = torch.tensor([[0, 1, 1, 2, 2, 3],
                                    [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        y3 = torch.tensor([0.8])
        graphs.append(Data(x=x3, pos=pos3, edge_index=edge_index3, y=y3))

        batch = Batch.from_data_list(graphs)
        return batch.to(self.device)

    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.decoder)

        # Check encoder has message passing layers
        self.assertTrue(hasattr(self.model.encoder, 'mp_layers'))

        # Check decoder has property conditioning
        self.assertTrue(hasattr(self.model.decoder, 'property_embedding'))

    def test_forward_pass(self):
        """Test the forward pass works without errors"""
        try:
            node_features, positions, mu, log_var, property_pred, num_nodes = self.model(self.test_batch)

            # Check output shapes
            self.assertEqual(mu.shape[1], self.model.latent_dim)
            self.assertEqual(log_var.shape[1], self.model.latent_dim)
            self.assertEqual(property_pred.shape[0], 3)  # 3 molecules
            self.assertEqual(property_pred.shape[1], 1)  # 1 property

            # Check number of reconstructed nodes matches input
            total_nodes = self.test_batch.x.shape[0]  # 5+7+4 = 16 nodes
            self.assertEqual(node_features.shape[0], total_nodes)
            self.assertEqual(positions.shape[0], total_nodes)

            print("Forward pass successful with shapes:")
            print(f"  node_features: {node_features.shape}")
            print(f"  positions: {positions.shape}")
            print(f"  mu: {mu.shape}")
            print(f"  log_var: {log_var.shape}")
            print(f"  property_pred: {property_pred.shape}")
            print(f"  num_nodes: {num_nodes}")
        except Exception as e:
            self.fail(f"Forward pass failed with error: {str(e)}")

    def test_loss_calculation(self):
        """Test that loss calculation works and produces reasonable values"""
        # First get model outputs
        node_features, positions, mu, log_var, property_pred, num_nodes = self.model(self.test_batch)

        # Calculate loss
        try:
            loss = self.model.loss_function(
                node_features, positions, num_nodes,
                self.test_batch, mu, log_var, property_pred
            )

            # Check loss is valid (not NaN, not infinite)
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())

            # Check loss has reasonable value (usually starts high)
            self.assertGreater(loss.item(), 0)
            print(f"Initial loss: {loss.item()}")

            # Check loss can be backpropagated
            loss.backward()
            print("Loss backpropagation successful")
        except Exception as e:
            self.fail(f"Loss calculation failed with error: {str(e)}")

    def test_numerical_stability(self):
        """Test for numerical stability issues (NaNs in intermediates)"""
        # Enable detection of NaNs
        torch.set_anomaly_enabled(True)

        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        try:
            # Run a few optimization steps
            for i in range(3):
                optimizer.zero_grad()

                # Forward pass
                node_features, positions, mu, log_var, property_pred, num_nodes = self.model(self.test_batch)

                # Check intermediates for NaN
                self.assertFalse(torch.isnan(mu).any(), f"NaN found in mu at step {i}")
                self.assertFalse(torch.isnan(log_var).any(), f"NaN found in log_var at step {i}")
                self.assertFalse(torch.isnan(node_features).any(), f"NaN found in node_features at step {i}")
                self.assertFalse(torch.isnan(positions).any(), f"NaN found in positions at step {i}")
                self.assertFalse(torch.isnan(property_pred).any(), f"NaN found in property_pred at step {i}")

                # Calculate loss
                loss = self.model.loss_function(
                    node_features, positions, num_nodes,
                    self.test_batch, mu, log_var, property_pred
                )

                # Check loss for NaN
                self.assertFalse(torch.isnan(loss).any(), f"NaN found in loss at step {i}")

                # Backprop
                loss.backward()

                # Check gradients for NaN
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.assertFalse(torch.isnan(param.grad).any(),
                                         f"NaN found in gradients of {name} at step {i}")

                optimizer.step()
                print(f"Optimization step {i} completed with loss: {loss.item()}")

        except Exception as e:
            self.fail(f"Numerical stability test failed with error: {str(e)}")
        finally:
            torch.set_anomaly_enabled(False)

    def test_property_prediction(self):
        """Test that property prediction is reasonable"""
        # Put model in eval mode
        self.model.eval()

        # Forward pass
        with torch.no_grad():
            _, _, _, _, property_pred, _ = self.model(self.test_batch)

            # Property predictions should be in a reasonable range
            true_properties = self.test_batch.y

            # Print property predictions
            print("Property prediction test:")
            print(f"  True properties: {true_properties.cpu().numpy()}")
            print(f"  Predicted properties: {property_pred.cpu().numpy()}")

            # Even untrained model shouldn't give extreme values
            self.assertTrue(property_pred.min() > -10)
            self.assertTrue(property_pred.max() < 10)

    def test_equivariance(self):
        """Test rotation equivariance of the model"""
        # Original forward pass
        node_features_orig, positions_orig, mu_orig, log_var_orig, property_pred_orig, num_nodes_orig = \
            self.model(self.test_batch)

        # Create rotated batch
        rotated_batch = self._create_rotated_batch(self.test_batch)

        # Forward pass with rotated data
        node_features_rot, positions_rot, mu_rot, log_var_rot, property_pred_rot, num_nodes_rot = \
            self.model(rotated_batch)

        # The latent representations should be similar
        mu_diff = torch.abs(mu_orig - mu_rot).mean()
        log_var_diff = torch.abs(log_var_orig - log_var_rot).mean()

        # Property predictions should be the same
        prop_diff = torch.abs(property_pred_orig - property_pred_rot).mean()

        print("Equivariance test results:")
        print(f"  Latent mu difference: {mu_diff.item()}")
        print(f"  Latent log_var difference: {log_var_diff.item()}")
        print(f"  Property prediction difference: {prop_diff.item()}")

        # The latent space and property predictions should be similar after rotation
        # (this checks for translation and rotation invariance)
        self.assertLess(prop_diff.item(), 0.1,
                       "Property predictions not invariant to rotation/translation")

    def _create_rotated_batch(self, batch):
        """Create a rotated version of the input batch"""
        # Copy the batch
        rotated_batch = batch.clone()

        # Define a random rotation matrix
        theta = np.pi/4  # 45 degrees rotation
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32).to(self.device)

        # Apply rotation to positions
        rotated_batch.pos = torch.matmul(rotated_batch.pos, rotation_matrix.T)

        # Apply translation
        translation = torch.tensor([1.0, 2.0, -1.0], device=self.device)
        rotated_batch.pos = rotated_batch.pos + translation

        return rotated_batch

    def test_gradient_flow(self):
        """Test that gradients flow through all parts of the model"""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Store initial parameters
        initial_params = {name: param.clone().detach()
                         for name, param in self.model.named_parameters()}

        # Training step
        optimizer.zero_grad()
        node_features, positions, mu, log_var, property_pred, num_nodes = self.model(self.test_batch)
        loss = self.model.loss_function(
            node_features, positions, num_nodes,
            self.test_batch, mu, log_var, property_pred
        )
        loss.backward()
        optimizer.step()

        # Check if parameters have been updated
        params_changed = []
        params_unchanged = []

        for name, param in self.model.named_parameters():
            if torch.equal(param.data, initial_params[name]):
                params_unchanged.append(name)
            else:
                params_changed.append(name)

        print(f"Parameters updated: {len(params_changed)}")
        print(f"Parameters unchanged: {len(params_unchanged)}")

        if len(params_unchanged) > 0:
            print("Warning: Some parameters were not updated:")
            for name in params_unchanged:
                print(f"  - {name}")

        # At least some parameters should have changed
        self.assertGreater(len(params_changed), 0,
                          "No parameters were updated during gradient step")

        # Most parameters should have changed (allow a few corner cases)
        self.assertLess(len(params_unchanged), len(params_changed),
                       "Too many parameters were not updated")

        # Check that both encoder and decoder parameters were updated
        encoder_updated = any('encoder' in name for name in params_changed)
        decoder_updated = any('decoder' in name for name in params_changed)

        self.assertTrue(encoder_updated, "No encoder parameters were updated")
        self.assertTrue(decoder_updated, "No decoder parameters were updated")

# Run tests
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
