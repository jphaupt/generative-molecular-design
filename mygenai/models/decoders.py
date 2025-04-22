import logging
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Sigmoid, Tanh, Softmax, LayerNorm
import torch.nn.functional as F
from mygenai.models.layers import EquivariantMPNNLayer
class ConditionalDecoder(Module):
    def __init__(self, latent_dim=32, emb_dim=64, out_node_dim=5, out_edge_dim=4, max_distance=2.0, min_distance=0.8):
        """
        Initialize the decoder model for generative molecular design.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space. Defaults to 32.
        emb_dim : int
            Dimensionality of the embedding space. Defaults to 64.
        out_node_dim : int
            Dimensionality of the node feature output (e.g., atom types). Defaults to 11.
        out_edge_dim : int
            Dimensionality of the edge feature output (e.g., bond types). Defaults to 4.
        max_distance : float, optional
            Fixed maximum distance for normalization, by default 2.0.
        min_distance : float, optional
            Fixed minimum distance for normalization, by default 0.8.

        Attributes
        ----------
        lin_latent : torch.nn.Linear
            Linear layer for projecting the latent space and property input to the embedding space.
        node_decoder : torch.nn.Sequential
            Sequential model for generating node features (e.g., atom types) from embeddings.
        edge_existence : torch.nn.Sequential
            Sequential model for predicting edge existence.
        distance_decoder : torch.nn.Sequential
            Sequential model for predicting scalar bond lengths (distances) between nodes.
        direction_decoder : torch.nn.Sequential
            Sequential model for predicting direction vectors (unit vectors) for relative orientation of connected atoms.
        edge_features : torch.nn.Sequential
            Sequential model for predicting edge properties (e.g., bond types) to ensure physicality.
        num_nodes_predictor : torch.nn.Sequential
            Sequential model for predicting the number of nodes in the graph.

        Notes
        -----
        - The model uses a complete graph representation, which assumes all nodes are connected. This simplifies the model
          and is suitable for small molecules, where the higher computational cost is not a significant issue.
        - Outputs are bounded to [0, 1] using sigmoid activations where appropriate.
        - Since I kept getting bugs with this class, I have added extensive commenting to explain each step (with help
          from LLMs).
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_distance = max_distance
        self.min_distance = min_distance

        # Increase model capacity
        hidden_dim = 128

        # Initial projection from latent+property space
        # Input: (batch_size, latent_dim + 1) -> Output: (batch_size, emb_dim)
        self.lin_latent = Linear(latent_dim + 1, emb_dim)

        # Node embedding transformation
        self.node_transform = Sequential(
            Linear(emb_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim)
        )

        # Node feature generation
        # Input: (n, emb_dim) -> Output: (n, out_node_dim)
        self.node_decoder = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, out_node_dim),
            Softmax(dim=-1)  # One-hot encoding
        )

        # Edge MLP with more capacity
        self.edge_mlp = Sequential(
            Linear(2 * emb_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim)
        )

        # Edge existence prediction
        # Input: (e, 2 * emb_dim) -> Output: (e, 1)
        self.edge_existence = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, 1),
            Sigmoid()  # Output in [0, 1]
        )

        # Distance prediction (scalar bond lengths)
        # Input: (e, 2 * emb_dim) -> Output: (e, 1)
        self.distance_decoder = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, 1),
            Sigmoid()  # Constrain output to [0, 1]
        )

        # Direction vector prediction (unit vectors)
        # Input: (e, 2 * emb_dim) -> Output: (e, 3)
        self.direction_decoder = Sequential(
            Linear(2 * emb_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 3)
        )
        # Initialize with random unit vectors
        with torch.no_grad():
            rand_dirs = torch.randn(self.direction_decoder[-1].weight.shape)
            self.direction_decoder[-1].weight.data = F.normalize(rand_dirs, dim=1)

        # Edge property prediction (e.g., bond type)
        # Input: (e, 2 * emb_dim) -> Output: (e, out_edge_dim)
        self.edge_features = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, out_edge_dim),
            Softmax(dim=-1)  # Predicts probabilities for each bond type
        )

        # Number of nodes prediction
        # Input: (batch_size, emb_dim) -> Output: (batch_size, 1)
        self.num_nodes_predictor = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, 1),
            ReLU()  # Predicts a positive scalar
        )

        # Add this projection for message passing compatibility
        self.edge_proj = Sequential(
            Linear(2 * emb_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, out_edge_dim)
        )

        # Equivariant MPNN layer for position refinement
        self.decoder_mpnn = EquivariantMPNNLayer(emb_dim, out_edge_dim)

    def forward(self, z, target_property, data):
        """
        Forward pass through the decoder.

        Args:
            z: Latent space embedding (batch_size, latent_dim)
            target_property: Target property (batch_size, 1)
            data: Input graph data with precomputed complete graph (PyG Data object)

        Returns:
            node_features: Predicted node features (n, out_node_dim)
            distances: Predicted distances for edges (e, 1)
            directions: Predicted direction vectors for edges (e, 3)
            edge_features: Predicted edge properties (e, out_edge_dim)
            num_nodes: Predicted number of nodes in the graph (batch_size,)
            edge_existence: Predicted edge existence probabilities (e, 1)
        """
        self.logger.debug(f"Input shapes - z: {z.shape}, target_property: {target_property.shape}")

        # Concatenate latent vector with property condition
        # Input: z (batch_size, latent_dim), target_property (batch_size, 1)
        # Output: z_cond (batch_size, latent_dim + 1)
        z_cond = torch.cat([z, target_property], dim=1)

        # Project latent space to embedding space
        # Input: z_cond (batch_size, latent_dim + 1)
        # Output: h (batch_size, emb_dim)
        h = self.lin_latent(z_cond)

        # Expand latent representation for all nodes in the graph
        # Input: h (batch_size, emb_dim), data.batch (n,)
        # Output: h_expanded (n, emb_dim)
        h_expanded = h[data.batch]

        # Transform node embeddings
        h_expanded = self.node_transform(h_expanded)

        # Predict node features
        # Input: h_expanded (n, emb_dim)
        # Output: node_features (n, out_node_dim)
        node_features = self.node_decoder(h_expanded)

        # Combine node embeddings for edge prediction
        src, dst = data.edge_index
        edge_inputs_original = torch.cat([h_expanded[src], h_expanded[dst]], dim=1)

        # Apply transformation for general edge embeddings if needed elsewhere
        edge_embedded = self.edge_mlp(edge_inputs_original)

        # Use ORIGINAL edge inputs for predictions
        edge_existence = self.edge_existence(edge_inputs_original)
        distances = self.distance_decoder(edge_inputs_original)
        edge_features = self.edge_features(edge_inputs_original)

        # Generate initial positions (randomly placed in space)
        batch_size = z.shape[0]
        radius = 1.0  # Initial radius
        pos_init = torch.randn(h_expanded.shape[0], 3).to(z.device)
        pos_init = radius * F.normalize(pos_init, dim=1)  # Normalize to unit sphere

        # Project edge features to the expected dimension
        edge_attr = self.edge_proj(edge_inputs_original)  # Project from 2*emb_dim to out_edge_dim

        # Refine positions using message passing
        pos = pos_init
        for i in range(6):
            h_update, pos_delta = self.decoder_mpnn(h_expanded, pos, data.edge_index, edge_attr)
            h_expanded = h_expanded + h_update

            # More controlled updates with each iteration
            step_size = 0.5 / (i + 1)
            pos = pos + step_size * pos_delta

        # Now extract directions from refined positions
        src, dst = data.edge_index
        rel_pos = pos[dst] - pos[src]
        directions = rel_pos / (torch.norm(rel_pos, dim=1, keepdim=True) + 1e-10)

        # Predict number of nodes
        # Input: h (batch_size, emb_dim)
        # Output: num_nodes (batch_size,)
        num_nodes = self.num_nodes_predictor(h).squeeze(-1)

        self.logger.debug(f"Output shapes - node_features: {node_features.shape}, distances: {distances.shape}, directions: {directions.shape}, edge_features: {edge_features.shape}, num_nodes: {num_nodes.shape}, edge_existence: {edge_existence.shape}")

        return node_features, distances, directions, edge_features, num_nodes, edge_existence
