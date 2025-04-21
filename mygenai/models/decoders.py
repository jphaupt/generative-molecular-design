import logging
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Sigmoid, Tanh, Softmax

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
        - Edge existence is not explicitly predicted, as the complete graph assumption is used.
        - Outputs are bounded to [0, 1] using sigmoid activations where appropriate.
        - Since I kept getting bugs with this class, I have added extensive commenting to explain each step (with help
          from LLMs).
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_distance = max_distance
        self.min_distance = min_distance

        # Initial projection from latent+property space
        # Input: (batch_size, latent_dim + 1) -> Output: (batch_size, emb_dim)
        self.lin_latent = Linear(latent_dim + 1, emb_dim)

        # Node feature generation
        # Input: (n, emb_dim) -> Output: (n, out_node_dim)
        self.node_decoder = Sequential(
            Linear(emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, out_node_dim),
            Softmax(dim=-1)  # One-hot encoding
        )

        # Distance prediction (scalar bond lengths)
        # Input: (e, 2 * emb_dim) -> Output: (e, 1)
        self.distance_decoder = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, 1),
            Sigmoid()  # Constrain output to [0, 1]
        )

        # Direction vector prediction (unit vectors)
        # Input: (e, 2 * emb_dim) -> Output: (e, 3)
        self.direction_decoder = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
            Linear(emb_dim, 3)
        )

        # Edge property prediction (e.g., bond type)
        # Input: (e, 2 * emb_dim) -> Output: (e, out_edge_dim)
        self.edge_features = Sequential(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            BatchNorm1d(emb_dim),
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

        # Predict node features
        # Input: h_expanded (n, emb_dim)
        # Output: node_features (n, out_node_dim)
        node_features = self.node_decoder(h_expanded)

        # Combine node embeddings for edge prediction
        # Input: h_expanded (n, emb_dim), data.edge_index (2, e)
        # Output: edge_inputs (e, 2 * emb_dim)
        src, dst = data.edge_index  # Precomputed complete graph
        edge_inputs = torch.cat([h_expanded[src], h_expanded[dst]], dim=1)

        # Predict distances
        # Input: edge_inputs (e, 2 * emb_dim)
        # Output: distances (e, 1)
        distances = self.distance_decoder(edge_inputs)
        distances = distances * (self.max_distance - self.min_distance) + self.min_distance

        # Predict direction vectors
        # Input: edge_inputs (e, 2 * emb_dim)
        # Output: directions (e, 3)
        directions = self.direction_decoder(edge_inputs)
        directions = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-10)  # Normalize to unit vectors

        # predict edge properties
        # Input: edge_inputs (e, 2 * emb_dim)
        # Output: edge_features (e, out_edge_dim)
        edge_features = self.edge_features(edge_inputs)

        # Predict number of nodes
        # Input: h (batch_size, emb_dim)
        # Output: num_nodes (batch_size,)
        num_nodes = self.num_nodes_predictor(h).squeeze(-1)

        self.logger.debug(f"Output shapes - node_features: {node_features.shape}, distances: {distances.shape}, directions: {directions.shape}, edge_features: {edge_features.shape}, num_nodes: {num_nodes.shape}")

        return node_features, distances, directions, edge_features, num_nodes
