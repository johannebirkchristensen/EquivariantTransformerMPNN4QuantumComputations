import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface as BaseModel

# Updated paths for sampling and smearing
from fairchem.core.models.uma.common.sampling import CalcSpherePoints
from fairchem.core.models.escaip.utils.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass
    logging.warning("e3nn not found. Please install e3nn to use EquiformerV2_OC20 model.")
    
    
    
from fairchem.core.datasets import AseReadDataset
from ase.db import connect

# Download QM9 database (if you don't have it)
# You can get it from: https://doi.org/10.6084/m9.figshare.978904

# Create ASE database from QM9
# Assuming you have qm9.db file
#dataset = AseReadDataset(config={
 #   'src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db',  # or qm9.xyz file
 #   'target': 'U0',  # or whatever property you want
#})






print("Loading QM9 dataset from ASE database...")

from fairchem.core.datasets import AseDBDataset

dataset = AseDBDataset(config={
    'src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_small_1000.db',
    'target': 'U0',
})

print(f"Loaded {len(dataset)} samples of QM9 dataset from ASE database")

# Access just one sample to test
print("First sample:")
print(dataset[0])





"""
EquiformerV2 for QM9 Dataset
Adapted from OC20 implementation for molecular property prediction

Key differences from OC20:
- No periodic boundary conditions
- 12 regression targets instead of energy/forces
- Smaller molecules (~18 atoms vs ~78)
- Only elements H, C, N, O, F (atomic numbers 1, 6, 7, 8, 9)
"""

import math
import torch
import torch.nn as nn
from typing import Optional

# We'll implement these modules step by step
# from .so3 import SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2, CoefficientMappingModule
# from .transformer_block import TransBlockV2, SO2EquivariantGraphAttention, FeedForwardNetwork
# from .input_block import EdgeDegreeEmbedding
# from .layer_norm import get_normalization_layer
# from .radial_function import RadialFunction
# from .edge_rot_mat import init_edge_rot_mat


class EquiformerV2_QM9(nn.Module):
    """
    EquiformerV2 adapted for QM9 molecular property prediction
    
    Args:
        max_num_elements: Maximum atomic number (10 for QM9: H, C, N, O, F)
        num_layers: Number of transformer blocks
        sphere_channels: Hidden dimension per spherical harmonic resolution
        
        attn_hidden_channels: Hidden dimension in attention
        num_heads: Number of attention heads
        attn_alpha_channels: Channels for alpha (scalar) attention component
        attn_value_channels: Channels for value vectors per head
        
        ffn_hidden_channels: Hidden dimension in feedforward network
        
        lmax_list: List of maximum spherical harmonic degrees per resolution
        mmax_list: List of maximum orders (truncation) per resolution
        
        max_radius: Cutoff radius for neighbor search (Ångströms)
        max_neighbors: Maximum neighbors per atom
        num_distance_basis: Number of radial basis functions
        
        num_targets: Number of prediction targets (12 for QM9)
        
        norm_type: Type of equivariant normalization
        use_sep_s2_act: Use separable S2 activation (EquiformerV2's improvement)
    """
    
    def __init__(
        self,
        # QM9 specific
        max_num_elements: int = 10,  # H, C, N, O, F (+ padding)
        num_targets: int = 12,  # QM9 properties
        
        # Architecture
        num_layers: int = 8,
        sphere_channels: int = 128,
        
        # Attention
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        
        # FFN
        ffn_hidden_channels: int = 512,
        
        # Spherical harmonics
        lmax_list: list = [4],  # Lower than OC20, sufficient for small molecules
        mmax_list: list = [2],
        
        # Graph construction
        max_radius: float = 5.0,
        max_neighbors: int = 50,
        num_distance_basis: int = 128,
        
        # Activations and regularization
        norm_type: str = 'rms_norm_sh',
        attn_activation: str = 'scaled_silu',
        ffn_activation: str = 'scaled_silu',
        use_sep_s2_act: bool = True,
        use_gate_act: bool = False,
        
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        # Store config
        self.max_num_elements = max_num_elements
        self.num_targets = num_targets
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels
        
        # QM9: No PBC, no force regression
        self.use_pbc = False
        self.regress_forces = False
        
        # ================================================================
        # 1. INITIAL EMBEDDINGS
        # ================================================================
        
        # Embed atomic numbers into sphere_channels_all dimensions
        # This initializes the l=0, m=0 component of each atom's SO(3) features
        self.sphere_embedding = nn.Embedding(
            max_num_elements, 
            self.sphere_channels_all
        )
        
        # Distance basis functions (Gaussian RBF)
        self.distance_expansion = GaussianRBF(
            num_basis=num_distance_basis,
            cutoff=max_radius
        )
        
        # ================================================================
        # 2. SO(3) INFRASTRUCTURE
        # ================================================================
        
        # TODO: Initialize SO3_Rotation for each resolution
        # This computes Wigner-D matrices for rotating spherical harmonics
        # self.SO3_rotation = nn.ModuleList([
        #     SO3_Rotation(lmax) for lmax in lmax_list
        # ])
        
        # TODO: Initialize coefficient mapping (l,m) layout conversions
        # self.mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)
        
        # TODO: Initialize SO3_Grid for each (l,m) pair
        # Enables conversion: spherical harmonics ↔ grid points on S²
        # self.SO3_grid = ...
        
        # ================================================================
        # 3. EDGE FEATURES
        # ================================================================
        
        # Edge channels: [distance_features, hidden, hidden]
        edge_channels_list = [
            num_distance_basis,
            128,
            128
        ]
        
        # TODO: Edge degree embedding
        # Aggregates neighbor information to initialize node features
        # self.edge_degree_embedding = EdgeDegreeEmbedding(...)
        
        # ================================================================
        # 4. TRANSFORMER BLOCKS
        # ================================================================
        
        # TODO: Stack of TransBlockV2
        # Each block: Attention + FFN with residuals
        # self.blocks = nn.ModuleList([
        #     TransBlockV2(...) for _ in range(num_layers)
        # ])
        
        # ================================================================
        # 5. OUTPUT HEADS
        # ================================================================
        
        # TODO: Final layer norm
        # self.norm = get_normalization_layer(...)
        
        # TODO: Output projection for each QM9 target
        # We need 12 separate heads or one multi-target head
        # self.output_heads = nn.ModuleList([
        #     FeedForwardNetwork(...) for _ in range(num_targets)
        # ])
        
    
    def forward(self, data):
        """
        Forward pass for QM9 molecules
        
        Args:
            data: Batch object containing:
                - atomic_numbers: [num_atoms] atomic numbers
                - pos: [num_atoms, 3] atomic positions
                - batch: [num_atoms] batch assignment
                - (optionally) edge_index if pre-computed
        
        Returns:
            predictions: [batch_size, num_targets] predicted properties
        """
        
        atomic_numbers = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        num_atoms = len(atomic_numbers)
        
        # ================================================================
        # 1. GRAPH CONSTRUCTION
        # ================================================================
        
        # Compute edges within cutoff radius
        # For QM9: no PBC, just Euclidean distance
        edge_index, edge_vec, edge_dist = self._build_graph(
            pos, batch, self.max_radius, self.max_neighbors
        )
        
        # ================================================================
        # 2. EDGE ROTATION MATRICES
        # ================================================================
        
        # Compute rotation matrix that aligns z-axis with edge vector
        # This is used to rotate spherical harmonics along edges
        # edge_rot_mat = init_edge_rot_mat(edge_vec)
        
        # Set Wigner-D matrices for all SO3_Rotation modules
        # for i in range(self.num_resolutions):
        #     self.SO3_rotation[i].set_wigner(edge_rot_mat)
        
        # ================================================================
        # 3. INITIALIZE NODE FEATURES
        # ================================================================
        
        # Create SO3_Embedding structure
        # Stores features as coefficients of spherical harmonics
        # x = SO3_Embedding(
        #     num_atoms,
        #     self.lmax_list,
        #     self.sphere_channels,
        #     device=pos.device,
        #     dtype=pos.dtype
        # )
        
        # Initialize l=0, m=0 coefficients from atomic embeddings
        # offset_res = 0
        # offset = 0
        # for i in range(self.num_resolutions):
        #     if self.num_resolutions == 1:
        #         x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
        #     else:
        #         x.embedding[:, offset_res, :] = self.sphere_embedding(
        #             atomic_numbers
        #         )[:, offset : offset + self.sphere_channels]
        #     offset += self.sphere_channels
        #     offset_res += (self.lmax_list[i] + 1) ** 2
        
        # ================================================================
        # 4. EDGE FEATURES
        # ================================================================
        
        # Expand distances with RBF
        edge_features = self.distance_expansion(edge_dist)
        
        # TODO: Add edge degree embedding
        # edge_degree = self.edge_degree_embedding(
        #     atomic_numbers, edge_features, edge_index
        # )
        # x.embedding = x.embedding + edge_degree.embedding
        
        # ================================================================
        # 5. TRANSFORMER BLOCKS
        # ================================================================
        
        # TODO: Pass through transformer blocks
        # for block in self.blocks:
        #     x = block(x, atomic_numbers, edge_features, edge_index, batch)
        
        # TODO: Final norm
        # x.embedding = self.norm(x.embedding)
        
        # ================================================================
        # 6. POOLING AND PREDICTION
        # ================================================================
        
        # Pool to graph-level features (sum or mean over atoms)
        # For QM9, we typically sum then normalize
        # graph_features = scatter(x.embedding[:, 0, :], batch, dim=0, reduce='sum')
        
        # TODO: Predict each target
        # predictions = torch.stack([
        #     head(x).mean(dim=0) for head in self.output_heads
        # ], dim=1)
        
        # Placeholder return
        predictions = torch.zeros(
            batch.max().item() + 1, 
            self.num_targets,
            device=pos.device
        )
        
        return predictions
    
    
    def _build_graph(self, pos, batch, cutoff, max_neighbors):
        """
        Construct graph from atomic positions
        No PBC for QM9
        """
        from torch_cluster import radius_graph
        
        # Find all edges within cutoff
        edge_index = radius_graph(
            pos, 
            r=cutoff, 
            batch=batch, 
            max_num_neighbors=max_neighbors
        )
        
        # Compute edge vectors and distances
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=1)
        
        return edge_index, edge_vec, edge_dist


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Functions for distance encoding
    """
    def __init__(self, num_basis=128, cutoff=5.0, start=0.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.start = start
        
        # Centers of Gaussians
        self.centers = nn.Parameter(
            torch.linspace(start, cutoff, num_basis),
            requires_grad=False
        )
        
        # Width of Gaussians
        self.width = nn.Parameter(
            torch.tensor((cutoff - start) / num_basis),
            requires_grad=False
        )
    
    def forward(self, distances):
        """
        Args:
            distances: [num_edges] edge lengths
        Returns:
            [num_edges, num_basis] RBF features
        """
        distances = distances.unsqueeze(-1)  # [num_edges, 1]
        
        # Compute Gaussian RBF
        rbf = torch.exp(
            -(distances - self.centers) ** 2 / (2 * self.width ** 2)
        )
        
        # Smooth cutoff
        cutoff_values = self._smooth_cutoff(distances.squeeze(-1))
        rbf = rbf * cutoff_values.unsqueeze(-1)
        
        return rbf
    
    def _smooth_cutoff(self, distances):
        """Smooth cutoff function (cosine)"""
        cutoffs = 0.5 * (torch.cos(distances * (torch.pi / self.cutoff)) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


# ================================================================
# PLACEHOLDER CLASSES
# We'll implement these step by step
# ================================================================

# class SO3_Embedding:
#     """Data structure for storing features in spherical harmonic basis"""
#     pass

# class SO3_Rotation:
#     """Computes Wigner-D matrices for rotating spherical harmonics"""
#     pass

# class TransBlockV2:
#     """Transformer block: Attention + FFN"""
#     pass