"""
MODEL ARCHITECTURE
==================
Purpose: Define PyTorch Geometric GNN model architecture.

Pipeline Position: Used by train.py and inference.py
- Input: None (defines structure only)
- Output: Model class definition

Key Operations:
1. Define graph neural network layers (SAGEConv)
2. Model initialization methods
3. Forward pass logic

This file only defines the model structure, doesn't handle data or training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GAE(torch.nn.Module):
    """
    Graph Autoencoder with configurable layer dimensions.
    
    Default architecture (hidden_dims=[64, 64, 32, 16]):
    - Encoder: in_dim -> 64 -> 64 -> 32 -> 16 -> latent_dim
    - Decoder: latent_dim -> 16 -> 32 -> 64 -> 64 -> in_dim
    
    Args:
        in_channels: Number of input features per node
        hidden_dims: List of hidden dimensions (default: [64, 64, 32, 16])
        latent_dim: Dimension of latent space (default: 2)
        dropout: Dropout probability (default: 0.2)
    """
    def __init__(self, in_channels, hidden_dims=None, latent_dim=2, dropout=0.2):
        super().__init__()
        
        # Default: 64 -> 64 -> 32 -> 16 -> latent
        if hidden_dims is None:
            hidden_dims = [64, 64, 32, 16]
        
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        
        # First encoder layer: in_channels -> hidden_dims[0]
        self.encoder_layers.append(SAGEConv(in_channels, hidden_dims[0], aggr='mean', project=True))
        self.encoder_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        self.encoder_dropouts.append(nn.Dropout(dropout))
        
        # Middle encoder layers
        for i in range(1, len(hidden_dims)):
            self.encoder_layers.append(SAGEConv(hidden_dims[i-1], hidden_dims[i], aggr='mean', project=True))
            self.encoder_norms.append(nn.BatchNorm1d(hidden_dims[i]))
            self.encoder_dropouts.append(nn.Dropout(dropout))
        
        # Final encoder layer: hidden_dims[-1] -> latent_dim
        self.encoder_layers.append(SAGEConv(hidden_dims[-1], latent_dim, aggr='mean', project=True))
        
        # Decoder layers (reverse of encoder)
        decoder_dims = list(reversed(hidden_dims))
        
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        
        # First decoder layer: latent_dim -> decoder_dims[0]
        self.decoder_layers.append(SAGEConv(latent_dim, decoder_dims[0], aggr='mean', project=True))
        self.decoder_norms.append(nn.BatchNorm1d(decoder_dims[0]))
        self.decoder_dropouts.append(nn.Dropout(dropout))
        
        # Middle decoder layers
        for i in range(1, len(decoder_dims)):
            self.decoder_layers.append(SAGEConv(decoder_dims[i-1], decoder_dims[i], aggr='mean', project=True))
            self.decoder_norms.append(nn.BatchNorm1d(decoder_dims[i]))
            self.decoder_dropouts.append(nn.Dropout(dropout))
        
        # Final decoder layer: decoder_dims[-1] -> in_channels
        self.decoder_layers.append(SAGEConv(decoder_dims[-1], in_channels, aggr='mean', project=True))
        
        print(f"GAE Architecture: {in_channels} -> {hidden_dims} -> {latent_dim} -> {decoder_dims} -> {in_channels}")

    def encode(self, x, edge_index):
        for i, (conv, norm, drop) in enumerate(zip(self.encoder_layers[:-1], self.encoder_norms, self.encoder_dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        
        # Final encoder layer (no activation - raw latent representation)
        x = self.encoder_layers[-1](x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        x = z
        for i, (conv, norm, drop) in enumerate(zip(self.decoder_layers[:-1], self.decoder_norms, self.decoder_dropouts)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        
        # Final decoder layer (no activation - raw reconstruction)
        x = self.decoder_layers[-1](x, edge_index)
        return x
       
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z


class GAESimple(torch.nn.Module):
    """
    Original simple Graph Autoencoder with 2 layers.
    Kept for backward compatibility.
    """
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()

        # Encoder layers
        self.encoder_conv1 = SAGEConv(in_channels, hidden_channels, aggr='add', project=True)
        self.encoder_conv2 = SAGEConv(hidden_channels, latent_dim, aggr='add', project=True)
        
        # Decoder layers
        self.decoder_conv1 = SAGEConv(latent_dim, hidden_channels, aggr='add', project=True)
        self.decoder_conv2 = SAGEConv(hidden_channels, in_channels, aggr='add', project=True)

    def encode(self, x, edge_index):
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = self.encoder_conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        x = self.decoder_conv1(z, edge_index)
        x = F.relu(x)
        x = self.decoder_conv2(x, edge_index)
        return x
       
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z 