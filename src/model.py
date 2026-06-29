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
from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool


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
        
        # Default: 64 -> 32 -> 16 -> 8 -> 4 -> latent
        if hidden_dims is None:
            hidden_dims = [64, 32, 16, 8, 4]
        
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


class VGAE(GAE):
    """
    Variational Graph Autoencoder.

    Final encoder layer produces (mu, log_var) of size latent_dim each.
    At training time, samples z via reparameterization: z = mu + eps * exp(0.5 * log_var).
    At eval time, z = mu (deterministic).

    Decoder and earlier encoder layers are identical to GAE.

    forward returns (x_recon, z, mu, log_var).
    """
    def __init__(self, in_channels, hidden_dims=None, latent_dim=2, dropout=0.2):
        super().__init__(in_channels, hidden_dims=hidden_dims, latent_dim=latent_dim, dropout=dropout)

        # Replace the final encoder SAGEConv with one that outputs 2 * latent_dim
        hidden_dims_used = self.hidden_dims
        self.encoder_layers[-1] = SAGEConv(hidden_dims_used[-1], 2 * latent_dim, aggr='mean', project=True)

        print(f"VGAE Architecture: {in_channels} -> {hidden_dims_used} -> 2*{latent_dim} (mu,logvar) -> {latent_dim} -> {list(reversed(hidden_dims_used))} -> {in_channels}")

    def encode(self, x, edge_index):
        # Run through all encoder layers except the final one (with activations)
        for conv, norm, drop in zip(self.encoder_layers[:-1], self.encoder_norms, self.encoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        # Final layer produces concatenated [mu | log_var]
        out = self.encoder_layers[-1](x, edge_index)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index):
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z, mu, log_var


class GAEVAE(torch.nn.Module):
    """
    Hybrid GAE + VAE: graph encoder (SAGEConv stack) -> per-node MLP VAE
    bottleneck (mu, log_var) -> reparameterize -> per-node MLP expansion ->
    graph decoder (SAGEConv stack) -> reconstruction.

    Distinct from VGAE: VGAE replaces the *last conv* with a SAGEConv that
    outputs 2*latent_dim, so the variational step is still graph-aware message
    passing. GAEVAE keeps the conv stack as plain SAGEConv layers and puts the
    variational bottleneck in a separate per-node MLP. Graph structure is
    handled by the SAGEConv stacks; the MLP handles the latent compression.

    forward returns (x_recon, z, mu, log_var) matching VGAE so the training
    loop does not need to branch by class.
    """
    def __init__(self, in_channels, hidden_dims=None, latent_dim=2, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32, 16]
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout

        # Graph encoder: in -> hd[0] -> ... -> hd[-1] (norm+activation+dropout each)
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        prev = in_channels
        for h in hidden_dims:
            self.encoder_layers.append(SAGEConv(prev, h, aggr='mean', project=True))
            self.encoder_norms.append(nn.BatchNorm1d(h))
            self.encoder_dropouts.append(nn.Dropout(dropout))
            prev = h

        # Per-node MLP VAE bottleneck
        self.vae_encoder = nn.Linear(hidden_dims[-1], 2 * latent_dim)
        self.vae_decoder = nn.Linear(latent_dim, hidden_dims[-1])

        # Graph decoder: hd[-1] -> hd[-2] -> ... -> hd[0] -> in_channels
        decoder_dims = list(reversed(hidden_dims))
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        prev = decoder_dims[0]
        for h in decoder_dims[1:]:
            self.decoder_layers.append(SAGEConv(prev, h, aggr='mean', project=True))
            self.decoder_norms.append(nn.BatchNorm1d(h))
            self.decoder_dropouts.append(nn.Dropout(dropout))
            prev = h
        self.decoder_layers.append(SAGEConv(prev, in_channels, aggr='mean', project=True))

        print(f"GAEVAE Architecture: {in_channels} -> {hidden_dims} -> "
              f"[MLP {hidden_dims[-1]}->2*{latent_dim}] -> {latent_dim} -> "
              f"[MLP {latent_dim}->{hidden_dims[-1]}] -> {decoder_dims} -> {in_channels}")

    def encode(self, x, edge_index):
        for conv, norm, drop in zip(self.encoder_layers, self.encoder_norms, self.encoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        mu_lv = self.vae_encoder(x)
        mu, log_var = mu_lv.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, edge_index):
        x = self.vae_decoder(z)
        x = F.leaky_relu(x, negative_slope=0.1)
        for conv, norm, drop in zip(self.decoder_layers[:-1], self.decoder_norms, self.decoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        x = self.decoder_layers[-1](x, edge_index)
        return x

    def forward(self, x, edge_index):
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, edge_index)
        return x_recon, z, mu, log_var


class GATVAE(torch.nn.Module):
    """
    Attention-based variational graph autoencoder: a GATv2 message-passing
    encoder + a per-node MLP VAE bottleneck + a GATv2 decoder.

    Same structure as GAEVAE, but every graph-convolution layer is a multi-head
    GATv2Conv (learned attention over the k-NN electrode neighbourhood) instead of
    a SAGEConv mean aggregator. Attention heads are averaged (concat=False) so each
    layer's output width stays hidden_dims[i] and the rest of the plumbing (config
    keys, BatchNorm, decoder mirror) is unchanged.

    forward returns (x_recon, z, mu, log_var) matching VGAE/GAEVAE so the training
    loop and _is_variational handling do not need a new branch. The constructor
    signature matches the other models (in_channels, hidden_dims, latent_dim,
    dropout); the number of attention heads is fixed internally.
    """
    HEADS = 4

    def __init__(self, in_channels, hidden_dims=None, latent_dim=2, dropout=0.2,
                 heads=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32, 16]
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        h_att = int(heads) if heads is not None else self.HEADS
        self.heads = h_att

        # Graph encoder: in -> hd[0] -> ... -> hd[-1] (GATv2, heads averaged).
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        prev = in_channels
        for h in hidden_dims:
            self.encoder_layers.append(
                GATv2Conv(prev, h, heads=h_att, concat=False, dropout=dropout))
            self.encoder_norms.append(nn.BatchNorm1d(h))
            self.encoder_dropouts.append(nn.Dropout(dropout))
            prev = h

        # Per-node MLP VAE bottleneck (identical to GAEVAE).
        self.vae_encoder = nn.Linear(hidden_dims[-1], 2 * latent_dim)
        self.vae_decoder = nn.Linear(latent_dim, hidden_dims[-1])

        # Graph decoder: hd[-1] -> ... -> hd[0] -> in_channels (GATv2).
        decoder_dims = list(reversed(hidden_dims))
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        prev = decoder_dims[0]
        for h in decoder_dims[1:]:
            self.decoder_layers.append(
                GATv2Conv(prev, h, heads=h_att, concat=False, dropout=dropout))
            self.decoder_norms.append(nn.BatchNorm1d(h))
            self.decoder_dropouts.append(nn.Dropout(dropout))
            prev = h
        self.decoder_layers.append(
            GATv2Conv(prev, in_channels, heads=h_att, concat=False, dropout=dropout))

        print(f"GATVAE Architecture: {in_channels} -> {hidden_dims} (GATv2 x{h_att} heads) -> "
              f"[MLP {hidden_dims[-1]}->2*{latent_dim}] -> {latent_dim} -> "
              f"[MLP {latent_dim}->{hidden_dims[-1]}] -> {decoder_dims} -> {in_channels}")

    def encode(self, x, edge_index):
        for conv, norm, drop in zip(self.encoder_layers, self.encoder_norms, self.encoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        mu, log_var = self.vae_encoder(x).chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, edge_index):
        x = self.vae_decoder(z)
        x = F.leaky_relu(x, negative_slope=0.1)
        for conv, norm, drop in zip(self.decoder_layers[:-1], self.decoder_norms, self.decoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        x = self.decoder_layers[-1](x, edge_index)
        return x

    def forward(self, x, edge_index):
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, edge_index)
        return x_recon, z, mu, log_var


def kl_divergence(mu, log_var):
    """Standard KL(q(z|x) || N(0, I)) per-sample, averaged."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


class GNNEncoder(torch.nn.Module):
    """
    Encoder-only architecture for contrastive (CEBRA-style) training.

    GraphSAGE encoder over the electrode graph -> global mean pool to one vector
    per graph -> fully-connected head -> latent_dim -> L2-normalized embedding
    (contrastive embeddings live on a unit hypersphere). There is NO decoder:
    this model is trained with the InfoNCE loss in `cebra_loss.py`, not MSE.

    forward(x, edge_index, batch) returns z of shape (num_graphs, latent_dim).
    `batch` is the PyG batch vector mapping each node to its graph; if None, all
    nodes are treated as a single graph (single-graph inference).

    Args mirror GAE so the pipeline's config plumbing is reused.
    """
    def __init__(self, in_channels, hidden_dims=None, latent_dim=2, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32, 16]
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout

        # GraphSAGE encoder stack (same pattern as GAE.encode).
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        prev = in_channels
        for h in hidden_dims:
            self.encoder_layers.append(SAGEConv(prev, h, aggr='mean', project=True))
            self.encoder_norms.append(nn.BatchNorm1d(h))
            self.encoder_dropouts.append(nn.Dropout(dropout))
            prev = h

        # Fully-connected projection head: pooled hidden -> latent_dim.
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dims[-1], latent_dim),
        )

        print(f"GNNEncoder Architecture: {in_channels} -> {hidden_dims} "
              f"-> meanpool -> [FC {hidden_dims[-1]}->{hidden_dims[-1]}->{latent_dim}] "
              f"-> L2-norm (latent_dim={latent_dim})")

    def encode_nodes(self, x, edge_index):
        for conv, norm, drop in zip(self.encoder_layers, self.encoder_norms, self.encoder_dropouts):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = drop(x)
        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode_nodes(x, edge_index)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        pooled = global_mean_pool(h, batch)        # (num_graphs, hidden_dims[-1])
        z = self.head(pooled)                       # (num_graphs, latent_dim)
        z = F.normalize(z, p=2, dim=-1)             # unit hypersphere
        return z


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