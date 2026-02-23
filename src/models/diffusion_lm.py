import torch
import torch.nn as nn
import math

from src.config.config import ModelConfig

class TimeEmbedding(nn.Module):
    """Time embedding layer for diffusion timesteps"""
    def __init__(self, time_dim: int, 
                 T: int = ModelConfig.max_diffusion_steps,
                 expansion: int = ModelConfig.hidden_multiplier, 
                 max_period: float = ModelConfig.max_period):
        super().__init__()
        self.time_dim = time_dim
        self.T = T # total number of diffusion steps

        half_dim = self.time_dim // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / float(half_dim))
        self.register_buffer("freqs", self.freqs, persistent=False)  # device/dtype-safe

        hidden = expansion * time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=time_dim, out_features=hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_dim)
        )
        
    def forward(self, t: torch.Tensor):
        """
        Creates sinusoidal time embeddings similar to transformer positional embeddings
        
        Args:
            t: timestep tensor of shape [B]
            
        Returns:
            emb: time embeddings of shape [B, time_dim]
        """
        # Cast & normalize t to [0,1]
        t = t.float() / (self.T - 1) 

        # Build sinusoidal features
        # [B, 1] * [1, half] -> [B, half]
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) 

        # pad if odd
        if emb.size(-1) < self.time_dim:
            pad = torch.zeros(emb.size(0), self.time_dim - emb.size(-1), device=emb.device)
            emb = torch.cat([emb, pad], dim=-1)
        return self.mlp(emb)
    

class DiTLM(nn.Module):
    def __init__(self, vocab_size: int, time_dim: int, n_layers: int, n_heads: int, context_window: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, time_dim)
        self.position_emb = nn.Embedding(context_window, time_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=time_dim,
                                                   nhead=n_heads, 
                                                   dim_feedforward=ModelConfig.hidden_multiplier*time_dim, 
                                                   batch_first=True)
        self.encoder_backbone = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                      num_layers=n_layers)
        self.time_proj = TimeEmbedding(time_dim)
        self.out = nn.Linear(time_dim, time_dim)

    def forward(self, token_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        x0 = self.token_emb(token_ids) # [B,T,D]
        position = self.position_emb(torch.arange(T))
        x = x0 + position

        # add time conditioning (FiLM)
        t_emb = self.time_proj(t).unsqueeze(0) # [B,1,D]
        x = x + t_emb
        h = self.encoder_backbone(x)
        pred = self.out(h) # [B,1,D]

        return pred
        
