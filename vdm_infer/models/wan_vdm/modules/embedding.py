import torch
import torch.nn as nn

from typing import Optional, Tuple


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        return sinusoidal_embedding_1d(self.num_channels, timesteps)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, sample_proj_bias=True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TextEmbedding(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.time_freq_dim = time_freq_dim

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(in_features=dim, out_features=time_proj_dim)
        self.text_embedder = TextEmbedding(in_features=text_embed_dim, hidden_size=dim)

        self.image_embedder = None

    def init_weights(self):
        """Initialize weights for the time/text/image embedding module."""
        # Initialize time embedding layers
        nn.init.normal_(self.time_embedder.linear_1.weight, std=0.02)
        if self.time_embedder.linear_1.bias is not None:
            nn.init.constant_(self.time_embedder.linear_1.bias, 0.0)
        nn.init.normal_(self.time_embedder.linear_2.weight, std=0.02)
        if self.time_embedder.linear_2.bias is not None:
            nn.init.constant_(self.time_embedder.linear_2.bias, 0.0)
        
        # Initialize time projection
        nn.init.normal_(self.time_proj.weight, std=0.02)
        if self.time_proj.bias is not None:
            nn.init.constant_(self.time_proj.bias, 0.0)
        
        # Initialize text embedder
        # This typically contains a linear layer for projection
        for module in self.text_embedder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].double() / dim))
    )
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return freqs_cis


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

    def init_weights(self):
        pass

    def forward(self, f, h, w, device="cuda") -> torch.Tensor:
        f_freqs_cis, h_freqs_cis, w_freqs_cis = precompute_freqs_cis_3d(self.attention_head_dim)
        freqs = torch.cat([
            f_freqs_cis[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs_cis[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_freqs_cis[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)
        return freqs