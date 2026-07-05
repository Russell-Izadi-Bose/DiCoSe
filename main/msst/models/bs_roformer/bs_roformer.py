from functools import partial
from typing import Sequence

import torch
from torch import nn, einsum, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend
try:
    from .attend_sage import Attend as AttendSage
except:
    pass
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable, Sequence
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack, repeat
from einops.layers.torch import Rearrange

# Import time embedding for diffusion
from audio_diffusion_pytorch_.modules import TimePositionalEmbedding

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# norm

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


def expand_mapping_for_packed_batch(mapping: Optional[Tensor], expand_size: int) -> Optional[Tensor]:
    """
    Expands mapping tensor to match packed batch size without mixing batch member information.

    Args:
        mapping: Tensor of shape [batch, features] or None
        expand_size: Number of times to repeat each batch member

    Returns:
        Expanded tensor of shape [batch * expand_size, features] where each batch member
        is repeated expand_size times consecutively (no mixing between batch members)

    Example:
        mapping: [2, 48] with expand_size=62
        output: [124, 48] where rows 0-61 are batch0 repeated, rows 62-123 are batch1 repeated
    """
    if mapping is None:
        return None
    return repeat(mapping, 'b d -> (b n) d', n=expand_size)


class MappingToScaleShift(Module):
    """Converts mapping to scale and shift for FiLM conditioning"""
    def __init__(self, features: int, dim: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=features, out_features=dim * 2),
        )

    def forward(self, mapping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, "b d -> b 1 d")
        scale, shift = scale_shift.chunk(2, dim=-1)
        return scale, shift


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.,
            context_mapping_features=None,
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.use_mapping = exists(context_mapping_features)

        # Build Sequential with same structure regardless of use_mapping
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

        # Optionally add FiLM conditioning
        if self.use_mapping:
            self.to_scale_shift = MappingToScaleShift(
                features=context_mapping_features,
                dim=dim
            )

    def forward(self, x, mapping=None):
        if self.use_mapping and exists(mapping):
            # Apply FiLM: manually run through layers with scale-shift after norm
            x = self.net[0](x)  # RMSNorm
            scale, shift = self.to_scale_shift(mapping)
            x = x * (scale + 1) + shift
            # Continue with rest of network
            for layer in self.net[1:]:
                x = layer(x)
            return x
        else:
            # Standard path - just run Sequential
            return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            sage_attention=False,
            context_mapping_features=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.use_mapping = exists(context_mapping_features)

        if sage_attention:
            self.attend = AttendSage(flash=flash, dropout=dropout)
        else:
            self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

        # Time conditioning via FiLM
        if self.use_mapping:
            self.to_scale_shift = MappingToScaleShift(
                features=context_mapping_features,
                dim=dim
            )

    def forward(self, x, mapping=None):
        x = self.norm(x)

        # Apply FiLM conditioning if mapping is provided
        if self.use_mapping and exists(mapping):
            scale, shift = self.to_scale_shift(mapping)
            x = x * (scale + 1) + shift

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.,
            sage_attention=False,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            self.attend = AttendSage(
                scale=scale,
                dropout=dropout,
                flash=flash
            )
        else:
            self.attend = Attend(
                scale=scale,
                dropout=dropout,
                flash=flash
            )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            sage_attention=False,
            context_mapping_features=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                    context_mapping_features=context_mapping_features,
                )

            self.layers.append(ModuleList([
                attn,
                FeedForward(
                    dim=dim,
                    mult=ff_mult,
                    dropout=ff_dropout,
                    context_mapping_features=context_mapping_features,
                )
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, mapping=None):

        for attn, ff in self.layers:
            x = attn(x, mapping=mapping) + x
            x = ff(x, mapping=mapping) + x

        return self.norm(x)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class ParallelFreqTimeAdapter(Module):
    """
    Parallel frequency and time linear adapter for STFT features.

    Two parallel paths:
    - Freq path: learns spectral patterns (harmonics, formants) across all frequencies
    - Time path: learns temporal patterns (transients, attacks) across all time steps

    Both paths are zero-initialized so the adapter starts with no impact.
    Much faster than Conv2d on full resolution while having more parameters.
    """
    def __init__(self, audio_channels, freq_dim, time_dim, hidden=128, use_time_net=True):
        super().__init__()
        self.audio_channels = audio_channels
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.use_time_net = use_time_net

        stft_channels = audio_channels * 2  # stereo × complex
        freq_flat = stft_channels * freq_dim  # 4 * 1025 = 4100

        # Freq path: (b, t, freq_flat) -> per-time-step spectral correction
        # bias=False: zero mixture input → zero output (no injection during true silence)
        self.freq_net = nn.Sequential(
            nn.Linear(freq_flat, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, freq_flat, bias=False),
        )
        nn.init.zeros_(self.freq_net[-1].weight)

        if use_time_net:
            time_flat = stft_channels * time_dim  # 4 * 1101 = 4404
            # Time path: (b, f, time_flat) -> per-frequency constant-across-time offset
            # NOTE: this naturally produces horizontal stripes; disable with use_time_net=False
            self.time_net = nn.Sequential(
                nn.Linear(time_flat, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, time_flat, bias=False),
            )
            nn.init.zeros_(self.time_net[-1].weight)

    def forward(self, x):
        # x: (b, stereo, freq, time, complex)
        b, s, f, t, c = x.shape

        xf = rearrange(x, 'b s f t c -> b t (s f c)')  # (b, time, 4*freq)
        xf = self.freq_net(xf)
        xf = rearrange(xf, 'b t (s f c) -> b s f t c', s=s, f=f, c=c)

        if self.use_time_net:
            xt = rearrange(x, 'b s f t c -> b f (s t c)')  # (b, freq, 4*time)
            xt = self.time_net(xt)
            xt = rearrange(xt, 'b f (s t c) -> b s f t c', s=s, t=t, c=c)
            return xf + xt

        return xf


class BSRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            zero_dc = True,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            # Diffusion parameters
            use_context_time=False,  # Match UNet1d naming
            time_emb_type="Positional",  # "Positional" or "LearnedPositional"
            # Feature conditioning parameters
            use_mixture_feature_conditioning=False,  # Enable mixture feature injection
            stft_adapter_type=None,  # "depthwise", "conv2d", "parallel_linear", or None (no STFT injection)
            stft_adapter_hidden=128,  # Hidden dim for parallel_linear adapter
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.use_context_time = use_context_time
        self.use_mixture_feature_conditioning = use_mixture_feature_conditioning

        # Time conditioning for diffusion
        if use_context_time:
            # Mapping features dimension (similar to UNet1d)
            context_mapping_features = dim * 4

            # Time positional embedding
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=dim,
                    out_features=context_mapping_features,
                    time_emb_type=time_emb_type
                ),
                nn.GELU(),
            )

            # Mapping network (processes time embeddings)
            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )
        else:
            context_mapping_features = None

        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
            context_mapping_features=context_mapping_features,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        # Store adapter config for later initialization
        self.stft_adapter_type = stft_adapter_type
        self.stft_adapter_hidden = stft_adapter_hidden

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # Feature conditioning projection layers (zero-initialized)
        if self.use_mixture_feature_conditioning:
            stft_channels = self.audio_channels * 2  # stereo × complex = 4

            # STFT adapter: select based on stft_adapter_type
            if stft_adapter_type == "depthwise":
                # OLD: Depthwise separable conv - fast (~8ms), small capacity (~188 params)
                stft_hidden_dim = 16
                self.stft_feature_adapter = nn.Sequential(
                    Rearrange('b s f t c -> b (s c) f t'),  # (b, 4, freq, time)
                    # Depthwise: process each channel's spatial patterns independently
                    nn.Conv2d(stft_channels, stft_channels, kernel_size=3, padding=1, groups=stft_channels),
                    nn.GELU(),
                    # Pointwise: mix information across channels
                    nn.Conv2d(stft_channels, stft_hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(stft_hidden_dim, stft_channels, kernel_size=1),
                    Rearrange('b (s c) f t -> b s f t c', s=self.audio_channels, c=2)
                )
                # Zero-initialize final pointwise conv
                nn.init.zeros_(self.stft_feature_adapter[-2].weight)
                nn.init.zeros_(self.stft_feature_adapter[-2].bias)
           
            elif stft_adapter_type == "conv2d":
                # stft_adapter_hidden controls intermediate channels, keeps memory manageable
                self.stft_feature_adapter = nn.Sequential(
                    Rearrange('b s f t c -> b (s c) f t'),  # (b, 4, freq, time)
                    nn.Conv2d(stft_channels, stft_adapter_hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(stft_adapter_hidden, stft_adapter_hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(stft_adapter_hidden, stft_adapter_hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(stft_adapter_hidden, stft_channels, kernel_size=3, padding=1),
                    Rearrange('b (s c) f t -> b s f t c', s=self.audio_channels, c=2)
                )
                # Zero-initialize final conv
                nn.init.zeros_(self.stft_feature_adapter[-2].weight)
                nn.init.zeros_(self.stft_feature_adapter[-2].bias)

            elif stft_adapter_type == "parallel_linear":
                # Parallel freq + time linear - fast, large capacity (~2.2M params)
                # WARNING: time_net produces horizontal stripes by construction; prefer freq_linear
                expected_time_dim = 1101  # Based on chunk_size=485100, hop_length=441
                self.stft_feature_adapter = ParallelFreqTimeAdapter(
                    audio_channels=self.audio_channels,
                    freq_dim=freqs,
                    time_dim=expected_time_dim,
                    hidden=stft_adapter_hidden,
                    use_time_net=True,
                )

            elif stft_adapter_type == "freq_linear":
                # Freq-only linear adapter: per-time-step spectral correction from mixture STFT
                # No time_net → no horizontal stripe artifact; silence-safe (bias=False)
                self.stft_feature_adapter = ParallelFreqTimeAdapter(
                    audio_channels=self.audio_channels,
                    freq_dim=freqs,
                    hidden=stft_adapter_hidden,
                    time_dim=0,         # unused, but required by __init__ signature
                    use_time_net=False,
                )

            elif stft_adapter_type is None:
                self.stft_feature_adapter = None  # consume mixture STFT feature but don't inject it

            else:
                raise ValueError(f"Unknown stft_adapter_type: {stft_adapter_type}. "
                                 f"Choose from 'depthwise', 'conv2d', 'parallel_linear', 'freq_linear', or None")

            # Projection for band split feature
            self.band_split_feature_adapter = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            nn.init.zeros_(self.band_split_feature_adapter[-1].weight)
            nn.init.zeros_(self.band_split_feature_adapter[-1].bias)

            # Projections for time and freq transformer features (one for each layer)
            # depth transformer blocks × 2 (time + freq) = total projections
            self.transformer_feature_adapters = nn.ModuleList()
            for _ in range(depth * 2):  # depth blocks, 2 features each (time + freq)
                adapter = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim)
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.transformer_feature_adapters.append(adapter)

        # whether to zero out dc

        self.zero_dc = zero_dc

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            time: Optional[Tensor] = None,
            s: Optional[Tensor] = None,  # For CTM (future use)
            *,
            features: Optional[Tensor] = None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
            target=None,
            return_loss_breakdown=False,
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        # Create a copy of mixture_features_channels_list to avoid modifying the original list
        if mixture_features_channels_list is not None:
            mixture_features_channels_list_ = mixture_features_channels_list.copy()
            # Track which transformer adapter to use
            transformer_adapter_idx = 0

        # Create time conditioning mapping if diffusion is enabled
        mapping = None
        if self.use_context_time:
            assert time is not None, "time parameter required when use_context_time=True"
            time_emb = self.to_time(time)
            mapping = self.to_mapping(time_emb)

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(
                device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # Feature injection 0: Add mixture feature in STFT space (before merging stereo)
        if mixture_features_channels_list is not None and len(mixture_features_channels_list_) > 0:
            mixture_feature = mixture_features_channels_list_.pop(0)
            if self.use_mixture_feature_conditioning and self.stft_feature_adapter is not None:
                stft_repr = stft_repr + self.stft_feature_adapter(mixture_feature)

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # Feature injection 1: Add mixture feature after band split
        if mixture_features_channels_list is not None and len(mixture_features_channels_list_) > 0:
            mixture_feature = mixture_features_channels_list_.pop(0)
            # Apply band split feature adapter (zero-initialized projection)
            if self.use_mixture_feature_conditioning:
                mixture_feature = self.band_split_feature_adapter(mixture_feature)
            x = x + mixture_feature

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                else:
                    x = linear_transformer(x, mapping=mapping)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            # TIME TRANSFORMER with feature injection
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            # Expand mapping for time_transformer: [b, d] -> [b*f, d]
            # Each batch member's mapping is repeated f times (no mixing between batches)
            mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
            else:
                x = time_transformer(x, mapping=mapping_time)

            x, = unpack(x, ps, '* t d')

            # Feature injection AFTER TIME transformer (skip connection style)
            if mixture_features_channels_list is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature_time = mixture_features_channels_list_.pop(0)
                # Apply time transformer feature adapter (zero-initialized projection)
                if self.use_mixture_feature_conditioning:
                    mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
                    transformer_adapter_idx += 1
                x = x + mixture_feature_time

            # FREQ TRANSFORMER with feature injection
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            # Expand mapping for freq_transformer: [b, d] -> [b*t, d]
            # Each batch member's mapping is repeated t times (no mixing between batches)
            mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
            else:
                x = freq_transformer(x, mapping=mapping_freq)

            x, = unpack(x, ps, '* f d')

            # Feature injection AFTER FREQ transformer (skip connection style)
            if mixture_features_channels_list is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature_freq = mixture_features_channels_list_.pop(0)
                # Apply freq transformer feature adapter (zero-initialized projection)
                if self.use_mixture_feature_conditioning:
                    mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
                    transformer_adapter_idx += 1
                x = x + mixture_feature_freq

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            # whether to dc filter
            stft_repr = stft_repr.index_fill(1, tensor(0, device = device), 0.)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)

    def get_feature(
            self,
            raw_audio,
    ) -> List[Tensor]:
        """
        Extracts features at various stages of the BS-Roformer network.
        This is used for feature extraction from the mixture (deterministic).
        NO time conditioning is applied - features are extracted without diffusion timesteps.

        Args:
            raw_audio: Input audio tensor

        Returns:
            List[Tensor]: List of feature maps at different stages:
                - fmap[0]: After STFT (frequency domain)
                - fmap[1]: After band split
                - fmap[2, 3, ...]: After time and freq transformers for each block
                  * For each transformer block i:
                    - fmap[2 + 2*i]: After time transformer
                    - fmap[2 + 2*i + 1]: After freq transformer
                - fmap[2 + 2*N]: After final norm (N = depth)
                - fmap[2 + 2*N + 1]: Mask estimator outputs
                - fmap[2 + 2*N + 2]: Final separated tracks
        """
        device = raw_audio.device

        # No time conditioning for feature extraction (deterministic)
        mapping = None

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # Feature map list
        fmap = []

        # to stft
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(
                device)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # Feature 0: STFT representation
        fmap.append(stft_repr)

        # merge stereo / mono into the frequency
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        # Band split
        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # Feature 1: After band split
        fmap.append(x)

        # axial / hierarchical attention
        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                else:
                    x = linear_transformer(x, mapping=mapping)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, mapping, use_reentrant=False)
            else:
                x = time_transformer(x, mapping=mapping)

            x, = unpack(x, ps, '* t d')

            # Feature 2 + 2*i: After TIME transformer
            fmap.append(x)

            # Freq transformer
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, mapping, use_reentrant=False)
            else:
                x = freq_transformer(x, mapping=mapping)

            x, = unpack(x, ps, '* f d')

            # Feature 2 + 2*i + 1: After FREQ transformer
            fmap.append(x)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Feature N+2: After final norm
        fmap.append(x)

        # Apply mask estimators to get separation masks
        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

        # Feature N+3: Mask estimator outputs (before reshaping)
        fmap.append(mask)

        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # Modulate frequency representation with masks
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # Complex number multiplication
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft to get separated tracks
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr = stft_repr.index_fill(1, tensor(0, device=device), 0.)

        # Inverse STFT
        stft_window = self.stft_window_fn(device=device)
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs,
                                    window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False,
                                    length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # Feature N+4: Final separated tracks
        fmap.append(recon_audio)

        return fmap


class BSRoformerStemsInStemsOut(BSRoformer):
    """
    Subclass of BSRoformer for diffusion-based stem enhancement.

    Takes all stems as input and outputs enhanced stems.
    Input: [B, num_stems, C, T] - all stems (possibly noisy/degraded)
    Output: [B, num_stems, C, T] - enhanced stems

    The forward pass:
    1. Flattens stems into batch: [B, num_stems, C, T] -> [B*num_stems, C, T]
    2. Processes through shared band_split + transformers
    3. Unflattens and routes each stem to its corresponding mask_estimator
    4. Outputs enhanced stems
    """

    ################### this is batched stems-in-stems-out forward pass ###################
    ################### we decided not to use it as it requires a lot of memory ###################
    
    # def forward(        
    #         self,
    #         raw_audio,
    #         time: Optional[Tensor] = None,
    #         s: Optional[Tensor] = None,
    #         *,
    #         features: Optional[Tensor] = None,
    #         channels_list: Optional[Sequence[Tensor]] = None,
    #         embedding: Optional[Tensor] = None,
    #         mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
    #         target=None,
    #         return_loss_breakdown=False,
    # ):
    #     """
    #     Forward pass for stems-in-stems-out processing.

    #     Args:
    #         raw_audio: Input tensor of shape [B, num_stems, C, T] where:
    #             - B: batch size
    #             - num_stems: number of stems (e.g., 4)
    #             - C: audio channels (1 for mono, 2 for stereo)
    #             - T: time samples
    #         time: Diffusion timestep tensor [B]
    #         mixture_features_channels_list: List of mixture features for conditioning
    #         target: Target tensor for loss computation [B, num_stems, C, T]
    #     """
    #     device = raw_audio.device

    #     # Handle input shape - expect [B, num_stems, C, T]
    #     assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

    #     batch_size, num_stems_in, channels, seq_len = raw_audio.shape
    #     assert num_stems_in == self.num_stems, \
    #         f"Input has {num_stems_in} stems but model expects {self.num_stems}"
    #     assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
    #         'stereo setting must match input channels'

    #     # ===== FLATTEN STEMS INTO BATCH =====
    #     # [B, num_stems, C, T] -> [B*num_stems, C, T]
    #     raw_audio_flat = rearrange(raw_audio, 'b n c t -> (b n) c t')
    #     B, N = batch_size, num_stems_in

    #     # Create a copy of mixture_features_channels_list and EXPAND for flattened batch
    #     if mixture_features_channels_list is not None:
    #         # Expand each feature: [B, ...] -> [B*num_stems, ...]
    #         mixture_features_channels_list_ = []
    #         for feat in mixture_features_channels_list:
    #             # feat has shape [B, ...], we need [B*N, ...]
    #             # Repeat each batch element N times
    #             feat_expanded = repeat(feat, 'b ... -> (b n) ...', n=N)
    #             mixture_features_channels_list_.append(feat_expanded)
    #         transformer_adapter_idx = 0
    #     else:
    #         mixture_features_channels_list_ = None

    #     # Create time conditioning mapping if diffusion is enabled
    #     mapping = None
    #     if self.use_context_time:
    #         assert time is not None, "time parameter required when use_context_time=True"
    #         # Expand time: [B] -> [B*num_stems]
    #         time_expanded = repeat(time, 'b -> (b n)', n=N)
    #         time_emb = self.to_time(time_expanded)
    #         mapping = self.to_mapping(time_emb)

    #     # defining whether model is loaded on MPS (MacOS GPU accelerator)
    #     x_is_mps = True if device.type == "mps" else False

    #     # ===== STFT =====
    #     raw_audio_flat, batch_audio_channel_packed_shape = pack_one(raw_audio_flat, '* t')

    #     stft_window = self.stft_window_fn(device=device)

    #     # RuntimeError: FFT operations are only supported on MacOS 14+
    #     # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
    #     try:
    #         stft_repr = torch.stft(raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True)
    #     except:
    #         stft_repr = torch.stft(
    #             raw_audio_flat.cpu() if x_is_mps else raw_audio_flat,
    #             **self.stft_kwargs,
    #             window=stft_window.cpu() if x_is_mps else stft_window,
    #             return_complex=True
    #         ).to(device)

    #     stft_repr = torch.view_as_real(stft_repr)

    #     stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

    #     # Feature injection 0: Add mixture feature in STFT space (before merging stereo)
    #     if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
    #         mixture_feature = mixture_features_channels_list_.pop(0)
    #         # Apply STFT feature adapter (zero-initialized projection)
    #         if self.use_mixture_feature_conditioning:
    #             mixture_feature = self.stft_feature_adapter(mixture_feature)
    #         stft_repr = stft_repr + mixture_feature

    #     # Store stft_repr for later masking (before band merging)
    #     # Shape: [B*N, C, f, t, 2]
    #     stft_repr_for_mask = stft_repr.clone()

    #     # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
    #     stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

    #     # ===== BAND SPLIT =====
    #     x = rearrange(stft_repr, 'b f t c -> b t (f c)')

    #     if self.use_torch_checkpoint:
    #         x = checkpoint(self.band_split, x, use_reentrant=False)
    #     else:
    #         x = self.band_split(x)

    #     # Feature injection 1: Add mixture feature after band split
    #     if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
    #         mixture_feature = mixture_features_channels_list_.pop(0)
    #         if self.use_mixture_feature_conditioning:
    #             mixture_feature = self.band_split_feature_adapter(mixture_feature)
    #         x = x + mixture_feature

    #     # ===== TRANSFORMER LAYERS =====
    #     # axial / hierarchical attention

    #     store = [None] * len(self.layers)
    #     for i, transformer_block in enumerate(self.layers):

    #         if len(transformer_block) == 3:
    #             linear_transformer, time_transformer, freq_transformer = transformer_block

    #             x, ft_ps = pack([x], 'b * d')
    #             if self.use_torch_checkpoint:
    #                 x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
    #             else:
    #                 x = linear_transformer(x, mapping=mapping)
    #             x, = unpack(x, ft_ps, 'b * d')
    #         else:
    #             time_transformer, freq_transformer = transformer_block

    #         if self.skip_connection:
    #             # Sum all previous
    #             for j in range(i):
    #                 x = x + store[j]

    #         # TIME TRANSFORMER with feature injection
    #         x = rearrange(x, 'b t f d -> b f t d')
    #         x, ps = pack([x], '* t d')

    #         # Expand mapping for time_transformer: [b, d] -> [b*f, d]
    #         # Each batch member's mapping is repeated f times (no mixing between batches)
    #         mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

    #         if self.use_torch_checkpoint:
    #             x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
    #         else:
    #             x = time_transformer(x, mapping=mapping_time)

    #         x, = unpack(x, ps, '* t d')

    #         # Feature injection AFTER TIME transformer (skip connection style)
    #         if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
    #             mixture_feature_time = mixture_features_channels_list_.pop(0)
    #             if self.use_mixture_feature_conditioning:
    #                 mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
    #                 transformer_adapter_idx += 1
    #             x = x + mixture_feature_time

    #         # FREQ TRANSFORMER with feature injection
    #         x = rearrange(x, 'b f t d -> b t f d')
    #         x, ps = pack([x], '* f d')

    #         # Expand mapping for freq_transformer: [b, d] -> [b*t, d]
    #         # Each batch member's mapping is repeated t times (no mixing between batches)
    #         mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

    #         if self.use_torch_checkpoint:
    #             x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
    #         else:
    #             x = freq_transformer(x, mapping=mapping_freq)

    #         x, = unpack(x, ps, '* f d')

    #         # Feature injection AFTER FREQ transformer (skip connection style)
    #         if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
    #             mixture_feature_freq = mixture_features_channels_list_.pop(0)
    #             # Apply freq transformer feature adapter (zero-initialized projection)
    #             if self.use_mixture_feature_conditioning:
    #                 mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
    #                 transformer_adapter_idx += 1
    #             x = x + mixture_feature_freq

    #         if self.skip_connection:
    #             store[i] = x

    #     x = self.final_norm(x)

    #     # ===== UNFLATTEN AND APPLY MASK ESTIMATORS =====
    #     # x shape: [B*N, t, f, d] -> [B, N, t, f, d]
    #     x = rearrange(x, '(b n) t f d -> b n t f d', b=B, n=N)

    #     # Apply each mask_estimator ONLY to its corresponding stem
    #     masks = []
    #     for stem_idx in range(N):
    #         stem_features = x[:, stem_idx]  # [B, t, f, d]
    #         if self.use_torch_checkpoint:
    #             mask = checkpoint(self.mask_estimators[stem_idx], stem_features, use_reentrant=False)
    #         else:
    #             mask = self.mask_estimators[stem_idx](stem_features)
    #         masks.append(mask)

    #     # Stack masks: [B, N, t, f*c]
    #     mask = torch.stack(masks, dim=1)
    #     mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

    #     # ===== APPLY MASKS AND iSTFT =====
    #     # Unflatten stft_repr_for_mask: [B*N, C, f, t, 2] -> [B, N, C, f, t, 2]
    #     # modulate frequency representation

    #     stft_repr_for_mask = rearrange(stft_repr_for_mask, '(b n) s f t c -> b n s f t c', b=B, n=N)

    #     # Merge stereo into freq for mask application
    #     stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b n s f t c -> b n (f s) t c')

    #     # complex number multiplication

    #     stft_repr_complex = torch.view_as_complex(stft_repr_for_mask)
    #     mask_complex = torch.view_as_complex(mask)

    #     stft_repr_masked = stft_repr_complex * mask_complex

    #     # istft

    #     stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

    #     if self.zero_dc:
    #         # whether to dc filter
    #         stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

    #     # same as torch.stft() fix for MacOS MPS above
    #     try:
    #         recon_audio = torch.istft(
    #             stft_repr_masked, **self.stft_kwargs, window=stft_window,
    #             return_complex=False, length=seq_len
    #         )
    #     except:
    #         recon_audio = torch.istft(
    #             stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
    #             **self.stft_kwargs,
    #             window=stft_window.cpu() if x_is_mps else stft_window,
    #             return_complex=False, length=seq_len
    #         ).to(device)

    #     # Reshape output: [B*N*C, T] -> [B, N, C, T]
    #     recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=B, n=N, s=self.audio_channels)

    #     # ===== LOSS COMPUTATION =====
    #     # if a target is passed in, calculate loss for learning

    #     if not exists(target):
    #         return recon_audio

    #     if self.num_stems > 1:
    #         assert target.ndim == 4 and target.shape[1] == self.num_stems

    #     # Ensure target shape matches [B, N, C, T]
    #     if target.ndim == 3:
    #         target = target.unsqueeze(1)

    #     target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

    #     loss = F.l1_loss(recon_audio, target)

    #     # Multi-resolution STFT loss
    #     multi_stft_resolution_loss = 0.

    #     for window_size in self.multi_stft_resolutions_window_sizes:
    #         res_stft_kwargs = dict(
    #             n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
    #             win_length=window_size,
    #             return_complex=True,
    #             window=self.multi_stft_window_fn(window_size, device=device),
    #             **self.multi_stft_kwargs,
    #         )

    #         recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
    #         target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

    #         multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

    #     weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

    #     total_loss = loss + weighted_multi_resolution_loss

    #     if not return_loss_breakdown:
    #         return total_loss

    #     return total_loss, (loss, multi_stft_resolution_loss)

    def forward(
            self,
            raw_audio,
            time: Optional[Tensor] = None,
            s: Optional[Tensor] = None,
            *,
            features: Optional[Tensor] = None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
            target=None,
            return_loss_breakdown=False,
    ):
        """
        Memory-efficient iterative forward pass for stems-in-stems-out processing.

        Processes each stem sequentially through the transformers instead of batching
        all stems together, reducing peak memory usage by ~4x at the cost of speed.

        Args:
            raw_audio: Input tensor of shape [B, num_stems, C, T]
            time: Diffusion timestep tensor [B]
            mixture_features_channels_list: List of mixture features for conditioning
            target: Target tensor for loss computation [B, num_stems, C, T]
        """
        device = raw_audio.device

        # Handle input shape - expect [B, num_stems, C, T]
        assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

        batch_size, num_stems_in, channels, seq_len = raw_audio.shape
        assert num_stems_in == self.num_stems, \
            f"Input has {num_stems_in} stems but model expects {self.num_stems}"
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo setting must match input channels'

        # Create time conditioning mapping (shared across all stems)
        mapping = None
        if self.use_context_time:
            assert time is not None, "time parameter required when use_context_time=True"
            time_emb = self.to_time(time)
            mapping = self.to_mapping(time_emb)

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        stft_window = self.stft_window_fn(device=device)

        # Collect masked STFT representations for all stems
        all_stft_masked = []

        # ===== PROCESS EACH STEM ITERATIVELY =====
        for stem_idx in range(num_stems_in):
            # Extract this stem: [B, C, T]
            stem_audio = raw_audio[:, stem_idx]

            # Create a fresh copy of mixture features for each stem
            if mixture_features_channels_list is not None:
                mixture_features_channels_list_ = list(mixture_features_channels_list)
                transformer_adapter_idx = 0
            else:
                mixture_features_channels_list_ = None

            # ===== STFT for this stem =====
            stem_audio_packed, batch_audio_channel_packed_shape = pack_one(stem_audio, '* t')

            try:
                stft_repr = torch.stft(stem_audio_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
            except:
                stft_repr = torch.stft(
                    stem_audio_packed.cpu() if x_is_mps else stem_audio_packed,
                    **self.stft_kwargs,
                    window=stft_window.cpu() if x_is_mps else stft_window,
                    return_complex=True
                ).to(device)

            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

            # Feature injection 0: STFT space
            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning and self.stft_feature_adapter is not None:
                    stft_repr = stft_repr + self.stft_feature_adapter(mixture_feature)

            # Store stft_repr for masking later
            stft_repr_for_mask = stft_repr.clone()

            # Merge stereo/mono into frequency
            stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

            # ===== BAND SPLIT =====
            x = rearrange(stft_repr, 'b f t c -> b t (f c)')

            if self.use_torch_checkpoint:
                x = checkpoint(self.band_split, x, use_reentrant=False)
            else:
                x = self.band_split(x)

            # Feature injection 1: After band split
            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning:
                    mixture_feature = self.band_split_feature_adapter(mixture_feature)
                x = x + mixture_feature

            # ===== TRANSFORMER LAYERS =====
            store = [None] * len(self.layers)
            for i, transformer_block in enumerate(self.layers):

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    x, ft_ps = pack([x], 'b * d')
                    if self.use_torch_checkpoint:
                        x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                    else:
                        x = linear_transformer(x, mapping=mapping)
                    x, = unpack(x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                if self.skip_connection:
                    for j in range(i):
                        x = x + store[j]

                # TIME TRANSFORMER
                x = rearrange(x, 'b t f d -> b f t d')
                x, ps = pack([x], '* t d')

                mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
                else:
                    x = time_transformer(x, mapping=mapping_time)

                x, = unpack(x, ps, '* t d')

                # Feature injection after TIME transformer
                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_time = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_time

                # FREQ TRANSFORMER
                x = rearrange(x, 'b f t d -> b t f d')
                x, ps = pack([x], '* f d')

                mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
                else:
                    x = freq_transformer(x, mapping=mapping_freq)

                x, = unpack(x, ps, '* f d')

                # Feature injection after FREQ transformer
                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_freq = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_freq

                if self.skip_connection:
                    store[i] = x

            x = self.final_norm(x)

            # ===== APPLY MASK ESTIMATOR FOR THIS STEM =====
            if self.use_torch_checkpoint:
                mask = checkpoint(self.mask_estimators[stem_idx], x, use_reentrant=False)
            else:
                mask = self.mask_estimators[stem_idx](x)

            mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)

            # ===== APPLY MASK =====
            # stft_repr_for_mask: [B, C, f, t, 2]
            stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')

            stft_complex = torch.view_as_complex(stft_repr_for_mask)
            mask_complex = torch.view_as_complex(mask)

            stft_masked = stft_complex * mask_complex  # [B, f*s, t]

            all_stft_masked.append(stft_masked)

        # ===== COMBINE ALL STEMS AND iSTFT =====
        # Stack: [B, N, f*s, t]
        stft_repr_masked = torch.stack(all_stft_masked, dim=1)

        # Reshape for iSTFT: [B*N*s, f, t]
        stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

        try:
            recon_audio = torch.istft(
                stft_repr_masked, **self.stft_kwargs, window=stft_window,
                return_complex=False, length=seq_len
            )
        except:
            recon_audio = torch.istft(
                stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False, length=seq_len
            ).to(device)

        # Reshape output: [B*N*C, T] -> [B, N, C, T]
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch_size, n=num_stems_in, s=self.audio_channels)

        # ===== LOSS COMPUTATION (same as batched version) =====
        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = target[..., :recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


class BSRoformerStemsInStemsOutStemCond(BSRoformerStemsInStemsOut):
    """
    Stem-conditioned version of BSRoformerStemsInStemsOut.

    Adds a learned stem embedding so the shared transformer knows which stem
    it is refining. The embedding is summed with the diffusion time embedding
    before to_mapping, giving each transformer layer stem-specific FiLM
    conditioning via to_scale_shift.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.use_context_time, "Stem conditioning requires use_context_time=True"
        context_mapping_features = kwargs['dim'] * 4
        self.stem_embedding = nn.Embedding(self.num_stems, context_mapping_features)

    def forward(
            self,
            raw_audio,
            time: Optional[Tensor] = None,
            s: Optional[Tensor] = None,
            *,
            features: Optional[Tensor] = None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
            target=None,
            return_loss_breakdown=False,
    ):
        device = raw_audio.device

        assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

        batch_size, num_stems_in, channels, seq_len = raw_audio.shape
        assert num_stems_in == self.num_stems, \
            f"Input has {num_stems_in} stems but model expects {self.num_stems}"
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo setting must match input channels'

        # Compute time embedding ONCE (shared across all stems)
        assert time is not None, "time parameter required when use_context_time=True"
        time_emb = self.to_time(time)

        x_is_mps = True if device.type == "mps" else False
        stft_window = self.stft_window_fn(device=device)

        all_stft_masked = []

        for stem_idx in range(num_stems_in):
            stem_audio = raw_audio[:, stem_idx]

            # Stem-conditioned mapping: time_emb + stem_embedding -> to_mapping
            stem_emb = self.stem_embedding(
                torch.tensor(stem_idx, device=device)
            ).unsqueeze(0).expand(batch_size, -1)
            mapping = self.to_mapping(time_emb + stem_emb)

            if mixture_features_channels_list is not None:
                mixture_features_channels_list_ = list(mixture_features_channels_list)
                transformer_adapter_idx = 0
            else:
                mixture_features_channels_list_ = None

            # ===== STFT =====
            stem_audio_packed, batch_audio_channel_packed_shape = pack_one(stem_audio, '* t')

            try:
                stft_repr = torch.stft(stem_audio_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
            except:
                stft_repr = torch.stft(
                    stem_audio_packed.cpu() if x_is_mps else stem_audio_packed,
                    **self.stft_kwargs,
                    window=stft_window.cpu() if x_is_mps else stft_window,
                    return_complex=True
                ).to(device)

            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning and self.stft_feature_adapter is not None:
                    stft_repr = stft_repr + self.stft_feature_adapter(mixture_feature)

            stft_repr_for_mask = stft_repr.clone()
            stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

            # ===== BAND SPLIT =====
            x = rearrange(stft_repr, 'b f t c -> b t (f c)')

            if self.use_torch_checkpoint:
                x = checkpoint(self.band_split, x, use_reentrant=False)
            else:
                x = self.band_split(x)

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning:
                    mixture_feature = self.band_split_feature_adapter(mixture_feature)
                x = x + mixture_feature

            # ===== TRANSFORMER LAYERS =====
            store = [None] * len(self.layers)
            for i, transformer_block in enumerate(self.layers):

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    x, ft_ps = pack([x], 'b * d')
                    if self.use_torch_checkpoint:
                        x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                    else:
                        x = linear_transformer(x, mapping=mapping)
                    x, = unpack(x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                if self.skip_connection:
                    for j in range(i):
                        x = x + store[j]

                x = rearrange(x, 'b t f d -> b f t d')
                x, ps = pack([x], '* t d')

                mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
                else:
                    x = time_transformer(x, mapping=mapping_time)

                x, = unpack(x, ps, '* t d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_time = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_time

                x = rearrange(x, 'b f t d -> b t f d')
                x, ps = pack([x], '* f d')

                mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
                else:
                    x = freq_transformer(x, mapping=mapping_freq)

                x, = unpack(x, ps, '* f d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_freq = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_freq

                if self.skip_connection:
                    store[i] = x

            x = self.final_norm(x)

            if self.use_torch_checkpoint:
                mask = checkpoint(self.mask_estimators[stem_idx], x, use_reentrant=False)
            else:
                mask = self.mask_estimators[stem_idx](x)

            mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)

            stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')
            stft_complex = torch.view_as_complex(stft_repr_for_mask)
            mask_complex = torch.view_as_complex(mask)
            stft_masked = stft_complex * mask_complex
            all_stft_masked.append(stft_masked)

        # ===== COMBINE ALL STEMS AND iSTFT =====
        stft_repr_masked = torch.stack(all_stft_masked, dim=1)
        stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

        try:
            recon_audio = torch.istft(
                stft_repr_masked, **self.stft_kwargs, window=stft_window,
                return_complex=False, length=seq_len
            )
        except:
            recon_audio = torch.istft(
                stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False, length=seq_len
            ).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch_size, n=num_stems_in, s=self.audio_channels)

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = target[..., :recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


class BSRoformerStemsInStemsOutStemCondRandomStem(BSRoformerStemsInStemsOut):
    """
    Stem-conditioned version with random stem training.

    During training: module_base picks 1 random stem, passes [B,1,C,T] input
    with stem index via `features` kwarg.
    During inference: receives [B,num_stems,C,T] with features=None, processes
    all stems sequentially.

    Uses a learned stem embedding (summed with time embedding before to_mapping)
    to give each transformer layer stem-specific FiLM conditioning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.use_context_time, "Stem conditioning requires use_context_time=True"
        context_mapping_features = kwargs['dim'] * 4
        self.stem_embedding = nn.Embedding(self.num_stems, context_mapping_features)

    def forward(
            self,
            raw_audio,
            time: Optional[Tensor] = None,
            s: Optional[Tensor] = None,
            *,
            features=None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
            target=None,
            return_loss_breakdown=False,
    ):
        device = raw_audio.device

        assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

        batch_size, num_stems_in, channels, seq_len = raw_audio.shape
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo setting must match input channels'

        # Compute time embedding ONCE (shared across all stems)
        assert time is not None, "time parameter required when use_context_time=True"
        time_emb = self.to_time(time)

        x_is_mps = True if device.type == "mps" else False
        stft_window = self.stft_window_fn(device=device)

        # features carries stem index during training (set by module_base)
        # features=None during eval -> process all stems
        if features is not None:
            # Training: single stem [B,1,C,T], features = stem index (int)
            assert num_stems_in == 1, f"When features (stem_idx) is provided, expected 1 stem, got {num_stems_in}"
            stem_id = features if isinstance(features, int) else int(features)
            # (position_in_input, stem_identity)
            stem_loop = [(0, stem_id)]
        else:
            # Eval: all stems [B,num_stems,C,T]
            assert num_stems_in == self.num_stems, \
                f"Input has {num_stems_in} stems but model expects {self.num_stems}"
            stem_loop = [(i, i) for i in range(num_stems_in)]

        n_processed = len(stem_loop)
        all_stft_masked = []

        for input_pos, stem_id in stem_loop:
            stem_audio = raw_audio[:, input_pos]

            # Stem-conditioned mapping: time_emb + stem_embedding -> to_mapping
            stem_emb = self.stem_embedding(
                torch.tensor(stem_id, device=device)
            ).unsqueeze(0).expand(batch_size, -1)
            mapping = self.to_mapping(time_emb + stem_emb)

            if mixture_features_channels_list is not None:
                mixture_features_channels_list_ = list(mixture_features_channels_list)
                transformer_adapter_idx = 0
            else:
                mixture_features_channels_list_ = None

            # ===== STFT =====
            stem_audio_packed, batch_audio_channel_packed_shape = pack_one(stem_audio, '* t')

            try:
                stft_repr = torch.stft(stem_audio_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
            except:
                stft_repr = torch.stft(
                    stem_audio_packed.cpu() if x_is_mps else stem_audio_packed,
                    **self.stft_kwargs,
                    window=stft_window.cpu() if x_is_mps else stft_window,
                    return_complex=True
                ).to(device)

            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning and self.stft_feature_adapter is not None:
                    stft_repr = stft_repr + self.stft_feature_adapter(mixture_feature)

            stft_repr_for_mask = stft_repr.clone()
            stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

            # ===== BAND SPLIT =====
            x = rearrange(stft_repr, 'b f t c -> b t (f c)')

            if self.use_torch_checkpoint:
                x = checkpoint(self.band_split, x, use_reentrant=False)
            else:
                x = self.band_split(x)

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning:
                    mixture_feature = self.band_split_feature_adapter(mixture_feature)
                x = x + mixture_feature

            # ===== TRANSFORMER LAYERS =====
            store = [None] * len(self.layers)
            for i, transformer_block in enumerate(self.layers):

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    x, ft_ps = pack([x], 'b * d')
                    if self.use_torch_checkpoint:
                        x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                    else:
                        x = linear_transformer(x, mapping=mapping)
                    x, = unpack(x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                if self.skip_connection:
                    for j in range(i):
                        x = x + store[j]

                x = rearrange(x, 'b t f d -> b f t d')
                x, ps = pack([x], '* t d')

                mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
                else:
                    x = time_transformer(x, mapping=mapping_time)

                x, = unpack(x, ps, '* t d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_time = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_time

                x = rearrange(x, 'b f t d -> b t f d')
                x, ps = pack([x], '* f d')

                mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
                else:
                    x = freq_transformer(x, mapping=mapping_freq)

                x, = unpack(x, ps, '* f d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_freq = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_freq

                if self.skip_connection:
                    store[i] = x

            x = self.final_norm(x)

            if self.use_torch_checkpoint:
                mask = checkpoint(self.mask_estimators[stem_id], x, use_reentrant=False)
            else:
                mask = self.mask_estimators[stem_id](x)

            mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)

            stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')
            stft_complex = torch.view_as_complex(stft_repr_for_mask)
            mask_complex = torch.view_as_complex(mask)
            stft_masked = stft_complex * mask_complex
            all_stft_masked.append(stft_masked)
            
            # stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')
            # stft_complex = torch.view_as_complex(stft_repr_for_mask.contiguous())
            # all_stft_masked.append(stft_complex)
            

        # ===== COMBINE ALL STEMS AND iSTFT =====
        stft_repr_masked = torch.stack(all_stft_masked, dim=1)
        stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

        try:
            recon_audio = torch.istft(
                stft_repr_masked, **self.stft_kwargs, window=stft_window,
                return_complex=False, length=seq_len
            )
        except:
            recon_audio = torch.istft(
                stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False, length=seq_len
            ).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch_size, n=n_processed, s=self.audio_channels)

        if not exists(target):
            return recon_audio


class BSRoformerStemsInStemsOutStemCondRandomStem_det2(BSRoformerStemsInStemsOut):
    """
    Stem-conditioned version with random stem training.

    During training: module_base picks 1 random stem, passes [B,1,C,T] input
    with stem index via `features` kwarg.
    During inference: receives [B,num_stems,C,T] with features=None, processes
    all stems sequentially.

    Uses a learned stem embedding (summed with time embedding before to_mapping)
    to give each transformer layer stem-specific FiLM conditioning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        context_mapping_features = kwargs['dim'] * 4
        self.stem_embedding = nn.Embedding(self.num_stems, context_mapping_features)

    def forward(
            self,
            raw_audio,
            time: Optional[Tensor] = None,
            s: Optional[Tensor] = None,
            *,
            features=None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
            target=None,
            return_loss_breakdown=False,
    ):
        device = raw_audio.device

        assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

        batch_size, num_stems_in, channels, seq_len = raw_audio.shape
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo setting must match input channels'

        x_is_mps = True if device.type == "mps" else False
        stft_window = self.stft_window_fn(device=device)

        # features carries stem index during training (set by module_base)
        # features=None during eval -> process all stems
        if features is not None:
            # Training: single stem [B,1,C,T], features = stem index (int)
            assert num_stems_in == 1, f"When features (stem_idx) is provided, expected 1 stem, got {num_stems_in}"
            stem_id = features if isinstance(features, int) else int(features)
            # (position_in_input, stem_identity)
            stem_loop = [(0, stem_id)]
        else:
            # Eval: all stems [B,num_stems,C,T]
            assert num_stems_in == self.num_stems, \
                f"Input has {num_stems_in} stems but model expects {self.num_stems}"
            stem_loop = [(i, i) for i in range(num_stems_in)]

        n_processed = len(stem_loop)
        all_stft_masked = []

        for input_pos, stem_id in stem_loop:
            stem_audio = raw_audio[:, input_pos]

            # Stem-conditioned mapping
            mapping = self.stem_embedding(
                torch.tensor(stem_id, device=device)
            ).unsqueeze(0).expand(batch_size, -1)

            if mixture_features_channels_list is not None:
                mixture_features_channels_list_ = list(mixture_features_channels_list)
                transformer_adapter_idx = 0
            else:
                mixture_features_channels_list_ = None

            # ===== STFT =====
            stem_audio_packed, batch_audio_channel_packed_shape = pack_one(stem_audio, '* t')

            try:
                stft_repr = torch.stft(stem_audio_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
            except:
                stft_repr = torch.stft(
                    stem_audio_packed.cpu() if x_is_mps else stem_audio_packed,
                    **self.stft_kwargs,
                    window=stft_window.cpu() if x_is_mps else stft_window,
                    return_complex=True
                ).to(device)

            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning and self.stft_feature_adapter is not None:
                    stft_repr = stft_repr + self.stft_feature_adapter(mixture_feature)

            stft_repr_for_mask = stft_repr.clone()
            stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

            # ===== BAND SPLIT =====
            x = rearrange(stft_repr, 'b f t c -> b t (f c)')

            if self.use_torch_checkpoint:
                x = checkpoint(self.band_split, x, use_reentrant=False)
            else:
                x = self.band_split(x)

            if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                mixture_feature = mixture_features_channels_list_.pop(0)
                if self.use_mixture_feature_conditioning:
                    mixture_feature = self.band_split_feature_adapter(mixture_feature)
                x = x + mixture_feature

            # ===== TRANSFORMER LAYERS =====
            store = [None] * len(self.layers)
            for i, transformer_block in enumerate(self.layers):

                if len(transformer_block) == 3:
                    linear_transformer, time_transformer, freq_transformer = transformer_block

                    x, ft_ps = pack([x], 'b * d')
                    if self.use_torch_checkpoint:
                        x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
                    else:
                        x = linear_transformer(x, mapping=mapping)
                    x, = unpack(x, ft_ps, 'b * d')
                else:
                    time_transformer, freq_transformer = transformer_block

                if self.skip_connection:
                    for j in range(i):
                        x = x + store[j]

                x = rearrange(x, 'b t f d -> b f t d')
                x, ps = pack([x], '* t d')

                mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
                else:
                    x = time_transformer(x, mapping=mapping_time)

                x, = unpack(x, ps, '* t d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_time = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_time

                x = rearrange(x, 'b f t d -> b t f d')
                x, ps = pack([x], '* f d')

                mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping

                if self.use_torch_checkpoint:
                    x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
                else:
                    x = freq_transformer(x, mapping=mapping_freq)

                x, = unpack(x, ps, '* f d')

                if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
                    mixture_feature_freq = mixture_features_channels_list_.pop(0)
                    if self.use_mixture_feature_conditioning:
                        mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
                        transformer_adapter_idx += 1
                    x = x + mixture_feature_freq

                if self.skip_connection:
                    store[i] = x

            x = self.final_norm(x)

            if self.use_torch_checkpoint:
                mask = checkpoint(self.mask_estimators[stem_id], x, use_reentrant=False)
            else:
                mask = self.mask_estimators[stem_id](x)

            mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)

            stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')
            stft_complex = torch.view_as_complex(stft_repr_for_mask)
            mask_complex = torch.view_as_complex(mask)
            stft_masked = stft_complex * mask_complex
            all_stft_masked.append(stft_masked)

        # ===== COMBINE ALL STEMS AND iSTFT =====
        stft_repr_masked = torch.stack(all_stft_masked, dim=1)
        stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

        try:
            recon_audio = torch.istft(
                stft_repr_masked, **self.stft_kwargs, window=stft_window,
                return_complex=False, length=seq_len
            )
        except:
            recon_audio = torch.istft(
                stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False, length=seq_len
            ).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch_size, n=n_processed, s=self.audio_channels)

        if not exists(target):
            return recon_audio
       
##### this one bathces inference but not training ##### Didn't give us acceleration 
# class BSRoformerStemsInStemsOutStemCondRandomStem(BSRoformerStemsInStemsOut):
#     """
#     Stem-conditioned version with random stem training.

#     During training: module_base picks 1 random stem, passes [B,1,C,T] input
#     with stem index via `features` kwarg. Runs one sequential transformer pass.
#     During inference: receives [B,num_stems,C,T] with features=None. Flattens
#     all stems into the batch dim [B*N,C,T] and runs a single batched transformer
#     pass, then splits per-stem for mask estimation. ~4x faster than sequential.

#     Uses a learned stem embedding (summed with time embedding before to_mapping)
#     to give each transformer layer stem-specific FiLM conditioning.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         assert self.use_context_time, "Stem conditioning requires use_context_time=True"
#         context_mapping_features = kwargs['dim'] * 4
#         self.stem_embedding = nn.Embedding(self.num_stems, context_mapping_features)

#     def forward(
#             self,
#             raw_audio,
#             time: Optional[Tensor] = None,
#             s: Optional[Tensor] = None,
#             *,
#             features=None,
#             channels_list: Optional[Sequence[Tensor]] = None,
#             embedding: Optional[Tensor] = None,
#             mixture_features_channels_list: Optional[Sequence[Tensor]] = None,
#             target=None,
#             return_loss_breakdown=False,
#     ):
#         device = raw_audio.device

#         assert raw_audio.ndim == 4, f"Expected 4D input [B, num_stems, C, T], got shape {raw_audio.shape}"

#         batch_size, num_stems_in, channels, seq_len = raw_audio.shape
#         assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
#             'stereo setting must match input channels'

#         # Compute time embedding ONCE (shared across all stems)
#         assert time is not None, "time parameter required when use_context_time=True"
#         time_emb = self.to_time(time)

#         x_is_mps = True if device.type == "mps" else False
#         stft_window = self.stft_window_fn(device=device)

#         # features carries stem index during training (set by module_base)
#         # features=None during eval -> process all stems
#         if features is not None:
#             # Training: single stem [B,1,C,T], features = stem index (int)
#             assert num_stems_in == 1, f"When features (stem_idx) is provided, expected 1 stem, got {num_stems_in}"
#             stem_id = features if isinstance(features, int) else int(features)
#             # (position_in_input, stem_identity)
#             stem_loop = [(0, stem_id)]
#         else:
#             # Eval: all stems [B,num_stems,C,T]
#             assert num_stems_in == self.num_stems, \
#                 f"Input has {num_stems_in} stems but model expects {self.num_stems}"
#             stem_loop = [(i, i) for i in range(num_stems_in)]

#         all_stft_masked = []

#         if features is not None:
#             # ===== TRAINING PATH: single stem, sequential =====
#             n_processed = 1
#             input_pos, stem_id = stem_loop[0]
#             stem_audio = raw_audio[:, input_pos]

#             stem_emb = self.stem_embedding(
#                 torch.tensor(stem_id, device=device)
#             ).unsqueeze(0).expand(batch_size, -1)
#             mapping = self.to_mapping(time_emb + stem_emb)

#             if mixture_features_channels_list is not None:
#                 mixture_features_channels_list_ = list(mixture_features_channels_list)
#                 transformer_adapter_idx = 0
#             else:
#                 mixture_features_channels_list_ = None

#             stem_audio_packed, batch_audio_channel_packed_shape = pack_one(stem_audio, '* t')
#             try:
#                 stft_repr = torch.stft(stem_audio_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
#             except:
#                 stft_repr = torch.stft(
#                     stem_audio_packed.cpu() if x_is_mps else stem_audio_packed,
#                     **self.stft_kwargs,
#                     window=stft_window.cpu() if x_is_mps else stft_window,
#                     return_complex=True
#                 ).to(device)

#             stft_repr = torch.view_as_real(stft_repr)
#             stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

#             if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                 mixture_feature = mixture_features_channels_list_.pop(0)
#                 if self.use_mixture_feature_conditioning:
#                     mixture_feature = self.stft_feature_adapter(mixture_feature)
#                 stft_repr = stft_repr + mixture_feature

#             stft_repr_for_mask = stft_repr.clone()
#             stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

#             x = rearrange(stft_repr, 'b f t c -> b t (f c)')
#             if self.use_torch_checkpoint:
#                 x = checkpoint(self.band_split, x, use_reentrant=False)
#             else:
#                 x = self.band_split(x)

#             if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                 mixture_feature = mixture_features_channels_list_.pop(0)
#                 if self.use_mixture_feature_conditioning:
#                     mixture_feature = self.band_split_feature_adapter(mixture_feature)
#                 x = x + mixture_feature

#             store = [None] * len(self.layers)
#             for i, transformer_block in enumerate(self.layers):
#                 if len(transformer_block) == 3:
#                     linear_transformer, time_transformer, freq_transformer = transformer_block
#                     x, ft_ps = pack([x], 'b * d')
#                     if self.use_torch_checkpoint:
#                         x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
#                     else:
#                         x = linear_transformer(x, mapping=mapping)
#                     x, = unpack(x, ft_ps, 'b * d')
#                 else:
#                     time_transformer, freq_transformer = transformer_block

#                 if self.skip_connection:
#                     for j in range(i):
#                         x = x + store[j]

#                 x = rearrange(x, 'b t f d -> b f t d')
#                 x, ps = pack([x], '* t d')
#                 mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping
#                 if self.use_torch_checkpoint:
#                     x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
#                 else:
#                     x = time_transformer(x, mapping=mapping_time)
#                 x, = unpack(x, ps, '* t d')

#                 if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                     mixture_feature_time = mixture_features_channels_list_.pop(0)
#                     if self.use_mixture_feature_conditioning:
#                         mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
#                         transformer_adapter_idx += 1
#                     x = x + mixture_feature_time

#                 x = rearrange(x, 'b f t d -> b t f d')
#                 x, ps = pack([x], '* f d')
#                 mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping
#                 if self.use_torch_checkpoint:
#                     x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
#                 else:
#                     x = freq_transformer(x, mapping=mapping_freq)
#                 x, = unpack(x, ps, '* f d')

#                 if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                     mixture_feature_freq = mixture_features_channels_list_.pop(0)
#                     if self.use_mixture_feature_conditioning:
#                         mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
#                         transformer_adapter_idx += 1
#                     x = x + mixture_feature_freq

#                 if self.skip_connection:
#                     store[i] = x

#             x = self.final_norm(x)

#             if self.use_torch_checkpoint:
#                 mask = checkpoint(self.mask_estimators[stem_id], x, use_reentrant=False)
#             else:
#                 mask = self.mask_estimators[stem_id](x)

#             mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)
#             stft_repr_for_mask = rearrange(stft_repr_for_mask, 'b s f t c -> b (f s) t c')
#             stft_complex = torch.view_as_complex(stft_repr_for_mask)
#             mask_complex = torch.view_as_complex(mask)
#             all_stft_masked.append(stft_complex * mask_complex)

#         else:
#             # ===== EVAL PATH: all stems batched in one transformer pass =====
#             N = self.num_stems
#             n_processed = N

#             # Flatten [B, N, C, T] -> [B*N, C, T]
#             raw_audio_flat = rearrange(raw_audio, 'b n c t -> (b n) c t')

#             # Expand time emb and build per-stem mappings: [B*N, mapping_dim]
#             time_emb_expanded = repeat(time_emb, 'b d -> (b n) d', n=N)
#             stem_ids = torch.arange(N, device=device)
#             stem_emb = self.stem_embedding(stem_ids)                          # [N, d]
#             stem_emb_expanded = repeat(stem_emb, 'n d -> (b n) d', b=batch_size)  # [B*N, d]
#             mapping = self.to_mapping(time_emb_expanded + stem_emb_expanded)

#             # Expand mixture features: [B, ...] -> [B*N, ...]
#             if mixture_features_channels_list is not None:
#                 mixture_features_channels_list_ = [
#                     repeat(f, 'b ... -> (b n) ...', n=N)
#                     for f in mixture_features_channels_list
#                 ]
#                 transformer_adapter_idx = 0
#             else:
#                 mixture_features_channels_list_ = None

#             # STFT on [B*N, C, T]
#             flat_packed, batch_audio_channel_packed_shape = pack_one(raw_audio_flat, '* t')
#             try:
#                 stft_repr = torch.stft(flat_packed, **self.stft_kwargs, window=stft_window, return_complex=True)
#             except:
#                 stft_repr = torch.stft(
#                     flat_packed.cpu() if x_is_mps else flat_packed,
#                     **self.stft_kwargs,
#                     window=stft_window.cpu() if x_is_mps else stft_window,
#                     return_complex=True
#                 ).to(device)

#             stft_repr = torch.view_as_real(stft_repr)
#             stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
#             # stft_repr: [B*N, C, f, t, 2]

#             if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                 mixture_feature = mixture_features_channels_list_.pop(0)
#                 if self.use_mixture_feature_conditioning:
#                     mixture_feature = self.stft_feature_adapter(mixture_feature)
#                 stft_repr = stft_repr + mixture_feature

#             stft_repr_for_mask = stft_repr.clone()  # [B*N, C, f, t, 2]
#             stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

#             x = rearrange(stft_repr, 'b f t c -> b t (f c)')
#             if self.use_torch_checkpoint:
#                 x = checkpoint(self.band_split, x, use_reentrant=False)
#             else:
#                 x = self.band_split(x)

#             if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                 mixture_feature = mixture_features_channels_list_.pop(0)
#                 if self.use_mixture_feature_conditioning:
#                     mixture_feature = self.band_split_feature_adapter(mixture_feature)
#                 x = x + mixture_feature

#             store = [None] * len(self.layers)
#             for i, transformer_block in enumerate(self.layers):
#                 if len(transformer_block) == 3:
#                     linear_transformer, time_transformer, freq_transformer = transformer_block
#                     x, ft_ps = pack([x], 'b * d')
#                     if self.use_torch_checkpoint:
#                         x = checkpoint(linear_transformer, x, mapping, use_reentrant=False)
#                     else:
#                         x = linear_transformer(x, mapping=mapping)
#                     x, = unpack(x, ft_ps, 'b * d')
#                 else:
#                     time_transformer, freq_transformer = transformer_block

#                 if self.skip_connection:
#                     for j in range(i):
#                         x = x + store[j]

#                 x = rearrange(x, 'b t f d -> b f t d')
#                 x, ps = pack([x], '* t d')
#                 mapping_time = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping
#                 if self.use_torch_checkpoint:
#                     x = checkpoint(time_transformer, x, mapping_time, use_reentrant=False)
#                 else:
#                     x = time_transformer(x, mapping=mapping_time)
#                 x, = unpack(x, ps, '* t d')

#                 if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                     mixture_feature_time = mixture_features_channels_list_.pop(0)
#                     if self.use_mixture_feature_conditioning:
#                         mixture_feature_time = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_time)
#                         transformer_adapter_idx += 1
#                     x = x + mixture_feature_time

#                 x = rearrange(x, 'b f t d -> b t f d')
#                 x, ps = pack([x], '* f d')
#                 mapping_freq = expand_mapping_for_packed_batch(mapping, ps[0][1]) if self.use_context_time else mapping
#                 if self.use_torch_checkpoint:
#                     x = checkpoint(freq_transformer, x, mapping_freq, use_reentrant=False)
#                 else:
#                     x = freq_transformer(x, mapping=mapping_freq)
#                 x, = unpack(x, ps, '* f d')

#                 if mixture_features_channels_list_ is not None and len(mixture_features_channels_list_) > 0:
#                     mixture_feature_freq = mixture_features_channels_list_.pop(0)
#                     if self.use_mixture_feature_conditioning:
#                         mixture_feature_freq = self.transformer_feature_adapters[transformer_adapter_idx](mixture_feature_freq)
#                         transformer_adapter_idx += 1
#                     x = x + mixture_feature_freq

#                 if self.skip_connection:
#                     store[i] = x

#             x = self.final_norm(x)  # [B*N, t, f, d]

#             # Split back by stem and apply per-stem mask estimators
#             x_per_stem = rearrange(x, '(b n) t f d -> b n t f d', b=batch_size, n=N)
#             stft_per_stem = rearrange(stft_repr_for_mask, '(b n) s f t c -> b n s f t c', b=batch_size, n=N)

#             for sid in range(N):
#                 stem_x = x_per_stem[:, sid]  # [B, t, f, d]
#                 if self.use_torch_checkpoint:
#                     mask = checkpoint(self.mask_estimators[sid], stem_x, use_reentrant=False)
#                 else:
#                     mask = self.mask_estimators[sid](stem_x)
#                 mask = rearrange(mask, 'b t (f c) -> b f t c', c=2)
#                 stft_repr_stem = rearrange(stft_per_stem[:, sid], 'b s f t c -> b (f s) t c')
#                 stft_complex = torch.view_as_complex(stft_repr_stem)
#                 mask_complex = torch.view_as_complex(mask)
#                 all_stft_masked.append(stft_complex * mask_complex)

#         # ===== COMBINE ALL STEMS AND iSTFT =====
#         stft_repr_masked = torch.stack(all_stft_masked, dim=1)
#         stft_repr_masked = rearrange(stft_repr_masked, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

#         if self.zero_dc:
#             stft_repr_masked = stft_repr_masked.index_fill(1, tensor(0, device=device), 0.)

#         try:
#             recon_audio = torch.istft(
#                 stft_repr_masked, **self.stft_kwargs, window=stft_window,
#                 return_complex=False, length=seq_len
#             )
#         except:
#             recon_audio = torch.istft(
#                 stft_repr_masked.cpu() if x_is_mps else stft_repr_masked,
#                 **self.stft_kwargs,
#                 window=stft_window.cpu() if x_is_mps else stft_window,
#                 return_complex=False, length=seq_len
#             ).to(device)

#         recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch_size, n=n_processed, s=self.audio_channels)

#         if not exists(target):
#             return recon_audio
