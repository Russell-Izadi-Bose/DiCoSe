from math import sqrt
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
import tqdm

from .utils import default, exists

""" Distributions """


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


""" Schedules """


class Schedule(nn.Module):
    """Interface used by different schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        
        if num_steps == 1:
            sigmas = torch.tensor([self.sigma_max], device=device, dtype=torch.float32)
        else:
            sigmas = (
                self.sigma_max ** rho_inv
                + (steps / (num_steps - 1))
                * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
            ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


""" Samplers """

""" Many methods inspired by https://github.com/crowsonkb/k-diffusion/ """


class Sampler(nn.Module):
    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")


class KarrasSampler(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 1"""

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(
        self, x: Tensor, fn: Callable, sigma: float, sigma_next: float, gamma: float
    ) -> Tensor:
        """Algorithm 2 (step)"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        # Evaluate ∂x/∂sigma at sigma_hat
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d
        # Second order correction
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, sqrt(2) - 1),
            0.0,
        )
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(
                x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i]  # type: ignore # noqa
            )

        return x


class AEulerSampler(Sampler):
    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Euler method
        x_next = x + d * (sigma_down - sigma)
        # Add randomness
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x


class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    def __init__(self, rho: float = 1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        # x = sigmas[0] * noise
        x = sigmas[0] * noise[0] + noise[1]
        
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        for i in range(num_steps - 1):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask


class MSDMSampler(Sampler):
    def __init__(self, num_resamples: int = 1, s_churn: float = 0.0):
        super().__init__()
        self.s_churn=s_churn 
        self.num_resamples=num_resamples

    def score_differential(self, x, sigma, denoise_fn):
        d = (x - denoise_fn(x, sigma=sigma)) / sigma 
        return d

    @torch.no_grad()
    def generate_track(
        self,
        denoise_fn: Callable,
        sigmas: torch.Tensor,
        noises: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = sigmas[0] * noises[0] + noises[1]
        # _, num_sources, _  = x.shape

        # Initialize default values
        source = torch.zeros_like(x) if source is None else source
        mask = torch.zeros_like(x) if mask is None else mask
        
        sigmas = sigmas.to(x.device)
        gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        
        # Iterate over all timesteps
        for i in tqdm.tqdm(range(len(sigmas) - 1)):
            sigma, sigma_next = sigmas[i], sigmas[i+1]

            # Noise source to current noise level
            noisy_source = source + sigma*torch.randn_like(source)
            
            for r in range(self.num_resamples):
                # Merge noisy source and current x
                x = mask*noisy_source + (1.0 - mask)*x 

                # Inject randomness
                sigma_hat = sigma * (gamma + 1)    
                x_hat = x + torch.randn_like(x) * (sigma_hat**2 - sigma**2)**0.5

                # Compute conditioned derivative
                d = self.score_differential(x=x_hat, sigma=sigma_hat, denoise_fn=denoise_fn)

                # Update integral
                x = x_hat + d*(sigma_next - sigma_hat)
                    
                # Renoise if not last resample step
                if r < self.num_resamples - 1:
                    x = x + torch.randn_like(x) * (sigma**2 - sigma_next**2)**0.5

        return mask*source + (1.0 - mask)*x


    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int ) -> Tensor:
        x = self.generate_track(fn,
                        sigmas=sigmas,
                        noises=noise,
                        )
        return x






""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


class Diffusion(nn.Module):
    """Elucidated Diffusion: https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
        loss_type: str = "mse",  # "mse" or "bsroformer"
        bridge_sigma_max: float = None,  # If set, enables bridge interpolation mode
    ):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold
        self.loss_type = loss_type
        self.bridge_sigma_max = bridge_sigma_max  # For bridge/interpolation mode

    def get_scale_weights(self, sigmas: Tensor, ndim: int = 3) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        # Pad sigmas to match input dimensions: [B] -> [B, 1, 1, ...]
        batch = sigmas.shape[0]
        sigmas_padded = sigmas.view(batch, *([1] * (ndim - 1)))
        c_skip = (sigma_data ** 2) / (sigmas_padded ** 2 + sigma_data ** 2)
        c_out = (
            sigmas_padded * sigma_data * (sigma_data ** 2 + sigmas_padded ** 2) ** -0.5
        )
        c_in = (sigmas_padded ** 2 + sigma_data ** 2) ** -0.5
        c_noise = torch.log(sigmas) * 0.25
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch, device = x_noisy.shape[0], x_noisy.device

        assert exists(sigmas) ^ exists(sigma), "Either sigmas or sigma must be provided"

        # If sigma provided use the same for all batch items (used for sampling)
        if exists(sigma):
            sigmas = torch.full(size=(batch,), fill_value=sigma).to(device)

        assert exists(sigmas)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, ndim=x_noisy.ndim)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        # Dynamic thresholding
        if self.dynamic_threshold == 0.0:
            return x_denoised.clamp(-1.0, 1.0)
        else:
            # Find dynamic threshold quantile for each batch
            x_flat = rearrange(x_denoised, "b ... -> b (...)")
            scale = torch.quantile(x_flat.abs(), self.dynamic_threshold, dim=-1)
            # Clamp to a min of 1.0
            scale.clamp_(min=1.0)
            # Clamp all values and scale
            scale = pad_dims(scale, ndim=x_denoised.ndim - scale.ndim)
            x_denoised = x_denoised.clamp(-scale, scale) / scale
            return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, target: Tensor = None, **kwargs) -> Tensor:
        """
        Forward pass for diffusion training.

        Args:
            x: Input tensor - source for bridge mode (extracted), or base for standard mode
            noise: Optional pre-generated noise
            target: Clean ground truth (defaults to x if not provided)

        Bridge mode (when bridge_sigma_max is set):
            - At high sigma: x_base ≈ x (extracted)
            - At low sigma: x_base ≈ target (clean)
            - Interpolates smoothly based on sigma
        """
        batch, device = x.shape[0], x.device

        # target defaults to x (standard diffusion), but can be overridden
        # to always reconstruct clean ground truth even when x is degraded
        if target is None:
            target = x

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch, device=device)
        # Pad sigmas to match x dimensions: [B] -> [B, 1, 1, ...]
        sigmas_padded = sigmas.view(batch, *([1] * (x.ndim - 1)))

        # Bridge interpolation mode: interpolate base signal based on sigma
        # At high sigma -> use x (extracted), at low sigma -> use target (clean)
        if self.bridge_sigma_max is not None and target is not None:
            # alpha = 1 at sigma >= bridge_sigma_max, alpha = 0 at sigma = 0
            alpha = (sigmas_padded / self.bridge_sigma_max).clamp(0, 1)
            x_base = alpha * x + (1 - alpha) * target
        else:
            x_base = x

        # Add noise to base signal
        noise = default(noise, lambda: torch.randn_like(x_base))
        x_noisy = x_base + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Reconstruction loss: always compared against target (clean ground truth)
        if self.loss_type == "mse":
            per_example_loss = self._mse_per_example(x_denoised, target)
            # EDM weighting only for MSE (theoretically grounded)
            weights = self.loss_weight(sigmas)   # [B]
            loss = (per_example_loss * weights).mean()
        else:
            per_example_loss = self._bsroformer_loss(x_denoised, target)
            # Adding EDM weighting for non-MSE losses showed better results
            weights = self.loss_weight(sigmas)   # [B]
            loss = (per_example_loss * weights).mean()

        return loss

    def _mse_per_example(self, x_denoised: Tensor, x: Tensor) -> Tensor:
        # [B, ...] -> [B]
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        return losses

    def _ensure_bct(self, x: Tensor) -> Tensor:
        """
        Normalize shapes to [B, C, T]:

        - [B, T]        -> [B, 1, T]
        - [B, C, T]     -> [B, C, T]
        - [B, N, C, T]  -> [B, N*C, T]
        """
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim == 3:
            pass
        elif x.ndim == 4:
            B, N, C, T = x.shape
            x = x.view(B, N * C, T)
        else:
            raise ValueError(f"Unsupported audio shape {x.shape}")
        return x

    def _bsroformer_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        """
        BSRoformer-style reconstruction loss:
        waveform L1 + multi-resolution STFT L1, per-example [B].

        Hyperparameters are taken from self.net if present, otherwise fall back to defaults.
        """

        device = recon.device
        B = recon.shape[0]

        # base waveform L1 per example
        wave_loss = F.l1_loss(recon, target, reduction="none")
        wave_loss = wave_loss.view(B, -1).mean(dim=1)  # [B]

        # read settings from the underlying net if they exist
        net = self.net
        multi_weight = getattr(net, "multi_stft_resolution_loss_weight", 1.0)
        window_sizes = getattr(
            net,
            "multi_stft_resolutions_window_sizes",
            (4096, 2048, 1024, 512, 256),
        )
        n_fft_base = getattr(net, "multi_stft_n_fft", 2048)
        window_fn = getattr(net, "multi_stft_window_fn", torch.hann_window)
        multi_kwargs = getattr(
            net,
            "multi_stft_kwargs",
            dict(hop_length=147, normalized=False),
        )

        recon_bct = self._ensure_bct(recon)   # [B, C, T]
        target_bct = self._ensure_bct(target) # [B, C, T]
        B, C, T = recon_bct.shape

        multi_stft_loss = recon.new_zeros(B)

        for window_size in window_sizes:
            n_fft = max(window_size, n_fft_base)

            res_stft_kwargs = dict(
                n_fft=n_fft,
                win_length=window_size,
                return_complex=True,
                window=window_fn(window_size, device=device),
                **multi_kwargs,
            )

            # [B, C, T] -> [B*C, T]
            recon_flat = recon_bct.reshape(B * C, T)
            target_flat = target_bct.reshape(B * C, T)

            recon_Y = torch.stft(recon_flat, **res_stft_kwargs)
            target_Y = torch.stft(target_flat, **res_stft_kwargs)

            # [B*C, F, T] -> [B, C, F, T]
            recon_Y = recon_Y.view(B, C, *recon_Y.shape[-2:])
            target_Y = target_Y.view(B, C, *target_Y.shape[-2:])

            diff = (recon_Y - target_Y).abs()
            # average over channels, freqs, time -> [B]
            stft_loss_b = diff.mean(dim=(1, 2, 3))
            multi_stft_loss = multi_stft_loss + stft_loss_b

        total = wave_loss + multi_weight * multi_stft_loss  # [B]
        return total


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps

    @torch.no_grad()
    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise[0].device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0)
        return x


class DiffusionInpainter(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        num_resamples: int,
        sampler: Sampler,
        sigma_schedule: Schedule,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor) -> Tensor:
        x = self.inpaint_fn(
            source=inpaint,
            mask=inpaint_mask,
            fn=self.denoise_fn,
            sigmas=self.sigma_schedule(self.num_steps, inpaint.device),
            num_steps=self.num_steps,
            num_resamples=self.num_resamples,
        )
        return x


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):
    def __init__(
        self,
        inpainter: DiffusionInpainter,
        *,
        num_spans: int,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool = False) -> Tensor:
        half_length = start.shape[2] // 2

        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        # Inpaint second half from first half
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)

        for i in range(self.num_spans):
            # Inpaint second half
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            # Replace first half with generated second half
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=2)
