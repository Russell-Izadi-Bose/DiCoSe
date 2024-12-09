
import torch
from einops import rearrange, reduce
import numpy as np
from torch import Tensor, nn
from audio_diffusion_pytorch_.modules import UNet1d
import pytorch_lightning as pl
import torch.nn.functional as F


class Model1d_simple(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.unet = UNet1d(**kwargs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.unet(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule,
        sampler,
        **kwargs
    ) -> Tensor:

        return self.unet(noise, **kwargs)


class Audio_DM_Model_simple(pl.LightningModule):
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, class_cond: bool, separation: bool = False, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.class_cond = class_cond
        self.separation = separation

        self.model = Model1d_simple(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    def get_input(self, batch):

        if isinstance(batch, (list, tuple)) and self.class_cond and self.separation:
            waveforms, class_indexes, stems  = batch

            batch_size, channels, feature_width = waveforms.shape
            mixture = stems.sum(1)

            # Desired output sizes for each layer
            target_sizes = [262144, 4096, 1024, 256, 128, 64, 32]  # TODO: this needs to be caclulated automaticaly somehow

            # Create downscaled versions of waveforms using interpolation
            channels_list = []
            for size in target_sizes:
                if feature_width == size:
                    # No need to resize if the current size matches the target
                    channels_list.append(mixture)
                else:
                    # Resize waveform to the target size
                    resized_mixture = F.interpolate(mixture, size=(size,), mode='linear', align_corners=False)
                    channels_list.append(resized_mixture)
                    # feature_width = size  # Update current length for the next iteration


            # Adjust channel dimensions to match `context_channels`
            # This involves expanding the channel dimension after downsampling
            channels_list = [
                torch.cat([channels_list[i]] * num, dim=1) if num != channels_list[i].shape[1]
                else channels_list[i]
                for i, num in enumerate([1, 512, 1024, 1024, 1024, 1024, 1024])
            ]

            # embedding = torch.randn(2, 4, 32).to(self.device)
            embedding = None

            
        elif isinstance(batch, (list, tuple)) and self.class_cond:
            waveforms, class_indexes, _ = batch
            channels_list = None
            embedding = None
        elif isinstance(batch, (list, tuple)) :
            waveforms, _, _= batch
            class_indexes = None
            channels_list = None  
            embedding = None          
        else:
            waveforms = batch
            class_indexes = None
            channels_list = None
            embedding = None
        return waveforms, class_indexes, channels_list, embedding

    def training_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
        mixtures = batch[-1].sum(1)
        predictions = self.model(mixtures, features = class_indexes, channels_list=channels_list, embedding = embedding)

        # print("predictions Tensor:")
        # print("Min:", predictions.min().item())
        # print("Max:", predictions.max().item())
        # print("Mean:", predictions.mean().item())
        # print("Std:", predictions.std().item())

        # # Print statistics for `target`
        # print("Target Tensor:")
        # print("Min:", waveforms.min().item())
        # print("Max:", waveforms.max().item())
        # print("Mean:", waveforms.mean().item())
        # print("Std:", waveforms.std().item())
        # print("\n\n\n")


        # Compute weighted loss
        losses = F.mse_loss(predictions, waveforms, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        loss = losses.mean()

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
        mixtures = batch[-1].sum(1)
        predictions = self.model(mixtures, features = class_indexes, channels_list=channels_list, embedding = embedding)

        # Compute weighted loss
        losses = F.mse_loss(predictions, waveforms, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        loss = losses.mean()
        self.log("valid_loss", loss, sync_dist=True)
        return loss
