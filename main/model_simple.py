
import torch
from einops import rearrange, reduce
import numpy as np
from torch import Tensor, nn
from audio_diffusion_pytorch_.modules import UNet1d
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model1d_simple(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        # check here if MSST and call model accordingly else UNet.
        model_type = kwargs.pop('model_type', None)
        self.model_type = model_type

        if model_type == 'bs_roformer':
            from main.msst.models.bs_roformer import BSRoformer
            self.unet = BSRoformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        elif model_type == 'mel_band_roformer':
            from main.msst.models.bs_roformer import MelBandRoformer
            self.unet = MelBandRoformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        elif model_type == 'bs_conformer':
            from main.msst.models.bs_roformer import BSConformer
            self.unet = BSConformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        elif model_type == 'mel_band_conformer':
            from main.msst.models.bs_roformer import MelBandConformer
            self.unet = MelBandConformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        elif model_type == 'bs_roformer_experimental':
            from main.msst.models.bs_roformer.bs_roformer_experimental import BSRoformer
            self.unet = BSRoformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        elif model_type == 'mel_band_roformer_experimental':
            from main.msst.models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
            self.unet = MelBandRoformer(**kwargs)
            print(f"Loaded MSST model: {model_type}")
        else:
            # Default: UNet1d (backward compatible)
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
            waveforms, class_indexes, stems  = batch[0], batch[1], batch[2]

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
            waveforms, class_indexes, _ = batch[0], batch[1], batch[2]
            channels_list = None
            embedding = None
        elif isinstance(batch, (list, tuple)) :
            waveforms, _, _= batch[0], batch[1], batch[2]
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
        mixtures = batch[2].sum(1)
        predictions = self.model(mixtures, features = class_indexes, channels_list=channels_list, embedding = embedding)

        # Compute weighted loss
        losses = F.mse_loss(predictions, waveforms, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        loss = losses.mean()
        self.log("valid_loss", loss, sync_dist=True)
        return loss


class Audio_MSST_Model_simple(pl.LightningModule):
    def __init__(
        self, learning_rate: float, patience: int = 3, reduce_factor: float = 0.95,
        load_pretrained: str = None, optimizer: str = 'adam', optimizer_params: dict = None,
        use_scheduler: bool = False, scheduler_monitor: str = 'valid_loss', scheduler_mode: str = 'max',
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.optimizer_type = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.use_scheduler = use_scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_mode = scheduler_mode

        self.model = Model1d_simple(**kwargs)

        # Load pretrained weights if provided (MSST checkpoint format)
        if load_pretrained:
            print(f"Loading pretrained weights from: {load_pretrained}")
            checkpoint = torch.load(load_pretrained, map_location='cpu', weights_only=False)

            # MSST checkpoints store weights under 'model_state_dict'
            self.model.unet.load_state_dict(checkpoint, strict=True)
            print(f"Successfully loaded pretrained weights from epoch {checkpoint.get('epoch', 'unknown')}")


    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        print(f'Optimizer type: {self.optimizer_type}')
        if self.optimizer_params:
            print(f'Optimizer params from config: {self.optimizer_params}')

        if self.optimizer_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'radam':
            optimizer = RAdam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'rmsprop':
            optimizer = RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'prodigy':
            from prodigyopt import Prodigy
            # you can choose weight decay value based on your problem, 0 by default
            # We recommend using lr=1.0 (default) for all networks.
            optimizer = Prodigy(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'adamw8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_type == 'sgd':
            print('Use SGD optimizer')
            optimizer = SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        else:
            print(f'Unknown optimizer: {self.optimizer_type}')
            raise ValueError(f'Unknown optimizer: {self.optimizer_type}')

        # Reduce LR if no improvements for several epochs
        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(
                optimizer,
                self.scheduler_mode,
                patience=self.patience,
                factor=self.reduce_factor,
                verbose=True
            )
            print(f'Using ReduceLROnPlateau scheduler: mode={self.scheduler_mode}, monitor={self.scheduler_monitor}, patience={self.patience}, factor={self.reduce_factor}')

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.scheduler_monitor,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        return optimizer

    def get_input(self, batch):
        waveforms, mixtures = batch[0], batch[1]
        return waveforms, mixtures

    def training_step(self, batch, batch_idx):
        ### there and validation step will change but later
        waveforms, mixtures = self.get_input(batch)
        loss = self.model(mixtures, target = waveforms)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        waveforms, mixtures = self.get_input(batch)
        loss = self.model(mixtures, target = waveforms)
        self.log("valid_loss", loss, sync_dist=True)
        return loss