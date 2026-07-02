from typing import *
import numpy as np

import pytorch_lightning as pl
import torch
import wandb
from audio_diffusion_pytorch_ import AudioDiffusionModel, AudioDiffusionConditional, Sampler, Schedule
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger #LoggerCollection, WandbLogger
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
import os
import torchaudio
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import shutil
# from audioldm_eval import EvaluationHelper  # unused: only referenced by SampleLogger.on_validation_epoch_end below, which ClassCondSeparateTrackSampleLogger overrides
from pathlib import Path
import json
import math
from main.model_simple import Audio_DM_Model_simple

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = AudioDiffusionModel(*args, **kwargs)

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

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss = self.model(waveforms)
        self.log("valid_loss", loss)
        return loss



class Audio_DM_Model(pl.LightningModule):
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
        self.mixture_features_channels = kwargs.pop('mixture_features_channels', None)
        self.pre_trained_mixture_feature_extractor = kwargs.pop('pre_trained_mixture_feature_extractor', None)
        # self.mixture_features_channels = mixture_features_channels

        if self.pre_trained_mixture_feature_extractor is not None:
            # Create a copy of kwargs
            simple_model_kwargs = kwargs.copy()

            # Remove items that shouldn't be passed to Audio_DM_Model_simple
            simple_model_kwargs.pop('diffusion_sigma_data', None)
            simple_model_kwargs.pop('diffusion_dynamic_threshold', None)
            simple_model_kwargs.pop('diffusion_sigma_distribution', None)
            # Add any additional arguments required by Audio_DM_Model_simple
            simple_model_kwargs['use_context_time'] = False
            
            # creating models for feature extraction
            self.pre_trained_mixture_feature_extractor_model = Audio_DM_Model_simple(learning_rate = learning_rate,
                                                                                    beta1 = beta1,
                                                                                    beta2 = beta2,
                                                                                    class_cond = class_cond, 
                                                                                    separation =  False,
                                                                                    **simple_model_kwargs
                                                                                    )
            # loading pre_trained models from checkpoint
            print("\nloading pre_trained model for feature extraction from checkpoint:", self.pre_trained_mixture_feature_extractor)
            self.pre_trained_mixture_feature_extractor_model.load_state_dict(torch.load(self.pre_trained_mixture_feature_extractor, map_location="cpu")["state_dict"])

            # Freeze parameters and set to eval mode
            for param in self.pre_trained_mixture_feature_extractor_model.parameters():
                param.requires_grad = False
            self.pre_trained_mixture_feature_extractor_model.eval()

        # if self.class_cond:
        #     self.model = AudioDiffusionConditional(*args, **kwargs)
        # else:
        self.model = AudioDiffusionModel(*args, **kwargs)

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

    def get_input(self, batch, current_class_indexes = None):

        if isinstance(batch, (list, tuple)) and self.class_cond and self.separation and self.pre_trained_mixture_feature_extractor is not None:
            waveforms, class_indexes, stems  = batch[0], batch[1], batch[2]

            batch_size, channels, feature_width = waveforms.shape
            mixture = stems.sum(1)
 
            # extract features form pre trained model
            with torch.no_grad():
                waveforms, class_indexes, channels_list, embedding = self.pre_trained_mixture_feature_extractor_model.get_input(batch)

                # Makeing sure this works well for sapler fustion where we mannually pass index of audio we wan to generate
                if current_class_indexes is not None:
                    class_indexes = current_class_indexes

                mixture_features_channels_list = self.pre_trained_mixture_feature_extractor_model.model.unet.get_feature(mixture, features = class_indexes, channels_list=channels_list, embedding = embedding)

            # Modify mixture_features_channels_list: add mixture in the beginign and remove last member
            mixture_features_channels_list = [mixture] + mixture_features_channels_list #[:-1]

            # embedding = torch.randn(2, 4, 32).to(self.device)
            channels_list = None
            embedding = None

        elif isinstance(batch, (list, tuple)) and self.class_cond and self.separation and self.mixture_features_channels:
            waveforms, class_indexes, stems  =  batch[0], batch[1], batch[2]

            batch_size, channels, feature_width = waveforms.shape
            mixture = stems.sum(1)

            # Desired output sizes for each layer
            target_sizes = [262144, 16384, 4096, 1024, 256, 128, 64, 32, 32, 64, 128, 256, 1024, 4096, 16384]  # TODO: this needs to be caclulated automaticaly somehow

            # Create downscaled versions of waveforms using interpolation
            mixture_features_channels_list = []
            for size in target_sizes:
                if feature_width == size:
                    # No need to resize if the current size matches the target
                    mixture_features_channels_list.append(mixture)
                else:
                    # Resize waveform to the target size
                    resized_mixture = F.interpolate(mixture, size=(size,), mode='linear', align_corners=False)
                    mixture_features_channels_list.append(resized_mixture)
                    # feature_width = size  # Update current length for the next iteration


            # Adjust channel dimensions to match `context_channels`
            # This involves expanding the channel dimension after downsampling
            mixture_features_channels_list = [
                torch.cat([mixture_features_channels_list[i]] * num, dim=1) if num != mixture_features_channels_list[i].shape[1]
                else mixture_features_channels_list[i]
                for i, num in enumerate(self.mixture_features_channels)
            ]

            # embedding = torch.randn(2, 4, 32).to(self.device)
            channels_list = None
            embedding = None



        elif isinstance(batch, (list, tuple)) and self.class_cond and self.separation:
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
            mixture_features_channels_list = None

            
        elif isinstance(batch, (list, tuple)) and self.class_cond:
            waveforms, class_indexes, _ = batch[0], batch[1], batch[2]
            channels_list = None
            embedding = None
            mixture_features_channels_list = None
        elif isinstance(batch, (list, tuple)) :
            waveforms, _, _= batch[0], batch[1], batch[2]
            class_indexes = None
            channels_list = None  
            embedding = None   
            mixture_features_channels_list = None       
        else:
            waveforms = batch
            class_indexes = None
            channels_list = None
            embedding = None
            mixture_features_channels_list = None
        return waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list

    def training_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = self.get_input(batch)
        loss = self.model(waveforms, features = class_indexes, channels_list=channels_list, embedding = embedding, mixture_features_channels_list = mixture_features_channels_list)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = self.get_input(batch)
        loss = self.model(waveforms, features = class_indexes, channels_list=channels_list, embedding = embedding, mixture_features_channels_list= mixture_features_channels_list)
        self.log("valid_loss", loss, sync_dist=True)
        return loss




""" Datamodule """

class DatamoduleWithValidation(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        *,
        batch_size: int,
        val_batch_size: int = None,  # Optional validation batch size
        num_workers: int,
        pin_memory: bool = False,
        # train_dataset_2 = None,  # Optional second dataset for alternating batches
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size  # Default to batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if isinstance(trainer.logger, LoggerCollection):
    #     for logger in trainer.logger:
    #         if isinstance(logger, WandbLogger):
    #             return logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: Union[List[int], int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_sampler = diffusion_sampler

        self.log_next = False
        
        
    #     # Check if diffusion_sampler is an instance of KarrasDenoiser
    #     if isinstance(self.diffusion_sampler, KarrasDenoiser):
    #         print("diffusion_sampler is an instance of KarrasDenoiser")
    #         schedule_sampler = create_named_schedule_sampler(self.diffusion_sampler.args, self.diffusion_sampler.args.schedule_sampler, self.diffusion_sampler.args.start_scales)
    #         diffusion_schedule_sampler = create_named_schedule_sampler(self.diffusion_sampler.args, self.diffusion_sampler.args.diffusion_schedule_sampler, self.diffusion_sampler.args.start_scales)
            
    #         self.diffusion_sampler.schedule_sampler = schedule_sampler
    #         self.diffusion_sampler.diffusion_schedule_sampler = diffusion_schedule_sampler
    #         self.scenario = "KarrasDenoiser"
    #     else:
    #         print("diffusion_sampler is NOT an instance of KarrasDenoiser")
    #         self.scenario = "ADPM2Sampler"

    # def sample_wrapper(self, model, noise, step, model_kwargs={}, sampler="heun"):
    #     if self.scenario == 'KarrasDenoiser':
    #         sample = karras_sample(
    #             diffusion=self.diffusion_sampler,
    #             model=model,
    #             shape=(noise.shape[0], self.channels, self.length),
    #             steps=step,
    #             model_kwargs=model_kwargs,  # in case of classes class goes here
    #             device=noise.device,
    #             clip_denoised=True,
    #             sampler=sampler,
    #             generator=None,
    #             teacher=False,
    #             ctm=True,
    #             x_T=noise,
    #             clip_output=True,
    #             sigma_min=self.diffusion_sampler.args.sigma_min,
    #             sigma_max=self.diffusion_sampler.args.sigma_max,
    #             train=False,
    #         )
    #     elif self.scenario == 'ADPM2Sampler':
    #         sample = model.sample(
    #             noise=noise,
    #             sampler=self.diffusion_sampler,
    #             sigma_schedule=self.diffusion_schedule,
    #             num_steps=step,
    #         )
    #     else:
    #         raise ValueError(f"Unknown scenario: {self.scenario}")
    #     return sample

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

        if batch_idx % 5 == 0 or trainer.state.fn == 'validate':            
            self.save_sample(trainer, pl_module, batch, batch_idx)

    def save_sample(self, trainer, pl_module, batch, batch_idx):
        current_epoch = trainer.current_epoch
        
        new_sampling_rate = 16000 # because FAD is calculated of 16000
        
        # Create base directory path
        base_dir = os.path.dirname(pl_module._trainer.checkpoint_callback.dirpath)
        resampler = torchaudio.transforms.Resample(self.sampling_rate, new_sampling_rate)
        
        # doing this for sweep to work
        if type(self.sampling_steps) == int:
            sampling_steps = [self.sampling_steps]
        else:
            sampling_steps = self.sampling_steps


        # Generate model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.diffusion_sampler, sampling_steps, "net", batch)

        # Get GPU identifier
        gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        for step_idx, step in enumerate(sampling_steps):
            step_dir = os.path.join(base_dir, f'audios_{current_epoch}_step{step}')
            generated_dir = os.path.join(step_dir, 'generated')
            original_dir = os.path.join(step_dir, 'original')

            # Create directories if they don't exist
            os.makedirs(generated_dir, exist_ok=True)
            os.makedirs(original_dir, exist_ok=True)
            
            for idx in range(batch[0].size(0)):
                audio = new_audios_to_log[step_idx][idx]  # Ensure tensor is on CPU
                original_audio = batch[0][idx].cpu()  # Ensure tensor is on CPU

                # Resample audio
                resampled_audio = resampler(torch.tensor(audio).permute(1,0))
                resampled_original_audio = resampler(original_audio.detach())

                # Define file names
                # generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # Define file names with GPU identifier
                generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                # Save audio files
                torchaudio.save(generated_file_name, resampled_audio, 16000)
                torchaudio.save(original_file_name, resampled_original_audio, 16000)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        base_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
        wandb_logger = get_wandb_logger(trainer).experiment
        evaluator = EvaluationHelper(sampling_rate=16000, device=pl_module.device)

        sampling_steps = self.sampling_steps if isinstance(self.sampling_steps, list) else [self.sampling_steps]

        for step in sampling_steps:
            step_dir = os.path.join(base_dir, f'audios_{current_epoch}_step{step}')
            if os.path.exists(step_dir):
                dir1, dir2 = Path(os.path.join(step_dir, "generated")), Path(os.path.join(step_dir, "original"))
                print("\nNow evaluating:", step_dir)
                metrics = evaluator.main(str(dir1), str(dir2))
                metrics_buffer = {f"step_{step}/{k}" if isinstance(self.sampling_steps, list) else k: float(v) for k, v in metrics.items()}

                if metrics_buffer:
                    for k, v in metrics_buffer.items():
                        wandb_logger.log({k: v}, commit=False)
                        print(k, v)
                    wandb_logger.log({}, commit=True)
                shutil.rmtree(dir1)
                shutil.rmtree(dir2)      
    
    def generate_model_output(self, model, sampler, steps, prefix, batch):

        audios_to_log = []
        captions = []
        
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = model.get_input(batch)
        
        model_kwargs = {}
        # if self.cfg.model.class_cond:
        model_kwargs["features"] = class_indexes
        model_kwargs["channels_list"] = channels_list
        model_kwargs["embedding"] = embedding
        model_kwargs["mixture_features_channels_list"] = mixture_features_channels_list
        
        batch_size = batch[0].size(0)

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=step, num_samples=1, batch_size=batch_size, ctm= False, class_idx = None, **model_kwargs)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix}_{step}_Steps"

            audios_to_log.append(xh.permute(0, 2, 1).cpu().numpy())
            captions.append(caption)

        return audios_to_log, captions

    @torch.no_grad()
    def sampling(self, model, sampler = 'exact', ctm=None, teacher=False, prefix="", step=-1, num_samples=-1, batch_size=-1, resize=False, generator=None, class_idx = None, **model_kwargs):
        # if not teacher:
        #     model.eval()
        if step == -1:
            step = 1
        if batch_size == -1:
            batch_size = model.cfg.datamodule.batch_size

        all_images = []
        number = 0
        
        # Dynamically select model based on prefix using getattr
        model_to_use = model.model

        while num_samples > number:
            # model_kwargs = {}
            is_train = model.training
            if is_train:
                model.eval()

            # Get start diffusion noise
            noise = torch.randn(
                (batch_size, self.channels, self.length), device=model.device
            )

            # samples = self.sample_wrapper(model, noise, step)

            samples = model_to_use.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=step,
                **model_kwargs
            )
            sample = samples #rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            if is_train:
                model.train()

            gathered_samples = sample.contiguous()
            all_images += [sample.cpu() for sample in gathered_samples]
            
            number += int(gathered_samples.shape[0])
        # if not teacher:
        #     model.train()

        arr = torch.stack(all_images, axis=0)

        return arr


    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )
        
        # doing this for sweep to work
        if type(self.sampling_steps) == int:
            sampling_steps = [self.sampling_steps]
        else:
            sampling_steps = self.sampling_steps

        for steps in sampling_steps:
            samples = model.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            wandb_logger.log(
                {
                    f"sample_{idx}_{steps}": wandb.Audio(
                        samples[idx],
                        caption=f"Sampled in {steps} steps",
                        sample_rate=self.sampling_rate,
                    )
                    for idx in range(self.num_items)
                }
            )

        if is_train:
            pl_module.train()


class MultiSourceSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str]
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )

        for steps in self.sampling_steps:
            samples = model.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            for i in range(samples.shape[-1]):
                wandb_logger.log(
                 {
                     f"sample_{self.stems[i]}_{idx}_{steps}": wandb.Audio(
                            samples[idx, :, i][..., np.newaxis],
                            caption=f"Sampled in {steps} steps",
                            sample_rate=self.sampling_rate,
                     ) for idx in range(self.num_items)
                    }
            )
            # log mixture
            wandb_logger.log(
                {
                    f"sample_mix_{idx}_{steps}": wandb.Audio(
                        samples[idx, :, :].sum(axis=-1, keepdims=True),
                        caption=f"Sampled in {steps} steps",
                        sample_rate=self.sampling_rate,
                    ) for idx in range(self.num_items)
                })
        if is_train:
            pl_module.train()



class ClassCondTrackSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str]
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )

        # Iterate over each diffusion step size
        for steps in self.sampling_steps:
            # Iterate over each one-hot encoded feature vector (each stem)
            for i, stem in enumerate(self.stems):
                # Create a feature tensor for the current stem for all items
                current_features = torch.zeros(self.num_items, len(self.stems)).to(pl_module.device)
                current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

                # Sample from the model using the noise and the current one-hot features
                samples = model.sample(
                    noise=noise,
                    features=current_features,
                    sampler=self.diffusion_sampler,
                    sigma_schedule=self.diffusion_schedule,
                    num_steps=steps,
                )
                samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

                # Log each sample for the current stem
                for idx in range(self.num_items):
                    audio_data = samples[idx, :, 0][..., np.newaxis]  # Reshape for mono audio
                    wandb_logger.log({
                        f"sample_{stem}_{idx}_{steps}": wandb.Audio(
                            audio_data,
                            caption=f"Sampled in {steps} steps",
                            sample_rate=self.sampling_rate
                        )
                    })



        # for steps in self.sampling_steps:
        #     samples = model.sample(
        #         noise=noise,
        #         sampler=self.diffusion_sampler,
        #         sigma_schedule=self.diffusion_schedule,
        #         num_steps=steps,
        #     )
        #     samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            # for i in range(samples.shape[-1]):
            #     wandb_logger.log(
            #      {
            #          f"sample_{self.stems[i]}_{idx}_{steps}": wandb.Audio(
            #                 samples[idx, :, i][..., np.newaxis],
            #                 caption=f"Sampled in {steps} steps",
            #                 sample_rate=self.sampling_rate,
            #          ) for idx in range(self.num_items)
            #         }
            # )
            # # log mixture
        #     wandb_logger.log(
        #         {
        #             f"sample_mix_{idx}_{steps}": wandb.Audio(
        #                 samples[idx, :, :].sum(axis=-1, keepdims=True),
        #                 caption=f"Sampled in {steps} steps",
        #                 sample_rate=self.sampling_rate,
        #             ) for idx in range(self.num_items)
        #         })
        if is_train:
            pl_module.train()

def _bss_eval_silent_frames(refs_nTC, ests_nTC, win, silence_threshold=1.5e-5, eps=1e-8):
    """
    Compute eps-regularised SIR and SAR for frames where each stem's reference is silent.
    refs_nTC, ests_nTC: (nsrc, T, C)
    Returns: {stem_idx: {'SIR': [float, ...], 'SAR': [float, ...]}} — one value per silent frame.
    """
    import numpy as np
    nsrc, T, C = refs_nTC.shape
    out = {j: {'SIR': [], 'SAR': []} for j in range(nsrc)}

    for frame_idx in range(T // win):
        s, e = frame_idx * win, (frame_idx + 1) * win
        refs_ch = refs_nTC[:, s:e, :]   # (nsrc, win, C)
        ests_ch = ests_nTC[:, s:e, :]

        for j in range(nsrc):
            ref_j = refs_ch[j]
            if np.linalg.norm(ref_j) / ref_j.size >= silence_threshold:
                continue  # active frame — skip

            SIR_ch, SAR_ch = [], []
            for ch in range(C):
                refs = refs_ch[:, :, ch]   # (nsrc, win)
                ej   = ests_ch[j, :, ch]   # (win,)

                S   = refs.T               # (win, nsrc)
                c   = np.linalg.solve(S.T @ S + eps * np.eye(nsrc), S.T @ ej)
                e_proj = S @ c

                sj      = refs[j]
                c_tgt   = (ej @ sj) / (np.sum(sj ** 2) + eps)
                s_tgt   = c_tgt * sj
                e_interf = e_proj - s_tgt
                e_artif  = ej - e_proj

                SIR_ch.append(10 * np.log10((np.sum(s_tgt ** 2) + eps) / (np.sum(e_interf ** 2) + eps)))
                SAR_ch.append(10 * np.log10((np.sum(e_proj  ** 2) + eps) / (np.sum(e_artif  ** 2) + eps)))

            out[j]['SIR'].append(float(np.mean(SIR_ch)))
            out[j]['SAR'].append(float(np.mean(SAR_ch)))

    return out


def _evaluate_batch_file_museval(file_tuple):
    """Load a batch file, evaluate all samples, and delete files.
    Returns (results, silent_results, track_names, file_error).
    """
    import os
    import pickle
    # Ensure worker doesn't spawn extra threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    refs_file, ests_file, win, hop, do_museval = file_tuple

    results = []
    silent_results = []
    batch_track_names = None
    file_error = None

    try:
        # Load the batch file (refs may be a dict with audio + track_names)
        with open(refs_file, 'rb') as f:
            refs_payload = pickle.load(f)
        if isinstance(refs_payload, dict):
            batch_refs = refs_payload['audio']
            batch_track_names = refs_payload.get('track_names', None)
        else:
            batch_refs = refs_payload  # backward compat
        with open(ests_file, 'rb') as f:
            batch_ests = pickle.load(f)

        # Process all samples in this batch
        for idx in range(len(batch_refs)):
            if do_museval:
                import museval
                try:
                    SDR, ISR, SIR, SAR = museval.evaluate(
                        batch_refs[idx],
                        batch_ests[idx],
                        win=win,
                        hop=hop
                    )
                    results.append((SDR.tolist(), ISR.tolist(), SIR.tolist(), SAR.tolist(), None))
                except Exception as e:
                    results.append((None, None, None, None, str(e)))
            else:
                results.append(None)

            try:
                silent_results.append(_bss_eval_silent_frames(batch_refs[idx], batch_ests[idx], win=win))
            except Exception as e:
                silent_results.append(None)

    except Exception as e:
        file_error = str(e)

    # Worker deletes its own files after processing
    try:
        if os.path.exists(refs_file):
            os.remove(refs_file)
        if os.path.exists(ests_file):
            os.remove(ests_file)
    except Exception:
        pass

    return results, silent_results, batch_track_names, file_error


class ClassCondSeparateTrackSampleLogger(SampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        stems: List[str],
        log_all: bool = False,
        run_museval: bool = True,
        msdm_style: bool = False,
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            diffusion_sampler=diffusion_sampler,
        )
        self.stems = stems
        self.log_all = log_all
        self.run_museval = run_museval
        self.msdm_style = msdm_style

        self.torch_si_snr = ScaleInvariantSignalNoiseRatio()
        self.torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        
        self.metrics_log = {
            stem: {
                'si_snr': [], 'si_sdr': [], 'msdm_si_sdr': [], 'sdr': [], 'sdr_silent': [], 'output_rms_silent': [],
                'museval': {"SDR": [], "ISR": [], "SIR": [], "SAR": []}
            } for stem in stems
        }

        # minimal additions to support per-epoch museval
        self._museval_batch_counter = 0
        self._museval_temp_dir = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True
        # reset batch counter for museval incremental saving
        self._museval_batch_counter = 0

        # Get wandb directory for saving museval data incrementally
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                wandb_logger = get_wandb_logger(trainer).experiment
                wandb_run_dir = wandb_logger.dir
                # Create subdirectory for museval temp files
                self._museval_temp_dir = os.path.join(wandb_run_dir, 'museval_temp')
                os.makedirs(self._museval_temp_dir, exist_ok=True)
            else:
                self._museval_temp_dir = None
            # Broadcast museval temp directory from rank 0 to all ranks
            temp_dir_list = [self._museval_temp_dir]
            dist.broadcast_object_list(temp_dir_list, src=0)
            self._museval_temp_dir = temp_dir_list[0]
        else:
            # Single GPU
            wandb_logger = get_wandb_logger(trainer).experiment
            wandb_run_dir = wandb_logger.dir
            # Create subdirectory for museval temp files
            self._museval_temp_dir = os.path.join(wandb_run_dir, 'museval_temp')
            os.makedirs(self._museval_temp_dir, exist_ok=True)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        # keep your original behavior (no duplicate loop)
        self.log_sample(trainer, pl_module, batch, batch_idx)

    def log_sample(self, trainer, pl_module, batch, batch_idx):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        original_samples, generated_samples, mixture_audios = self.generate_sample(trainer, pl_module, batch)

        # Extract track names if the dataset provides them (3rd element, list of strings)
        track_names = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            candidate = batch[-1]
            if isinstance(candidate, (list, tuple)) and len(candidate) > 0 and isinstance(candidate[0], str):
                track_names = list(candidate)

        if batch_idx == 0 and trainer.is_global_zero:
            self.log_audio(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)

        # fast metrics + accumulation for museval (museval itself deferred to epoch end)
        self.update_metrics(original_samples, generated_samples, mixture_audios, wandb_logger, trainer,
                            track_names=track_names)

        if is_train:
            pl_module.train()
            
    @torch.no_grad()
    def generate_sample(self, trainer, pl_module, batch):
        model = pl_module.model
        
        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch)

        # Dictionary to store generated samples
        generated_samples = {stem: [] for stem in self.stems}
        
        # Iterate over each one-hot encoded feature vector (each stem)
        steps = self.sampling_steps
        for i, stem in enumerate(self.stems):
            # one-hot class features
            current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
            current_features[:, i] = 1

            # extract corresponding mixture features for current stem if available
            if pl_module.pre_trained_mixture_feature_extractor is not None:
                waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch, current_features)

            # start diffusion noise
            noise = torch.randn(
                (waveforms.size(0), self.channels, self.length), device=pl_module.device
            )

            noise = [noise, mixture_features_channels_list.pop()]

            # Sample from the model
            samples = model.sample(
                noise=noise,
                features=current_features,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
                channels_list=channels_list,
                mixture_features_channels_list=mixture_features_channels_list,
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            # Store the generated samples
            for idx in range(waveforms.size(0)):
                generated_samples[stem].append(samples[idx])

        # get original stems
        original_samples = {stem: {} for stem in self.stems}
        original_stems = batch[2]
        for i, stem in enumerate(self.stems):
            stem_data = original_stems[:, i]
            stem_data =rearrange(stem_data, "b c t -> b t c").detach().cpu().numpy()
            original_samples[stem] = []
            for idx in range(waveforms.size(0)):
                original_samples[stem].append(stem_data[idx])  
        
        mixture_audios = batch[2].sum(1)[:, 0, :].detach().cpu().numpy()[..., np.newaxis]
        return original_samples, generated_samples, mixture_audios

    @torch.no_grad()
    def log_audio(self, original_samples, generated_samples, mixture_audio, wandb_logger, trainer):
        n = len(mixture_audio) if self.log_all else self.num_items
        for idx in range(n):
            logging_data = {}
            logging_data[f"Mixture_audio"] = wandb.Audio(
                mixture_audio[idx], caption=f"Mixture Audio {idx}", sample_rate=self.sampling_rate
            )
            for stem in self.stems:
                original_audio = original_samples[stem][idx]
                logging_data[f"original_{stem}"] = wandb.Audio(
                    original_audio, caption=f"Original {stem} Audio {idx}", sample_rate=self.sampling_rate
                )
            for stem in self.stems:
                generated_audio = generated_samples[stem][idx]
                logging_data[f"generated_{stem}"] = wandb.Audio(
                    generated_audio,
                    caption=f"{stem} Sampled in {self.sampling_steps} steps (idx: {idx})",
                    sample_rate=self.sampling_rate
                )
            mix_audio = sum(generated_samples[stem][idx] for stem in self.stems)
            logging_data[f"generated_mix"] = wandb.Audio(
                mix_audio,
                caption=f"Sampled in {self.sampling_steps} steps (Mix) (idx: {idx})",
                sample_rate=self.sampling_rate
            )
            wandb_logger.log(logging_data)

    def sdr(self, preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        s_target = torch.norm(target, dim=-1)**2 + eps
        s_error = torch.norm(target - preds, dim=-1)**2 + eps
        return 10 * torch.log10(s_target/s_error)
            
    def sisdr(self, preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
        target_scaled = alpha * target
        noise = target_scaled - preds
        s_target = torch.sum(target_scaled**2, dim=-1) + eps
        s_error = torch.sum(noise**2, dim=-1) + eps
        return 10 * torch.log10(s_target / s_error)

    def sliding_window(self, tensor, window_size=1024, step_size=512):
        num_windows = (tensor.size(-1) - window_size) // step_size + 1
        windows = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            windows.append(tensor[..., start:end])
        return torch.stack(windows, dim=0)        

    def assert_is_audio(self, *signal: torch.Tensor):
        for s in signal:
            assert len(s.shape) == 2
            assert s.shape[0] == 1 or s.shape[0] == 2

    def is_silent(self, signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
        self.assert_is_audio(signal)
        num_samples = signal.shape[-1]
        return torch.linalg.norm(signal) / num_samples < silence_threshold

    @torch.no_grad()
    def update_metrics(self, original_samples, generated_samples, mixture_audios, wandb_logger, trainer,
                       chunk_duration=4.0, overlap_duration=2.0, eps=1e-8, track_names=None):
        # fast metrics - use the current device instead of root_device
        device = next(iter(trainer.model.parameters())).device  # Use model's device (each rank's own GPU)
        self.torch_si_snr = self.torch_si_snr.to(device)
        self.torch_si_sdr = self.torch_si_sdr.to(device)

        chunk_samples = int(chunk_duration * self.sampling_rate)
        overlap_samples = int(overlap_duration * self.sampling_rate)
        step_size = chunk_samples - overlap_samples
        num_eval_chunks = math.ceil((self.length - overlap_samples) / step_size)

        for idx in range(len(mixture_audios)):
            for j in range(num_eval_chunks):
                start_sample = j * step_size
                end_sample = start_sample + chunk_samples
                if end_sample > self.length:
                    end_sample = self.length

                num_active_signals = 0
                for stem in self.stems:
                    o = original_samples[stem][idx][start_sample:end_sample, :]
                    if not self.is_silent(torch.tensor(o).permute(1, 0)):
                        num_active_signals += 1
                if num_active_signals <= 1:
                    continue

                for stem in self.stems:
                    original_audio = original_samples[stem][idx][start_sample:end_sample, :]
                    generated_audio = generated_samples[stem][idx][start_sample:end_sample, :]
                    mixture_audio   = mixture_audios[idx, start_sample:end_sample, :]

                    o = torch.tensor(original_audio, device=device).permute(1, 0)
                    g = torch.tensor(generated_audio, device=device).permute(1, 0)
                    m = torch.tensor(mixture_audio,  device=device).permute(1, 0)

                    # sdr (eps-regularised) is well-defined even on silent frames
                    sdr = self.sdr(g, o).mean()
                    self.metrics_log[stem]['sdr'].append(sdr.item())

                    silent = self.is_silent(o)

                    if silent:
                        self.metrics_log[stem]['sdr_silent'].append(sdr.item())
                        rms = g.norm() / (g.numel() ** 0.5)
                        rms_db = 20.0 * torch.log10(rms + 1e-10).item()
                        self.metrics_log[stem]['output_rms_silent'].append(rms_db)

                    # Skip SI-SDR for this stem if its reference is silent —
                    # avoids unfair penalty on total silences and also prevents potential numerical issues
                    if not silent:
                        si_snr = self.torch_si_snr(g, o)
                        si_sdr = self.torch_si_sdr(g, o)
                        self.metrics_log[stem]['si_snr'].append(si_snr.item())
                        self.metrics_log[stem]['si_sdr'].append(si_sdr.item())

                    # msdm_si_sdr: msdm_style=True includes both silent and non-silent stems
                    # (matches the original MSDM evaluation pipeline convention). msdm_style=False
                    # (default) only counts non-silent stems.
                    if self.msdm_style or not silent:
                        msdm_si_sdr_s = self.sisdr(g, o).mean()
                        msdm_si_sdr_o = self.sisdr(m, o).mean()
                        msdm_si_sdr   = msdm_si_sdr_s - msdm_si_sdr_o
                        self.metrics_log[stem]['msdm_si_sdr'].append(msdm_si_sdr.item())

        # Move metrics back to CPU to save GPU memory
        self.torch_si_snr = self.torch_si_snr.to('cpu')
        self.torch_si_sdr = self.torch_si_sdr.to('cpu')

        # Save audio data to disk for museval / silent BSS processing at epoch end
        import pickle
        import torch.distributed as dist

        # Get rank for filename
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        take = len(mixture_audios)
        batch_refs = []
        batch_ests = []

        for idx in range(take):
            ref_stems, est_stems = [], []
            for stem in self.stems:
                ref = np.asarray(original_samples[stem][idx], dtype=np.float32)  # (T, C)
                est = np.asarray(generated_samples[stem][idx], dtype=np.float32)

                ref_stems.append(ref)
                est_stems.append(est)
            batch_refs.append(np.stack(ref_stems, axis=0))  # (nsrc, T, C)
            batch_ests.append(np.stack(est_stems, axis=0))  # (nsrc, T, C)

        # Write this batch's data to disk immediately
        batch_file_refs = os.path.join(self._museval_temp_dir, f'museval_refs_rank_{rank}_epoch_{trainer.current_epoch}_batch_{self._museval_batch_counter}.pkl')
        batch_file_ests = os.path.join(self._museval_temp_dir, f'museval_ests_rank_{rank}_epoch_{trainer.current_epoch}_batch_{self._museval_batch_counter}.pkl')

        # Include track names so epoch-end can do per-track median-of-medians
        batch_names = track_names if track_names is not None else [None] * len(batch_refs)
        refs_payload = {'audio': batch_refs, 'track_names': batch_names}
        with open(batch_file_refs, 'wb') as f:
            pickle.dump(refs_payload, f)
        with open(batch_file_ests, 'wb') as f:
            pickle.dump(batch_ests, f)

        self._museval_batch_counter += 1

        # Clear batch data from memory
        del batch_refs, batch_ests

    def on_validation_epoch_end(self, trainer, pl_module):
        log_dict = {}
        num_stems = len(self.stems)

        # --- GATHER METRICS FROM ALL GPUS ---
        # Gather metrics_log from all ranks
        from pytorch_lightning.utilities import rank_zero_only
        import torch.distributed as dist

        # Prepare data to gather: convert lists to tensors
        gathered_metrics = {stem: {'si_snr': [], 'si_sdr': [], 'msdm_si_sdr': [], 'sdr': [], 'sdr_silent': [], 'output_rms_silent': []} for stem in self.stems}

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

            # Gather fast metrics from all GPUs
            for stem in self.stems:
                for metric_name in ['si_snr', 'si_sdr', 'msdm_si_sdr', 'sdr', 'sdr_silent', 'output_rms_silent']:
                    # Convert local list to tensor
                    local_values = self.metrics_log[stem][metric_name]
                    if local_values:
                        local_tensor = torch.tensor(local_values, dtype=torch.float32, device=pl_module.device)
                    else:
                        local_tensor = torch.tensor([], dtype=torch.float32, device=pl_module.device)

                    # Gather sizes from all ranks
                    local_size = torch.tensor([local_tensor.shape[0]], device=pl_module.device)
                    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
                    dist.all_gather(size_list, local_size)

                    # Gather actual tensors with proper padding
                    max_size = max([s.item() for s in size_list])
                    if max_size > 0:
                        # Pad local tensor to max_size
                        if local_tensor.shape[0] < max_size:
                            padding = torch.zeros(max_size - local_tensor.shape[0], device=pl_module.device)
                            local_tensor = torch.cat([local_tensor, padding])

                        # Gather from all ranks
                        gathered_list = [torch.zeros(max_size, device=pl_module.device) for _ in range(world_size)]
                        dist.all_gather(gathered_list, local_tensor)

                        # Combine results, removing padding
                        for rank_idx, (gathered_tensor, size) in enumerate(zip(gathered_list, size_list)):
                            actual_size = size.item()
                            if actual_size > 0:
                                gathered_metrics[stem][metric_name].extend(gathered_tensor[:actual_size].cpu().tolist())
        else:
            # Single GPU: just use local metrics
            for stem in self.stems:
                for metric_name in ['si_snr', 'si_sdr', 'msdm_si_sdr', 'sdr', 'sdr_silent', 'output_rms_silent']:
                    gathered_metrics[stem][metric_name] = self.metrics_log[stem][metric_name]

        # fast metrics aggregates (using gathered data)
        total_msdm_si_sdr = 0.0
        if trainer.is_global_zero:
            for stem in self.stems:
                si_snr_vals = gathered_metrics[stem]['si_snr']
                si_sdr_vals = gathered_metrics[stem]['si_sdr']
                msdm_vals   = gathered_metrics[stem]['msdm_si_sdr']
                sdr_vals    = gathered_metrics[stem]['sdr']

                mean_si_snr = sum(si_snr_vals)/len(si_snr_vals) if si_snr_vals else float('nan')
                mean_si_sdr = sum(si_sdr_vals)/len(si_sdr_vals) if si_sdr_vals else float('nan')
                mean_msdm   = sum(msdm_vals)/len(msdm_vals)     if msdm_vals   else float('nan')
                sdr_silent_vals = gathered_metrics[stem]['sdr_silent']
                mean_sdr        = sum(sdr_vals)/len(sdr_vals)             if sdr_vals        else float('nan')
                median_sdr      = np.median(sdr_vals)                     if sdr_vals        else float('nan')
                mean_sdr_silent   = sum(sdr_silent_vals)/len(sdr_silent_vals) if sdr_silent_vals else float('nan')
                median_sdr_silent = np.median(sdr_silent_vals)                if sdr_silent_vals else float('nan')

                log_dict[f'si_snr/{stem}'] = mean_si_snr
                log_dict[f'si_sdr/{stem}'] = mean_si_sdr
                log_dict[f'msdm_si_sdr/{stem}'] = mean_msdm
                log_dict[f'sdr/{stem}'] = mean_sdr
                log_dict[f'sdr/{stem}/median'] = median_sdr
                log_dict[f'sdr_silent/{stem}'] = mean_sdr_silent
                log_dict[f'sdr_silent/{stem}/median'] = median_sdr_silent

                rms_silent_vals = gathered_metrics[stem]['output_rms_silent']
                mean_rms_silent   = sum(rms_silent_vals)/len(rms_silent_vals) if rms_silent_vals else float('nan')
                median_rms_silent = np.median(rms_silent_vals)                if rms_silent_vals else float('nan')
                log_dict[f'output_rms_silent/{stem}'] = mean_rms_silent
                log_dict[f'output_rms_silent/{stem}/median'] = median_rms_silent

                if not np.isnan(mean_msdm):
                    total_msdm_si_sdr += mean_msdm

        # --- PROCESS MUSEVAL DATA INCREMENTALLY FROM BATCH FILES ---
        # Free GPU memory before file operations
        torch.cuda.empty_cache()

        total_median_museval_sdr = 0.0
        total_per_track_museval_sdr_medians = 0.0
        ran_museval = False

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            # Synchronize - ensure all ranks have finished writing batch files
            dist.barrier()

        # Only rank 0 processes batch files
        if trainer.is_global_zero:
            import museval
            from tqdm import tqdm
            import multiprocessing as mp
            import pickle
            import glob

            win = hop = int(self.sampling_rate)  # 1s frames

            # Keyed by track name so we can compute per-track median → median-of-medians
            per_stem_SDR = {s: {} for s in self.stems}
            per_stem_ISR = {s: {} for s in self.stems}
            per_stem_SIR = {s: {} for s in self.stems}
            per_stem_SAR = {s: {} for s in self.stems}
            per_stem_SIR_silent = {s: [] for s in self.stems}
            per_stem_SAR_silent = {s: [] for s in self.stems}

            # Get all batch files from all ranks
            temp_dir = self._museval_temp_dir
            all_batch_files = []

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

            for r in range(world_size):
                refs_pattern = os.path.join(temp_dir, f'museval_refs_rank_{r}_epoch_{trainer.current_epoch}_batch_*.pkl')
                ests_pattern = os.path.join(temp_dir, f'museval_ests_rank_{r}_epoch_{trainer.current_epoch}_batch_*.pkl')

                refs_files = sorted(glob.glob(refs_pattern))
                ests_files = sorted(glob.glob(ests_pattern))

                all_batch_files.extend(zip(refs_files, ests_files))

            total_samples = 0

            # Process batch files with parallel workers - workers load files themselves!
            num_workers = min(12, max(1, mp.cpu_count() // 4))

            # Set environment to avoid OpenMP conflicts
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

            print(f"\nProcessing museval from {len(all_batch_files)} batch files using {num_workers} parallel workers...")

            try:
                # Use 'spawn' context to avoid inheriting CUDA state
                ctx = mp.get_context('spawn')

                # Prepare tasks: each task is (refs_file, ests_file, win, hop)
                batch_tasks = [(refs_file, ests_file, win, hop, self.run_museval) for refs_file, ests_file in all_batch_files]

                with ctx.Pool(processes=num_workers) as pool:
                    # Workers load, process, and delete batch files in parallel!
                    # Process results as they stream in (no accumulation)
                    for batch_results, batch_silent, batch_track_names, file_error in tqdm(pool.imap_unordered(_evaluate_batch_file_museval, batch_tasks),
                                                         total=len(batch_tasks),
                                                         desc="Processing museval batches"):
                        if file_error:
                            print(f"\nError loading/processing batch file: {file_error}")
                            continue

                        # Accumulate results from this batch, keyed by track name
                        for i, museval_result in enumerate(batch_results):
                            tn = (batch_track_names[i]
                                  if (batch_track_names and i < len(batch_track_names) and batch_track_names[i] is not None)
                                  else f"__item_{total_samples}")
                            if self.run_museval and museval_result is not None:
                                SDR, ISR, SIR, SAR, sample_error = museval_result
                                if sample_error:
                                    print(f"\nError during museval evaluation: {sample_error}")
                                elif SDR is not None:
                                    for s_idx, stem in enumerate(self.stems):
                                        per_stem_SDR[stem].setdefault(tn, []).extend(SDR[s_idx])
                                        per_stem_ISR[stem].setdefault(tn, []).extend(ISR[s_idx])
                                        per_stem_SIR[stem].setdefault(tn, []).extend(SIR[s_idx])
                                        per_stem_SAR[stem].setdefault(tn, []).extend(SAR[s_idx])

                            if batch_silent and i < len(batch_silent) and batch_silent[i] is not None:
                                for s_idx, stem in enumerate(self.stems):
                                    per_stem_SIR_silent[stem].extend(batch_silent[i][s_idx]['SIR'])
                                    per_stem_SAR_silent[stem].extend(batch_silent[i][s_idx]['SAR'])

                            total_samples += 1

                print(f"Parallel museval completed for {total_samples} samples!")

            except Exception as e:
                print(f"\nParallel museval failed ({e}), falling back to sequential processing...")

                # Sequential fallback - process batch files one at a time
                for refs_file, ests_file in tqdm(all_batch_files, desc="Processing museval batches (sequential)"):
                    try:
                        # Load one batch at a time
                        with open(refs_file, 'rb') as f:
                            refs_payload = pickle.load(f)
                        if isinstance(refs_payload, dict):
                            batch_refs = refs_payload['audio']
                            seq_track_names = refs_payload.get('track_names', None)
                        else:
                            batch_refs = refs_payload
                            seq_track_names = None
                        with open(ests_file, 'rb') as f:
                            batch_ests = pickle.load(f)

                        # Process this batch sequentially
                        for idx in range(len(batch_refs)):
                            tn = (seq_track_names[idx]
                                  if (seq_track_names and idx < len(seq_track_names) and seq_track_names[idx] is not None)
                                  else f"__item_{total_samples}")
                            if self.run_museval:
                                try:
                                    SDR, ISR, SIR, SAR = museval.evaluate(
                                        batch_refs[idx],
                                        batch_ests[idx],
                                        win=win,
                                        hop=hop
                                    )
                                    for s_idx, stem in enumerate(self.stems):
                                        per_stem_SDR[stem].setdefault(tn, []).extend(SDR[s_idx].tolist())
                                        per_stem_ISR[stem].setdefault(tn, []).extend(ISR[s_idx].tolist())
                                        per_stem_SIR[stem].setdefault(tn, []).extend(SIR[s_idx].tolist())
                                        per_stem_SAR[stem].setdefault(tn, []).extend(SAR[s_idx].tolist())
                                except Exception as e:
                                    print(f"\nError during museval evaluation: {e}")

                            try:
                                sr = _bss_eval_silent_frames(batch_refs[idx], batch_ests[idx], win=win)
                                for s_idx, stem in enumerate(self.stems):
                                    per_stem_SIR_silent[stem].extend(sr[s_idx]['SIR'])
                                    per_stem_SAR_silent[stem].extend(sr[s_idx]['SAR'])
                            except Exception:
                                pass

                            total_samples += 1

                        # Clear batch data from memory
                        del batch_refs, batch_ests

                        # Delete batch files after processing
                        os.remove(refs_file)
                        os.remove(ests_file)

                    except Exception as e:
                        print(f"\nWarning: Failed to process batch file {refs_file}: {e}")
                        try:
                            if os.path.exists(refs_file):
                                os.remove(refs_file)
                            if os.path.exists(ests_file):
                                os.remove(ests_file)
                        except:
                            pass

                print(f"Sequential museval completed for {total_samples} samples!")
                    
            def filt(x):
                x = np.asarray(x, dtype=np.float32)
                return x[np.isfinite(x)]

            def track_medians(track_dict):
                """Per-track median of frame-level values → list of one float per track."""
                medians = []
                for frames in track_dict.values():
                    f = filt(frames)
                    if f.size:
                        medians.append(float(np.median(f)))
                return medians

            def flat_frames(track_dict):
                """All frame-level values flattened across tracks."""
                return filt([v for frames in track_dict.values() for v in frames])

            if self.run_museval:
                for stem in self.stems:
                    sdr_f = flat_frames(per_stem_SDR[stem])
                    isr_f = flat_frames(per_stem_ISR[stem])
                    sir_f = flat_frames(per_stem_SIR[stem])
                    sar_f = flat_frames(per_stem_SAR[stem])

                    sdr_m = track_medians(per_stem_SDR[stem])
                    isr_m = track_medians(per_stem_ISR[stem])
                    sir_m = track_medians(per_stem_SIR[stem])
                    sar_m = track_medians(per_stem_SAR[stem])

                    if sdr_f.size:
                        log_dict[f'museval_sdr/{stem}/mean'] = float(np.mean(sdr_f))
                        log_dict[f'museval_sdr/{stem}/median'] = float(np.median(sdr_f))
                        total_median_museval_sdr += float(np.median(sdr_f))
                    if sdr_m:
                        log_dict[f'museval_sdr/{stem}/per_track_mean'] = float(np.mean(sdr_m))
                        log_dict[f'museval_sdr/{stem}/per_track_median'] = float(np.median(sdr_m))
                        log_dict[f'museval_sdr/{stem}/num_tracks'] = len(sdr_m)
                        total_per_track_museval_sdr_medians += float(np.median(sdr_m))

                    if isr_f.size:
                        log_dict[f'museval_isr/{stem}/mean'] = float(np.mean(isr_f))
                        log_dict[f'museval_isr/{stem}/median'] = float(np.median(isr_f))
                    if isr_m:
                        log_dict[f'museval_isr/{stem}/per_track_mean'] = float(np.mean(isr_m))
                        log_dict[f'museval_isr/{stem}/per_track_median'] = float(np.median(isr_m))

                    if sir_f.size:
                        log_dict[f'museval_sir/{stem}/mean'] = float(np.mean(sir_f))
                        log_dict[f'museval_sir/{stem}/median'] = float(np.median(sir_f))
                    if sir_m:
                        log_dict[f'museval_sir/{stem}/per_track_mean'] = float(np.mean(sir_m))
                        log_dict[f'museval_sir/{stem}/per_track_median'] = float(np.median(sir_m))

                    if sar_f.size:
                        log_dict[f'museval_sar/{stem}/mean'] = float(np.mean(sar_f))
                        log_dict[f'museval_sar/{stem}/median'] = float(np.median(sar_f))
                    if sar_m:
                        log_dict[f'museval_sar/{stem}/per_track_mean'] = float(np.mean(sar_m))
                        log_dict[f'museval_sar/{stem}/per_track_median'] = float(np.median(sar_m))

                ran_museval = True

            for stem in self.stems:
                sir_s = filt(per_stem_SIR_silent[stem])
                sar_s = filt(per_stem_SAR_silent[stem])
                if sir_s.size:
                    log_dict[f'sir_silent/{stem}/mean']   = float(np.mean(sir_s))
                    log_dict[f'sir_silent/{stem}/median'] = float(np.median(sir_s))
                if sar_s.size:
                    log_dict[f'sar_silent/{stem}/mean']   = float(np.mean(sar_s))
                    log_dict[f'sar_silent/{stem}/median'] = float(np.median(sar_s))

        # Synchronize all ranks - ensure GPU 0 finishes museval before others proceed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # averaged summaries (computed on rank 0, will be broadcasted later)
        if trainer.is_global_zero:
            mean_msdm_si_sdr_avg = total_msdm_si_sdr / num_stems if num_stems else float('nan')
            # Don't add to log_dict here - will be logged via broadcast below

            if ran_museval and num_stems:
                sdr_avg_value = total_median_museval_sdr / num_stems
                sdr_avg_per_track_medians_value = total_per_track_museval_sdr_medians / num_stems
            else:
                sdr_avg_per_track_medians_value = float('nan')
                # fallback: median of fast SDR per stem
                s = 0.0; c = 0
                for stem in self.stems:
                    vals = self.metrics_log[stem]['sdr']
                    if vals:
                        s += float(np.median(vals)); c += 1
                sdr_avg_value = (s / c) if c else float('nan')
        else:
            mean_msdm_si_sdr_avg = float('nan')
            sdr_avg_value = float('nan')
            sdr_avg_per_track_medians_value = float('nan')

        # final log (per-stem metrics only, not averaged ones)
        # Only log on rank 0 since we manually gathered all data there
        # if trainer.is_global_zero:
        pl_module.log_dict(log_dict, sync_dist=False, on_epoch=True)

        # reset per-stem buffers on ALL ranks (not just rank 0) to prevent memory leaks
        for stem in self.stems:
            self.metrics_log[stem]['si_snr'].clear()
            self.metrics_log[stem]['si_sdr'].clear()
            self.metrics_log[stem]['msdm_si_sdr'].clear()
            self.metrics_log[stem]['sdr'].clear()
            self.metrics_log[stem]['sdr_silent'].clear()
            self.metrics_log[stem]['output_rms_silent'].clear()
            self.metrics_log[stem]['museval']["SDR"].clear()
            self.metrics_log[stem]['museval']["ISR"].clear()
            self.metrics_log[stem]['museval']["SIR"].clear()
            self.metrics_log[stem]['museval']["SAR"].clear()

        # Save gathered metrics to JSON file (only on rank 0)
        if trainer.is_global_zero:
            wandb_logger = get_wandb_logger(trainer).experiment
            with open(os.path.join(wandb_logger.dir, f'metrics_log_epoch_{trainer.current_epoch}.json'), 'w') as f:
                json.dump(gathered_metrics, f)

        # Broadcast averaged metrics from rank 0 to all ranks for ModelCheckpoint
        # Use CPU for broadcast tensors (they're just scalars, no need for GPU)
        sdr_tensor       = torch.tensor(0.0 if np.isnan(sdr_avg_value) else sdr_avg_value, device='cpu')
        msdm_tensor      = torch.tensor(0.0 if np.isnan(mean_msdm_si_sdr_avg) else mean_msdm_si_sdr_avg, device='cpu')
        sdr_ptmed_tensor = torch.tensor(0.0 if np.isnan(sdr_avg_per_track_medians_value) else sdr_avg_per_track_medians_value, device='cpu')

        # Broadcast from rank 0 so all ranks have identical numbers
        sdr_tensor     = trainer.strategy.broadcast(sdr_tensor,     src=0)
        msdm_tensor    = trainer.strategy.broadcast(msdm_tensor,    src=0)
        sdr_ptmed_tensor = trainer.strategy.broadcast(sdr_ptmed_tensor, src=0)

        pl_module.log('sdr_avg',                  float(sdr_tensor.item()),       on_epoch=True, prog_bar=False, sync_dist=False)
        pl_module.log('msdm_si_sdr_avg',          float(msdm_tensor.item()),      on_epoch=True, prog_bar=False, sync_dist=False)
        pl_module.log('sdr_median_of_medians',    float(sdr_ptmed_tensor.item()), on_epoch=True, prog_bar=False, sync_dist=False)



class ClassCondSeparateTrackSampleLogger_simple(ClassCondSeparateTrackSampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        # sampling_steps: List[int],
        # diffusion_schedule: Schedule,
        # diffusion_sampler: Sampler,
        stems = ['bass', 'drums', 'guitar', 'piano'],
        msdm_style: bool = False,
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=None,
            diffusion_schedule=None,
            diffusion_sampler=None,
            stems = stems,
            msdm_style=msdm_style,
        )

    def generate_sample(self, trainer, pl_module, batch):
        
        model = pl_module.model
        
        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding = pl_module.get_input(batch)


        mixtures = batch[2].sum(1)

        # Get start diffusion noise for whole batch
        noise = mixtures

        # Dictionary to store generated samples
        generated_samples = {stem: [] for stem in self.stems}
        
        # Iterate over each one-hot encoded feature vector (each stem)
        for i, stem in enumerate(self.stems):
            # Create a feature tensor for the current stem for all items
            current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
            current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

            # Sample from the model using the noise and the current one-hot features
            samples = model.sample(
                noise=noise,
                features=current_features,
                sampler=None,
                sigma_schedule=None,
                num_steps=None,
                channels_list=channels_list
            )
            samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

            # Store the generated samples
            for idx in range(waveforms.size(0)):
                # if steps not in generated_samples[stem]:
                #     generated_samples[stem][steps] = []
                generated_samples[stem].append(samples[idx])

        # get original stems
        original_samples = {stem: {} for stem in self.stems}
        
        original_stems = batch[2]
        
        for i, stem in enumerate(self.stems):
            stem_data = original_stems[:, i]
            stem_data =rearrange(stem_data, "b c t -> b t c") .detach().cpu().numpy()
            original_samples[stem] = []
            for idx in range(waveforms.size(0)):
                original_samples[stem].append(stem_data[idx])  
        
        mixture_audios = batch[2].sum(1)[:, 0, :].detach().cpu().numpy()[..., np.newaxis] #channels_list[0][idx, 0, :].detach().cpu().numpy()[..., np.newaxis]
        
        return  original_samples, generated_samples, mixture_audios