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
from audioldm_eval import EvaluationHelper
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
            waveforms, class_indexes, stems  = batch

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
            waveforms, class_indexes, stems  = batch

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
            mixture_features_channels_list = None

            
        elif isinstance(batch, (list, tuple)) and self.class_cond:
            waveforms, class_indexes, _ = batch
            channels_list = None
            embedding = None
            mixture_features_channels_list = None
        elif isinstance(batch, (list, tuple)) :
            waveforms, _, _= batch
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
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.batch_size = batch_size
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
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
        
        self.torch_si_snr = ScaleInvariantSignalNoiseRatio()
        self.torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        
        self.metrics_log = {stem: {'si_snr': [], 'si_sdr': [], 'msdm_si_snr': []} for stem in stems}

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        if batch_idx % 5 == 0 or trainer.state.fn == 'validate':
            self.log_sample(trainer, pl_module, batch, batch_idx)

    def log_sample(self, trainer, pl_module, batch, batch_idx):
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        original_samples, generated_samples, mixture_audios = self.generate_sample(trainer, pl_module, batch)
        
        if  batch_idx ==0 and trainer.is_global_zero:
            self.log_audio(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)
        
        # log metrics into dict
        self.update_metrics(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)       
        
        if is_train:
            pl_module.train()

    def generate_sample(self, trainer, pl_module, batch):
        
        model = pl_module.model
        
        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch)

        # # Get start diffusion noise for whole batch
        # noise = torch.randn(
        #     (waveforms.size(0), self.channels, self.length), device=pl_module.device
        # )

        # Dictionary to store generated samples
        generated_samples = {stem: [] for stem in self.stems}
        
        
        # Iterate over each diffusion step size
        # for steps in self.sampling_steps:
        steps = self.sampling_steps
        # Iterate over each one-hot encoded feature vector (each stem)
        for i, stem in enumerate(self.stems):
            # Create a feature tensor for the current stem for all items
            current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
            current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

            # extract coresponding featires from the pre-trained model for cuccrent stem we are generating
            if pl_module.pre_trained_mixture_feature_extractor is not None:
                waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch, current_features)

            # Get start diffusion noise for whole batch
            noise = torch.randn(
                (waveforms.size(0), self.channels, self.length), device=pl_module.device
            )

            noise = [noise, mixture_features_channels_list.pop()]

            # Sample from the model using the noise and the current one-hot features
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


    @torch.no_grad()
    def log_audio(self, original_samples, generated_samples, mixture_audio, wandb_logger, trainer):
        
        # Log the first item of the batch
        for idx in range(self.num_items):
            # Prepare the logging data
            logging_data = {}

            # Prepare the original mixture log
            logging_data[f"Mixture_audio"] = wandb.Audio(
                mixture_audio[idx], 
                caption=f"Mixture Audio {idx}", 
                sample_rate=self.sampling_rate
            )

            # Prepare the original stems logs
            for stem in self.stems:
                original_audio = original_samples[stem][idx]
                logging_data[f"original_{stem}"] = wandb.Audio(
                    original_audio,
                    caption=f"Original {stem} Audio {idx}",
                    sample_rate=self.sampling_rate
                )

            # Prepare each generated sample for the current stem by number of steps
            # for steps in self.sampling_steps:
            for stem in self.stems:
                generated_audio = generated_samples[stem][idx]
                logging_data[f"generated_{stem}"] = wandb.Audio(
                    generated_audio,
                    caption=f"{stem} Sampled in {self.sampling_steps} steps (idx: {idx})",
                    sample_rate=self.sampling_rate
                )

            # Prepare the mixture of the generated samples
            mix_audio = sum(generated_samples[stem][idx] for stem in self.stems)
            logging_data[f"generated_mix"] = wandb.Audio(
                mix_audio,
                caption=f"Sampled in {self.sampling_steps} steps (Mix) (idx: {idx})",
                sample_rate=self.sampling_rate
            )

            # Log all accumulated data
            wandb_logger.log(logging_data) #step = trainer.global_step+idx)
            
    def sisnr(self, preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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
    def update_metrics(self, original_samples, generated_samples, mixture_audios, wandb_logger, trainer, chunk_duration = 4.0, overlap_duration = 2.0, eps = 1e-8):

        chunk_samples = int(chunk_duration * self.sampling_rate)
        overlap_samples = int(overlap_duration * self.sampling_rate)

        # Calculate the step size between consecutive sub-chunks
        step_size = chunk_samples - overlap_samples

        # Determine the number of evaluation chunks based on step_size
        num_eval_chunks = math.ceil((self.length - overlap_samples) / step_size)

        for idx in range(len(mixture_audios)):

            for j in range(num_eval_chunks):

                start_sample = j * step_size
                end_sample = start_sample + chunk_samples

                # Ensure the end_sample does not exceed the timesteps
                if end_sample > self.length:
                    end_sample = self.length

                # Determine number of active signals in sub-chunk
                num_active_signals = 0
                for stem in self.stems:
                    o = original_samples[stem][idx][start_sample:end_sample, : ]
                    if not self.is_silent(torch.tensor(o).permute(1, 0)):
                        num_active_signals += 1
                
                # Skip sub-chunk if necessary
                if num_active_signals <= 1:
                    continue

                # for steps in self.sampling_steps:
                for stem in self.stems:
                    original_audio = original_samples[stem][idx][start_sample:end_sample, :]
                    generated_audio = generated_samples[stem][idx][start_sample:end_sample, :]
                    mixture_audio = mixture_audios[idx, start_sample:end_sample, :]

                    original_audio = torch.tensor(original_audio).permute(1, 0)
                    generated_audio = torch.tensor(generated_audio).permute(1, 0)
                    mixture_audio = torch.tensor(mixture_audio).permute(1, 0)

                    si_snr = self.torch_si_snr(generated_audio, original_audio)
                    si_sdr = self.torch_si_sdr(generated_audio, original_audio)
                    msdm_si_snr_s = self.sisnr(generated_audio, original_audio) 
                    msdm_si_snr_o = self.sisnr(mixture_audio, original_audio)
                    msdm_si_snr = msdm_si_snr_s - msdm_si_snr_o

                    # Append computed metrics to the corresponding lists
                    self.metrics_log[stem]['si_snr'].append(si_snr.item())
                    self.metrics_log[stem]['si_sdr'].append(si_sdr.item())
                    self.metrics_log[stem]['msdm_si_snr'].append(msdm_si_snr.item())

        if trainer.is_global_zero:
            with open(os.path.join(wandb_logger.dir, f'metrics_log_epoch_{trainer.current_epoch}.json'), 'w') as f:
                json.dump(self.metrics_log, f)

    
    def on_validation_epoch_end(self, trainer, pl_module):
        log_dict = {}
        total_msdm_si_snr = 0
        num_stems = len(self.stems)
        
        for stem in self.stems:
            mean_si_snr = sum(self.metrics_log[stem]['si_snr']) / len(self.metrics_log[stem]['si_snr'])
            mean_si_sdr = sum(self.metrics_log[stem]['si_sdr']) / len(self.metrics_log[stem]['si_sdr'])
            mean_msdm_si_snr = sum(self.metrics_log[stem]['msdm_si_snr']) / len(self.metrics_log[stem]['msdm_si_snr'])

            log_dict[f'si_snr/{stem}'] = mean_si_snr
            log_dict[f'si_sdr/{stem}'] = mean_si_sdr
            log_dict[f'msdm_si_snr/{stem}'] = mean_msdm_si_snr

            # Accumulate total msdm_si_snr for averaging later
            total_msdm_si_snr += mean_msdm_si_snr

            # Reset metrics for current stem
            self.metrics_log[stem]['si_snr'] = []
            self.metrics_log[stem]['si_sdr'] = []
            self.metrics_log[stem]['msdm_si_snr'] = []
        
        # Calculate the average of msdm_si_snr across all instruments
        mean_msdm_si_snr_avg = total_msdm_si_snr / num_stems
        log_dict[f'msdm_si_snr_avg'] = mean_msdm_si_snr_avg

        # Log the results
        pl_module.log_dict(log_dict, sync_dist=True) # step=trainer.global_step)



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
        stems = ['bass', 'drums', 'guitar', 'piano']
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
        )

    def generate_sample(self, trainer, pl_module, batch):
        
        model = pl_module.model
        
        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding = pl_module.get_input(batch)


        mixtures = batch[-1].sum(1)

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