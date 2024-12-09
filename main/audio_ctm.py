import pytorch_lightning as pl
# from networks import VEPrecond, VPPrecond, EDMPrecond, CTPrecond, iCTPrecond
# from loss import VELoss, VPLoss, EDMLoss, CTLoss, iCTLoss
from torch import optim
import numpy as np
import torch
# from sampler import multistep_consistency_sampling
from torchvision.utils import make_grid, save_image
import copy
from torchmetrics.image.inception import InceptionScore
# from sampler import multistep_consistency_sampling
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from ctm.utils import EMAAndScales_Initialiser, create_model_and_diffusion_audio
from ctm.enc_dec_lib import load_feature_extractor
from ctm.sample_util import karras_sample
import torch.nn as nn
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback, Trainer
import torch.nn.functional as F
from typing import *
from einops import rearrange
import torchaudio
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from audioldm_eval import EvaluationHelper
import shutil
from pathlib import Path
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
import json
import math
from main.model_simple import Audio_DM_Model_simple
import soundfile as sf

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class Audio_CTM_Model(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.diffusion.preconditioning in ['ctm', 'cd'] :

            self.ema_scale_fn = EMAAndScales_Initialiser(target_ema_mode=self.cfg.diffusion.target_ema_mode,
                                                        start_ema=self.cfg.diffusion.start_ema,
                                                        scale_mode=self.cfg.diffusion.scale_mode,
                                                        start_scales=self.cfg.diffusion.start_scales,
                                                        end_scales=self.cfg.diffusion.end_scales,
                                                        total_steps=self.cfg.trainer.max_steps,
                                                        distill_steps_per_iter=self.cfg.diffusion.distill_steps_per_iter,
                                                        ).get_ema_and_scales

            # Load Feature Extractor
            feature_extractor = load_feature_extractor(self.cfg.diffusion, eval=True)


            # Extracting the values
            self.mixture_features_channels = getattr(self.cfg.model, 'mixture_features_channels', None)
            self.pre_trained_mixture_feature_extractor = getattr(self.cfg.model, 'pre_trained_mixture_feature_extractor', None)

            # Deleting the attributes from the Namespace
            if hasattr(self.cfg.model, 'mixture_features_channels'):
                delattr(self.cfg.model, 'mixture_features_channels')

            if hasattr(self.cfg.model, 'pre_trained_mixture_feature_extractor'):
                delattr(self.cfg.model, 'pre_trained_mixture_feature_extractor')


            if self.pre_trained_mixture_feature_extractor is not None:
                # Create a copy of kwargs
                simple_model_kwargs = vars(self.cfg.model).copy()

                # Add or modify any additional arguments required by Audio_DM_Model_simple
                simple_model_kwargs['separation'] = False
                simple_model_kwargs['use_context_time'] = False
                
                # creating models for feature extraction
                self.pre_trained_mixture_feature_extractor_model = Audio_DM_Model_simple(learning_rate = 1.e-4,
                                                                                        beta1 = 0.9,
                                                                                        beta2 = 0.99,
                                                                                        # class_cond = self.cfg.model.get('class_cond', None), 
                                                                                        # separation =  False,
                                                                                        **simple_model_kwargs
                                                                                        )
                # loading pre_trained models from checkpoint
                print("\nloading pre_trained model for feature extraction from checkpoint:", self.pre_trained_mixture_feature_extractor)
                self.pre_trained_mixture_feature_extractor_model.load_state_dict(torch.load(self.pre_trained_mixture_feature_extractor, map_location="cpu")["state_dict"])

                # Freeze parameters and set to eval mode
                for param in self.pre_trained_mixture_feature_extractor_model.parameters():
                    param.requires_grad = False
                self.pre_trained_mixture_feature_extractor_model.eval()
            
            # Load main model
            self.net, self.diffusion = create_model_and_diffusion_audio(self.cfg, feature_extractor)

            if self.cfg.diffusion.teacher_model_path is not None and not self.cfg.diffusion.self_learn:
                print(f"loading the teacher model from {self.cfg.diffusion.teacher_model_path}")
                self.teacher_model, _ = create_model_and_diffusion_audio(self.cfg, teacher=True)
                # if not self.cfg.diffusion.edm_nn_ncsn and not self.cfg.diffusion.edm_nn_ddpm:
                ckpt = torch.load(self.cfg.diffusion.teacher_model_path, map_location="cpu")
                # ckpt = self.adjust_state_dict(ckpt["state_dict"])
                self.teacher_model.load_state_dict(ckpt["state_dict"],strict=False)
                self.teacher_model.eval()
                self.copy_teacher_params_to_model(self.cfg.diffusion)
                self.teacher_model.requires_grad_(False)
                # if self.cfg.diffusion.edm_nn_ncsn:
                #     self.net.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs
                if self.cfg.diffusion.use_fp16:
                    self.teacher_model.convert_to_fp16()
            else:
                self.teacher_model = None

            self.target_model, _ = create_model_and_diffusion_audio(self.cfg)
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            # for dst, src in zip(self.target_model.parameters(), self.net.parameters()):
            #     dst.data.copy_(src.data)

            # if self.cfg.diffusion.edm_nn_ncsn:
            #     self.target_model.model.map_noise.freqs = self.teacher_model.model.model.map_noise.freqs

            # Initialize EMA models
            self.ema_models = nn.ModuleList()
            for ema_rate in self.cfg.diffusion.ema_rate:
                ema_model, _ = create_model_and_diffusion_audio(self.cfg)
                for param in ema_model.parameters():
                    param.requires_grad = False
                ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
                ema_model.eval()
                self.ema_models.append(ema_model)
            self.ema_models.eval()

            self.diffusion.teacher_model = self.teacher_model

        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')

        # extract class cond and separation arguments from sub-networks
        self.class_cond = self.net.model.class_cond
        self.separation = self.net.model.separation
        # # FID and Inception Score instances are not registered as modules
        # self.evaluation_attrs = {}
        # for step in self.cfg.diffusion.denoise_steps_to_log:            
        #     if self.cfg.testing.calc_inception:
        #         self.evaluation_attrs[f'inception_student_{step}'] = InceptionScore().eval()
        #         for i in self.cfg.diffusion.ema_rate:
        #             self.evaluation_attrs[f"inception_ema_{i}_{step}"] = InceptionScore().eval()
        #         self.evaluation_attrs[f'inception_target_{step}'] = InceptionScore().eval()

        #     if self.cfg.testing.calc_fid:
        #         self.evaluation_attrs[f'fid_student_{step}'] = FrechetInceptionDistance(feature=2048).eval()
        #         for i in self.cfg.diffusion.ema_rate:
        #             self.evaluation_attrs[f"fid_ema_{i}_{step}"] = FrechetInceptionDistance(feature=2048)
        #         self.evaluation_attrs[f'fid_target_{step}'] = FrechetInceptionDistance(feature=2048).eval()


    # def adjust_state_dict(self, state_dict):
    #     # adjust pretrained teacher's format to ours/
    #     new_state_dict = {}
    #     for key in state_dict.keys():
    #         if key.startswith("model.unet."):
    #             new_key = key.replace("model.unet.", "model.")
    #         # else:
    #         #     new_key = key
    #         new_state_dict[new_key] = state_dict[key]
    #     return new_state_dict

    # def move_evaluation_attrs_to_device(self):
    #     # Move all eval attributes to the correct device
    #     for key, value in self.evaluation_attrs.items():
    #         self.evaluation_attrs[key] = value.to(self.device)

    # def on_train_start(self):
    #     self.move_evaluation_attrs_to_device()

    # def on_validation_start(self):
    #     self.move_evaluation_attrs_to_device()


    def copy_teacher_params_to_model(self, args):
        def filter_(dst_name):
            dst_ = dst_name.split('.')
            for idx, name in enumerate(dst_):
                if '_train' in name:
                    dst_[idx] = ''.join(name.split('_train'))
            return '.'.join(dst_)

        for dst_name, dst in self.net.named_parameters():
            for src_name, src in self.teacher_model.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    if args.linear_probing:
                        dst.requires_grad = False
                    break
                if args.linear_probing:
                    if filter_(dst_name) in ['.'.join(src_name.split('.')[1:]), src_name]:
                        dst.data.copy_(src.data)
                        break


    def get_input(self, batch, current_class_indexes = None):

        if isinstance(batch, (list, tuple)) and self.class_cond and self.separation and self.pre_trained_mixture_feature_extractor is not None:
            waveforms, class_indexes, stems  = batch

            batch_size, channels, feature_width = waveforms.shape
            mixture = stems.sum(1)
 
            # extract features form pre trained model
            with torch.no_grad():
                waveforms, class_indexes, channels_list, embedding = self.pre_trained_mixture_feature_extractor_model.get_input(batch)

                # Makeing sure this works well for sampler funtion where we mannually pass index of audio we wan to generate
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


    def calculate_loss(self, waveforms, features, channels_list, embedding, mixture_features_channels_list, split="train"):

        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.cfg.diffusion.num_heun_step)]
        diffusion_training_ = [np.random.rand() < self.cfg.diffusion.diffusion_training_frequency]

        model_kwargs = {}
        # if self.cfg.model.class_cond:
        model_kwargs["features"] = features
        model_kwargs["channels_list"] = channels_list
        model_kwargs["embedding"] = embedding
        model_kwargs["mixture_features_channels_list"] = mixture_features_channels_list

        if split == "val":
            apply_adaptive_weight_original_value = self.diffusion.args.apply_adaptive_weight
            self.diffusion.args.apply_adaptive_weight = False
            
        losses = self.diffusion.ctm_losses(
            step=self.global_step,
            model=self.net,
            x_start=waveforms,
            model_kwargs=model_kwargs,
            target_model=self.target_model,
            discriminator=None,
            init_step=0,              # TODO: might need to change to train start stpe (resume_step) if we adopt any schedulers for GAN ir something similar 
            ctm=True if self.diffusion.args.training_mode=="ctm" else False,
            num_heun_step=num_heun_step[0],
            gan_num_heun_step=-1,
            diffusion_training_=diffusion_training_[0],
            gan_training_=False
        )

        if split == "val":
            self.diffusion.args.apply_adaptive_weight = apply_adaptive_weight_original_value

        if 'consistency_loss' in list(losses.keys()):
            # print("Consistency learning")
            loss = self.cfg.diffusion.consistency_weight * losses["consistency_loss"].mean()

            if 'denoising_loss' in list(losses.keys()):
                loss = loss + self.cfg.diffusion.denoising_weight * losses['denoising_loss'].mean()

            self.log_loss_dict( {k: v.view(-1) for k, v in losses.items()}, split)

            # self.mp_trainer.backward(loss)
           
        elif 'denoising_loss' in list(losses.keys()):
            loss = losses['denoising_loss'].mean()
            self.log_loss_dict({k: v.view(-1) for k, v in losses.items()}, split)
            # self.mp_trainer.backward(loss)

        return loss
    
    def log_loss_dict(self, losses, split):
        for key, values in losses.items():
            self.log(f"{split}/{key} mean", values.mean().item(), sync_dist=True)
            # Log the quantiles (four quartiles, in particular).
            self.log(f"{split}/{key} std", values.std().item(), sync_dist=True)

    def training_step(self, batch, _):
        waveforms, features, channels_list, embedding, mixture_features_channels_list = self.get_input(batch)

        loss = self.calculate_loss(waveforms, features, channels_list, embedding, mixture_features_channels_list = mixture_features_channels_list, split = "train")


        return loss

    # def on_train_batch_end(self, out, batch, batch_idx):
    def on_before_zero_grad(self,optimizer):
        # Update EMA models manually at the end of each batch

        # Update target model
        target_ema, _ = self.ema_scale_fn(self.global_step)
        self.update_ema(self.target_model, self.net, target_ema)

        # Update all the EMA models
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self.update_ema(ema_model, self.net, ema_rate)

    def update_ema(self, ema_model, model, decay):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)



    def validation_step(self, batch, batch_idx):
        waveforms, features, channels_list, embedding, mixture_features_channels_list = self.get_input(batch)
        # images, cond = batch

        loss = self.calculate_loss(waveforms, features, channels_list, embedding, mixture_features_channels_list = mixture_features_channels_list, split = "val")
        self.log("val_loss", loss, sync_dist=True)




    # def on_validation_epoch_end(self):

    #     # Log Inception scores
    #     if self.cfg.testing.calc_inception:
    #         for step in self.cfg.diffusion.denoise_steps_to_log:
    #             iscore = self.evaluation_attrs[f'inception_student_{step}'].compute()[0]
    #             self.log(f'iscore/student_{step}', iscore, sync_dist=True)
    #             self.evaluation_attrs[f'inception_student_{step}'].reset()
                
    #             for i in self.cfg.diffusion.ema_rate:
    #                 iscore_ema = self.evaluation_attrs[f"inception_ema_{i}_{step}"].compute()[0]
    #                 self.log(f'iscore/ema_{i}_{step}', iscore_ema, sync_dist=True)
    #                 self.evaluation_attrs[f"inception_ema_{i}_{step}"].reset()
                
    #             iscore_target = self.evaluation_attrs[f'inception_target_{step}'].compute()[0]
    #             self.log(f'iscore/target_{step}', iscore_target, sync_dist=True)
    #             self.evaluation_attrs[f'inception_target_{step}'].reset()

    #     # Log FID scores
    #     if self.cfg.testing.calc_fid:
    #         for step in self.cfg.diffusion.denoise_steps_to_log:
    #             fid = self.evaluation_attrs[f'fid_student_{step}'].compute().item()
    #             self.log(f'fid/student_{step}', fid, sync_dist=True)
    #             self.evaluation_attrs[f'fid_student_{step}'].reset()
                
    #             for i in self.cfg.diffusion.ema_rate:
    #                 fid_ema = self.evaluation_attrs[f"fid_ema_{i}_{step}"].compute().item()
    #                 self.log(f'fid/ema_{i}_{step}', fid_ema, sync_dist=True)
    #                 self.evaluation_attrs[f"fid_ema_{i}_{step}"].reset()
                
    #             fid_target = self.evaluation_attrs[f'fid_target_{step}'].compute().item()
    #             self.log(f'fid/target_{step}', fid_target, sync_dist=True)
    #             self.evaluation_attrs[f'fid_target_{step}'].reset()

    #     return super().on_validation_epoch_end()

    
    def configure_optimizers(self):
        cfg = self.cfg.optim
        if cfg.optimizer == 'radam':
            optimizer = optim.RAdam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr)

        # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.cfg.optim.warmup_epochs, max_iters=self.cfg.trainer.max_epochs)

        return optimizer
        # return {
        # "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": lr_scheduler,
        #     },
        # }
    
    
    
    
    
    
    
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


class UncondSampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampler: str = "exact",
        check_ctm_denoising_ability: bool = True,
        clip_output: bool = True,
        clip_denoised: bool = False,
        denoise_steps_to_log: List[int] = [1],
        # diffusion_schedule = None,  # Assuming type Schedule
        # diffusion_sampler = None,  # Assuming type Sampler
        model_to_calculate_metrics: str = "",
        sampler_to_calculate_metrics: str = "",
        steps_to_calculate_metrics: int = 0,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.check_ctm_denoising_ability = check_ctm_denoising_ability
        self.clip_output = clip_output
        self.clip_denoised = clip_denoised
        self.denoise_steps_to_log = denoise_steps_to_log
        self.sampler = sampler
        # self.diffusion_schedule = diffusion_schedule
        # self.diffusion_sampler = diffusion_sampler
        self.model_to_calculate_metrics = model_to_calculate_metrics
        self.sampler_to_calculate_metrics = sampler_to_calculate_metrics
        self.steps_to_calculate_metrics = steps_to_calculate_metrics

        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, 
        # dataloader_idx
    ):
        # if self.log_next:
        #     self.log_sample(trainer, pl_module, batch)
        #     self.log_next = False
        
        # if self.log_next == False and batch_idx % 5 == 0:
        #     self.update_metrics(trainer, pl_module, batch)

        if batch_idx == 0:
            self.log_sample(trainer, pl_module, batch, batch_idx)
            
        # this will save audios for FAD and oter metrcis calculation
        if batch_idx % 5 == 0 or trainer.state.fn == 'validate':
            if self.model_to_calculate_metrics is not None:
                self.generate_and_save_model_samples(trainer, pl_module, self.sampler_to_calculate_metrics, self.steps_to_calculate_metrics, batch, batch_idx, prefix=self.model_to_calculate_metrics)
                    
            
    def generate_and_save_model_samples(self, trainer, pl_module, sampler, denoise_steps_to_log, batch, batch_idx, prefix=""):   
        
        current_epoch = trainer.current_epoch
        
        new_sampling_rate = 16000 # because FAD is calculated of 16000
        
        # Create base directory path
        base_dir = os.path.dirname(pl_module._trainer.checkpoint_callback.dirpath)
        resampler = torchaudio.transforms.Resample(self.sampling_rate, new_sampling_rate)
        
        # doing this for sweep to work
        if type(denoise_steps_to_log) == int:
            sampling_steps = [denoise_steps_to_log]
        else:
            sampling_steps = denoise_steps_to_log
                     
        # Generate model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, sampler, sampling_steps, prefix)

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

                # # Define file names
                # generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # Define file names with GPU identifier
                generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                # Save audio files
                torchaudio.save(generated_file_name, resampled_audio, 16000)
                torchaudio.save(original_file_name, resampled_original_audio, 16000)

    def get_nested_attr(self, obj, attr):
        """ Get a nested attribute, supporting indexed access """
        parts = attr.split('[')
        for part in parts:
            if ']' in part:
                key = int(part.rstrip(']'))
                obj = obj[key]
            else:
                obj = getattr(obj, part)
        return obj

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch, batch_idx):
        # is_train = pl_module.training
        # if is_train:
        #     pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        # model = pl_module.model

        # name = self.cfg.data.name
        audios_to_log = []
        captions = []

        # doing this for sweep to work
        if type(self.denoise_steps_to_log) == int:
            sampling_steps = [self.denoise_steps_to_log]
        else:
            sampling_steps = self.denoise_steps_to_log

        # Log teacher model denoises with heun and 18 steps (if we need to see)
        if self.check_ctm_denoising_ability:
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'heun', [18], "teacher_model")
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Log student model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, "net")
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)

        # Log target model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, "target_model")
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)


        # Log EMA models outputs
        for i, ema_rate in enumerate(pl_module.cfg.diffusion.ema_rate):
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, f"ema_models[{i}]")
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch)

        # log original data
        original_audio = waveforms[:self.num_items].clamp(-1.0, 1.0)
        audios_to_log.append(original_audio.permute(0, 2, 1).cpu().numpy())
        captions.append("Original")
        ##################################


        # Log one sample of the original and generated images for the first batch in the epoch
        if trainer.is_global_zero and batch_idx==0:            
            for idx in range(self.num_items):
                logging_data = {}
                for i, caption in enumerate(captions):
                    audio = audios_to_log[i][idx]
                    logging_data[caption] = wandb.Audio(audio, sample_rate=self.sampling_rate, caption=f"batch_{batch_idx}_N_{idx}") # TODO: Make validation epoch number here instaed of batch index
                wandb_logger.log(logging_data)
        
         
    @torch.no_grad()
    def sampling(self, model, sampler = 'exact', ctm=None, teacher=False, prefix="", step=-1, num_samples=-1, batch_size=-1, resize=False, generator=None, class_idx = None, **model_kwargs):
        if not teacher:
            model.eval()
        if step == -1:
            step = 1
        if batch_size == -1:
            batch_size = model.cfg.datamodule.batch_size

        all_images = []
        number = 0
        
        # Dynamically select model based on prefix using getattr
        model_to_use = self.get_nested_attr(model, prefix)

        while num_samples > number:
            # model_kwargs = {}
            # if self.cfg.data.class_cond:
            #     if class_idx == None:                
            #         model_kwargs["y"] = torch.randint(0, self.cfg.data.num_classes, size=(batch_size, ), device=self.device)
            #     else:
            #         model_kwargs["y"] = torch.full((batch_size,), class_idx, device=self.device)

            # print(f"{number} number samples complete")
            if generator != None:
                x_T = generator.randn(*(batch_size, self.channels, self.length),
                            device=self.device) * model.cfg.diffusion.sigma_max
            else:
                x_T = None

            if sampler == "ADPM2Sampler":
                import audio_diffusion_pytorch_
                from main.module_base import Audio_DM_Model
                diffusion_sampler = audio_diffusion_pytorch_.ADPM2Sampler(rho = 1)
                sigma_schedule = audio_diffusion_pytorch_.KarrasSchedule(sigma_min = model.cfg.diffusion.sigma_min, sigma_max =  model.cfg.diffusion.sigma_max,  rho = model.cfg.diffusion.rho)

                # model_to_use = Audio_DM_Model(learning_rate = 1.e-4, beta1 = 0.9, beta2 = 0.99, **vars(model.cfg.model)).to(model.device)
                # model_to_use.eval()
                diffusion_sigma_distribution = audio_diffusion_pytorch_.LogNormalDistribution(mean = -3.0, std = 1.0)
                
                diffusion = audio_diffusion_pytorch_.Diffusion(
                    net=model_to_use.model.unet,
                    sigma_distribution=diffusion_sigma_distribution,
                    sigma_data=model.cfg.diffusion.sigma_data,
                    dynamic_threshold=0.0,
                )


                # Get start diffusion noise
                noise = torch.randn(
                    (batch_size, self.channels, self.length), device=model.device
                )
               
                diffusion_sampler = audio_diffusion_pytorch_.DiffusionSampler(
                        diffusion=diffusion,
                        sampler=diffusion_sampler,
                        sigma_schedule=sigma_schedule,
                        num_steps=step,
                    )
                sample = diffusion_sampler(noise, **model_kwargs)
            else:        
                sample = karras_sample(
                    diffusion=model.diffusion,
                    model=model_to_use,
                    shape=(batch_size, self.channels, self.length),
                    steps=step,
                    model_kwargs=model_kwargs, # in case of classes class goes here???
                    device=model.device,
                    clip_denoised=True if teacher else self.clip_denoised,
                    sampler=sampler,
                    generator=None,
                    teacher=teacher,
                    ctm= ctm,
                    x_T=x_T if generator != None else self.x_T.to(self.device) if num_samples == -1 else None,
                    clip_output=self.clip_output,
                    sigma_min=model.cfg.diffusion.sigma_min,
                    sigma_max=model.cfg.diffusion.sigma_max,
                    train=False,
                )

            gathered_samples = sample.contiguous()
            all_images += [sample.cpu() for sample in gathered_samples]
            
            number += int(gathered_samples.shape[0])
        if not teacher:
            model.train()

        arr = torch.stack(all_images, axis=0)

        return arr

    def generate_model_output(self, model, sampler, steps, prefix):

        audios_to_log = []
        captions = []

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=step, num_samples=1, batch_size=model.cfg.datamodule.batch_size, ctm=True if model.cfg.diffusion.training_mode=="ctm" and prefix != "teacher_model" else False, class_idx = None)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix} {step} Steps"

            audios_to_log.append(xh.permute(0, 2, 1).cpu().numpy())
            captions.append(caption)

        return audios_to_log, captions
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        base_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
        wandb_logger = get_wandb_logger(trainer).experiment
        evaluator = EvaluationHelper(sampling_rate=16000, device=pl_module.device)

        steps_to_calculate_metrics = self.steps_to_calculate_metrics if isinstance(self.steps_to_calculate_metrics, list) else [self.steps_to_calculate_metrics]

        for step in steps_to_calculate_metrics:
            step_dir = os.path.join(base_dir, f'audios_{current_epoch}_step{step}')
            if os.path.exists(step_dir):
                dir1, dir2 = Path(os.path.join(step_dir, "generated")), Path(os.path.join(step_dir, "original"))
                print("\nNow evaluating:", step_dir)
                metrics = evaluator.main(str(dir1), str(dir2))
                metrics_buffer = {f"step_{step}/{k}" if isinstance(self.steps_to_calculate_metrics, list) else k: float(v) for k, v in metrics.items()}

                if metrics_buffer:
                    wandb_logger.log(metrics_buffer, commit=True)
                    # for k, v in metrics_buffer.items():
                    #     wandb_logger.log({k: v}, commit=False)
                    #     print(k, v)
                    # wandb_logger.log({}, commit=True)
                shutil.rmtree(dir1)
                shutil.rmtree(dir2)

class ClassCondTrackSampleLoggerCTM(UncondSampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampler: List[str] = ["exact"],
        check_ctm_denoising_ability: bool = True,
        clip_output: bool = True,
        clip_denoised: bool = False,
        denoise_steps_to_log: List[int] = [1],
        model_to_calculate_metrics: str = "",
        sampler_to_calculate_metrics: str = "",
        steps_to_calculate_metrics: int = 0,
        # Additional parameters for the new class
        stems: List[str] = ['bass', 'drums', 'guitar', 'piano'],
        models_to_log: List[str] = ["teacher_model", "net"]
    ) -> None:
        # Call the parent class's __init__ method
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampler="exact",
            check_ctm_denoising_ability=check_ctm_denoising_ability,
            clip_output=clip_output,
            clip_denoised=clip_denoised,
            denoise_steps_to_log=denoise_steps_to_log,
            model_to_calculate_metrics=model_to_calculate_metrics,
            sampler_to_calculate_metrics=sampler_to_calculate_metrics,
            steps_to_calculate_metrics=steps_to_calculate_metrics
        )
        
        self.sampler = sampler
        self.stems = stems
        self.models_to_log = models_to_log                

    def generate_model_output(self, model, sampler, steps, prefix, batch):

        audios_to_log = []
        captions = []
        
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = model.get_input(batch)
        
        model_kwargs = {}
        # if self.cfg.model.class_cond:
        model_kwargs["features"] = class_indexes
        model_kwargs["channels_list"] = channels_list
        model_kwargs["embedding"] = embedding
        # model_kwargs["mixture_features_channels_list"] = mixture_features_channels_list
        
        batch_size = batch[0].size(0)

        # generate random grid class conditional or unconditional
        for step in steps:
            xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=step, num_samples=1, batch_size=batch_size, ctm= True if model.cfg.diffusion.training_mode=="ctm" and prefix != "teacher_model" else False, **model_kwargs)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            caption = f"{prefix} {step} Steps"

            audios_to_log.append(xh.permute(0, 2, 1).cpu().numpy())
            captions.append(caption)

        return audios_to_log, captions                

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch, batch_idx):
        # is_train = pl_module.training
        # if is_train:
        #     pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        # model = pl_module.model

        # name = self.cfg.data.name
        audios_to_log = []
        captions = []

        # doing this for sweep to work
        if type(self.denoise_steps_to_log) == int:
            sampling_steps = [self.denoise_steps_to_log]
        else:
            sampling_steps = self.denoise_steps_to_log

        # Log teacher model denoises with heun and 18 steps (if we need to see)
        if self.check_ctm_denoising_ability:
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, 'heun', [18], "teacher_model", batch)
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Log student model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, "net", batch)
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)

        # Log target model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, "target_model", batch)
        audios_to_log.extend(new_audios_to_log)
        captions.extend(new_captions)


        # Log EMA models outputs
        for i, ema_rate in enumerate(pl_module.cfg.diffusion.ema_rate):
            new_audios_to_log, new_captions = self.generate_model_output(pl_module, self.sampler,  sampling_steps, f"ema_models[{i}]", batch)
            audios_to_log.extend(new_audios_to_log)
            captions.extend(new_captions)

        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch)

        # log original data
        original_audio = waveforms[:self.num_items].clamp(-1.0, 1.0)
        audios_to_log.append(original_audio.permute(0, 2, 1).cpu().numpy())
        captions.append("Original")
        ##################################


        # Log one sample of the original and generated images for the first batch in the epoch
        if trainer.is_global_zero and batch_idx==0:            
            for idx in range(self.num_items):
                logging_data = {}
                for i, caption in enumerate(captions):
                    audio = audios_to_log[i][idx]
                    logging_data[caption] = wandb.Audio(audio, sample_rate=self.sampling_rate, caption=f"batch_{batch_idx}_N_{idx}") # TODO: Make validation epoch number here instaed of batch index
                wandb_logger.log(logging_data)

    def generate_and_save_model_samples(self, trainer, pl_module, sampler, denoise_steps_to_log, batch, batch_idx, prefix=""):   
        
        current_epoch = trainer.current_epoch
        
        new_sampling_rate = 16000 # because FAD is calculated of 16000
        
        # Create base directory path
        base_dir = os.path.dirname(pl_module._trainer.checkpoint_callback.dirpath)
        resampler = torchaudio.transforms.Resample(self.sampling_rate, new_sampling_rate)
        
        # doing this for sweep to work
        if type(denoise_steps_to_log) == int:
            sampling_steps = [denoise_steps_to_log]
        else:
            sampling_steps = denoise_steps_to_log
                     
        # Generate model outputs
        new_audios_to_log, new_captions = self.generate_model_output(pl_module, sampler, sampling_steps, prefix, batch)

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

                # # Define file names
                # generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}.wav')
                # Define file names with GPU identifier
                generated_file_name = os.path.join(generated_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                original_file_name = os.path.join(original_dir, f'audio_epoch_{current_epoch}_batch_{batch_idx}_sample_{idx}_gpu_{gpu_id}.wav')
                # Save audio files
                torchaudio.save(generated_file_name, resampled_audio, 16000)
                torchaudio.save(original_file_name, resampled_original_audio, 16000)


class ClassCondSeparateTrackSampleLoggerCTM(UncondSampleLogger):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampler: List[str] = ["exact"],
        check_ctm_denoising_ability: bool = True,
        clip_output: bool = True,
        clip_denoised: bool = False,
        denoise_steps_to_log: List[int] = [1],
        model_to_calculate_metrics: str = "",
        sampler_to_calculate_metrics: str = "",
        steps_to_calculate_metrics: int = 0,
        # Additional parameters for the new class
        stems: List[str] = ['bass', 'drums', 'guitar', 'piano'],
        models_to_log: List[str] = ["teacher_model", "net"]
    ) -> None:
        # Call the parent class's __init__ method
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampler="exact",
            check_ctm_denoising_ability=check_ctm_denoising_ability,
            clip_output=clip_output,
            clip_denoised=clip_denoised,
            denoise_steps_to_log=denoise_steps_to_log,
            model_to_calculate_metrics=model_to_calculate_metrics,
            sampler_to_calculate_metrics=sampler_to_calculate_metrics,
            steps_to_calculate_metrics=steps_to_calculate_metrics
        )
        
        self.sampler = sampler
        self.stems = stems
        self.models_to_log = models_to_log
        
        self.torch_si_snr = ScaleInvariantSignalNoiseRatio()
        self.torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        
        # Initialize metrics dictionaries for each stem
        self.metrics_log = {
            stem: {'si_snr': [], 'si_sdr': [], 'msdm_si_snr': []} for stem in stems
        }

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

        if  batch_idx ==0 and trainer.is_global_zero:
            original_samples = {} 
            generated_samples = {}
            mixture_audios = {}
            
            # Generate samples from all the models and log in pairs
            for step, prefix, sampler in zip(self.denoise_steps_to_log, self.models_to_log, self.sampler):
                new_original_samples, new_generated_samples, new_mixture_audios = self.generate_model_output(trainer, pl_module, sampler, step, prefix, batch) 
                original_samples[f"original_samples"] = new_original_samples
                generated_samples[f"{prefix}_{step}"] = new_generated_samples
                mixture_audios[f"mixtures"] = new_mixture_audios
            self.log_audio(original_samples, generated_samples, mixture_audios, wandb_logger)



        # generate samples form the model that we calculate metrics
        original_samples, generated_samples, mixture_audios = self.generate_model_output(trainer, pl_module, self.sampler_to_calculate_metrics, self.steps_to_calculate_metrics, self.model_to_calculate_metrics, batch)       
        self.update_metrics(original_samples, generated_samples, mixture_audios, wandb_logger, trainer)       

        # # Save original_samples
        # self.save_audio_samples(original_samples, os.path.join(wandb_logger.dir, "original_samples"), batch_idx*len(original_samples["bass"]))

        # # Save generated_samples
        # self.save_audio_samples(generated_samples, os.path.join(wandb_logger.dir, "generated_samples"), batch_idx*len(original_samples["bass"]))
        
        if is_train:
            pl_module.train()

    def save_audio_samples(self, audio_samples, folder_name, index_shift=0):
        """
        Save audio samples from a dictionary into corresponding subfolders.
        
        Parameters:
        - audio_samples: The dictionary containing the audio data (e.g., original_samples).
        - folder_name: The base folder where the files will be saved.
        """
        os.makedirs(folder_name, exist_ok=True)

        for i in range(len(audio_samples['bass'])):  # Assuming all instruments have the same number of samples
            sub_folder_name = os.path.join(folder_name, str(i + index_shift))  # Folders 0, 1, 2
            os.makedirs(sub_folder_name, exist_ok=True)

            for stem in audio_samples.keys():
                filename = os.path.join(sub_folder_name, f"{stem}.wav")
                sf.write(filename, audio_samples[stem][i], samplerate=22050)



    @torch.no_grad()
    def generate_model_output(self, trainer, pl_module, sampler, steps, prefix, batch):

        # Extract mixture and original audio from the batch
        waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch)

        # Dictionary to store generated samples
        generated_samples = {stem: [] for stem in self.stems}


        # Iterate over each one-hot encoded feature vector (each stem)
        for i, stem in enumerate(self.stems):
            # Create a feature tensor for the current stem for all items
            current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
            current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)
            
            
            # Extract mixture and original audio from the batch
            waveforms, class_indexes, channels_list, embedding, mixture_features_channels_list = pl_module.get_input(batch, current_features)

            model_kwargs = {}
            # if self.cfg.model.class_cond:
            model_kwargs["features"] = current_features
            model_kwargs["channels_list"] = channels_list
            model_kwargs["embedding"] = embedding
            model_kwargs["mixture_features_channels_list"] = mixture_features_channels_list
            model_kwargs["features"] = current_features

            # Sample from the model using the noise and the current one-hot features
            xh = self.sampling(model=pl_module, sampler=sampler, teacher= True if prefix == "teacher_model" else False, prefix=prefix, step=steps, num_samples=1, batch_size=waveforms.size(0), ctm=True if pl_module.cfg.diffusion.training_mode=="ctm" and prefix != "teacher_model" else False, **model_kwargs) # class_idx = None, features = current_features, channels_list = channels_list, embedding = None)
            xh.clamp(-1.0, 1.0) # (xh * 0.5 + 0.5).clamp(0, 1)

            samples = rearrange(xh, "b c t -> b t c").detach().cpu().numpy()

            # Store the generated samples
            for idx in range(waveforms.size(0)):
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

    def log_audio(self, original_samples, generated_samples, mixture_audio, wandb_logger):
        
        # Log the first item of the batch
        for idx in range(self.num_items):
            # Prepare the logging data
            logging_data = {}

            # Prepare the original mixture log
            logging_data[f"Mixture_audio"] = wandb.Audio(
                mixture_audio["mixtures"][idx], 
                caption=f"Mixture Audio {idx}", 
                sample_rate=self.sampling_rate
            )

            # Prepare the original stems logs
            for stem in self.stems:
                original_audio = original_samples["original_samples"][stem][idx]
                logging_data[f"original_{stem}"] = wandb.Audio(
                    original_audio,
                    caption=f"Original {stem} Audio {idx}",
                    sample_rate=self.sampling_rate
                )

            for generated_samples_key in generated_samples.keys():
                # Prepare each generated sample for the current stem by number of steps
                for stem in self.stems:
                    generated_audio = generated_samples[generated_samples_key][stem][idx]
                    logging_data[f"generated_{stem}_{generated_samples_key}"] = wandb.Audio(
                        generated_audio,
                        caption=f"{stem} Sampled: {idx})",
                        sample_rate=self.sampling_rate
                    )

                # Prepare the mixture of the generated samples
                mix_audio = sum(generated_samples[generated_samples_key][stem][idx] for stem in self.stems)
                logging_data[f"generated_mix_{generated_samples_key}"] = wandb.Audio(
                    mix_audio,
                    caption=f"Sampled Mix (idx: {idx})",
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

            
    # @torch.no_grad()
    # def update_metrics(self, original_samples, generated_samples, mixture_audios, wandb_logger, trainer):

    #     for stem in self.stems:
    #         for idx in range(len(original_samples[stem])):  # Iterate over each sample in the batch
    #             original_audio = original_samples[stem][idx]
    #             generated_audio = generated_samples[stem][idx]
    #             mixture_audio = mixture_audios[idx]

    #             original_audio = torch.tensor(original_audio).permute(1, 0)
    #             generated_audio = torch.tensor(generated_audio).permute(1, 0)
    #             mixture_audio = torch.tensor(mixture_audio).permute(1, 0)

    #             si_snr = self.torch_si_snr(generated_audio, original_audio)
    #             si_sdr = self.torch_si_sdr(generated_audio, original_audio)
    #             msdm_si_snr = self.sisnr(generated_audio, original_audio) - self.sisnr(mixture_audio, original_audio)

    #             # Append computed metrics to the corresponding lists
    #             self.metrics_log[stem]['si_snr'].append(si_snr.item())
    #             self.metrics_log[stem]['si_sdr'].append(si_sdr.item())
    #             self.metrics_log[stem]['msdm_si_snr'].append(msdm_si_snr.item())

    
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
