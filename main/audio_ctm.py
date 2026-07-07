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
from ctm.utils import EMAAndScales_Initialiser, create_model_and_diffusion_audio, create_model_and_diffusion_audio_msst
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
# from audioldm_eval import EvaluationHelper  # unused: only referenced by UncondSampleLogger.on_validation_epoch_end, which ClassCondSeparateTrackSampleLoggerCTM overrides
import shutil
from pathlib import Path
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
import json
import math
from main.model_simple import Audio_DM_Model_simple
import soundfile as sf
from main.module_base import ClassCondSeparateTrackSampleLogger, ClassCondSeparateTrackSampleLogger_MUSDB_MSST_stems_in_out

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
            waveforms, class_indexes, stems  = batch[0], batch[1], batch[2]

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
            waveforms, class_indexes, stems  = batch[0], batch[1], batch[2]

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
            waveforms, _, _ = batch[0], batch[1], batch[2]
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
    
    
    
    
    
class Audio_MSST_CTM_Model(pl.LightningModule):
    """
    Consistency distillation (CD/CTM) training for BSRoformer-based MSST diffusion model.

    Mirrors Audio_CTM_Model but uses:
    - create_model_and_diffusion_audio_msst  (EDMPrecond_MSST_CTM) instead of UNet1d
    - Audio_MSST_DM_Model-style data pipeline (4-D stems, mixture features, random-stem mode)
    - Teacher checkpoint loaded from an Audio_MSST_DM_Model .ckpt (strict=False)
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        assert cfg.diffusion.preconditioning in ('ctm', 'cd'), \
            f"Unsupported preconditioning: {cfg.diffusion.preconditioning}"

        self.ema_scale_fn = EMAAndScales_Initialiser(
            target_ema_mode=cfg.diffusion.target_ema_mode,
            start_ema=cfg.diffusion.start_ema,
            scale_mode=cfg.diffusion.scale_mode,
            start_scales=cfg.diffusion.start_scales,
            end_scales=cfg.diffusion.end_scales,
            total_steps=cfg.trainer.max_steps,
            distill_steps_per_iter=cfg.diffusion.distill_steps_per_iter,
        ).get_ema_and_scales

        # ── Feature extractor (frozen pre-trained deterministic model) ──────────
        from main.module_base import instantiate_from_config
        self.diffusion_input_mix_prob = getattr(cfg.model, 'diffusion_input_mix_prob', 1.0)
        self.stem_to_diffuse = getattr(cfg.model, 'stem_to_diffuse', None)
        self.pre_trained_mixture_feature_extractor_path = getattr(cfg.model, 'pre_trained_mixture_feature_extractor', None)
        pre_trained_cfg = getattr(cfg.model, 'pre_trained_mixture_feature_extractor_model_config', None)

        if pre_trained_cfg is not None:
            self.pre_trained_mixture_feature_extractor_model = instantiate_from_config(pre_trained_cfg)
            print("\nLoading feature extractor from:", self.pre_trained_mixture_feature_extractor_path)
            self.pre_trained_mixture_feature_extractor_model.load_state_dict(
                torch.load(self.pre_trained_mixture_feature_extractor_path, map_location='cpu')['state_dict']
            )
            for param in self.pre_trained_mixture_feature_extractor_model.parameters():
                param.requires_grad = False
            self.pre_trained_mixture_feature_extractor_model.eval()
        else:
            self.pre_trained_mixture_feature_extractor_model = None

        # ── Student model ────────────────────────────────────────────────────────
        feature_extractor = load_feature_extractor(cfg.diffusion, eval=True)
        self.net, self.diffusion = create_model_and_diffusion_audio_msst(cfg, feature_extractor=feature_extractor)

        # ── Teacher model (loaded from diffusion checkpoint) ─────────────────────
        if cfg.diffusion.teacher_model_path is not None and not cfg.diffusion.self_learn:
            print(f"Loading teacher model from {cfg.diffusion.teacher_model_path}")
            self.teacher_model, _ = create_model_and_diffusion_audio_msst(cfg, teacher=True)
            ckpt = torch.load(cfg.diffusion.teacher_model_path, map_location='cpu')
            self.teacher_model.load_state_dict(ckpt['state_dict'], strict=False)
            self.teacher_model.eval()
            self.copy_teacher_params_to_model(cfg.diffusion)
            self.teacher_model.requires_grad_(False)
            if cfg.diffusion.use_fp16:
                self.teacher_model.convert_to_fp16()
        else:
            self.teacher_model = None

        self.diffusion.teacher_model = self.teacher_model

        # ── Target model (EMA copy of student) ──────────────────────────────────
        self.target_model, _ = create_model_and_diffusion_audio_msst(cfg)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

        # ── Optional additional EMA models ───────────────────────────────────────
        self.ema_models = nn.ModuleList()
        for ema_rate in cfg.diffusion.ema_rate:
            ema_model, _ = create_model_and_diffusion_audio_msst(cfg)
            for param in ema_model.parameters():
                param.requires_grad = False
            ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
            ema_model.eval()
            self.ema_models.append(ema_model)
        self.ema_models.eval()

    # ── Param copy helpers ────────────────────────────────────────────────────────

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

    # ── Data pipeline (mirrors Audio_MSST_DM_Model.get_input) ────────────────────

    def get_input(self, batch):
        waveforms, mixtures = batch[0], batch[1]
        batch_size, num_stems, c, t = waveforms.shape
        mixture = mixtures

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                mixture_features_channels_list = self.pre_trained_mixture_feature_extractor_model.model.unet.get_feature(mixture)
                separated_tracks = mixture_features_channels_list[-1]

        if self.training and self.stem_to_diffuse == 'random':
            stem_idx = torch.randint(0, num_stems, (1,)).item()
            waveforms = waveforms[:, stem_idx:stem_idx + 1, :, :]
            separated_track_to_diffuse = separated_tracks[:, stem_idx:stem_idx + 1, :, :]
            features = stem_idx
        elif self.stem_to_diffuse is not None and self.stem_to_diffuse != 'random':
            idx = int(self.stem_to_diffuse)
            waveforms = waveforms[:, idx, :, :]
            separated_tracks_all = separated_tracks
            separated_track_to_diffuse = separated_tracks_all[:, idx, :, :]
            features = None
        else:
            # All-stems mode or eval
            separated_track_to_diffuse = separated_tracks
            features = None

        return waveforms, features, mixture_features_channels_list, separated_track_to_diffuse

    def apply_training_mixing(self, waveforms, extracted_waveforms):
        if extracted_waveforms is not None:
            batch_size = waveforms.shape[0]
            # Create binary mask with shape [B, 1, 1, ...] matching waveforms dimensions
            mask_shape = (batch_size,) + (1,) * (waveforms.ndim - 1)
            mask = (torch.rand(mask_shape, device=waveforms.device) < self.diffusion_input_mix_prob).float()

            # Mix: waveforms = mask * extracted + (1 - mask) * clean
            waveforms = mask * extracted_waveforms + (1 - mask) * waveforms

        return waveforms

    # ── Loss ─────────────────────────────────────────────────────────────────────

    def calculate_loss(self, waveforms, features, mixture_features_channels_list, split='train', target=None):
        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.cfg.diffusion.num_heun_step)]
        diffusion_training_ = [np.random.rand() < self.cfg.diffusion.diffusion_training_frequency]

        model_kwargs = {
            'features': features,
            'channels_list': None,
            'embedding': None,
            'mixture_features_channels_list': mixture_features_channels_list,
        }

        if split == 'val':
            orig_adaptive = self.diffusion.args.apply_adaptive_weight
            self.diffusion.args.apply_adaptive_weight = False

        losses = self.diffusion.ctm_losses(
            step=self.global_step,
            model=self.net,
            x_start=waveforms,
            model_kwargs=model_kwargs,
            target_model=self.target_model,
            discriminator=None,
            init_step=0,
            ctm=self.cfg.diffusion.training_mode == 'ctm',
            num_heun_step=num_heun_step[0],
            gan_num_heun_step=-1,
            diffusion_training_=diffusion_training_[0],
            gan_training_=False,
            target=target if target is not None else waveforms,
        )

        if split == 'val':
            self.diffusion.args.apply_adaptive_weight = orig_adaptive

        if 'consistency_loss' in losses:
            loss = self.cfg.diffusion.consistency_weight * losses['consistency_loss'].mean()
            if 'denoising_loss' in losses:
                loss = loss + self.cfg.diffusion.denoising_weight * losses['denoising_loss'].mean()
        elif 'denoising_loss' in losses:
            loss = losses['denoising_loss'].mean()

        for key, values in losses.items():
            self.log(f'{split}/{key} mean', values.mean().item(), sync_dist=True)
            self.log(f'{split}/{key} std', values.std().item(), sync_dist=True)

        return loss

    # ── Training / validation ─────────────────────────────────────────────────────

    def training_step(self, batch, _):
        waveforms, features, mixture_features_channels_list, separated_track_to_diffuse = self.get_input(batch)
        clean_target = waveforms
        x_start = self.apply_training_mixing(waveforms, separated_track_to_diffuse)
        return self.calculate_loss(x_start, features, mixture_features_channels_list, split='train', target=clean_target)

    def validation_step(self, batch, batch_idx):
        return

    def on_before_zero_grad(self, optimizer):
        target_ema, _ = self.ema_scale_fn(self.global_step)
        self._update_ema(self.target_model, self.net, target_ema)
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self._update_ema(ema_model, self.net, ema_rate)

    def _update_ema(self, ema_model, model, decay):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def configure_optimizers(self):
        cfg = self.cfg.optim
        if cfg.optimizer == 'radam':
            optimizer = optim.RAdam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr)
        return optimizer


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


class ClassCondSeparateTrackSampleLoggerCTM(ClassCondSeparateTrackSampleLogger):
    """
    Validation logger for Audio_CTM_Model (U-Net-based CD/CTM training).

    Inherits all metrics, multiprocessing museval, track-name-aware per-song
    logic, and msdm_style handling from ClassCondSeparateTrackSampleLogger.
    Only generate_sample is overridden, to sample via karras_sample using the
    CD model's onestep/multistep sampler instead of the diffusion model's
    plain model.sample().
    """

    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        stems: List[str],
        clip_output: bool = True,
        clip_denoised: bool = False,
        log_all: bool = False,
        model_to_calculate_metrics: str = "net",
        sampler_to_calculate_metrics: str = "onestep",
        steps_to_calculate_metrics: int = 1,
        sigma_max_sampling: Optional[float] = None,
        sigma_min_sampling: Optional[float] = None,
        rho_sampling: Optional[float] = None,
        run_museval: bool = True,
        msdm_style: bool = False,
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=steps_to_calculate_metrics,
            diffusion_schedule=None,
            diffusion_sampler=None,
            stems=stems,
            log_all=log_all,
            run_museval=run_museval,
            msdm_style=msdm_style,
        )
        self.clip_output = clip_output
        self.clip_denoised = clip_denoised
        self.model_to_calculate_metrics = model_to_calculate_metrics
        self.sampler_to_calculate_metrics = sampler_to_calculate_metrics
        self.steps_to_calculate_metrics = steps_to_calculate_metrics
        self.sigma_max_sampling = sigma_max_sampling
        self.sigma_min_sampling = sigma_min_sampling
        self.rho_sampling = rho_sampling

    @torch.no_grad()
    def generate_sample(self, trainer, pl_module, batch):
        model = getattr(pl_module, self.model_to_calculate_metrics)

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
            batch_size = waveforms.size(0)

            model_kwargs = {
                "features": current_features,
                "channels_list": channels_list,
                "embedding": embedding,
                "mixture_features_channels_list": mixture_features_channels_list,
            }

            # Sample from the model using the CD model's onestep/multistep sampler
            sample = karras_sample(
                diffusion=pl_module.diffusion,
                model=model,
                shape=(batch_size, self.channels, self.length),
                steps=self.steps_to_calculate_metrics,
                model_kwargs=model_kwargs,
                device=pl_module.device,
                clip_denoised=self.clip_denoised,
                sampler=self.sampler_to_calculate_metrics,
                generator=None,
                teacher=False,
                ctm=pl_module.cfg.diffusion.training_mode == "ctm",
                x_T=None,
                clip_output=self.clip_output,
                sigma_min=self.sigma_min_sampling if self.sigma_min_sampling is not None else pl_module.cfg.diffusion.sigma_min,
                sigma_max=self.sigma_max_sampling if self.sigma_max_sampling is not None else pl_module.cfg.diffusion.sigma_max,
                rho=self.rho_sampling if self.rho_sampling is not None else pl_module.cfg.diffusion.rho,
                train=False,
                progress=True,
            )
            sample = sample.clamp(-1.0, 1.0)
            samples = rearrange(sample, "b c t -> b t c").detach().cpu().numpy()

            # Store the generated samples
            for idx in range(batch_size):
                generated_samples[stem].append(samples[idx])

        # get original stems
        original_samples = {stem: [] for stem in self.stems}
        original_stems = batch[2]
        for i, stem in enumerate(self.stems):
            stem_data = original_stems[:, i]
            stem_data = rearrange(stem_data, "b c t -> b t c").detach().cpu().numpy()
            for idx in range(waveforms.size(0)):
                original_samples[stem].append(stem_data[idx])

        mixture_audios = batch[2].sum(1)[:, 0, :].detach().cpu().numpy()[..., np.newaxis]

        return original_samples, generated_samples, mixture_audios


class ClassCondSeparateTrackSampleLoggerCTM_MUSDB_MSST(ClassCondSeparateTrackSampleLogger_MUSDB_MSST_stems_in_out):
    """
    Validation logger for Audio_MSST_CTM_Model (BSRoformer-based CD/CTM training).

    Inherits all metrics, multiprocessing museval, limit_val_batches-aware
    on_validation_batch_start from ClassCondSeparateTrackSampleLogger_MUSDB_MSST_stems_in_out.
    Only generate_sample is overridden to use karras_sample instead of model.sample.
    """

    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        stems: List[str],
        clip_output: bool = True,
        clip_denoised: bool = False,
        log_deterministic: bool = False,
        log_all: bool = False,
        model_to_calculate_metrics: str = "net",
        sampler_to_calculate_metrics: str = "onestep",
        steps_to_calculate_metrics: int = 1,
        sigma_max_sampling: Optional[float] = None,
        sigma_min_sampling: Optional[float] = None,
        rho_sampling: Optional[float] = None,
        run_museval: bool = True,
    ) -> None:
        super().__init__(
            num_items=num_items,
            channels=channels,
            sampling_rate=sampling_rate,
            length=length,
            sampling_steps=steps_to_calculate_metrics,
            diffusion_schedule=None,
            diffusion_sampler=None,
            stems=stems,
            log_deterministic=log_deterministic,
            log_all=log_all,
            run_museval=run_museval,
        )
        self.clip_output = clip_output
        self.clip_denoised = clip_denoised
        self.model_to_calculate_metrics = model_to_calculate_metrics
        self.sampler_to_calculate_metrics = sampler_to_calculate_metrics
        self.steps_to_calculate_metrics = steps_to_calculate_metrics
        self.sigma_max_sampling = sigma_max_sampling
        self.sigma_min_sampling = sigma_min_sampling
        self.rho_sampling = rho_sampling

    @torch.no_grad()
    def generate_sample(self, trainer, pl_module, batch):
        waveforms, _, mixture_features_channels_list, separated_track_to_diffuse = pl_module.get_input(batch)
        batch_size = waveforms.shape[0]
        num_stems = len(self.stems)
        model = getattr(pl_module, self.model_to_calculate_metrics)

        model_kwargs = {
            'features': None,
            'channels_list': None,
            'embedding': None,
            'mixture_features_channels_list': list(mixture_features_channels_list),
        }

        with torch.cuda.amp.autocast(enabled=trainer.precision == "16-mixed"):
            sample = karras_sample(
                diffusion=pl_module.diffusion,
                model=model,
                shape=(batch_size, num_stems, self.channels, self.length),
                steps=self.steps_to_calculate_metrics,
                model_kwargs=model_kwargs,
                device=pl_module.device,
                clip_denoised=self.clip_denoised,
                sampler=self.sampler_to_calculate_metrics,
                progress=True,
                generator=None,
                teacher=False,
                ctm=pl_module.cfg.diffusion.training_mode == 'ctm',
                x_T=None,  # x_det is NOT needed here — it is already inside mixture_features_channels_list[-1]
                clip_output=self.clip_output,
                sigma_min=self.sigma_min_sampling if self.sigma_min_sampling is not None else pl_module.cfg.diffusion.sigma_min,
                sigma_max=self.sigma_max_sampling if self.sigma_max_sampling is not None else pl_module.cfg.diffusion.sigma_max,
                rho=self.rho_sampling if self.rho_sampling is not None else pl_module.cfg.diffusion.rho,
                train=False,
            )

        sample = sample.clamp(-1.0, 1.0)
        samples_np = rearrange(sample, "b s c t -> b s t c").detach().cpu().numpy()

        generated_samples = {stem: [] for stem in self.stems}
        for i, stem in enumerate(self.stems):
            for idx in range(batch_size):
                generated_samples[stem].append(samples_np[idx, i])

        original_samples = {stem: [] for stem in self.stems}
        for i, stem in enumerate(self.stems):
            stem_data = rearrange(waveforms[:, i], "b c t -> b t c").detach().cpu().numpy()
            for idx in range(batch_size):
                original_samples[stem].append(stem_data[idx])

        deterministic_samples = None
        if self.log_deterministic:
            deterministic_samples = {stem: [] for stem in self.stems}
            for i, stem in enumerate(self.stems):
                stem_data = rearrange(separated_track_to_diffuse[:, i], "b c t -> b t c").detach().cpu().numpy()
                for idx in range(batch_size):
                    deterministic_samples[stem].append(stem_data[idx])

        mixture_audios = rearrange(batch[1], "b c t -> b t c").detach().cpu().numpy()

        return original_samples, generated_samples, mixture_audios, deterministic_samples
