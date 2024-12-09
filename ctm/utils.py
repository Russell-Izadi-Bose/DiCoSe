import sys
import os

# Get the current directory of the file
current_dir = os.path.dirname(os.path.abspath(__file__))

# # # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# # Add the parent directory to the sys.path
sys.path.append(current_dir)

# Add the subdirectory to sys.path
# subdirectory = os.path.join(current_dir, 'ctm')
# sys.path.append(subdirectory)

import numpy as np
from networks import EDMPrecond_CTM, EDMPrecond_Audio_CTM
from resample import create_named_schedule_sampler
from karras_diffusion import KarrasDenoiser

class EMAAndScales_Initialiser:
    def __init__(self, target_ema_mode, start_ema, scale_mode, start_scales, end_scales, total_steps, distill_steps_per_iter):
        self.target_ema_mode = target_ema_mode
        self.start_ema = start_ema
        self.scale_mode = scale_mode
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.total_steps = total_steps
        self.distill_steps_per_iter = distill_steps_per_iter

    def get_ema_and_scales(self, step):
        if self.target_ema_mode == "fixed" and self.scale_mode == "fixed":
            target_ema = self.start_ema
            scales = self.start_scales
        elif self.target_ema_mode == "fixed" and self.scale_mode == "progressive":
            target_ema = self.start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / self.total_steps) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
                    + self.start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1
        elif self.target_ema_mode == "adaptive" and self.scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / self.total_steps) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
                    + self.start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(self.start_ema) * self.start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif self.target_ema_mode == "fixed" and self.scale_mode == "progdist":
            distill_stage = step // self.distill_steps_per_iter
            scales = self.start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - self.distill_steps_per_iter * (np.log2(self.start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (self.distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)



def create_model_and_diffusion(args, feature_extractor=None, discriminator_feature_extractor=None, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args.diffusion, args.diffusion.schedule_sampler, args.diffusion.start_scales)
    diffusion_schedule_sampler = create_named_schedule_sampler(args.diffusion, args.diffusion.diffusion_schedule_sampler, args.diffusion.start_scales)


    # This will work only for "cifar10" other part of the code is removed for simplicity!! 
    model = EDMPrecond_CTM(img_resolution=args.data.img_resolution, 
                           img_channels=args.data.img_channels,
                            label_dim= 10 if args.data.class_cond else 0, 
                            use_fp16= args.diffusion.use_fp16,
                            sigma_min=args.diffusion.sigma_min, 
                            sigma_max=args.diffusion.sigma_max,
                            sigma_data=args.diffusion.sigma_data,
                            model_type='SongUNet',
                            teacher=teacher, 
                            teacher_model_path=args.diffusion.teacher_model_path or args.diffusion.model_path,
                            training_mode=args.diffusion.training_mode, 
                            arch='ddpmpp',
                            linear_probing=args.diffusion.linear_probing
                            )

    diffusion = KarrasDenoiser(
        args=args.diffusion, 
        schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        feature_extractor=feature_extractor,
        discriminator_feature_extractor=discriminator_feature_extractor,
    )

    return model, diffusion


def create_model_and_diffusion_audio(args, feature_extractor=None, discriminator_feature_extractor=None, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args.diffusion, args.diffusion.schedule_sampler, args.diffusion.start_scales)
    diffusion_schedule_sampler = create_named_schedule_sampler(args.diffusion, args.diffusion.diffusion_schedule_sampler, args.diffusion.start_scales)


    # This will work only for "cifar10" other part of the code is removed for simplicity!! 
    model = EDMPrecond_Audio_CTM(
                            # img_resolution=args.data.img_resolution, 
                        #    img_channels=args.data.img_channels,
                            args.model,
                            label_dim= 4, 
                            use_fp16= args.diffusion.use_fp16,
                            sigma_min=args.diffusion.sigma_min, 
                            sigma_max=args.diffusion.sigma_max,
                            sigma_data=args.diffusion.sigma_data,
                            model_type='UNet1d',
                            teacher=teacher, 
                            teacher_model_path=args.diffusion.teacher_model_path or args.diffusion.model_path,
                            training_mode=args.diffusion.training_mode, 
                            arch='ddpmpp',
                            linear_probing=args.diffusion.linear_probing
                            )

    diffusion = KarrasDenoiser(
        args=args.diffusion, 
        schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        feature_extractor=feature_extractor,
        discriminator_feature_extractor=discriminator_feature_extractor,
    )

    return model, diffusion
