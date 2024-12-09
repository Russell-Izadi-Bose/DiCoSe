"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from . import logger
import torch.distributed as dist

from nn import mean_flat, append_dims
# import cm.script_util as script_util
import blobfile as bf
import os
from torchvision.utils import make_grid, save_image
import torch



def save(x, save_dir, name, npz=False):
    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid((x + 1.) / 2., nrow, padding=2)
    with bf.BlobFile(os.path.join(save_dir, f"{name}.png"), "wb") as fout:
        save_image(image_grid, fout)
    if npz:
        sample = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().detach()
        os.makedirs(os.path.join(save_dir, 'targets'), exist_ok=True)
        np.savez(os.path.join(save_dir, f"targets/{name}.npz"), sample.numpy())


def get_weightings(weight_schedule, snrs, sigma_data, t, s, schedule_multiplier=None,):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    elif weight_schedule == "uniform_g":
        return 1./(1. - s / t) ** schedule_multiplier
    elif weight_schedule == "karras_weight":
        sigma = snrs ** -0.5
        weightings = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    elif weight_schedule == "karras_weight_s":
        sigma = snrs ** -0.5 #t #(t+s)*0.5
        weightings = (sigma * 2 + sigma_data * 2) / (sigma * sigma_data) ** 2
        weightings = weightings * schedule_multiplier
    elif weight_schedule == "sq-t-inverse":
        weightings = 1. / snrs ** 0.25
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        args,
        schedule_sampler,
        diffusion_schedule_sampler,
        feature_extractor=None,
        discriminator_feature_extractor=None,
    ):
        self.args = args
        self.schedule_sampler = schedule_sampler
        self.diffusion_schedule_sampler = diffusion_schedule_sampler
        self.feature_extractor = feature_extractor
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.num_timesteps = args.start_scales
        self.dist = nn.MSELoss(reduction='none')

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.args.sigma_data**2) ** 0.5

    def get_inner_scalings(self, t, inner_parametrization='no'):
        if inner_parametrization == 'edm':
            c_skip, c_out = self.get_edm_scalings(t)
        elif inner_parametrization == 'cm':
            c_skip, c_out = self.get_cm_scalings(t)
        elif inner_parametrization == 'no':
            c_skip, c_out = th.zeros_like(t), th.ones_like(t)
        return c_skip, c_out

    def get_outer_scalings(self, t, s=None, outer_parametrization='euler'):
        if outer_parametrization == 'euler':
            c_skip = s / t
        elif outer_parametrization == 'variance':
            c_skip = (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) / (
                        (t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt()
        elif outer_parametrization == 'euler_variance_mixed':
            c_skip = s / (t + 1.) + \
                     (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) /
                      ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt() / (t + 1.)
        c_out = (1. - s / t)
        return c_skip, c_out

    def get_edm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (sigma**2 + self.args.sigma_data**2)
        c_out = sigma * self.args.sigma_data / (sigma**2 + self.args.sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_cm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (
            (sigma - self.args.sigma_min) ** 2 + self.args.sigma_data**2
        )
        c_out = (
            (sigma - self.args.sigma_min)
            * self.args.sigma_data
            / (sigma**2 + self.args.sigma_data**2) ** 0.5
        )
        return c_skip, c_out

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def rescaling_t(self, t):
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        return rescaled_t

    def get_t(self, ind):
        if self.args.time_continuous:
            t = self.args.sigma_max ** (1 / self.args.rho) + ind * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        else:
            t = self.args.sigma_max ** (1 / self.args.rho) + ind / (self.args.start_scales - 1) * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        return t

    def get_num_heun_step(self, start_scales=-1, num_heun_step=-1, num_heun_step_random=None, heun_step_strategy='', time_continuous=None):
        if start_scales == -1:
            start_scales = self.args.start_scales
        if num_heun_step == -1:
            num_heun_step = self.args.num_heun_step
        if num_heun_step_random == None:
            num_heun_step_random = self.args.num_heun_step_random
        if heun_step_strategy == '':
            heun_step_strategy = self.args.heun_step_strategy
        if time_continuous == None:
            time_continuous = self.args.time_continuous
        if num_heun_step_random:
            if time_continuous:
                num_heun_step = np.random.rand() * num_heun_step / start_scales
            else:
                if heun_step_strategy == 'uniform':
                    num_heun_step = np.random.randint(1,1+num_heun_step)
                elif heun_step_strategy == 'weighted':
                    p = np.array([i ** self.args.heun_step_multiplier for i in range(1,1+num_heun_step)])
                    p = p / sum(p)
                    num_heun_step = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
        else:
            if time_continuous:
                num_heun_step = num_heun_step / start_scales
            else:
                num_heun_step = num_heun_step
        return num_heun_step

    def get_gan_time(self, x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step):
        if gan_num_heun_step != -1:
            indices, _ = self.schedule_sampler.sample_t(x_start.shape[0], x_start.device, gan_num_heun_step,
                                                            self.args.time_continuous)
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device,
                                                             indices,
                                                             gan_num_heun_step, self.args.time_continuous,
                                                             N=self.args.start_scales)
            t = self.get_t(indices)
            x_t = x_start + noise * append_dims(t, x_start.ndim)
            t_dt = self.get_t(indices + gan_num_heun_step)
            s = self.get_t(new_indices)
            num_heun_step = gan_num_heun_step
        return x_t, t, t_dt, s, indices, num_heun_step

    @th.no_grad()
    def heun_solver(self, target_model, x, ind, dims, t, t_dt, ctm=True, num_step=1, **model_kwargs):
        with th.no_grad():
            if self.args.self_learn:
                if self.args.self_learn_iterative:
                    for k in range(num_step):
                        t = self.get_t(ind + k)
                        t2 = self.get_t(ind + k + 1)
                        _, x = self.get_denoised_and_G(target_model, x, t, s=t2, ctm=ctm, **model_kwargs)
                else:
                    _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, **model_kwargs)
            else:
                for k in range(num_step):
                    t = self.get_t(ind + k)
                    denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=True, **model_kwargs)
                    d = (x - denoised) / append_dims(t, dims)
                    t2 = self.get_t(ind + k + 1)
                    x_phi_ODE_1st = x + d * append_dims(t2 - t, dims)
                    denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=True, **model_kwargs)
                    next_d = (x_phi_ODE_1st - denoised2) / append_dims(t2, dims)
                    x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
                    x = x_phi_ODE_2nd
            return x

    def get_ctm_estimate(self, x_t, t, t_dt, s, model, target_model, ctm, outer_type, inner_type, target_matching, **model_kwargs):
        if self.args.large_log:
            print("CTM estimate inner type, outer type, ctm: ", inner_type, outer_type, ctm)
        if target_matching:
            s = t_dt
        if inner_type == 'model':
            _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'target_model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(target_model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        else:
            raise NotImplementedError
        
        # if self.args.match_point == 'zs':
        #     return estimate
        # else:
        if self.args.training_mode == 'ctm':
            if outer_type == 'model':
                _, estimate = self.get_denoised_and_G(model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                                             ctm=ctm, **model_kwargs)
            elif outer_type == 'target_model_sg':
                _, estimate = self.get_denoised_and_G(target_model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                                             ctm=ctm, **model_kwargs)
            else:
                raise NotImplementedError
        return estimate

    @th.no_grad()
    def get_ctm_target(self, x_t_dt, t_dt, s, model, target_model, ctm, inner_type, **model_kwargs):
        if self.args.large_log:
            print("CTM target inner type, ctm: ", inner_type, ctm)
        with th.no_grad():
            if inner_type == 'model_sg':
                _, target = self.get_denoised_and_G(model, x_t_dt, t_dt, s=s, ctm=ctm, **model_kwargs)
            elif inner_type == 'target_model_sg':
                _, target = self.get_denoised_and_G(target_model, x_t_dt, t_dt, s=s, ctm=ctm, **model_kwargs)
            elif inner_type == 'no':
                target = x_t_dt
                s = t_dt
            else:
                raise NotImplementedError
            if self.args.training_mode == 'ctm':
                _, target = self.get_denoised_and_G(target_model, target, s, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, **model_kwargs)
            return target.detach()

    def get_denoised(self, g_theta, x_t ,t):
        if self.args.outer_parametrization.lower() == 'euler':
            denoised = g_theta
        elif self.args.outer_parametrization.lower() == 'variance':
            denoised = g_theta + append_dims((self.args.sigma_min ** 2 + self.args.sigma_data ** 2
                                              - self.args.sigma_min * t) / \
                                             ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2),
                                             x_t.ndim) * x_t
        elif self.args.outer_parametrization.lower() == 'euler_variance_mixed':
            denoised = g_theta + x_t - append_dims(t / (t + 1.) * (1. + (t - self.args.sigma_min) /
                                                                   ((
                                                                                t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)),
                                                   x_t.ndim) * x_t
        else:
            raise NotImplementedError
        return denoised

    def get_denoised_and_G(self, model, x_t, t, s=None, ctm=False, teacher=False, **model_kwargs):
        rescaled_t = self.rescaling_t(t)
        if s != None:
            rescaled_s = self.rescaling_t(s)
        else:
            rescaled_s = None
        c_in = append_dims(self.get_c_in(t), x_t.ndim)
        model_output = model(c_in * x_t, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)

        if ctm:
            if self.args.target_subtract:
                with th.no_grad():
                    teacher_denoised = self.teacher_model(c_in * x_t, rescaled_t, s=None, teacher=True, **model_kwargs)
                if self.args.rescaling:
                    model_output = model_output * append_dims((t ** 2 - s ** 2) ** 0.5 / t, x_t.ndim)
                model_output = model_output + teacher_denoised
            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_inner_scalings(t, self.args.inner_parametrization)
            ]
            g_theta = c_out * model_output + c_skip * x_t
            #z = th.randn_like(x_t)
            #x_t_ = x_t + 0.001 * z
            #model_output_ = model(c_in * x_t_, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)
            #g_theta_ = c_out * model_output_ + c_skip * x_t
            #np.savez(bf.join(logger.get_dir(), f"g/{np.random.randint(10000000)}.npz"),
            #         {'x_t': x_t, 't': t, 's': s, 'g': g_theta, 'g_': g_theta_, 'z': z})
            denoised = self.get_denoised(g_theta, x_t, t)
            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_outer_scalings(t, s, self.args.outer_parametrization)
            ]
            G_theta = c_out * g_theta + c_skip * x_t
        else:
            if teacher:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim) for x in self.get_edm_scalings(t)
                ]
            else:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_cm_scalings(t)
                ]
            denoised = c_out * model_output + c_skip * x_t
            G_theta = denoised
        return denoised, G_theta

    def get_CTM_loss(self, estimate, target, weights, step):
        if self.args.loss_norm == 'lpips':
            # print("Estimate Tensor:")
            # print("Min:", estimate.min().item())
            # print("Max:", estimate.max().item())
            # print("Mean:", estimate.mean().item())
            # print("Std:", estimate.std().item())

            # # Print statistics for `target`
            # print("Target Tensor:")
            # print("Min:", target.min().item())
            # print("Max:", target.max().item())
            # print("Mean:", target.mean().item())
            # print("Std:", target.std().item())

            if estimate.shape[-2] < 256:
                estimate = F.interpolate(estimate, size=224, mode="bilinear")
                target = F.interpolate(
                    target, size=224, mode="bilinear"
                )
            consistency_loss = (self.feature_extractor(
                (estimate + 1) / 2.0,
                (target + 1) / 2.0, ) * weights)
        elif self.args.loss_norm == 'mse':
            # print("Estimate Tensor:")
            # print("Min:", estimate.min().item())
            # print("Max:", estimate.max().item())
            # print("Mean:", estimate.mean().item())
            # print("Std:", estimate.std().item())

            # # Print statistics for `target`
            # print("Target Tensor:")
            # print("Min:", target.min().item())
            # print("Max:", target.max().item())
            # print("Mean:", target.mean().item())
            # print("Std:", target.std().item())
            # print("\n\n\n")
            # consistency_loss = self.feature_extractor((estimate + 1) / 2.0, (target + 1) / 2.0, reduction="none").mean(dim=[1, 2]) #* weights
            consistency_loss = self.feature_extractor(estimate, target, reduction="none").mean(dim=[1, 2])
            
            # Apply weights to the loss
            consistency_loss = consistency_loss * weights
                  
        else:
            raise NotImplementedError
        return consistency_loss

    def extract_last_layer(self, model):
        """ Function to extract the last layer and its name from a model """
        last_layer = None
        last_name = None
        for name, module in model.named_modules():
            last_layer = module
            last_name = name
        return last_layer

    def get_DSM_loss(self, model, x_start, model_kwargs, consistency_loss,
                           step, init_step):
        sigmas, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], device = x_start.device) #dist_util.dev())
        noise = th.randn_like(x_start)
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        if self.args.training_mode == 'ctm':
            denoised, _ = self.get_denoised_and_G(model, x_t, sigmas, s=sigmas, ctm=True, teacher=True, **model_kwargs)
        elif self.args.training_mode == 'cd':
            denoised, _ = self.get_denoised_and_G(model, x_t, sigmas, s=None, ctm=False, teacher=False, **model_kwargs)
        snrs = self.get_snr(sigmas)
        denoising_weights = append_dims(get_weightings(self.args.diffusion_weight_schedule, snrs, self.args.sigma_data, None, None), dims)
        denoising_loss = mean_flat(denoising_weights * (denoised - x_start) ** 2)
        if self.args.apply_adaptive_weight:
            last_layer = self.extract_last_layer(model)

            balance_weight = self.calculate_adaptive_weight(
                consistency_loss.mean(),
                denoising_loss.mean(),
                last_layer=last_layer.weight
            )
            
            # balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
            #                                                 last_layer=model.model.dec[
            #                                                     '32x32_aux_conv'].weight)
        else:
            balance_weight = 1.
        # if self.args.large_log:
        #     logger.log("denoising weight: ", balance_weight)
        balance_weight = self.adopt_weight(balance_weight, step, threshold=init_step, value=1.)
        denoising_loss = denoising_loss * balance_weight
        return denoising_loss

    def ctm_losses(
        self,
        step,
        model,
        x_start,
        model_kwargs=None,
        target_model=None,
        noise=None,
        discriminator=None,
        init_step=0,
        ctm=True,
        num_heun_step=-1,
        gan_num_heun_step=-1,
        diffusion_training_=False,
        gan_training_=False,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        s = None
        terms = {}
        if num_heun_step == -1:
            num_heun_step = [self.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
            num_heun_step = num_heun_step[0]
        indices, _ = self.schedule_sampler.sample_t(x_start.shape[0], x_start.device, num_heun_step,
                                                    self.args.time_continuous)
        t = self.get_t(indices)
        t_dt = self.get_t(indices + num_heun_step)
        if ctm:
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                         num_heun_step, self.args.time_continuous,
                                                         N=self.args.start_scales)
            s = self.get_t(new_indices)
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        ctm_estimate = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm=ctm,
                                             outer_type=self.args.ctm_estimate_outer_type,
                                             inner_type=self.args.ctm_estimate_inner_type,
                                             target_matching=self.args.ctm_target_matching,
                                             **model_kwargs)

        x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                    **model_kwargs).detach()
        ctm_target = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm=ctm,
                                            inner_type=self.args.ctm_target_inner_type, **model_kwargs)

        # if self.args.save_png and step % self.args.save_period == 0:
        #     save(ctm_estimate, logger.get_dir(), f'ctm_estimate_{step}')  # _{r}')
        #     save(ctm_target, logger.get_dir(), f'ctm_target_{step}')  # _{r}')
        #     save(x_t, logger.get_dir(), f'non_denoised_{step}')  # _{r}')
        #     save(x_t_dt, logger.get_dir(), f'denoised_{step}')  # _{r}')

        snrs = self.get_snr(t)
        weights = get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s, self.args.weight_schedule_multiplier)

        terms["consistency_loss"] = self.get_CTM_loss(ctm_estimate, ctm_target, weights, step - init_step,)

        if self.args.diffusion_training:
            if diffusion_training_:
                terms['denoising_loss'] = self.get_DSM_loss(model, x_start, model_kwargs,
                                                                    terms["consistency_loss"],
                                                                    step, init_step)

        return terms
