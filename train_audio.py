import wandb
import argparse
import yaml
import argparse
import importlib
import audio_diffusion_pytorch_

from main.module_base import Model, DatamoduleWithValidation, MultiSourceSampleLogger
from main.data import MultiSourceDataset
import pytorch_lightning as pl
import click
import os
import datetime
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint


@rank_zero_only
def create_directories(path):
    '''
    Makes sure to create directoris for only process 0 in multi-GPU scenarios
    '''
    os.makedirs(path, exist_ok=True)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# def instantiate_from_config(config):
#     # Get the module and class from the configuration
#     module_path, class_name = config._target_.rsplit('.', 1)
#     module = importlib.import_module(module_path)
#     cls = getattr(module, class_name)
    
#     # Remove _target_ from the config dictionary and instantiate the object
#     config_dict = vars(config)
#     config_dict.pop('_target_')
#     return cls(**config_dict)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/train_audiodm_conditional.yaml', help='Configuration File')
    args, unknown = parser.parse_known_args()

    cli_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            if '=' in unknown[i]:
                key, value = unknown[i].split('=', 1)
                key = key.lstrip("--")
            else:
                key = unknown[i].lstrip("--")
                if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    value = unknown[i + 1]
                    i += 1
                else:
                    value = True
            cli_args[key] = value
        i += 1

    return args.cfg, cli_args

def update_config_with_args(config, cli_args):
    for key, value in cli_args.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Convert value to appropriate type
        if isinstance(d[keys[-1]], bool):
            value = bool(value)
        elif isinstance(d[keys[-1]], int):
            value = int(value)
        elif isinstance(d[keys[-1]], float):
            value = float(value)
        d[keys[-1]] = value

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def instantiate_from_config(config, **kwargs):
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    
    module_path, class_name = config['_target_'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    # Remove _target_ from the config dictionary and instantiate the object
    config_dict = {k: v for k, v in config.items() if k != '_target_'}
    return cls(**config_dict, **kwargs)

import torch
class CheckGradientsCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        max_grad_norm = 2.0  # Upper threshold for detecting exploding gradients
        min_grad_norm = 1e-5  # Lower threshold for detecting vanishing gradients
        
        total_grad_norm = 0.0
        param_count = 0
        
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                print(f"Parameter {name} has no gradient.")
            else:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                if grad_norm >= max_grad_norm:
                    print(f"Parameter {name} gradient: {grad_norm:.6f} (Exploding)")
                # elif grad_norm <= min_grad_norm:
                #     print(f"Parameter {name} gradient: {grad_norm:.6f} (Vanishing)")
                # else:
                    # print(f"Parameter {name} gradient: {grad_norm:.6f}")
        
        if param_count > 0:
            avg_grad_norm = total_grad_norm / param_count
            print(f"Average gradient norm: {avg_grad_norm:.6f}")
        else:
            print("No gradients found for any parameters.")
        
        print()  # Separate outputs for readability

# @click.command()
# @click.option('--cfg', default='configs/train_audiodm_conditional.yaml', help='Configuration File')
def main():
    
    cfg_path, cli_args = parse_cli_args()
           
    config = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    # with open(cfg, "r") as f:
    #     cfg = yaml.safe_load(f)

    # Update the configuration with command-line arguments if provided
    update_config_with_args(config, cli_args)
       
    cfg = dict2namespace(config)

    log_path = cfg.log_directory
    create_directories(log_path)

    # adding a random number of seconds so that exp folder names coincide less often
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    nowname = "%s_%s_%s" % (
        now,
        cfg.id.name,
        cfg.id.version,
        # int(time.time())
    )

    print("\nName of the run is:", nowname, "\n")

    run_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
    )

    create_directories(run_path)

    # Flatten configuration for logging
    flattened_config = flatten_dict(config)
    
    wandb_logger = WandbLogger(
        save_dir=run_path,
        # version=nowname,
        project= cfg.project_name,
        config=flattened_config,
        name=nowname,
        entity='tornike_karchkha',
    )
    wandb_logger._project = ""  # prevent naming experiment nama 2 time in logginf vals

    checkpoint_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
        "checkpoints",
    )
    create_directories(checkpoint_path)

    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=2,
        monitor="msdm_si_snr_avg",
        mode="max",
        save_last=True,
        filename='{epoch}-{msdm_si_snr_avg:.4f}',
        every_n_train_steps=None
    )

    callbacks = [ckpt_callback]

    # # Init Model
    # diffusion_sigma_distribution = audio_diffusion_pytorch.LogNormalDistribution(**vars(cfg.diffusion_sigma_distribution))
    # model = Model(**vars(cfg.model), diffusion_sigma_distribution = diffusion_sigma_distribution)

    # Init Model
    diffusion_sigma_distribution = instantiate_from_config(cfg.diffusion_sigma_distribution)
    model = instantiate_from_config(cfg.model, diffusion_sigma_distribution = diffusion_sigma_distribution)

    # init datasets
    train_dataset = instantiate_from_config(cfg.train_dataset)
    validation_dataset = instantiate_from_config(cfg.val_dataset)

    # Instantiate the datamodule
    datamodule = DatamoduleWithValidation(train_dataset = train_dataset,
                                        val_dataset = validation_dataset,
                                            **vars(cfg.datamodule),
                                        )


    ### Init Sampler
    diffusion_sampler = instantiate_from_config(cfg.diffusion_sampler)
    diffusion_schedule = instantiate_from_config(cfg.diffusion_schedule)
    audio_samples_logger = instantiate_from_config(cfg.audio_samples_logger, diffusion_sampler = diffusion_sampler, diffusion_schedule = diffusion_schedule)

    callbacks.append(audio_samples_logger)

    # Initialize all callbacks (e.g. fancy modelsummary and progress bar)
    if "callbacks" in cfg:
        for _, cb_conf in vars(cfg.callbacks).items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate_from_config(cb_conf))
                
    # callbacks.append(CheckTrainingStateCallback())
    # callbacks.append(CheckGradientsCallback())

    # Initialize trainer
    trainer = pl.Trainer(**vars(cfg.trainer), callbacks=callbacks, logger=wandb_logger)

    # Start training
    if cfg.mode in ["test", "validate"]:
        # Evaluation / Validation
        trainer.validate(model, datamodule.val_dataloader(), ckpt_path = cfg.resume_from_checkpoint)
    elif cfg.mode == "train":
        trainer.fit(model, datamodule, ckpt_path = cfg.resume_from_checkpoint)



if __name__ == '__main__':
    main()