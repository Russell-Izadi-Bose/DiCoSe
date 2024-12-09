import wandb
import argparse
import yaml
import argparse
import importlib
import audio_diffusion_pytorch_

from main.module_base import Model, DatamoduleWithValidation, MultiSourceSampleLogger, ClassCondSeparateTrackSampleLogger, ClassCondSeparateTrackSampleLogger_simple
from main.data import MultiSourceDataset
import pytorch_lightning as pl
import click
import os
import datetime
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, nn
from audio_diffusion_pytorch_.modules import UNet1d
import torch.nn.functional as F
from einops import rearrange, reduce

from main.model_simple import Audio_DM_Model_simple


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

class CheckTrainingStateCallback(pl.Callback):
    def __init__(self):
        self.initial_teacher_params = None
        self.initial_net_params = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Save initial parameters on CPU if not already saved
        if self.initial_net_params is None:
            self.initial_net_params = {name: param.clone().detach().cpu() for name, param in pl_module.named_parameters()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        # Ensure net model is in training mode
        assert pl_module.training, "Net model should be in training mode"
        for name, param in pl_module.named_parameters():
            assert param.requires_grad, "Net model parameters should be trainable"
            if torch.equal(param.cpu(), self.initial_net_params[name]):
                print(f"Parameter {name} has not been updated.")
            else:
                print(f"Parameter {name} has been updated.")


        # Update initial parameters for the next step on CPU
        self.initial_net_params = {name: param.clone().detach().cpu() for name, param in pl_module.named_parameters()}



# class ClassCondSeparateTrackSampleLogger_simple(ClassCondSeparateTrackSampleLogger):
#     def __init__(
#         self,
#         num_items: int,
#         channels: int,
#         sampling_rate: int,
#         length: int,
#         # sampling_steps: List[int],
#         # diffusion_schedule: Schedule,
#         # diffusion_sampler: Sampler,
#         stems = ['bass', 'drums', 'guitar', 'piano']
#     ) -> None:
#         super().__init__(
#             num_items=num_items,
#             channels=channels,
#             sampling_rate=sampling_rate,
#             length=length,
#             sampling_steps=None,
#             diffusion_schedule=None,
#             diffusion_sampler=None,
#             stems = stems,
#         )

#     def generate_sample(self, trainer, pl_module, batch):
        
#         model = pl_module.model
        
#         # Extract mixture and original audio from the batch
#         waveforms, class_indexes, channels_list, embedding = pl_module.get_input(batch)


#         mixtures = batch[-1].sum(1)

#         # Get start diffusion noise for whole batch
#         noise = mixtures

#         # Dictionary to store generated samples
#         generated_samples = {stem: [] for stem in self.stems}
        
#         # Iterate over each one-hot encoded feature vector (each stem)
#         for i, stem in enumerate(self.stems):
#             # Create a feature tensor for the current stem for all items
#             current_features = torch.zeros(waveforms.size(0), len(self.stems)).to(pl_module.device)
#             current_features[:, i] = 1  # Set the current stem feature to 1 (one-hot)

#             # Sample from the model using the noise and the current one-hot features
#             samples = model.sample(
#                 noise=noise,
#                 features=current_features,
#                 sampler=None,
#                 sigma_schedule=None,
#                 num_steps=None,
#                 channels_list=channels_list
#             )
#             samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()

#             # Store the generated samples
#             for idx in range(waveforms.size(0)):
#                 # if steps not in generated_samples[stem]:
#                 #     generated_samples[stem][steps] = []
#                 generated_samples[stem].append(samples[idx])

#         # get original stems
#         original_samples = {stem: {} for stem in self.stems}
        
#         original_stems = batch[2]
        
#         for i, stem in enumerate(self.stems):
#             stem_data = original_stems[:, i]
#             stem_data =rearrange(stem_data, "b c t -> b t c") .detach().cpu().numpy()
#             original_samples[stem] = []
#             for idx in range(waveforms.size(0)):
#                 original_samples[stem].append(stem_data[idx])  
        
#         mixture_audios = batch[2].sum(1)[:, 0, :].detach().cpu().numpy()[..., np.newaxis] #channels_list[0][idx, 0, :].detach().cpu().numpy()[..., np.newaxis]
        
#         return  original_samples, generated_samples, mixture_audios


# class Model1d_simple(nn.Module):
#     def __init__(
#         self,
#         **kwargs
#     ):
#         super().__init__()

#         self.unet = UNet1d(**kwargs)

#     def forward(self, x: Tensor, **kwargs) -> Tensor:
#         return self.unet(x, **kwargs)

#     def sample(
#         self,
#         noise: Tensor,
#         num_steps: int,
#         sigma_schedule,
#         sampler,
#         **kwargs
#     ) -> Tensor:

#         return self.unet(noise, **kwargs)


# class Audio_DM_Model_simple(pl.LightningModule):
#     def __init__(
#         self, learning_rate: float, beta1: float, beta2: float, class_cond: bool, separation: bool = False, *args, **kwargs
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2

#         self.class_cond = class_cond
#         self.separation = separation

#         self.model = Model1d_simple(*args, **kwargs)

#     @property
#     def device(self):
#         return next(self.model.parameters()).device

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             list(self.parameters()),
#             lr=self.learning_rate,
#             betas=(self.beta1, self.beta2),
#         )
#         return optimizer

#     def get_input(self, batch):

#         if isinstance(batch, (list, tuple)) and self.class_cond and self.separation:
#             waveforms, class_indexes, stems  = batch

#             batch_size, channels, feature_width = waveforms.shape
#             mixture = stems.sum(1)

#             # Desired output sizes for each layer
#             target_sizes = [262144, 4096, 1024, 256, 128, 64, 32]  # TODO: this needs to be caclulated automaticaly somehow

#             # Create downscaled versions of waveforms using interpolation
#             channels_list = []
#             for size in target_sizes:
#                 if feature_width == size:
#                     # No need to resize if the current size matches the target
#                     channels_list.append(mixture)
#                 else:
#                     # Resize waveform to the target size
#                     resized_mixture = F.interpolate(mixture, size=(size,), mode='linear', align_corners=False)
#                     channels_list.append(resized_mixture)
#                     # feature_width = size  # Update current length for the next iteration


#             # Adjust channel dimensions to match `context_channels`
#             # This involves expanding the channel dimension after downsampling
#             channels_list = [
#                 torch.cat([channels_list[i]] * num, dim=1) if num != channels_list[i].shape[1]
#                 else channels_list[i]
#                 for i, num in enumerate([1, 512, 1024, 1024, 1024, 1024, 1024])
#             ]

#             # embedding = torch.randn(2, 4, 32).to(self.device)
#             embedding = None

            
#         elif isinstance(batch, (list, tuple)) and self.class_cond:
#             waveforms, class_indexes, _ = batch
#             channels_list = None
#             embedding = None
#         elif isinstance(batch, (list, tuple)) :
#             waveforms, _, _= batch
#             class_indexes = None
#             channels_list = None  
#             embedding = None          
#         else:
#             waveforms = batch
#             class_indexes = None
#             channels_list = None
#             embedding = None
#         return waveforms, class_indexes, channels_list, embedding

#     def training_step(self, batch, batch_idx):
#         waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
#         mixtures = batch[-1].sum(1)
#         predictions = self.model(mixtures, features = class_indexes, channels_list=channels_list, embedding = embedding)

#         # print("predictions Tensor:")
#         # print("Min:", predictions.min().item())
#         # print("Max:", predictions.max().item())
#         # print("Mean:", predictions.mean().item())
#         # print("Std:", predictions.std().item())

#         # # Print statistics for `target`
#         # print("Target Tensor:")
#         # print("Min:", waveforms.min().item())
#         # print("Max:", waveforms.max().item())
#         # print("Mean:", waveforms.mean().item())
#         # print("Std:", waveforms.std().item())
#         # print("\n\n\n")


#         # Compute weighted loss
#         losses = F.mse_loss(predictions, waveforms, reduction="none")
#         losses = reduce(losses, "b ... -> b", "mean")
#         loss = losses.mean()

#         self.log("train_loss", loss, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         waveforms, class_indexes, channels_list, embedding = self.get_input(batch)
#         mixtures = batch[-1].sum(1)
#         predictions = self.model(mixtures, features = class_indexes, channels_list=channels_list, embedding = embedding)

#         # Compute weighted loss
#         losses = F.mse_loss(predictions, waveforms, reduction="none")
#         losses = reduce(losses, "b ... -> b", "mean")
#         loss = losses.mean()
#         self.log("valid_loss", loss, sync_dist=True)
#         return loss



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
        # entity='tornike_karchkha',
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

    # Init Model
    model = instantiate_from_config(cfg.model)

    # init datasets
    train_dataset = instantiate_from_config(cfg.train_dataset)
    validation_dataset = instantiate_from_config(cfg.val_dataset)

    # Instantiate the datamodule
    datamodule = DatamoduleWithValidation(train_dataset = train_dataset,
                                        val_dataset = validation_dataset,
                                            **vars(cfg.datamodule),
                                        )


    ### Init Sampler
    audio_samples_logger = instantiate_from_config(cfg.audio_samples_logger)

    callbacks.append(audio_samples_logger)

    # Initialize all callbacks (e.g. fancy modelsummary and progress bar)
    if "callbacks" in cfg:
        for _, cb_conf in vars(cfg.callbacks).items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate_from_config(cb_conf))

    # callbacks.append(CheckTrainingStateCallback())

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