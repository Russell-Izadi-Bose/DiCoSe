import yaml
import os
import datetime
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from main.cli_utils import (
    create_directories,
    dict2namespace,
    parse_cli_args,
    update_config_with_args,
    flatten_dict,
    instantiate_from_config,
)
from main.module_base import DatamoduleWithValidation, ClassCondSeparateTrackSampleLogger_simple, ClassCondSeparateTrackSampleLogger_simple_MSST
from main.model_simple import Audio_DM_Model_simple, Audio_MSST_Model_simple


def main():

    cfg_path, cli_args = parse_cli_args()

    config = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

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
        project= cfg.project_name,
        config=flattened_config,
        name=nowname,
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
        monitor="msdm_si_sdr_avg",
        mode="max",
        save_last=True,
        filename='{epoch}-{msdm_si_sdr_avg:.4f}',
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
