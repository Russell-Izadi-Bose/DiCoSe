import argparse
import importlib

from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os


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
        current = d.get(keys[-1])
        if isinstance(current, bool):
            value = str(value).lower() in ('true', '1', 'yes')
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list) and isinstance(value, str):
            # e.g. --trainer.devices=0 or --trainer.devices=0,1,2
            value = [int(v) for v in value.strip('[]').split(',') if v != '']
        elif current is None and isinstance(value, str):
            # no existing value to match a type against (e.g. a yaml key left blank) - infer one
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
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


def resolve_target_class(config):
    '''
    Resolve and return the class named by a config's `_target_` field
    (e.g. "main.audio_ctm.Audio_MSST_CTM_Model"), removing that field from
    the config in place. Some classes (e.g. CD/CTM models) are handed the
    whole cfg object and read fields out of it well after construction, so
    _target_ must not linger as a leftover attribute on the source object -
    not just be excluded from a filtered copy.
    '''
    target = vars(config).pop('_target_') if isinstance(config, argparse.Namespace) else config.pop('_target_')
    module_path, class_name = target.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_from_config(config, **kwargs):
    cls = resolve_target_class(config)
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    return cls(**config, **kwargs)
