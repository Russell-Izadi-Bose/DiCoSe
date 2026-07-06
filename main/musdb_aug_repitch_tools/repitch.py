# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility for on the fly pitch/tempo change for data augmentation."""

import random
import subprocess as sp
import tempfile

import torch
import torchaudio as ta
from pathlib import Path
import typing as tp


      
def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav

def save_audio(wav: torch.Tensor,
               path: tp.Union[str, Path],
               samplerate: int,
               bitrate: int = 320,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False,
               preset: tp.Literal[2, 3, 4, 5, 6, 7] = 2):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`. Use `preset` to set mp3 quality:
    2 for highest quality, 7 for fastest speed
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    elif suffix == ".flac":
        ta.save(str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")


class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset, proba=0.2, max_pitch=2, max_tempo=12,
                 tempo_std=5, vocals=[3], same=True):
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.tempo_std = tempo_std
        self.same = same
        self.vocals = vocals

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int(in_length / (1 + 0.01 * self.max_tempo)+1) #int((1 - 0.01 * self.max_tempo) * in_length)

        if random.random() < self.proba:
            outs = []
            for idx, stream in enumerate(streams):
                if idx == 0 or not self.same:
                    delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
                    delta_tempo = random.gauss(0, self.tempo_std)
                    delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)
                stream = repitch(
                    stream,
                    delta_pitch,
                    delta_tempo,
                    voice=idx in self.vocals)
                outs.append(stream[:, :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams


def repitch(wav, pitch, tempo, voice=False, quick=False, samplerate=44100):
    """
    tempo is a relative delta in percentage, so tempo=10 means tempo at 110%!
    pitch is in semi tones.
    Requires `soundstretch` to be installed, see
    https://www.surina.net/soundtouch/soundstretch.html
    """
    infile = tempfile.NamedTemporaryFile(suffix=".wav")
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(wav, infile.name, samplerate, clip='clamp')
    command = [
        "soundstretch",
        # "/data/reach/karchkhadze/anaconda3/envs/ctm/bin/soundstretch",
        infile.name,
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr = ta.load(outfile.name)
    assert sr == samplerate
    return wav
