# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params


def transform(filename):
  audio, sr = T.load(filename)    # audio为二维的数据，[声道数，样本点数]
  audio = torch.clamp(audio[0], -1.0, 1.0)     # 将样本值限制到-1到1之间

  if params.sample_rate != sr:            # 如果样本采样率和代码预设的采样率不一致，则报错
    raise ValueError(f'Invalid sample rate {sr}.')
  mel_args = {                          # 梅尔谱的参数
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)   # 定义梅尔谱的transform

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)                # 得到样本的梅尔谱
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20   #零处理，取对数
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)  # 将梅尔谱的值限制在 [0,1] 之间
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())    # 将处理好的梅尔谱保存下来


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)   
  with ProcessPoolExecutor() as executor:     #  并行运行transform函数，使对每个wav文件的处理并行进行
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('--dir', default='/pubdata/zhongjiafeng/aishell3/train/wav_16k',
      help='directory containing .wav files for training')
  main(parser.parse_args())
