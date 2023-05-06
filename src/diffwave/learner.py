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
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import from_path, from_gtzan
from model import DiffWave
from params import AttrDict


def _nested_map(struct, map_fn):   # map_fn是一个lambda函数（匿名函数）
  if isinstance(struct, tuple):   # 如果struct是一个元组，对元组里的每个元素调用map_fn函数，返回元组结构
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):      # 如果sturct是一个列表，对列表中的每个元素调用map_fn函数，并返回列表结构
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):       #  如果sturct是一个字典，对字典中的所有值使用map_fn函数，返回字典
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))   # 混合精度训练
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))   # 混合精度训练
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)  # 从0.0001到0.05等间隔分布的50个数  
    noise_level = np.cumprod(1 - beta)           # 1-beta的累乘,对应于论文中的alpha
    self.noise_level = torch.tensor(noise_level.astype(np.float32))  
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    """
      分别装载model,optimizer,sacler,step。
    """
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])          
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device   # 待优化参数所在的设备位置
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:   # 如果是主进程，则显示进度条
        if max_steps is not None and self.step >= max_steps:   # 训练步数达到最大步数，直接退出循环
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x) # 将feature中所有tensor数据移动到device上
        loss = self.train_step(features)    # 训练一次
        if torch.isnan(loss).any():         # 检查loss中是否有Nan的情况出现
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:  
            self._write_summary(self.step, features, loss)    # 每隔50个step记录一次
          if self.step % len(self.dataset) == 0:                
            self.save_to_checkpoint()              # 每隔样本总量个step保存一次模型
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():      # 将模型的参数的梯度手动置0，其效果应该与optimize.zero_grad()效果一致
      param.grad = None

    audio = features['audio']                   # waveform 数据 
    spectrogram = features['spectrogram']       # mel谱 数据

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)    # 随机选取一个t
      noise_scale = self.noise_level[t].unsqueeze(1)      # noise_scale为[1,1]的tensor
      noise_scale_sqrt = noise_scale**0.5       
      noise = torch.randn_like(audio)              #  随机生成一个与audio一样的均值为0，方差为1的正态随机噪声noise
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise   # diffusion process 得到最终的加噪图片xT，一步到位

      predicted = self.model(noisy_audio, t, spectrogram)   # 输入最终的加噪图片xT，和t，以及条件melspectrogram
      loss = self.loss_fn(noise, predicted.squeeze(1))     # 计算得到的预测噪声和的所加噪声之间的L1loss

    self.scaler.scale(loss).backward()     # 利用混合精度训练的sacle对loss进行缩放处理再回传
    self.scaler.unscale_(self.optimizer)    # 将优化器的梯度张量除以比例因子
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)   # 利用梯度裁剪解决梯度爆炸的问题
    self.scaler.step(self.optimizer)     # 优化参数
    self.scaler.update()                  # 更新GradScaler的缩放因子
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer

 
def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True   # 引入CUDA算法的不确定性
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)   # 将多种参数封装在一个类中
  learner.is_master = (replica_id == 0)      # 将主进程的is_master参数设置为True，方便输出信息
  learner.restore_from_checkpoint()          # 装载checkpoint
  learner.train(max_steps=args.max_steps)    # 训练


def train(args, params):
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)   # 获取Dataloader
  model = DiffWave(params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)
