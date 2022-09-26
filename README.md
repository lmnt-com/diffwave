# DiffWave
![PyPI Release](https://img.shields.io/pypi/v/diffwave?label=release) [![License](https://img.shields.io/github/license/lmnt-com/diffwave)](https://github.com/lmnt-com/diffwave/blob/master/LICENSE)

**We're hiring!**
If you like what we're building here, [come join us at LMNT](https://explore.lmnt.com).

DiffWave is a fast, high-quality neural vocoder and waveform synthesizer. It starts with Gaussian noise and converts it into speech via iterative refinement. The speech can be controlled by providing a conditioning signal (e.g. log-scaled Mel spectrogram). The model and architecture details are described in [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf).

## What's new (2021-11-09)
- unconditional waveform synthesis (thanks to [Andrechang](https://github.com/Andrechang)!)

## What's new (2021-04-01)
- fast sampling algorithm based on v3 of the DiffWave paper

## What's new (2020-10-14)
- new pretrained model trained for 1M steps
- updated audio samples with output from new model

## Status (2021-11-09)
- [x] fast inference procedure
- [x] stable training
- [x] high-quality synthesis
- [x] mixed-precision training
- [x] multi-GPU training
- [x] command-line inference
- [x] programmatic inference API
- [x] PyPI package
- [x] audio samples
- [x] pretrained models
- [x] unconditional waveform synthesis

Big thanks to [Zhifeng Kong](https://github.com/FengNiMa) (lead author of DiffWave) for pointers and bug fixes.

## Audio samples
[22.05 kHz audio samples](https://lmnt.com/assets/diffwave)

## Pretrained models
[22.05 kHz pretrained model](https://lmnt.com/assets/diffwave/diffwave-ljspeech-22kHz-1000578.pt) (31 MB, SHA256: `d415d2117bb0bba3999afabdd67ed11d9e43400af26193a451d112e2560821a8`)

This pre-trained model is able to synthesize speech with a real-time factor of 0.87 (smaller is faster).

### Pre-trained model details
- trained on 4x 1080Ti
- default parameters
- single precision floating point (FP32)
- trained on LJSpeech dataset excluding LJ001&ast; and LJ002&ast;
- trained for 1000578 steps (1273 epochs)

## Install

Install using pip:
```
pip install diffwave
```

or from GitHub:
```
git clone https://github.com/lmnt-com/diffwave.git
cd diffwave
pip install .
```

### Training
Before you start training, you'll need to prepare a training dataset. The dataset can have any directory structure as long as the contained .wav files are 16-bit mono (e.g. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [VCTK](https://pytorch.org/audio/_modules/torchaudio/datasets/vctk.html)). By default, this implementation assumes a sample rate of 22.05 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

```
python -m diffwave.preprocess /path/to/dir/containing/wavs
python -m diffwave /path/to/model/dir /path/to/dir/containing/wavs

# in another shell to monitor training progress:
tensorboard --logdir /path/to/model/dir --bind_all
```

You should expect to hear intelligible (but noisy) speech by ~8k steps (~1.5h on a 2080 Ti).

#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

### Inference API
Basic usage:

```python
from diffwave.inference import predict as diffwave_predict

model_dir = '/path/to/model/dir'
spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(spectrogram, model_dir, fast_sampling=True)

# audio is a GPU tensor in [N,T] format.
```

### Inference CLI
```
python -m diffwave.inference --fast /path/to/model /path/to/spectrogram -o output.wav
```

## References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
