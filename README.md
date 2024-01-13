# DiffWave

DiffWave is a fast, high-quality neural vocoder and waveform synthesizer. It starts with Gaussian noise and converts it into speech via iterative refinement. The speech can be controlled by providing a conditioning signal (e.g. log-scaled Mel spectrogram). The model and architecture details are described in [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf).

## Install

(First install pytorch)

From GitHub:
```
git clone https://github.com/dillfrescott/diffwave.git
cd diffwave
pip install .
```

### Training

```
python -m diffwave.preprocess /path/to/dir/containing/wavs
python -m diffwave /path/to/model/dir /path/to/dir/containing/wavs

# in another shell to monitor training progress:
tensorboard --logdir /path/to/model/dir --bind_all
```

You should expect to hear intelligible (but noisy) speech by ~8k steps (~1.5h on a 2080 Ti).

### Inference CLI
```
python -m diffwave.inference /path/to/model --spectrogram_path /path/to/spectrogram -o output.wav [--fast]
```
