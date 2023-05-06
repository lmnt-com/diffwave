import librosa
import soundfile as sf
import os 
from glob import glob
from tqdm import tqdm

if __name__  == '__main__':
    dir = '/pubdata/zhongjiafeng/aishell3/train/wav'
    save_dir = '/pubdata/zhongjiafeng/aishell3/train/wav_16k/'

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    filenames = glob(f'{dir}/**/*.wav', recursive=True)  

    for path in tqdm(filenames):
        y, sr = librosa.load(path,sr=44100)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)

        name = path.split('/')[-1]
        sample_save = save_dir + name
        sf.write(sample_save, y_resampled, 16000, format='wav')
    print(len(filenames))