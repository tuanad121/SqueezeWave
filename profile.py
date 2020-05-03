import torch
from scipy.io.wavfile import read
import numpy as np

from TacotronSTFT import TacotronSTFT

from timeit import default_timer as timer

## RTF is the real-time factor which tells how many seconds of speech are generated in 1 second of wall time

MAX_WAV_VALUE = 32768.0

n_audio_channel = 128

stft = TacotronSTFT(filter_length=1024,
                    hop_length=256,
                    win_length=1024,
                    sampling_rate=22050,
                    mel_fmin=0.0, mel_fmax=8000.0,
                    n_group=n_audio_channel)

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def get_mel(audio):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

squeezewave = torch.load('pretrain_models/L128_large_pretrain')['model']
for m in squeezewave.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')
squeezewave = squeezewave.remove_weightnorm(squeezewave)
squeezewave.cuda().eval()

is_fp16 = True
sigma=0.6

if is_fp16:
    from apex import amp
    squeezewave, _ = amp.initialize(squeezewave, [], opt_level="O3")

fin = open('test_files.txt')
arr = []
for line in fin:
    print(line)
    filename = line.strip()
    audio, sampling_rate = load_wav_to_torch(filename)
    assert sampling_rate==22050
    melspectrogram = get_mel(audio)
    # print('***', melspectrogram.shape)
    melspectrogram = torch.autograd.Variable(melspectrogram.cuda())
    # print('---', melspectrogram.shape)
    melspectrogram = torch.unsqueeze(melspectrogram, 0)
    # print('+++', melspectrogram.shape)
    melspectrogram = melspectrogram.half() if is_fp16 else melspectrogram
    st = timer ()
    with torch.no_grad():
        audio = squeezewave.infer(melspectrogram, sigma=sigma).float()
        # if denoiser_strength > 0:
        #     audio = denoiser(audio, denoiser_strength)
        audio = audio * MAX_WAV_VALUE
    # print('---', audio.shape)
    audio = audio.squeeze()
    # print('+++', audio.shape)
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    en = timer()
    print('RTF = ', (len(audio) / sampling_rate) / (en - st), len(audio))
    arr.append((len(audio) / sampling_rate) / (en - st))
print('average RTF = ', np.mean(arr))  # 144.54