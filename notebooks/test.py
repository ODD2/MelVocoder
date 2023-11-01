import torch
import librosa
import torchaudio
raw, freq = librosa.load("dataset/audio/Soprano-3#你把我灌醉#0041.wav")
melTorch = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    n_mels=80,
    hop_length=256,
    win_length=1024,
    f_min=0,
    f_max=8000,
    pad=int((1024 - 256) / 2),
    center=False
).to("cpu")
spec = melTorch(torch.from_numpy(raw))
