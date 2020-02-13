import time
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import numpy as np

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1]
        out = self.fc(F.relu(out))
        return out


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

net = GRUNet(13, 10, 3, 1).to(device)
net.load_state_dict(torch.load('gru.pt'))
net.eval()

path = input("Enter path to audio: ")
samples, sample_rate = librosa.load(path)

# sample rate of each audio file (librosa mixes each file to 22050 mono)
default_sr = 22050

# number of samples in each frame
frame_size = int(default_sr * 0.025)

# number of samples to progress between each iteration
frame_step = int(default_sr * 0.010)

# length of sequence of mfccs
seq_len = 50

sd.play(samples, sample_rate)
tstart = time.time()
duration = len(samples) / sample_rate
predictions = {0:0, 1:0, 2:0}
predicted_class = 0

while time.time() - tstart < duration:
    tnow = time.time() - tstart
    current = int(tnow*sample_rate)

    rms = librosa.feature.rms(samples[current:current + (seq_len-1)*frame_step + frame_size])[0]
    if min(rms) < 0.01:
        continue

    mfccs = librosa.feature.mfcc(samples[current:current + (seq_len-1)*frame_step + frame_size], sample_rate, win_length=frame_size, hop_length=frame_step, n_mfcc=26)
    mfccs = mfccs[:13]
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)
    mfccs = (mfccs - mean) / std
    mfccs = mfccs.T[:seq_len]

    if len(mfccs) != 50:
        continue

    mfccs = torch.Tensor(mfccs).view(1, 50, 13)
    with torch.no_grad():
        output = net(mfccs.to(device))
        predicted_class = torch.argmax(output)
    predictions[predicted_class.item()] += 1
    print(predicted_class.item())

print(predictions)
