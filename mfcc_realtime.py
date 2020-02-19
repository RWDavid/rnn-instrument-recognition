import time
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import animation
import warnings
warnings.simplefilter("ignore")


# define neural network model
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


# import model
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
net = GRUNet(13, 10, 7, 1).to(device)
net.load_state_dict(torch.load('gru.pt', map_location=device))
net.eval()

# receive user input
path = input("Enter path to audio: ")
samples, sample_rate = librosa.load(path)
buffer_size = int(input("Enter buffer size: "))
buffer = np.array([-1 for x in range(buffer_size)])

# sample rate of each audio file (librosa mixes each file to 22050 mono)
default_sr = 22050

# number of samples in each frame
frame_size = int(default_sr * 0.025)

# number of samples to progress between each iteration
frame_step = int(default_sr * 0.010)

# length of sequence of mfccs
seq_len = 50

# play audio file
sd.play(samples, sample_rate)
tstart = time.time()
duration = len(samples) / sample_rate

# set up plot animation
labels = ["clarinet", "flute", "trumpet", "trombone", "violin", "guitar", "piano"]
fig = plt.figure()
ax = plt.gca()
ax.set_title('Instrument Prediction')
ax.set_xlabel('Instrument')
ax.set_ylabel('Confidence')
bars = plt.bar(labels, [0 for x in range(len(labels))])
bar_heights = [0 for x in range(len(labels))]
plt.ylim(0, buffer_size)

def animate(frame):
    for i, b in enumerate(bars):
        b.set_height(bar_heights[i])
    return bars

anim = animation.FuncAnimation(fig, animate, repeat=False, blit=True, interval=1000//60)
plt.show(block=False)

while time.time() - tstart < duration:

    # update plot
    plt.pause(.001)

    # obtain current sample index based on time
    tnow = time.time() - tstart
    current = int(tnow*sample_rate)

    # detect and skip silent audio
    rms = librosa.feature.rms(samples[current:current + (seq_len-1)*frame_step + frame_size])[0]
    if min(rms) < 0.01:
        continue

    # extract mfccs from current audio excerpt
    mfccs = librosa.feature.mfcc(samples[current:current + (seq_len-1)*frame_step + frame_size], sample_rate, n_fft=frame_size, hop_length=frame_step, n_mfcc=26)
    mfccs = mfccs[:13]
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)
    mfccs = (mfccs - mean) / std
    mfccs = mfccs.T[:seq_len]

    # skip this audio excerpt if there is insufficient data
    if len(mfccs) != 50:
        continue

    # send data to model for prediction
    mfccs = torch.Tensor(mfccs).view(1, 50, 13)
    with torch.no_grad():
        output = net(mfccs.to(device))
        predicted_class = torch.argmax(output)

    # update prediction buffer
    buffer = np.roll(buffer, 1)
    buffer[0] = predicted_class.item()

    # obtain occurrences of each prediction
    c = Counter(buffer)
    for i in range(len(labels)):
        bar_heights[i] = c[i]
