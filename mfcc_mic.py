import argparse
import queue
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import animation


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
net = GRUNet(13, 10, 4, 1).to(device)
net.load_state_dict(torch.load('gru.pt'))
net.eval()


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
args = parser.parse_args(remaining)

if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    args.samplerate = device_info['default_samplerate']


# number of samples in each frame
frame_size = int(args.samplerate * 0.025)

# number of samples to progress between each iteration
frame_step = int(args.samplerate * 0.010)

# length of sequence of mfccs
seq_len = 50

# receive user input
buffer_size = int(input("Enter buffer size: "))
buffer = np.array([-1 for x in range(buffer_size)])

# set up plot animation
labels = ["clarinet", "flute", "trumpet", "violin"]
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

# set up callback function for microphone stream
sample_queue = queue.Queue()
def audio_callback(indata, frames, time, status):

    # only obtain samples from the 0th channel
    sample_data = indata[:, 0]
    sample_queue.put(sample_data.flatten())


stream = sd.InputStream(device=args.device, channels=max(args.channels), samplerate=args.samplerate, callback=audio_callback)
stream.start()

current_samples = np.array([0.0 for x in range(frame_step*(seq_len-1) + frame_size)])
while True:

    # update plot
    plt.pause(.001)

    # read in sample data
    while True:
        try:
            data = sample_queue.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        current_samples = np.roll(current_samples, -shift, axis=0)
        current_samples[-shift:] = data


    # detect and skip silent audio
    rms = librosa.feature.rms(current_samples)[0]
    if min(rms) < 0.001:
        continue

    # extract mfccs from current audio excerpt
    mfccs = librosa.feature.mfcc(current_samples, args.samplerate, win_length=frame_size, hop_length=frame_step, n_mfcc=26)
    mfccs = mfccs[:13]
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)
    mfccs = (mfccs - mean) / std
    mfccs = mfccs.T[:seq_len]

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
