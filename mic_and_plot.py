#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation


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
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()

# GRU MODEL
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

net = GRUNet(200, 128, 3, 1).to(device)
net.load_state_dict(torch.load('gru.pt'))
net.eval()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global current_samples

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        current_samples = np.roll(current_samples, -shift, axis=0)
        current_samples[-shift:, :] = data

    spectrums = []
    for i in range(0, len(current_samples) - frame_size + 1, frame_step):

        # calculate and split frequency spectrum into bins
        x = int(np.ceil(freq_top * frame_size / args.samplerate))
        x -= x % bins

        data = np.zeros(x)
        for c in range(len(args.channels)):
            data += abs(np.fft.fft(current_samples[i:i + frame_size][c])[:x])

        a = np.arange(len(data)) // (x // bins)
        data = np.bincount(a, data) / np.bincount(a)

        # normalize data to [0, 1]
        max_data = max(data)
        if (max_data != 0):
            data /= max_data

        spectrums.append(np.array(data))

    spectrums = torch.Tensor(spectrums)
    with torch.no_grad():
        output = net(spectrums.view(1, 10, 200).to(device))
        predicted_class = torch.argmax(output)

    prediction = np.eye(3)[predicted_class]

    for column, line in enumerate(lines):
        line.set_ydata(prediction)
    return lines

def read_next_samples():
    global current_samples
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        current_samples = np.roll(current_samples, -shift, axis=0)
        current_samples[-shift:, :] = data

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    frame_size = int(args.samplerate / 4)
    frame_step = int(args.samplerate / 10)
    freq_top = 4000
    bins = 200

    current_prediction = np.array([0, 0, 0])
    fig, ax = plt.subplots()
    lines = ax.plot(current_prediction)
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(current_prediction), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    length = frame_step * 9 + frame_size
    current_samples = np.zeros((length, len(args.channels)))

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
