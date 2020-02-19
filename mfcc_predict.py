import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

path = input("Enter path to audio: ")
samples, sample_rate = librosa.load(path)

# number of samples in each frame
frame_size = int(sample_rate * 0.025)

# number of samples to progress between each iteration
frame_step = int(sample_rate * 0.010)

# length of sequence of mfccs
seq_len = 50

# count predictions for each label
labels = ["clarinet", "flute", "trumpet", "trombone", "violin", "guitar", "piano"]
counts = [0 for x in range(len(labels))]

# split audio into audible phrases (to avoid silence)
# note: lowering top_db leads to stricter spliting
intervals = librosa.effects.split(samples, top_db=25, frame_length=5133, hop_length=2048)
non_silent = []
for begin, end in intervals:

    # skip phrases which are less than one second
    if end - begin < sample_rate:
        continue
    non_silent.append(samples[begin:end])

# iterate through each audible phrase
for phrase in non_silent:
    mfccs = librosa.feature.mfcc(phrase, sample_rate, n_fft=frame_size, hop_length=frame_step, n_mfcc=26)
    mfccs = mfccs[:13]
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)
    mfccs = (mfccs - mean)/std
    mfccs = mfccs.T

    # concatenate every 50 adjacent mfccs to form a sequence
    for i in range(0, len(mfccs) - seq_len + 1, seq_len):
        sequence = [x for x in mfccs[i:i + seq_len]]
        sequence = torch.Tensor(sequence).view(1, 50, 13)
        with torch.no_grad():
            output = net(sequence.to(device))
            predicted_class = torch.argmax(output).item()
        counts[predicted_class] += 1

guess = np.argmax(np.array(counts))
print("Prediction: " + labels[guess])
print("Ratio: " + str(counts[guess] / sum(counts)))

print("\nAll instrument predictions (with ratios):")
for guess, instrument in enumerate(labels):
    print(instrument, counts[guess] / sum(counts))
