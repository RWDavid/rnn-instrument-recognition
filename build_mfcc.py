import os
import librosa
import numpy as np
from tqdm import tqdm

# sample rate of each audio file (librosa mixes each file to 22050 mono)
default_sr = 22050

# number of samples in each frame
frame_size = int(default_sr * 0.025)

# number of samples to progress between each iteration
frame_step = int(default_sr * 0.010)

# train/test data with corresponding labels
dataset = []

# classes and their corresponding labels
labels = {"clarinet": 0, "flute": 1, "trumpet": 2}
counts = {"clarinet": 0, "flute": 0, "trumpet": 0}

# determine whether train/test data should be generated
dataset_type = ""
while dataset_type not in ("train", "test"):
    dataset_type = input("What data should be generated? (\"train\", \"test\") ")

# go through all classes/folders in the selectory directory
for label in labels:
    directory = os.path.join(dataset_type, label)

    # iterate through each audio file in each class
    print("Processing " + label + " files:")
    for filename in tqdm(os.listdir(directory)):

        # construct audio file path
        path = os.path.join(directory, filename)

        # load audio samples
        samples, sample_rate = librosa.load(path, sr=default_sr)

        # split audio into audible phrases (to avoid silence)
        # note: lowering top_db leads to stricter spliting
        intervals = librosa.effects.split(samples, top_db=30, frame_length=5133, hop_length=2048)
        non_silent = []
        for begin, end in intervals:

            # skip phrases which are less than one second
            if end - begin < sample_rate:
                continue
            non_silent.append(samples[begin:end])

        # iterate through each audible phrase
        for phrase in non_silent:
            mfccs = librosa.feature.mfcc(phrase, sample_rate, win_length=frame_size, hop_length=frame_step, n_mfcc=26)
            mfccs = mfccs[:13]
            mean = np.mean(mfccs, axis=0)
            std = np.std(mfccs, axis=0)
            mfccs = (mfccs - mean)/std
            mfccs = mfccs.T

            # concatenate every 10 adjacent mfccs to form a sequence
            for i in range(0, len(mfccs) - 9, 10):
                sequence = [x for x in mfccs[i:i + 10]]
                dataset.append([np.array(sequence), np.eye(len(labels))[labels[label]]])
                counts[label] += 1

np.random.shuffle(dataset)
np.save(dataset_type + ".npy", dataset)
print("Data Distribution:")
for label in labels.keys():
    print(label + ": " + str(counts[label]))
