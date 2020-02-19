import os
import librosa
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

# sample rate of each audio file (librosa mixes each file to 22050 mono)
default_sr = 22050

# number of samples in each frame
frame_size = int(default_sr * 0.025)

# number of samples to progress between each iteration
frame_step = int(default_sr * 0.010)

# length of sequence of mfccs
seq_len = 50

# classes and their corresponding labels
labels = {"clarinet": 0, "flute": 1, "trumpet": 2, "trombone": 3, "violin": 4, "guitar": 5, "piano": 6}
counts = {"clarinet": 0, "flute": 0, "trumpet": 0, "trombone": 0, "violin": 0, "guitar": 0, "piano": 0}

# train/test data with corresponding labels
dataset = [[] for x in range(len(labels))]

# determine whether train/test data should be generated
dataset_type = ""
while dataset_type not in ("train", "test"):
    dataset_type = input("What data should be generated? (\"train\", \"test\") ")

# go through all classes/folders in the selectory directory
for label_num, label in enumerate(labels):
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
                dataset[label_num].append([np.array(sequence), np.eye(len(labels))[labels[label]]])
                counts[label] += 1

# randomize data examples
for x in dataset:
    np.random.shuffle(x)

# ensure that the distribution of each instrument is equal
min_examples = 999999999999
for x in dataset:
    min_examples = min(min_examples, len(x))
for x in range(len(dataset)):
    dataset[x] = dataset[x][:min_examples]

# combine different instrumental examples into a single dataset
full_data = []
for x in dataset:
    full_data.extend(x)
np.random.shuffle(full_data)

# print distribution of dataset
print("Original Data Distribution:")
for label in labels.keys():
    print(label + ": " + str(counts[label]))

print("\nAltered Data Distribution:")
for label_num, label in enumerate(labels):
    print(label + ": " + str(len(dataset[label_num])))

# save dataset
np.save(dataset_type + ".npy", full_data)
