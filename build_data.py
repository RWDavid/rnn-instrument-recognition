import os
import librosa
import numpy as np
from tqdm import tqdm

default_sr = 22050

# number of samples in each frame
frame_size = int(default_sr / 4)

# number of samples to progress between each iteration
frame_step = int(default_sr / 10)

# the highest frequency that will be considered in the frequency spectrum
freq_top = 4000

# the number of bins to condense the frequency spectrum into
bins = 200

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
        samples, sample_rate = librosa.load(path)

        # split audio into audible phrases (to avoid silence)
        # note: lowering top_db leads to stricter spliting
        intervals = librosa.effects.split(samples, top_db=30, frame_length=frame_size, hop_length=frame_step)
        non_silent = []
        for begin, end in intervals:

            # skip phrases which are less than one second
            if end - begin < sample_rate:
                continue
            non_silent.append(samples[begin:end])

        # record frequency spectrums
        spectrums = []

        # iterate through each audible phrase
        for phrase in non_silent:

            # iterate through frames of the sample data
            for i in range(0, len(phrase) - frame_size, frame_step):

                # calculate and split frequency spectrum into bins
                x = int(np.ceil(freq_top * frame_size / sample_rate))
                x -= x % bins
                data = abs(np.fft.fft(phrase[i:i + frame_size])[:x])
                a = np.arange(len(data)) // (x // bins)
                data = np.bincount(a, data) / np.bincount(a)

                # normalize data to [0, 1]
                max_data = max(data)
                data /= max_data

                spectrums.append(np.array(data))

        # concatenate every 10 adjacent spectrums to form a sequence
        for i in range(0, len(spectrums) - 9, 10):
            sequence = [x for x in spectrums[i:i + 10]]
            dataset.append([np.array(sequence), np.eye(len(labels))[labels[label]]])
            counts[label] += 1

np.random.shuffle(dataset)
np.save(dataset_type + ".npy", dataset)
print("Data Distribution:")
for label in labels.keys():
    print(label + ": " + str(counts[label]))
