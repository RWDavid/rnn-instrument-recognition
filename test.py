labels = {"clarinet": 0, "flute": 1, "trumpet": 2}
counts = {"clarinet": 0, "flute": 0, "trumpet": 0}

for label in labels.keys():
    print(label + ": " + str(counts[label]))