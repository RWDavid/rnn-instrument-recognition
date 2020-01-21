import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

def train(net, train_X, train_y, epochs, batch_size):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_X), batch_size)):
            batch_X = train_X[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}. Loss: {loss}")

def test(net, test_X, test_y):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            output = net(test_X[i].view(1, 10, 100))
            predicted_class = torch.argmax(output)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct/total, 3))

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

training_data = np.load("training_data.npy", allow_pickle=True)
X = torch.Tensor([i[0] for i in training_data])
y = torch.Tensor([i[1] for i in training_data])

test_size = int(len(X) * 0.1)
train_X = X[:-test_size]
train_y = y[:-test_size]
test_X = X[-test_size:]
test_y = y[-test_size:]

net = GRUNet(100, 64, 3, 1)
train(net, train_X, train_y, 50, 32)
test(net, test_X, test_y)