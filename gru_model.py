import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # weight initialization
        for name in self.gru.named_parameters():
            if 'weight' in name[0]:
                torch.nn.init.xavier_uniform_(name[1])
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1]
        out = self.fc(F.relu(out))
        return out


def train(net, train_X, train_y, test_X, test_y, epochs, batch_size):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        for phase in ("train", "test"):
            if phase == "train":
                net.train()
                batch_count = 0
                running_loss = 0.0
                for i in range(0, len(train_X), batch_size):
                    batch_X = train_X[i:i + batch_size].to(device)
                    batch_y = train_y[i:i + batch_size].to(device)
                    net.zero_grad()
                    outputs = net(batch_X)
                    loss = loss_function(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    batch_count += 1
                print(f"Epoch: {epoch}. Loss: {loss}")
                train_loss_history.append(running_loss/batch_count)

            else:
                with torch.no_grad():
                    net.eval()
                    batch_count = 0
                    running_loss = 0.0
                    for i in range(0, len(test_X), batch_size):
                        batch_X = test_X[i:i + batch_size].to(device)
                        batch_y = test_y[i:i + batch_size].to(device)
                        outputs = net(batch_X)
                        loss = loss_function(outputs, batch_y)
                        running_loss += loss.item()
                        batch_count += 1
                    test_loss_history.append(running_loss/batch_count)

    return train_loss_history, test_loss_history


def test(net, test_X, test_y):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            output = net(test_X[i].view(1, 50, 13).to(device))
            predicted_class = torch.argmax(output)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct/total, 3))


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

training_data = np.load("train.npy", allow_pickle=True)
X = torch.Tensor([i[0] for i in training_data])
y = torch.Tensor([i[1] for i in training_data])

test_data = np.load("test.npy", allow_pickle=True)
test_X = torch.Tensor([i[0] for i in test_data])
test_y = torch.Tensor([i[1] for i in test_data])

net = GRUNet(13, 10, 7, 1).to(device)

train_loss, test_loss = train(net, X, y, test_X, test_y, 500, 1000)

test(net, X, y)
test(net, test_X, test_y)

plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['training loss', 'test loss'], loc='upper left')
plt.show()

torch.save(net.state_dict(), "gru.pt")
