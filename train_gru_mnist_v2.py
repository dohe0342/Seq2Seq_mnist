"""
Example code of a simple RNN, GRU, LSTM on the MNIST dataset.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-09 Initial coding

"""

# Imports
import torch
import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!

import random
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_length = 28
learning_rate = 0.000001
batch_size = 64
num_epochs = 20000

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class GRU_ENC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU_ENC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, h):
        out, h_out = self.gru(x, h)

        return out, h_out

    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class GRU_DEC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU_DEC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_classes, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
        
        print(self.gru)

    def forward(self, x, h):
        input = self.emb(x).view(batch_size, 1, -1)
        out, h_out = self.gru(input, h)

        # Decode the hidden state of the last time step
        out = self.softmax(self.fc(out.reshape(out.shape[0], -1)))
        return out, h_out

    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


# Load Data
train_dataset = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
#model = RNN_GRU(input_size, hidden_size, num_layers, num_classes).to(device)
enc = GRU_ENC(input_size, hidden_size, num_layers, num_classes).to(device)
dec = GRU_DEC(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer_enc = optim.Adam(enc.parameters(), lr=learning_rate)
optimizer_dec = optim.Adam(enc.parameters(), lr=learning_rate)

# Train Network
'''
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
'''
for epoch in range(num_epochs):
    seq_data = None
    seq_targets = None
    loss = None
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, leave=False)):
        # Get data to cuda if possible
        forcing_prob = 0.5

        data = data.to(device=device).squeeze(1)
        data = data.unsqueeze(0)
        targets = targets.to(device=device).unsqueeze(0)
        
        if (batch_idx) % 10 == 0:
            seq_data = data
            seq_targets = targets
        elif (batch_idx) % 10 == 9: 
            # forward
            loss = 0
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            seq_data = torch.cat([seq_data, data], dim=0)
            seq_targets = torch.cat([seq_targets, targets], dim=0)
            #print('seq data size = ', seq_data.size())
            
            enc_h = enc.init_hidden()
            
            for i in range(10):
                enc_out, enc_h = enc(seq_data[i], enc_h)
                print(enc_out.size())
                print(enc_h.size())
            
            use_forcing = True if random.random() < forcing_prob else False

            dec_h = enc_h
            dec_input = torch.zeros([batch_size], device=device)
            dec_input = dec_input.type(torch.cuda.LongTensor)

            if use_forcing:
                for i in range(10):
                    dec_out, dec_h = dec(dec_input, dec_h)
                    dec_input = seq_targets[i]
                    loss += criterion(dec_out, seq_targets[i])
            else:
                for i in range(10):
                    dec_out, dec_h = dec(dec_input, dec_h)
                    loss += criterion(dec_out, seq_targets[i])
                    topv, topi = dec_out.topk(1)
                    dec_input = topi.squeeze().detach()
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
        
        else:
            seq_data = torch.cat([seq_data, data], dim=0)
            seq_targets = torch.cat([seq_targets, targets], dim=0)
    print(loss)

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
