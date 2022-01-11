import torch
import torchvision  
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 4
num_classes = 10
sequence_length = 28
learning_rate = 0.0005
batch_size = 64
num_epochs = 6000
seq_dynamic = True

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

random_seed(777)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(
            x, (h0, c0)
        )          
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out


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
        self.emb = nn.Linear(num_classes, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        input = self.emb(x).view(batch_size, 1, -1)
        out, h_out = self.gru(input, h)

        out = self.softmax(self.fc(out.reshape(out.shape[0], -1)))
        return out, h_out

    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


train_dataset = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

enc = GRU_ENC(input_size, hidden_size, num_layers, num_classes).to(device)
dec = GRU_DEC(input_size, hidden_size, num_layers, num_classes).to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer_enc = optim.Adam(enc.parameters(), lr=learning_rate)
optimizer_dec = optim.Adam(dec.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    seq_data = None
    seq_targets = None
    loss = None
    acc = 0
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, leave=False)):
        # Get data to cuda if possible
        forcing_prob = 0.5
        seq_length = 2
        #seq_length = 5

        data = data.to(device=device).squeeze(1)
        data = data.unsqueeze(0)
        targets = targets.to(device=device).unsqueeze(0)
        
        if (batch_idx) % seq_length == 0:
            seq_data = data
            seq_targets = targets
        elif (batch_idx) % seq_length == seq_length-1: 
            # forward
            loss = 0
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            seq_data = torch.cat([seq_data, data], dim=0)
            seq_targets = torch.cat([seq_targets, targets], dim=0)
            
            enc_h = enc.init_hidden()
            
            for i in range(seq_length):
                enc_out, enc_h = enc(seq_data[i], enc_h)
            
            use_forcing = True if random.random() < forcing_prob else False

            dec_h = enc_h
            dec_input = torch.zeros([batch_size, num_classes], device=device)
            dec_input = dec_input.type(torch.cuda.FloatTensor)
            #dec_input = torch.LongTensor([11]*batch_size).to(device)
            #print(dec_input.size())

            if use_forcing:
                for i in range(seq_length):
                    dec_out, dec_h = dec(dec_input, dec_h)
                    dec_input = F.one_hot(seq_targets[i], num_classes=num_classes).type(torch.cuda.FloatTensor)
                    loss += criterion(dec_out, seq_targets[i])
            else:
                for i in range(seq_length):
                    dec_out, dec_h = dec(dec_input, dec_h)
                    loss += criterion(dec_out, seq_targets[i])
                    #topv, topi = dec_out.topk(1)
                    #dec_input = topi.squeeze().detach()
                    dec_input = dec_out.detach()

            _, top1 = dec_out.max(1)
            acc += (top1 == targets).sum()
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            
            if seq_dynamic:
                seq_length = random.randint(2, 10)
        
        else:
            seq_data = torch.cat([seq_data, data], dim=0)
            seq_targets = torch.cat([seq_targets, targets], dim=0)
    #print(loss)
    print(f'acc = {acc.item()/60000.} %')


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
