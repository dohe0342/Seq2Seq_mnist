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
num_layers = 1
num_classes = 10
sequence_length = 28
learning_rate = 0.0005
batch_size = 64
num_epochs = 6000
MAX_LENGTH = 10
attention = False

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


class GRU_DEC_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_p=0.1):
        super(GRU_DEC_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.emb = nn.Linear(num_classes, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, enc_out):
        print(h.size())
        attn_score = torch.zeros(enc_out.size()[0])
        input = self.emb(x).view(batch_size, 1, -1)
        out, h_out = self.gru(input, h)
        
        attn_h = h_out.view(batch_size, self.num_layers, self.hidden_size)
        for i in range(enc_out.size()[0]):
            attn_score[i] = torch.dot(attn_h.flatten(), enc_out[i].flatten())
        attn_distribution = F.softmax(attn_score, dim=1)
        attn_value = torch.zeros(enc_out.size()[1], enc_out.size()[2], enc_out.size()[3]).to(device)

        for i in range(enc_out.size()[0]):
            attn_value += attn_distribution[i]*enc_out[i]
        attn_value = attn_value.view(batch_size, -1)
        h_out_ = h_out.view(batch_size, -1)
        s_bar = torch.cat([h_out, attn_value])

        out = self.softmax(self.fc(out.reshape(out.shape[0], -1)))
        return out, h_out

    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class GRU_DEC_ATTN(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(GRU_DEC_ATTN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.emb = nn.Linear(output_size, hidden_size*num_layers)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.emb(input).view(batch_size, -1)
        embedded = self.dropout(embedded)

        print(embedded.size())
        print(hidden.size())

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)

        print(encoder_outputs.size())
        attn_weights = attn_weights.T.view(10, 64, 1, 1)
        print(attn_weights.size())

        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)
        print(attn_applied.size())

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


train_dataset = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

enc = GRU_ENC(input_size, hidden_size, num_layers, num_classes).to(device)
if attention:
    dec = GRU_DEC_Attention(input_size, hidden_size, num_layers, num_classes).to(device)
else:
    dec = GRU_DEC(input_size, hidden_size, num_layers, num_classes).to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer_enc = optim.Adam(enc.parameters(), lr=learning_rate)
optimizer_dec = optim.Adam(dec.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
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
            enc_out_list = torch.zeros(MAX_LENGTH, batch_size, sequence_length, hidden_size).to(device)

            for i in range(seq_length):
                enc_out, enc_h = enc(seq_data[i], enc_h)
                enc_out_list[i] = enc_out
            
            use_forcing = True if random.random() < forcing_prob else False

            dec_h = enc_h
            dec_input = torch.zeros([batch_size, num_classes], device=device)

            if use_forcing:
                for i in range(seq_length):
                    if attention:
                        dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                    else:
                        dec_out, dec_h = dec(dec_input, dec_h)
                    dec_input = F.one_hot(seq_targets[i], num_classes=num_classes).type(torch.cuda.FloatTensor)
                    loss += criterion(dec_out, seq_targets[i])
            else:
                for i in range(seq_length):
                    if attention:
                        dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                    else:
                        dec_out, dec_h = dec(dec_input, dec_h)
                    
                    loss += criterion(dec_out, seq_targets[i])
                    dec_input = dec_out.detach()

            _, top1 = dec_out.max(1)
            acc += (top1 == targets).sum()
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            seq_length = random.randint(2, MAX_LENGTH)
        
        else:
            seq_data = torch.cat([seq_data, data], dim=0)
            seq_targets = torch.cat([seq_targets, targets], dim=0)
    #print(loss)
    print(f'corr num = {acc.item()}')
    print(f'acc = {acc.item()/60000.} %')
    print('')


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            #x = x.to(device=device).squeeze(1)
            #y = y.to(device=device)
            data = data.to(device=device).squeeze(1)
            data = data.unsqueeze(0)
            targets = targets.to(device=device).unsqueeze(0)

            seq_length = 2
            #seq_length = 5

            if (batch_idx) % seq_length == 0:
                seq_data = data
                seq_targets = targets
            elif (batch_idx) % seq_length == seq_length-1: 
                loss = 0
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
                
                seq_data = torch.cat([seq_data, data], dim=0)
                seq_targets = torch.cat([seq_targets, targets], dim=0)

                enc_h = enc.init_hidden()
                enc_out_list = torch.zeros(MAX_LENGTH, batch_size, sequence_length, hidden_size).to(device)

                for i in range(seq_length):
                    enc_out, enc_h = enc(seq_data[i], enc_h)
                    enc_out_list[i] = enc_out
                
                use_forcing = True if random.random() < forcing_prob else False

                dec_h = enc_h
                dec_input = torch.zeros([batch_size, num_classes], device=device)

                if use_forcing:
                    for i in range(seq_length):
                        if attention:
                            dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                        else:
                            dec_out, dec_h = dec(dec_input, dec_h)
                        dec_input = F.one_hot(seq_targets[i], num_classes=num_classes).type(torch.cuda.FloatTensor)
                        loss += criterion(dec_out, seq_targets[i])
                else:
                    for i in range(seq_length):
                        if attention:
                            dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                        else:
                            dec_out, dec_h = dec(dec_input, dec_h)
                        
                        loss += criterion(dec_out, seq_targets[i])
                        dec_input = dec_out.detach()

                _, top1 = dec_out.max(1)
                acc += (top1 == targets).sum()
                loss.backward()
                optimizer_enc.step()
                optimizer_dec.step()

                seq_length = random.randint(2, MAX_LENGTH)
            
            else:
                seq_data = torch.cat([seq_data, data], dim=0)
                seq_targets = torch.cat([seq_targets, targets], dim=0)
        #print(loss)
        print(f'corr num = {acc.item()}')
        print(f'acc = {acc.item()/60000.} %')
        print('')

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
