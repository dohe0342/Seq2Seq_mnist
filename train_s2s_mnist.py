import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import random

MAX_LENGTH=10
batch_size = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        input = input.view(1, batch_size, -1)
        #print('hidden_size = ', hidden.size())
        output, hidden = self.gru(input, hidden)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output = input.view(1, batch_size, -1)
        #output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train_s2s(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    teacher_forcing_ratio = 0.5
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[1]
    batch_size = input_tensor.size()[1]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.zeros([1, batch_size, 10], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    target_tensor = target_tensor.T
    target_tensor = F.one_hot(target_tensor, num_classes=10)
    target_tensor = target_tensor.type(torch.FloatTensor)
    
    if use_teacher_forcing:
        # Teacher forcing 포함: 목표를 다음 입력으로 전달
        for di in range(target_length):
            target = torch.unsqueeze(target_tensor[di], 0)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            
            target_squeeze = target.squeeze()
            loss += criterion(decoder_output, target_squeeze)
            decoder_input = target[:]

    else:
        # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
        for di in range(target_length):
            target = torch.unsqueeze(target_tensor[di], 0)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            print(decoder_output)
            #decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리
            decoder_input = decoder_output.detach()
            target_squeeze = target.squeeze()
            loss += criterion(decoder_output, target_squeeze)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, dataloader, print_every=1000, plot_every=100, learning_rate=0.1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every 마다 초기화
    plot_loss_total = 0  # plot_every 마다 초기화

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #criterion = nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    
    for iter in range(1, n_iters + 1):
        count = 0
        input_tensor = None
        target_tensor = None
        for enum, (data, label) in enumerate(dataloader):
            N, C, H , W = data.size()[0], data.size()[1], data.size()[2], data.size()[3]
            if count == 0:
                input_tensor = data.view(1, N, H*W)
                target_tensor = torch.unsqueeze(label, 1)
            else:
                data = data.view(1, N, H*W)
                label = torch.unsqueeze(label, 1)
                input_tensor = torch.cat([input_tensor, data], dim=0)
                target_tensor = torch.cat([target_tensor, label], dim=1)
            
            count += 1 
            
            if count == 10:
                count = 0
                input_tensor = input_tensor.view([10, N, H*W])
                loss = train_s2s(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)

                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg))

                if iter % plot_every == 0 and 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

    #showPlot(plot_losses)


def s2s_train(dataloader):
    hidden_size = 784
    input_size = 784
    output_size = 10	
    
    encoder1 = EncoderRNN(input_size, hidden_size).to(device)
    decoder1 = DecoderRNN(input_size, output_size).to(device)
    attn_decoder1 = AttnDecoderRNN(input_size, output_size, dropout_p=0.1).to(device)
    
    trainIters(encoder1, decoder1, 75000, dataloader, print_every=5000)
    #trainIters(encoder1, attn_decoder1, 75000, print_every=5000)


def prepare_mnist(train, test):
    count = 0
    batch_size = 10

    new_train = [[], []]
    
    new_data = None
    new_label = None

    for data, label in train:
        if count == 10:
            new_train[0].append(new_data)
            new_train[1].append(new_label)
            count = 0
        
        if count == 0:
            new_data = data
            new_label = label
        else:
            new_data = torch.cat([new_data, data], dim=3)
            new_label = torch.cat([new_label, label])
        
        count += 1
    
    new_batch_data = None
    new_batch_label = None
    count = 0

    new_batch_train = [[], []]
    for new in new_train[0]:
        if count == batch_size:
            new_batch_train[0].append(new_batch_data)
            count = 0
        
        if count == 0:
            new_batch_data = new
        else:
            new_batch_data = torch.cat([new_batch_data, new], dim=0)
        count += 1
    
    count = 0
    for new in new_train[1]:
        if count == batch_size:
            new_batch_train[1].append(new_batch_label)
            count = 0
        
        if count == 0:
            new_batch_label = new
            new_batch_label = new_batch_label.view([-1, 10])
        else:
            new = new_label.view([-1, 10])
            new_batch_label = torch.cat([new_batch_label, new], dim=0)
        count += 1
    
    print('----sequence mnist data shape-----')
    print(new_batch_train[0][1].shape)
    print(new_batch_train[1][1].shape)
    print(len(new_batch_train[0]))

    return new_train, new_batch_train


def main():
    #batch_size = 20
    learning_rate = 0.001
    training_epochs = 15

    mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                              train=True, # True를 지정하면 훈련 데이터로 다운로드
                              transform=transforms.ToTensor(), # 텐서로 변환
                              download=False)

    mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                             train=False, # False를 지정하면 테스트 데이터로 다운로드
                             transform=transforms.ToTensor(), # 텐서로 변환
                             download=False)

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    
    s2s_train(train_loader)


if __name__ == '__main__':
    main()
