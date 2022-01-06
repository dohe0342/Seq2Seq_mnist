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
SOS_token = 0
EOS_token = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out


def cnn_train():
    model = CNN().to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_batch = len(data_loader)
    print('총 배치의 수 : {}'.format(total_batch))

    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        #output, hidden = self.gru(output, hidden)
        #input = input.flatten().view([1, -1, 784])
        #input = input.flatten()

        input = input.view(1, 1, -1)
        
        output, hidden = self.gru(input, hidden)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        #input = input.flatten()
        #output = F.relu(output)
        #output = input.view(1, 1, -1)

        output = F.relu(output)
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
    teacher_forcing_ratio = 1.
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input_length = input_tensor.size(0)
    #target_length = target_tensor.size(0)
    #input_length = len(input_tensor)
    #target_length = len(target_tensor)
    input_length = 10
    target_length = 10

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        #print(ei)
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    #decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.zeros([10], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    target_tensor = F.one_hot(target_tensor, num_classes=10)
    target_tensor = target_tensor.type(torch.FloatTensor)

    if use_teacher_forcing:
        # Teacher forcing 포함: 목표를 다음 입력으로 전달
        for di in range(target_length):
            target = torch.unsqueeze(target_tensor[di], 0)
            #print('target size', target.size())
            #print(target)
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target)
            print(loss)
            print(decoder_output)
            #decoder_input = target  # Teacher forcing
            decoder_input = decoder_output

    else:
        # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
        for di in range(target_length):
            target = torch.unsqueeze(target_tensor[di], 0)
            #print('target size', target.size())
            #print(target)
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden = decoder(
            #    decoder_input, decoder_hidden, False)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리

            #loss += criterion(decoder_output, target_tensor[di])
            loss += criterion(decoder_output, target)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, dataloader, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every 마다 초기화
    plot_loss_total = 0  # plot_every 마다 초기화

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #criterion = nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    
    for iter in range(1, n_iters + 1):
        #input_tensor = training_pair[0]
        #target_tensor = training_pair[1]
        print(iter)
        count = 0
        input_tensor = None
        target_tensor = None
        for data, label in dataloader:
            if count == 0:
                input_tensor = data.flatten().view(-1, 784)
                target_tensor = label.flatten()
            else:
                data = data.flatten().view(-1, 784)
                label = label.flatten()
                input_tensor = torch.cat([input_tensor, data], dim=0)
                target_tensor = torch.cat([target_tensor, label], dim=0)
            
            count += 1 
            
            if count == 10:
                count = 0
                #print('input tensor size1', input_tensor.size())
                input_tensor = input_tensor.view([10, 1, 784])
                #print('input tensor size2', input_tensor.size())
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
    
    #print(encoder1)
    #print(decoder1)
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
    batch_size = 1
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
                                              shuffle=False,
                                              drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    
    #new_train, new_train_data = prepare_mnist(train_loader, test_loader)
    s2s_train(train_loader)
    #model = CNN().to(device)
    #print(model)



if __name__ == '__main__':
    main()
